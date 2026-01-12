import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import numpy as np
import wandb
from model_2 import Projector


# ---------------------------------------------------------
# 2. Dataset for Alignment
# ---------------------------------------------------------
class RelationAlignmentDataset(Dataset):
    def __init__(self, llm_emb, gnn_vec):
        """
        Args:
            llm_emb_dict: {rel_id: tensor(llm_dim)}
            gnn_vec_dict: {rel_id: tensor(gnn_dim)} (Pretrained GNN weights)
        """
        # self.rel_ids = list(llm_emb.keys())
        self.num_rels = llm_emb.shape[0]
        self.llm_embeddings = llm_emb
        self.gnn_vectors = gnn_vec
        self.num_layers = gnn_vec.shape[0]  # Assuming shape is (n_layers, n_rels, gnn_dim)
        self.rel_ids = np.arange(self.num_rels)

    def __len__(self):
        return len(self.rel_ids)

    def __getitem__(self, idx):
        anchor_id = self.rel_ids[idx]
        
        # Common anchor for all layers
        anchor_llm = self.llm_embeddings[anchor_id]
        
        # Gather Positives and Negatives for EACH layer
        pos_list = []
        neg_list = []
        
        # Determine a negative relation ID (same for all layers in this triplet)
        neg_id = anchor_id
        while neg_id == anchor_id:
            neg_id = np.random.choice(self.rel_ids)
        
        for layer_idx in range(self.num_layers):
            # Positive at this layer
            pos_list.append(self.gnn_vectors[layer_idx, anchor_id])
            # Negative at this layer
            neg_list.append(self.gnn_vectors[layer_idx, neg_id])
        
        return anchor_llm, torch.stack(pos_list), torch.stack(neg_list)

# ---------------------------------------------------------
# 3. Training Function
# ---------------------------------------------------------
def run_alignment_training(llm_data, gnn_data, epochs=100, batch_size=32, lr=1e-4, 
                         patience=20, min_delta=1e-4, project_name="relation_alignment",
                         run_name=None, val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = gnn_data.shape[0]
    llm_dim = llm_data.shape[1]
    gnn_dim = gnn_data.shape[2]
    # print(llm_data.shape, gnn_data.shape)
    # Initialize wandb
    wandb.init(project=project_name, name=run_name, config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_layers": num_layers,
        "llm_dim": llm_dim,
        "gnn_dim": gnn_dim,
        "patience": patience,
        "min_delta": min_delta,
        "val_split": val_split
    })
    
    # Using dimensions from the data: 5120 (LLM) and 64 (GNN)
    projector = Projector(llm_dim, [1024, 512], gnn_dim, num_layers).to(device)
    dataset = RelationAlignmentDataset(llm_data, gnn_data)
    
    # Split dataset for validation
    dataset_size = len(dataset)
    print(f"Total dataset size: {dataset_size}")
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(projector.parameters(), lr=lr)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Starting Multi-Layer Alignment ({num_layers} layers) on {device}...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        projector.train()
        train_loss = 0
        
        for anchors, pos_stack, neg_stack in train_loader:
            anchors = anchors.to(device) # (B, llm_dim)
            pos_stack = pos_stack.to(device) # (B, L, gnn_dim)
            neg_stack = neg_stack.to(device) # (B, L, gnn_dim)
            
            optimizer.zero_grad()
            
            layer_losses = 0
            for l_idx in range(num_layers):
                # Project for this layer
                # print(anchors.shape)
                proj_a = projector(l_idx, anchors)
                
                p = pos_stack[:, l_idx, :]
                n = neg_stack[:, l_idx, :]
                
                layer_losses += criterion(proj_a, p, n)
            
            loss = layer_losses / num_layers
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        projector.eval()
        val_loss = 0
        with torch.no_grad():
            for anchors, pos_stack, neg_stack in val_loader:
                anchors = anchors.to(device)
                pos_stack = pos_stack.to(device)
                neg_stack = neg_stack.to(device)
                
                layer_losses = 0
                for l_idx in range(num_layers):
                    proj_a = projector(l_idx, anchors)
                    p = pos_stack[:, l_idx, :]
                    n = neg_stack[:, l_idx, :]
                    layer_losses += criterion(proj_a, p, n)
                
                loss = layer_losses / num_layers
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "patience_counter": patience_counter
        })
        
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in projector.state_dict().items()}
            print(f"Epoch [{epoch+1}/{epochs}] - New best validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state is not None:
        projector.load_state_dict(best_model_state)
        print("Restored best model weights")
    
    wandb.finish()
    return projector

# ---------------------------------------------------------
# Example Usage (Mock Data)
# ---------------------------------------------------------
if __name__ == "__main__":
    llm_data = torch.from_numpy(np.load("LLM_relation_embeddings.npy")).float()
    gnn_data = torch.from_numpy(np.load("GNN_relation_embeddings.npy")).float()
    trained_projector = run_alignment_training(llm_data, gnn_data)
    
    # Save the projector for Phase 2
    torch.save(trained_projector.state_dict(), "relation_projector_phase1.pt")
    print("Projector saved successfully.")