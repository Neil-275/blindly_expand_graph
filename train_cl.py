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
    def __init__(self, llm_embs, rel_labels, gnn_vec, num_negatives=5):
        """
        Args:
            llm_embs: Tensor of shape (total_variations, llm_dim)
            rel_labels: LongTensor of shape (total_variations,) containing the relation ID for each emb
            gnn_vec: Tensor of shape (n_layers, n_rels, gnn_dim)
            num_negatives: Number of negative samples per positive sample
        """
        self.llm_embeddings = llm_embs
        self.rel_labels = rel_labels
        self.gnn_vectors = gnn_vec
        self.num_layers = gnn_vec.shape[0]
        self.num_rels = 474
        self.num_negatives = num_negatives
        self.unique_rel_ids = np.arange(self.num_rels)

    def __len__(self):
        # Now the length is the total number of LLM variations, not just unique relations
        return self.llm_embeddings.shape[0]

    def __getitem__(self, idx):
        # The anchor is a specific textual variation
        anchor_llm = self.llm_embeddings[idx]
        # Find which relation ID this variation belongs to
        anchor_rel_id = self.rel_labels[idx]
        
        # Positive structural target for this relation at all layers
        pos_stack = self.gnn_vectors[:, anchor_rel_id, :]
        
        # Sample K unique negative relation IDs
        neg_rel_ids = []
        while len(neg_rel_ids) < self.num_negatives:
            nid = np.random.choice(self.unique_rel_ids)
            if nid != anchor_rel_id:
                neg_rel_ids.append(nid)
        
        # Gather Negatives for all layers: [K_Negs, Layers, GNN_Dim]
        neg_stacks = torch.stack([self.gnn_vectors[:, nid, :] for nid in neg_rel_ids])
        
        return anchor_llm, pos_stack, neg_stacks

# ---------------------------------------------------------
# 3. Training Function
# ---------------------------------------------------------
def run_alignment_training(llm_data, rel_labels, gnn_data, epochs=200, batch_size=32, lr=5e-5, 
                         num_negatives=5, patience=20, project_name="relation_alignment"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers, _, gnn_dim = gnn_data.shape
    llm_dim = llm_data.shape[1]
    
    wandb.init(project=project_name, config={
        "lr": lr, "num_negatives": num_negatives, "batch_size": batch_size, "total_samples": llm_data.shape[0]
    })
    
    projector = Projector(llm_dim, [512, 256], gnn_dim, num_layers).to(device)
    dataset = RelationAlignmentDataset(llm_data, rel_labels, gnn_data, num_negatives=num_negatives)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    criterion = nn.TripletMarginLoss(margin=1.2, p=2)
    optimizer = optim.AdamW(projector.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        projector.train()
        train_loss = 0
        for anchors, pos_stack, neg_stacks in train_loader:
            anchors, pos_stack, neg_stacks = anchors.to(device), pos_stack.to(device), neg_stacks.to(device)
            optimizer.zero_grad()
            
            total_batch_loss = 0
            for l in range(num_layers):
                proj_a = projector(l, anchors) # [B, GNN_Dim]
                pos = pos_stack[:, l, :]       # [B, GNN_Dim]
                
                # Implementation of Hard Negative Mining: 
                # Instead of averaging all negatives, we focus on the ones closest to the anchor
                layer_neg_losses = []
                for k in range(num_negatives):
                    neg = neg_stacks[:, k, l, :] # [B, GNN_Dim]
                    layer_neg_losses.append(criterion(proj_a, pos, neg))
                
                # Take the top 50% hardest negatives in the batch for this relation
                sorted_losses, _ = torch.sort(torch.stack(layer_neg_losses), descending=True)
                hard_loss = sorted_losses[:max(1, num_negatives // 2)].mean()
                
                total_batch_loss += hard_loss
            
            loss = total_batch_loss / num_layers
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        projector.eval()
        val_loss = 0
        with torch.no_grad():
            for anchors, pos_stack, neg_stacks in val_loader:
                anchors, pos_stack, neg_stacks = anchors.to(device), pos_stack.to(device), neg_stacks.to(device)
                batch_l = 0
                for l in range(num_layers):
                    proj_a = projector(l, anchors)
                    pos = pos_stack[:, l, :]
                    lnl = sum([criterion(proj_a, pos, neg_stacks[:, k, l, :]) for k in range(num_negatives)])
                    batch_l += (lnl / num_negatives)
                val_loss += (batch_l / num_layers).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        wandb.log({"train_loss": avg_train, "val_loss": avg_val})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(projector.state_dict(), "best_projector.pt")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train {avg_train:.4f}, Val {avg_val:.4f}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    wandb.finish()
    return projector

import pickle as pkl

if __name__ == "__main__":
    # Ensure files exist before loading
    try:
        with open("knowledge_graph/KG_data/FB15k-237-betae/id2rel.pkl", "rb") as f:
            id2rel = pkl.load(f)
        rel_labels = []
        for k, rel in id2rel.items():
            rel_labels.extend([k] * 20)
        llm_data = torch.load("llm_description_all_emb.pt")
        gnn_data = torch.from_numpy(np.load("GNN_relation_embeddings.npy")).float()
        trained_projector = run_alignment_training(llm_data, rel_labels, gnn_data)
        
        torch.save(trained_projector.state_dict(), "relation_projector_phase1.pt")
        print("Projector saved successfully.")
    except FileNotFoundError:
        print("Error: .npy or .pkl files not found. Please ensure embeddings and labels are generated first.")