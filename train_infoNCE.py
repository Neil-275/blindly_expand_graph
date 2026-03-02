import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model_2 import Projector
import pickle as pkl
import torch.nn.functional as F

class ContrastiveDataset(Dataset):
    def __init__(self, n_ver, n_rel, data, n_neg = 4):
        self.n_neg = n_neg
        self.n_rel = n_rel
        self.n_ver = n_ver
        
        self.data = torch.stack(list(data.values()), dim=0) ## (474, ver, dim)
        # print("Dataset shape: ", self.data.shape)

    def __len__(self):
        return self.data.shape[0] * self.data.shape[1]

    def __getitem__(self, idx):
        rel_idx = idx // self.n_ver
        ver_idx = idx % self.n_ver

        neg_samps = []
        for _ in range(self.n_neg):
            while True:
                i_neg = torch.randint(0, self.n_rel, (1,)).item()
                if i_neg != rel_idx:
                    i_ver = torch.randint(0, self.n_ver, (1,)).item()
                    neg_samps.append(self.data[i_neg, i_ver, :])
                    break
        
        neg_samps = torch.stack(neg_samps)

        pos_samps = []
        
        while True:
            pos_ver = torch.randint(0, self.n_ver, (1,)).item()
            if pos_ver != ver_idx:
                pos_samps.append(self.data[rel_idx, pos_ver, :])
                break
        pos_samps = torch.stack(pos_samps)

        return self.data[rel_idx, ver_idx].unsqueeze(0), pos_samps, neg_samps
    

def info_nce_loss(anchor, positive, negatives, temp):
    anchor = F.normalize(anchor, dim = -1)
    positive = F.normalize(positive, dim = -1)
    negatives = F.normalize(negatives, dim=-1)

    pos_logits = torch.sum(anchor * positive, dim=-1) / temp

    neg_logits = torch.sum(anchor * negatives, dim=-1) / temp
    
    logits = torch.cat([pos_logits, neg_logits], dim =-1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(anchor.device)

    return F.cross_entropy(logits, labels)


def train(model, train_loader, val_loader, val_only = False):

    device = Training_config.device
    temp = Training_config.temperature
    optimizer = optim.Adam(model.parameters(), lr=Training_config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    model.train()

    patience = 0
    best_loss = torch.inf

    for epoch in range(Training_config.n_epoch):
        epoch_loss = 0
        val_loss = 0
        if val_only == False:
            for data in train_loader:
                anchor, pos, neg = data
                
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                batch_loss = 0 

                for i_layer in range(Projector_config.active_layers):
                    anchor_proj = projector(i_layer, anchor)
                    pos_proj = projector(i_layer, pos)
                    neg_proj = projector(i_layer, neg)

                    # print("shapes: ", anchor_proj.shape, pos_proj.shape, neg_proj.shape)

                    
                    loss = info_nce_loss(anchor_proj, pos_proj, neg_proj, temp)
                    batch_loss += loss
                
                batch_loss /= Projector_config.active_layers
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss

            epoch_loss /= len(train_loader)


        model.eval()

        for data in val_loader:
            
            anchor, pos, neg = data
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # print("Anchor shape:", anchor.shape)
            # print("pos shape:", pos.shape)
            # print("neg shape:", neg.shape)

            batch_loss = 0

            for i_layer in range(Projector_config.active_layers):

                anchor_proj = projector(i_layer, anchor)
                pos_proj = projector(i_layer, pos)
                neg_proj = projector(i_layer, neg)

                
                loss = info_nce_loss(anchor_proj, pos_proj, neg_proj, temp)
                batch_loss += loss

            batch_loss /= Projector_config.active_layers

            val_loss += batch_loss

        val_loss /= len(val_loader)

        print(f"Epoch_{epoch + 1}: Train: {epoch_loss:.3f}, Eval: {val_loss:.3f}")

        if val_only:
            break
        
        if val_loss < best_loss:
            print(f"New best found: {best_loss:.3f} to {val_loss:.3f}")
            best_loss = val_loss
            patience = 0
        

        patience += 1
        
        if patience > Training_config.bearing:
            print("early stopping triggered...")
            break

        scheduler.step(val_loss)

class Training_config:
    lr = 5e-4
    device = "cuda:0"
    train_ratio = 0.01
    n_epoch = 8
    bearing = 8
    temperature = 0.07

class Projector_config:
    in_dim = 4096
    hidden_dims = [512, 256]
    out_dim = 64
    n_layers = 8
    active_layers = 4

if __name__ == "__main__":
    data = pkl.load(open("data_for_CL/llm_description_aligned_emb.pkl", "rb"))

    dataset = ContrastiveDataset(20, len(data), data)
    train_size = int(Training_config.train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # print(dataset[train_size])

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    projector = Projector(
        in_dim = Projector_config.in_dim,
        hidden_dims = Projector_config.hidden_dims,
        out_dim = Projector_config.out_dim,
        n_layers = Projector_config.n_layers
    ).to(Training_config.device)

    # Load pretrained weights
    # checkpoint = torch.load("weights/finetune/vocal-gorge-97.pt", weights_only=False)
    # projector.load_state_dict(checkpoint["projector"])
    checkpoint = torch.load("weights/finetune/vague-planet-123.pt", weights_only=False)
    projector.load_state_dict(checkpoint['projector'])



    train(projector, train_loader, val_loader, val_only=True)





    