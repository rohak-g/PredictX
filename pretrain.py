# ------------------ pretrain_gae.py ------------------
import os
import torch
import json
import random
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GAE

# ---------- Device ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnlabeledNetlistDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.root_dir = root_dir
        self.graph_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))]

    def len(self):
        return len(self.graph_dirs)

    def get(self, idx):
        netlist_dir = self.graph_dirs[idx]
        node_feat_path = os.path.join(netlist_dir, "node_features.csv")
        node_feat_df = pd.read_csv(node_feat_path)
        node_feat_df = node_feat_df.drop(columns=[c for c in node_feat_df.columns
                                                  if 'Unnamed' in c or c in ['node_id', 'gate_type']], errors='ignore')
        node_feat_df = node_feat_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        x = torch.tensor(node_feat_df.values, dtype=torch.float)

        adj_path = os.path.join(netlist_dir, "adjacency.csv")
        adj_df = pd.read_csv(adj_path, comment='#', index_col=0)
        adj_df = adj_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        adj_tensor = torch.tensor(adj_df.values, dtype=torch.float)
        edge_index = (adj_tensor > 0).nonzero(as_tuple=False).t().contiguous()

        return Data(x=x, edge_index=edge_index)

# Load dataset
unlabeled_root = "C:/Users/gupta/Desktop/vlsi/delay_predictor/Dataset/ordered/data2"
unlabeled_dataset = UnlabeledNetlistDataset(unlabeled_root)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=True)

# Determine num_features
num_features = unlabeled_dataset[0].x.shape[1]

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

encoder = GCNEncoder(in_channels=num_features, hidden_channels=128, latent_channels=64).to(device)
model_gae = GAE(encoder).to(device)
optimizer = torch.optim.Adam(model_gae.parameters(), lr=0.01)

# GAE Pretraining Loop
if __name__ == '__main__':
    for epoch in range(1, 76):
        model_gae.train()
        total_loss = 0
        for data in unlabeled_loader:
            data = data.to(device)
            optimizer.zero_grad()
            z = model_gae.encode(data.x, data.edge_index)
            loss = model_gae.recon_loss(z, data.edge_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"GAE Epoch {epoch}, Recon Loss: {total_loss/len(unlabeled_loader):.4f}")

    torch.save(model_gae.encoder.state_dict(), "pretrained_encoder.pt")
