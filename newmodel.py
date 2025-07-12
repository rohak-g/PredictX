import os
import torch
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labeled_root = "/path/to/data1"      # 213 labeled netlists
unlabeled_root = "/path/to/data2"    # 20 unlabeled netlists

class NetlistDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.root_dir = root_dir
        self.graph_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))]

    def len(self):
        return len(self.graph_dirs)

    def get(self, idx):
        netlist_dir = self.graph_dirs[idx]

        # ---------- Load Node Features ----------
        node_feat_path = os.path.join(netlist_dir, "node_features.csv")
        node_feat_df = pd.read_csv(node_feat_path)

        # Drop 'node_id' and 'gate_type' columns (already one-hot encoded)
        node_feat_df = node_feat_df.drop(columns=[c for c in node_feat_df.columns
                                                  if 'Unnamed' in c or c in ['node_id', 'gate_type']], errors='ignore')
        node_feat_df = node_feat_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        x = torch.tensor(node_feat_df.values, dtype=torch.float)

        # ---------- Load Adjacency ----------
        adj_path = os.path.join(netlist_dir, "adjacency.csv")
        adj_df = pd.read_csv(adj_path, comment='#', index_col=0)
        adj_df = adj_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        adj_tensor = torch.tensor(adj_df.values, dtype=torch.float)
        edge_index = (adj_tensor > 0).nonzero(as_tuple=False).t().contiguous()

        # ---------- Load Labels ----------
        label_path = os.path.join(netlist_dir, "label.json")
        y = torch.tensor([0.0], dtype=torch.float)  # default delay
        comb_flag = torch.tensor([0], dtype=torch.float)  # 1 if combinational

        if os.path.exists(label_path):
            with open(label_path) as f:
                label = json.load(f)

            cpd = label.get("critical_path_delay")
            slack = label.get("worst_slack")

            if cpd is not None and cpd != "Infinity":
                y = torch.tensor([float(cpd)], dtype=torch.float)
            if slack is None or str(slack).lower() == "infinity":
                comb_flag = torch.tensor([1], dtype=torch.float)

        # ---------- Create Data object ----------
        data = Data(x=x, edge_index=edge_index, y=y)
        data.comb = comb_flag
        return data


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

        # ---------- Load Node Features ----------
        node_feat_path = os.path.join(netlist_dir, "node_features.csv")
        node_feat_df = pd.read_csv(node_feat_path)

        # Drop 'node_id' and 'gate_type' columns (already one-hot encoded)
        node_feat_df = node_feat_df.drop(columns=[c for c in node_feat_df.columns
                                                  if 'Unnamed' in c or c in ['node_id', 'gate_type']], errors='ignore')
        node_feat_df = node_feat_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        x = torch.tensor(node_feat_df.values, dtype=torch.float)

        # ---------- Load Adjacency ----------
        adj_path = os.path.join(netlist_dir, "adjacency.csv")
        adj_df = pd.read_csv(adj_path, comment='#', index_col=0)
        adj_df = adj_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        adj_tensor = torch.tensor(adj_df.values, dtype=torch.float)
        edge_index = (adj_tensor > 0).nonzero(as_tuple=False).t().contiguous()

        # ---------- Create Data object ----------
        data = Data(x=x, edge_index=edge_index)
        return data


# Define GCN encoder and GAE model
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Create GAE with inner-product decoder
encoder = GCNEncoder(in_channels=num_features, hidden_channels=128, latent_channels=64).to(device)
model_gae = GAE(encoder).to(device)
optimizer = torch.optim.Adam(model_gae.parameters(), lr=0.01)

# Pretrain GAE on unlabeled netlists (data2)
for epoch in range(1, 76):
    model_gae.train()
    total_loss = 0
    for data in unlabeled_loader:
        data = data.to(device)
        optimizer.zero_grad()
        z = model_gae.encode(data.x, data.edge_index)
        loss = model_gae.recon_loss(z, data.edge_index)  # BCE loss on edges:contentReference[oaicite:2]{index=2}
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"GAE Epoch {epoch}, Recon Loss: {total_loss/len(unlabeled_loader):.4f}")
# Save pretrained encoder
pretrained_encoder = model_gae.encoder

# Define supervised GNN model
class GNNRegressor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # pretrained GCN
        # Additional GCN layers
        self.gcn1 = GCNConv(64, 64)
        self.gcn2 = GCNConv(64, 64)
        self.gcn3 = GCNConv(64, 64)
        # SGC layer (K-hop aggregation)
        self.sg = SGConv(64, 64, K=4)
        # MLP for final predictions
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out_cpd = nn.Linear(16, 1)   # regression output
        self.out_comb = nn.Linear(16, 1)  # logits for combinational flag

    def forward(self, x, edge_index, batch):
        # Encode with pretrained layers
        z = self.encoder(x, edge_index)
        # Additional GCN layers
        z = F.relu(self.gcn1(z, edge_index))
        z = F.relu(self.gcn2(z, edge_index))
        z = F.relu(self.gcn3(z, edge_index))
        # Apply SGC (propagate K hops)
        z = self.sg(z, edge_index)
        # Global pooling to graph vector
        z = global_max_pool(z, batch)
        # MLP -> outputs
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        cpd_pred = self.out_cpd(z)
        comb_logits = self.out_comb(z)
        return cpd_pred, comb_logits

model = GNNRegressor(pretrained_encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_reg = nn.MSELoss()
loss_cls = nn.BCEWithLogitsLoss()

best_rmse = float('inf')
best_epoch = 0
for epoch in range(1, 151):
    model.train()
    total_loss = 0
    # Training step
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred_cpd, pred_comb = model(data.x, data.edge_index, data.batch)
        true_cpd = data.y.view(-1,1)
        true_comb = data.comb.view(-1,1)
        loss = loss_reg(pred_cpd, true_cpd) + loss_cls(pred_comb, true_comb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        mse_pct_sum, count = 0.0, 0
        for data in test_loader:
            data = data.to(device)
            pred_cpd, pred_comb = model(data.x, data.edge_index, data.batch)
            pred_cpd = pred_cpd.item(); pred_logit = pred_comb.item()
            true_cpd = data.y.item()
            if true_cpd > 0:
                err = (pred_cpd - true_cpd) / true_cpd
                mse_pct_sum += err*err
                count += 1
        rmse_pct = (mse_pct_sum/count)**0.5 if count>0 else 0.0
        if rmse_pct < best_rmse:
            best_rmse, best_epoch = rmse_pct, epoch
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, RMSE%={(100*rmse_pct):.2f}%")
print(f"Best RMSE%={(100*best_rmse):.2f}% at epoch {best_epoch}")
