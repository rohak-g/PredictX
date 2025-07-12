import os
import torch
import json
import random
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SGConv, global_max_pool

# ---------- Device ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Dataset Definition ----------
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

        label_path = os.path.join(netlist_dir, "label.json")
        y = torch.tensor([0.0], dtype=torch.float)
        comb_flag = torch.tensor([0], dtype=torch.float)

        if os.path.exists(label_path):
            with open(label_path) as f:
                label = json.load(f)
            cpd = label.get("critical_path_delay")
            slack = label.get("worst_slack")
            if cpd is not None and cpd != "Infinity":
                y = torch.tensor([float(cpd)], dtype=torch.float)
            if slack is None or str(slack).lower() == "infinity":
                comb_flag = torch.tensor([1], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.comb = comb_flag
        return data

# ---------- Load Dataset ----------
labeled_root = "/path/to/data1"
labeled_dataset = NetlistDataset(labeled_root)

# Determine num_features
num_features = labeled_dataset[0].x.shape[1]

# ---------- Data Split ----------
indices = list(range(len(labeled_dataset)))
random.shuffle(indices)
split_idx = int(0.85 * len(indices))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

train_dataset = [labeled_dataset[i] for i in train_indices]
test_dataset = [labeled_dataset[i] for i in test_indices]

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------- GCN Encoder Definition (Must Match Pretraining) ----------
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Load pretrained encoder
encoder = GCNEncoder(in_channels=num_features, hidden_channels=128, latent_channels=64).to(device)
encoder.load_state_dict(torch.load("pretrained_encoder.pt"))

# ---------- GNN Regressor ----------
class GNNRegressor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.gcn1 = GCNConv(64, 64)
        self.gcn2 = GCNConv(64, 64)
        self.gcn3 = GCNConv(64, 64)
        self.sg = SGConv(64, 64, K=4)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out_cpd = nn.Linear(16, 1)
        self.out_comb = nn.Linear(16, 1)

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index)
        z = F.relu(self.gcn1(z, edge_index))
        z = F.relu(self.gcn2(z, edge_index))
        z = F.relu(self.gcn3(z, edge_index))
        z = self.sg(z, edge_index)
        z = global_max_pool(z, batch)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        cpd_pred = self.out_cpd(z)
        comb_logits = self.out_comb(z)
        return cpd_pred, comb_logits

# ---------- Train Model ----------
model = GNNRegressor(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_reg = nn.MSELoss()
loss_cls = nn.BCEWithLogitsLoss()

best_rmse = float('inf')
best_epoch = 0
for epoch in range(1, 151):
    model.train()
    total_loss = 0
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
                mse_pct_sum += err * err
                count += 1
        rmse_pct = (mse_pct_sum / count)**0.5 if count > 0 else 0.0
        if rmse_pct < best_rmse:
            best_rmse, best_epoch = rmse_pct, epoch
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, RMSE%={(100*rmse_pct):.2f}%")

print(f"Best RMSE%={(100*best_rmse):.2f}% at epoch {best_epoch}")
