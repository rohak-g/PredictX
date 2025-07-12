import os
import json
import math
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset ----------------
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
        edge_index = (torch.tensor(adj_df.values, dtype=torch.float) > 0).nonzero(as_tuple=False).t().contiguous()

        label_path = os.path.join(netlist_dir, "label.json")
        y = torch.tensor([0.0], dtype=torch.float)
        if os.path.exists(label_path):
            with open(label_path) as f:
                label = json.load(f)
            cpd = label.get("critical_path_delay")
            if cpd is not None and cpd != "Infinity":
                y = torch.tensor([float(cpd)], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data

def compute_topological_levels(data):
    num_nodes = data.num_nodes
    indegree = torch.zeros(num_nodes, dtype=torch.int)
    for src, dst in data.edge_index.t():
        indegree[dst] += 1
    queue = [i for i in range(num_nodes) if indegree[i] == 0]
    levels = torch.zeros(num_nodes)
    while queue:
        u = queue.pop(0)
        for i, (src, dst) in enumerate(data.edge_index.t()):
            if src == u:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst.item())
                    levels[dst] = levels[u] + 1
    return levels

def encode_levels(levels, max_level=None, num_freqs=3):
    if max_level is None:
        max_level = float(levels.max().item()) + 1e-5
    levels = levels.unsqueeze(1) / max_level
    features = [levels]
    for i in range(num_freqs):
        freq = 2.0 ** i
        features.append(torch.sin(math.pi * freq * levels))
        features.append(torch.cos(math.pi * freq * levels))
    return torch.cat(features, dim=1)

# ---------------- Model ----------------
class GATRegressor(torch.nn.Module):
    def __init__(self, input_dim, level_enc_dim, hidden_dim=64, heads=4, num_layers=3):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim + level_enc_dim, hidden_dim)

        self.gat_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))

        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        for gat in self.gat_layers:
            residual = x
            x = F.elu(gat(x, edge_index))
            x = x + residual
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)

# ---------------- Train ----------------
def train():
    print("Preparing dataset and loading graphs...")
    root = r"C:/Users/gupta/Desktop/vlsi/delay_predictor/Dataset/ordered/data1"
    dataset = NetlistDataset(root)
    data_list = []

    for i in range(len(dataset)):
        try:
            data = dataset.get(i)
            levels = compute_topological_levels(data)
            level_enc = encode_levels(levels, num_freqs=3)
            data.x = torch.cat([data.x, level_enc], dim=1)
            data_list.append(data)
        except Exception as e:
            print(f"Skipping graph {i} due to error: {e}")

    print(f"Loaded {len(data_list)} graphs.")
    if len(data_list) == 0:
        raise ValueError("No valid circuit graphs loaded. Check directory paths and filenames.")

    print("Splitting into train and test sets...")
    random_split = torch.randperm(len(data_list))
    split = int(0.8 * len(data_list))
    train_data = [data_list[i] for i in random_split[:split]]
    test_data = [data_list[i] for i in random_split[split:]]

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4)

    in_dim = data_list[0].x.shape[1]
    model = GATRegressor(input_dim=in_dim, level_enc_dim=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("Starting training loop...")
    for epoch in range(1, 51):  # Run for 50 epochs only
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"[Epoch {epoch}] Train MSE: {total_loss / len(train_data):.4f}")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            all_preds = []
            all_targets = []
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch)
                all_preds.append(pred.cpu())
                all_targets.append(batch.y.cpu())
                total_loss += F.mse_loss(pred, batch.y, reduction="sum").item()
            test_mse = total_loss / len(test_data)
            print(f"          Test MSE: {test_mse:.4f}")

    print("Evaluating final test accuracy...")
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    relative_error = torch.abs(all_preds - all_targets) / (all_targets + 1e-8)
    accuracy_within_5_percent = (relative_error <= 0.05).float().mean().item() * 100

    print("\n================ Final Test Accuracy ================")
    print(f"Final Test MSE: {test_mse:.4f}")
    print(f"Final Test RMSE: {test_mse ** 0.5:.4f}")
    print(f"Accuracy within ±5%: {accuracy_within_5_percent:.2f}%\n")

    torch.save(model.state_dict(), "gat_regressor.pt")
    print("Model saved as gat_regressor.pt")

if __name__ == "__main__":
    train()