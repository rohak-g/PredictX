import os
import json
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, SGConv, global_mean_pool, GAE
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import warnings

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Data Loading and Preprocessing ---

class NetlistDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.graph_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))]

    def len(self):
        return len(self.graph_dirs)

    def get(self, idx):
        netlist_dir = self.graph_dirs[idx]
        
        # Node features
        node_feat_path = os.path.join(netlist_dir, 'node_features.csv')
        node_feat_df = pd.read_csv(node_feat_path)
        if 'node_id' in node_feat_df.columns:
            node_feat_df = node_feat_df.drop(columns=['node_id'])
        node_feat_df = node_feat_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        x = torch.tensor(node_feat_df.values, dtype=torch.float)

        # Adjacency matrix
        adj_path = os.path.join(netlist_dir, 'adjacency.csv')
        adj_df = pd.read_csv(adj_path, index_col=0)
        edge_index = torch.tensor(np.array(adj_df.to_records().tolist()), dtype=torch.long).t().contiguous()


        # Labels
        label_path = os.path.join(netlist_dir, 'label.json')
        y = torch.tensor([0.0, 0.0], dtype=torch.float) # Default: [slack, delay]
        is_combinational = False
        if os.path.exists(label_path):
            with open(label_path) as f:
                label_dict = json.load(f)
            worst_slack = label_dict.get('worst_slack')
            critical_path_delay = label_dict.get('critical_path_delay')

            if worst_slack == 'Infinity':
                is_combinational = True
            
            # We only care about critical_path_delay for the model's prediction
            if critical_path_delay is not None and critical_path_delay != 'Infinity':
                 y = torch.tensor([float(critical_path_delay)], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.is_combinational = is_combinational
        
        # Add logic depth calculation here to avoid re-computing
        data.logic_depth = self._calculate_logic_depth(data)
        
        return data
    
    def _calculate_logic_depth(self, data):
        G = nx.DiGraph()
        G.add_edges_from(data.edge_index.t().tolist())
        if nx.is_directed_acyclic_graph(G):
            try:
                return nx.algorithms.dag.dag_longest_path_length(G)
            except Exception:
                return 1 # Fallback
        return 3 # Fallback for non-DAG


# --- 2. Model Architecture ---

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class DelayPredictor(nn.Module):
    def __init__(self, encoder, hidden_dim, sgc_k_max=4):
        super().__init__()
        self.encoder = encoder
        self.sgc_k_max = sgc_k_max

        # The input dimension to the first GCN layer is the output of the encoder
        encoder_out_dim = hidden_dim

        self.conv1 = GCNConv(encoder_out_dim, hidden_dim * 2)
        self.conv2 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.conv3 = GCNConv(hidden_dim * 4, hidden_dim * 2)
        
        # SGC layer - its K is dynamic, so we just define the layer itself here
        self.sgc = SGConv(hidden_dim * 2, hidden_dim)

        # Dense layers for regression
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, 1) # Output is a single value (delay)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Pre-trained Encoder
        x = self.encoder(x, edge_index)

        # 2. GCN layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()

        # 3. Normalization and SGC layer
        x = F.normalize(x, p=2, dim=1)
        
        k = min(self.sgc_k_max, data.logic_depth)
        self.sgc.K = k # Dynamically set K
        x = self.sgc(x, edge_index)
        
        # 4. Pooling
        x = global_mean_pool(x, batch)

        # 5. Dense network
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        
        return x

# --- 3. Training and Evaluation ---

def pretrain_gae(gae_model, data_loader, epochs=75):
    print("--- Starting GAE Pre-training ---")
    optimizer = torch.optim.Adam(gae_model.parameters(), lr=0.01)
    
    for epoch in range(1, epochs + 1):
        gae_model.train()
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            z = gae_model.encode(data.x, data.edge_index)
            loss = gae_model.recon_loss(z, data.edge_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"GAE Epoch: {epoch:02d}, Loss: {avg_loss:.4f}")
    print("--- GAE Pre-training Finished ---\n")

def train_regressor(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)

def test_regressor(model, loader):
    model.eval()
    total_rmse_percentage = 0
    num_samples = 0
    with torch.no_grad():
        for data in loader:
            pred = model(data)
            # Ensure actual is not zero to avoid division by zero
            actual = data.y
            # Clamp actual to a small positive number if it's zero
            actual = torch.clamp(actual, min=1e-6)
            
            # Calculate RMSE percentage for the batch
            rmse_perc = torch.sqrt(torch.mean(((pred - actual) / actual) ** 2))
            total_rmse_percentage += rmse_perc.item() * data.num_graphs
            num_samples += data.num_graphs

    return total_rmse_percentage / num_samples if num_samples > 0 else 0


if __name__ == '__main__':
    # --- Dummy Data Generation (for demonstration) ---
    def generate_dummy_data(root, num_graphs, has_labels):
        if not os.path.exists(root):
            os.makedirs(root)
        for i in range(num_graphs):
            graph_dir = os.path.join(root, f'graph_{i}')
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
            
            num_nodes = np.random.randint(5, 50) # smaller for demo
            num_features = 10

            # Node features
            features = pd.DataFrame(np.random.rand(num_nodes, num_features),
                                    columns=[f'f_{j}' for j in range(num_features)])
            features.to_csv(os.path.join(graph_dir, 'node_features.csv'), index=False)
            
            # Adjacency
            adj = pd.DataFrame(np.random.randint(0, 2, size=(num_nodes, num_nodes)))
            # ensure no self loops for simplicity in demo
            np.fill_diagonal(adj.values, 0)
            adj.to_csv(os.path.join(graph_dir, 'adjacency.csv'))

            if has_labels:
                is_comb = np.random.rand() > 0.5
                delay = np.random.uniform(1.0, 9.0)
                slack = 10 - delay if not is_comb else 'Infinity'
                label = {'worst_slack': slack, 'critical_path_delay': delay}
                with open(os.path.join(graph_dir, 'label.json'), 'w') as f:
                    json.dump(label, f)

    # Generate dummy data if directories don't exist
    # if not os.path.exists('data1'):
    #     print("Generating dummy labeled data (data1)...")
    #     generate_dummy_data('data1', 213, True)
    # if not os.path.exists('data2'):
    #     print("Generating dummy unlabeled data (data2)...")
    #     generate_dummy_data('data2', 20, False)


    # --- Main Execution ---
    
    # 1. Load unlabeled data for GAE pre-training
    unlabeled_dataset = NetlistDataset(root_dir='data2')
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=10, shuffle=True)

    # Determine input feature dimension from the first graph
    num_node_features = unlabeled_dataset[0].num_node_features
    ENCODER_OUT_DIM = 64 # The dimensionality of the learned embeddings

    # 2. Initialize and pre-train the GAE
    encoder_model = GCNEncoder(num_node_features, ENCODER_OUT_DIM)
    gae = GAE(encoder_model)
    pretrain_gae(gae, unlabeled_loader, epochs=75)

    # 3. Load labeled data for supervised training
    full_dataset = NetlistDataset(root_dir='data1')
    
    # Split data: 85% train, 15% test
    train_indices, test_indices = train_test_split(list(range(len(full_dataset))), test_size=0.15, random_state=42)
    train_dataset = full_dataset[train_indices]
    test_dataset = full_dataset[test_indices]
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 4. Initialize the final regressor model with the pre-trained encoder
    predictor_model = DelayPredictor(encoder=gae.encoder, hidden_dim=ENCODER_OUT_DIM)

    # 5. Supervised training loop
    optimizer = torch.optim.Adam(predictor_model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_epoch = -1
    
    print("\n--- Starting Supervised Training for Delay Prediction ---")
    for epoch in range(1, 151):
        train_loss = train_regressor(predictor_model, train_loader)
        test_rmse_perc = test_regressor(predictor_model, test_loader)
        
        print(f"Epoch: {epoch:03d}, Train Loss (MSE): {train_loss:.4f}, Test RMSE %: {test_rmse_perc*100:.2f}%")
        
        if test_rmse_perc < best_loss:
            best_loss = test_rmse_perc
            best_epoch = epoch
            # You would typically save your best model here
            # torch.save(predictor_model.state_dict(), 'best_delay_model.pth')
            
    print("\n--- Supervised Training Finished ---")
    print(f"Best Test RMSE Percentage: {best_loss*100:.2f}% at Epoch {best_epoch}")

    # --- Example Prediction and Slack Calculation ---
    print("\n--- Example Prediction ---")
    predictor_model.eval()
    sample_data = test_dataset[0] 
    
    predicted_delay = predictor_model(sample_data).item()
    
    is_comb = sample_data.is_combinational
    if is_comb:
        predicted_slack = "Infinity"
    else:
        predicted_slack = 10.0 - predicted_delay

    print(f"Sample Graph: Test data point 0")
    print(f"Is Combinational: {is_comb}")
    print(f"Actual Critical Path Delay: {sample_data.y.item():.4f}")
    print(f"Predicted Critical Path Delay: {predicted_delay:.4f}")
    print(f"Calculated Slack: {predicted_slack if isinstance(predicted_slack, str) else f'{predicted_slack:.4f}'}")