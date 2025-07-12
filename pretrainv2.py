# ------------------ pretrain_gae.py ------------------
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, VGAE
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch import sigmoid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

# ---------- Device ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnlabeledNetlistDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.graph_dirs = sorted(
            [os.path.join(root_dir, d) for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]
        )
        if not self.graph_dirs:
            raise ValueError(f"No valid graph folders in {root_dir}")

    def len(self):
        return len(self.graph_dirs)

    def get(self, idx):
        folder = self.graph_dirs[idx]
        # load node features
        df = pd.read_csv(os.path.join(folder, 'node_features.csv'))
        # drop non-numeric
        df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c or c in ['node_id', 'gate_type']], errors='ignore')
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        # add degree features
        # load adjacency
        adj = pd.read_csv(os.path.join(folder, 'adjacency.csv'), comment='#', index_col=0)
        adj = adj.apply(pd.to_numeric, errors='coerce').fillna(0)
        in_deg = torch.tensor(adj.sum(axis=0).values, dtype=torch.float)
        out_deg = torch.tensor(adj.sum(axis=1).values, dtype=torch.float)
        # normalize features
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
        x = torch.tensor(scaled, dtype=torch.float)
        # append degree
        x = torch.cat([x, in_deg.unsqueeze(1), out_deg.unsqueeze(1)], dim=1)
        # edge_index
        A = torch.tensor(adj.values, dtype=torch.float)
        edge_index = (A > 0).nonzero(as_tuple=False).t().contiguous()
        return Data(x=x, edge_index=edge_index)

# paths & loader
root = r"C:/Users/gupta/Desktop/vlsi/delay_predictor/Dataset/ordered/data2"
unlabeled = UnlabeledNetlistDataset(root)
loader = DataLoader(unlabeled, batch_size=1, shuffle=True)

# feature size
feat_dim = unlabeled[0].x.size(1)

# ---------- Stronger Encoder: 2-layer GAT ----------
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid_dim, heads=4, concat=False)
        self.gat2 = GATConv(hid_dim, lat_dim * 2, heads=1)
    def forward(self, x, edge_index):
        h = F.elu(self.gat1(x, edge_index))
        out = self.gat2(h, edge_index)
        mu, logstd = out.chunk(2, dim=-1)
        return mu, logstd

encoder = GATEncoder(feat_dim, 128, 64).to(device)
model = VGAE(encoder).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# training
epochs = 100
print(f"Pretraining VGAE with GAT encoder for {epochs} epochs")
start = time.time()
for ep in range(1, epochs+1):
    model.train()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        # Encode to latent embeddings z directly
        z = model.encode(data.x, data.edge_index)
        # Reconstruction + KL
        loss = model.recon_loss(z, data.edge_index) + (ep/50 if ep<50 else 1.0) * model.kl_loss()
        loss.backward()
        opt.step()
        loss_all += loss.item()
        count += 1
    print(f"Epoch {ep}/{epochs} - Loss: {loss_all/count:.4f} - Time: {time.time()-start:.1f}s")

# Save pretrained encoder
torch.save(model.encoder.state_dict(), 'pretrained_encoder.pt')
print("Saved pretrained_encoder.pt")
print("Saved pretrained_encoder.pt")

# visualization & ROC
os.makedirs('plots', exist_ok=True)
print("\n--- Visualization & ROC ---")
for i in range(min(5, len(unlabeled))):
    data = unlabeled[i].to(device)
    model.eval()
    with torch.no_grad():
        mu, logstd = model.encode(data.x, data.edge_index)
        z = model.reparametrize(mu, logstd)
        # sharpen
        logits = (z @ z.t()) * 2.0
        prob = torch.sigmoid(logits).cpu().numpy()
    # true
    N = data.num_nodes
    true = torch.zeros(N,N)
    true[data.edge_index[0], data.edge_index[1]] = 1
    true = true.numpy().flatten()
    # ROC
    auc = roc_auc_score(true, prob.flatten())
    # plot
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
    ax1.imshow(true.reshape(N,N), cmap='Greys'); ax1.set_title('True')
    ax2.imshow(prob, cmap='viridis'); ax2.set_title(f'Prob (AUC={auc:.2f})')
    fig.savefig(f'plots/graph_{i}_compare.png')
    plt.close(fig)
    print(f"Graph {i} AUC={auc:.2f}, saved plots/graph_{i}_compare.png")

# Summary of changes:
# 1) Enhanced node features by adding in-/out-degree and MinMax normalization
# 2) Switched to GATConv encoder for directional relation modeling
# 3) Used VGAE with sharper dot-product decoding (temperature scale=2.0)
# 4) Gradual KL-weight schedule for better latent regularization
# 5) Evaluate with ROC-AUC and save combined visualizations

