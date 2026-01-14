from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from lottery.utils.logger import logger
from lottery.engine.predictor import BasePredictor
from lottery.config import GNNConfig
from lottery.pbt import Member, ModelAdapter

# Simple GCN Layer implementation to avoid heavy dependency if not needed
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [Nodes, In]
        # adj: [Nodes, Nodes] (Normalized)
        x = self.linear(x)
        return torch.matmul(adj, x)


# Graph Attention Layer for dynamic edge weighting
class GATLayer(nn.Module):
    """Graph Attention Network layer with learned attention."""
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_per_head = out_features // n_heads
        
        # Linear transformations for each head
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention parameters [2 * out_per_head] -> 1 for each head
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.out_per_head))
        nn.init.xavier_uniform_(self.a)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            Updated node features [N, out_features]
        """
        N = x.size(0)
        
        # Linear transform
        h = self.W(x)  # [N, out_features]
        h = h.view(N, self.n_heads, self.out_per_head)  # [N, heads, out_per_head]
        
        # Compute attention scores
        # For each pair (i, j), concatenate [h_i || h_j] and apply attention
        h_repeat = h.unsqueeze(1).repeat(1, N, 1, 1)  # [N, N, heads, out_per_head]
        h_repeat_inter = h.unsqueeze(0).repeat(N, 1, 1, 1)  # [N, N, heads, out_per_head]
        
        concat = torch.cat([h_repeat, h_repeat_inter], dim=-1)  # [N, N, heads, 2*out_per_head]
        
        # Attention scores
        e = (concat * self.a).sum(dim=-1)  # [N, N, heads]
        e = self.leaky_relu(e)
        
        # Mask with adjacency (only attend to neighbors)
        mask = (adj > 0).unsqueeze(-1)  # [N, N, 1]
        e = e.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attention = F.softmax(e, dim=1)  # [N, N, heads]
        attention = self.dropout(attention)
        
        # Aggregate
        h_prime = torch.einsum('ijh,jhf->ihf', attention, h)  # [N, heads, out_per_head]
        
        return h_prime.reshape(N, -1)  # [N, out_features]


class LotteryGNN(nn.Module):
    def __init__(self, num_nodes=35, hidden_dim=64, num_layers=2, dropout=0.5, use_gat=True):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes + 1, hidden_dim)  # 1-based indexing
        self.layers = nn.ModuleList()
        self.use_gat = use_gat
        
        for _ in range(num_layers):
            if use_gat:
                self.layers.append(GATLayer(hidden_dim, hidden_dim, n_heads=4, dropout=dropout))
            else:
                self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        # Predict probability for each node being selected
        self.out = nn.Linear(hidden_dim, 1) 

    def forward(self, x, adj):
        # x: Node indices [0, 1, ..., 34]
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, adj)
            h = F.relu(h)
            h = self.dropout(h)
        return torch.sigmoid(self.out(h)).squeeze(-1)  # [Nodes]

class GNNPredictor(BasePredictor):
    def __init__(self, config: GNNConfig):
        super().__init__(config)
        self.cfg = config
        self.model = None
        self.adj_matrix = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_graph(self, df: pd.DataFrame):
        """Build co-occurrence graph with weighted edges and temporal decay."""
        # Size 35 to include blue ball (node 34 = blue ball range 1-16 mapped to single node)
        co_counts = np.zeros((35, 35))
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols].to_numpy(dtype=int)
        blue_data = df["blue"].to_numpy(dtype=int) if "blue" in df.columns else None
        
        n = len(data)
        decay_rate = 0.99  # Temporal decay factor
        
        for t, row in enumerate(data):
            # Temporal weight: more recent = higher weight
            weight = decay_rate ** (n - t - 1)
            
            # Red-Red edges
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    u, v = row[i], row[j]
                    co_counts[u, v] += weight
                    co_counts[v, u] += weight
            
            # Red-Blue edges (blue mapped to node 34)
            if blue_data is not None:
                for r in row:
                    co_counts[r, 34] += weight
                    co_counts[34, r] += weight
                    
        # Apply log transformation to normalize weights
        adj = np.log1p(co_counts)
        np.fill_diagonal(adj, 1.0)  # Self-loop
        
        # Normalize: D^-0.5 A D^-0.5
        rowsum = adj.sum(axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        self.adj_matrix = torch.FloatTensor(norm_adj).to(self.device)

    def train(self, df: pd.DataFrame) -> None:
        logger.info("Training GNN...")
        self._build_graph(df)
        
        # Check resume
        save_dir = "models/GNN" # Matches pipeline convention
        loaded = False
        if self.cfg.resume and not self.cfg.fresh:
            if self.load(save_dir):
                logger.info(f"Resumed GNN model from {save_dir}")
                loaded = True
        
        if not loaded:
            self.model = LotteryGNN(
                num_nodes=35,  # 1-33 red + node 0 unused + node 34 blue
                hidden_dim=self.cfg.hidden_channels, 
                num_layers=self.cfg.num_layers, 
                dropout=self.cfg.dropout
            ).to(self.device)
        
        self.model.train() # Ensure training mode
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        criterion = nn.BCELoss()
        
        # Prepare targets: For each draw, target is boolean vector of length 34
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols].to_numpy(dtype=int)
        
        # We can train to predict "next" draw based on graph structure?
        # Or simpler: Is this node likely to be active?
        # Let's frame it as: Predict the probability of each number appearing in the NEXT draw, 
        # driven by the static graph structure + node features (maybe embedding learns "current heat").
        # Limitation: Static graph doesn't capture sequence. 
        # Improvement: Input could be node features representing "recent frequency".
        
        # Dynamic Features: Last 10 frequency
        # For simplicity in this GNN demo, we'll train with a static embedding 
        # but using 'recent' draws to optimize the embedding for 'next' prediction.
        
        x = torch.arange(35).to(self.device) # Nodes 0-34 (node 34 = blue)
        
        # Training loop
        self.model.train()
        for epoch in range(self.cfg.epochs):
            total_loss = 0
            # Sample batch of draws
            indices = np.random.choice(len(data), size=min(len(data), 64), replace=False)
            
            for idx in indices:
                target_nums = data[idx]
                target = torch.zeros(35).to(self.device)
                target[target_nums] = 1.0
                # Mask 0 (unused)
                target[0] = 0
                
                optimizer.zero_grad()
                pred = self.model(x, self.adj_matrix)
                
                # loss computation (ignore index 0, and handle dimension matching)
                loss = criterion(pred[1:34], target[1:34])  # Only red balls 1-33
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 100 == 0:
                logger.info(f"GNN Epoch {epoch}: Loss {total_loss/len(indices):.4f}")

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
             from lottery.utils.exceptions import ModelNotFittedError
             raise ModelNotFittedError("GNN model not trained")
        
        # In a real dynamic GNN, we would update node features based on 'df'.
        # Here we rely on the learned embeddings and graph structure.
        
        self.model.eval()
        with torch.no_grad():
            x = torch.arange(35).to(self.device)
            # Use stored adj matrix (assuming graph structure is relatively stable or built from train_df)
            if self.adj_matrix is None:
                self._build_graph(df)
            
            probs = self.model(x, self.adj_matrix).cpu().numpy()[1:34] # 1-33 (red balls)
            
        # Top red predictions
        top_indices = np.argsort(probs)[::-1]
        top_reds = [(i + 1, float(probs[i])) for i in top_indices[:6]]
        
        # GNN doesn't predict Blue in this simple logic. Return generic blue or random.
        # Or we can build a separate graph for blue. For now, empty blue or frequency based?
        # Let's return frequency-based blue fallback.
        blue_counts = df["blue"].value_counts()
        best_blue = blue_counts.idxmax() if not blue_counts.empty else 1
        
        return {
            "red": {i: top_reds for i in range(6)}, # Format mismatch with others? 
            # Others return Dict[pos, List], here we just have a bag of numbers.
            # We will supply the same list for all positions to indicate "set prediction"
            "flat_reds": top_reds, 
            "blue": [(best_blue, 1.0)],
            "sum_pred": 100 # Dummy
        }

    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "gnn.pt"
        torch.save(self.model.state_dict(), file_path)

    def load(self, save_dir: str) -> bool:
        path = Path(save_dir) / "gnn.pt"
        if not path.exists(): return False
        try:
            self.model = LotteryGNN(
                num_nodes=35, 
                hidden_dim=self.cfg.hidden_channels, 
                num_layers=self.cfg.num_layers, 
                dropout=self.cfg.dropout
            ).to(self.device)
            self.model.load_state_dict(torch.load(path))
            return True
        except Exception:
            return False
# PBT Adapter
class GNNModelAdapter(ModelAdapter):
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        
    def train_step(self, member: Member, dataset: pd.DataFrame, steps: int) -> Tuple[Any, float]:
        """Train GNN for 'steps' epochs and return (new_model_state, loss)."""
        cfg = member.config
        
        # Build model
        model = LotteryGNN(
            num_nodes=35,
            hidden_dim=cfg.get("hidden_channels", 64),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.5)
        ).to(self.device)
        
        # Load existing state if available
        if member.model_state is not None:
            model.load_state_dict(member.model_state)
        
        # Build graph
        co_counts = np.zeros((35, 35))
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = dataset[red_cols].to_numpy(dtype=int)
        n = len(data)
        decay_rate = 0.99
        
        for t, row in enumerate(data):
            weight = decay_rate ** (n - t - 1)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    u, v = row[i], row[j]
                    co_counts[u, v] += weight
                    co_counts[v, u] += weight
                    
        adj = np.log1p(co_counts)
        np.fill_diagonal(adj, 1.0)
        rowsum = adj.sum(axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        adj_matrix = torch.FloatTensor(norm_adj).to(self.device)
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.01))
        criterion = nn.BCELoss()
        x = torch.arange(35).to(self.device)
        
        model.train()
        total_loss = 0.0
        for _ in range(steps):
            indices = np.random.choice(len(data), size=min(len(data), 32), replace=False)
            for idx in indices:
                target_nums = data[idx]
                target = torch.zeros(35).to(self.device)
                target[target_nums] = 1.0
                
                optimizer.zero_grad()
                pred = model(x, adj_matrix)
                loss = criterion(pred[1:], target[1:])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return model.state_dict(), total_loss / (steps * 32)
    
    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        """Evaluate by computing Top6 hit rate on last 10% of data."""
        cfg = member.config
        model = LotteryGNN(num_nodes=35, hidden_dim=cfg.get("hidden_channels", 64),
                           num_layers=cfg.get("num_layers", 2), dropout=cfg.get("dropout", 0.5)).to(self.device)
        if member.model_state is not None:
            model.load_state_dict(member.model_state)
        
        model.eval()
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = dataset[red_cols].to_numpy(dtype=int)
        
        # Use last 10%
        n_test = max(1, len(data) // 10)
        test_data = data[-n_test:]
        
        hits = 0
        x = torch.arange(35).to(self.device)
        # Assume adj_matrix needs to be rebuilt (simplify: static)
        adj = torch.eye(35).to(self.device)  # Simplified for eval
        
        with torch.no_grad():
            for row in test_data:
                probs = model(x, adj).cpu().numpy()[1:34]
                top6 = np.argsort(probs)[::-1][:6] + 1
                hits += len(set(top6) & set(row))
        
        return hits / (n_test * 6)  # Hit rate
    
    def perturb_config(self, config: Any) -> Any:
        """Randomly perturb hyperparameters."""
        new_config = config.copy()
        new_config["hidden_channels"] = max(16, config.get("hidden_channels", 64) + np.random.randint(-16, 17))
        new_config["num_layers"] = max(1, min(4, config.get("num_layers", 2) + np.random.randint(-1, 2)))
        new_config["dropout"] = np.clip(config.get("dropout", 0.5) + np.random.uniform(-0.1, 0.1), 0.1, 0.7)
        new_config["lr"] = np.clip(config.get("lr", 0.01) * np.random.choice([0.8, 1.0, 1.25]), 1e-4, 0.1)
        return new_config
    
    def save(self, member: Member, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(member.model_state, path / "gnn_pbt.pt")
    
    def load(self, path: Path) -> Member:
        state = torch.load(path / "gnn_pbt.pt", weights_only=True)
        return Member(config={}, model_state=state)

