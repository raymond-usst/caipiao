"""Multi-Task Learning model for joint lottery prediction.

Shared encoder with task-specific heads for red, blue, and sum predictions.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from lottery.engine.predictor import BasePredictor
from lottery.utils.logger import logger


class SharedEncoder(nn.Module):
    """Shared encoder backbone for multi-task learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        return h


class RedHead(nn.Module):
    """Task-specific head for red ball prediction."""
    
    def __init__(self, hidden_dim: int, num_positions: int = 6, num_balls: int = 33):
        super().__init__()
        self.num_positions = num_positions
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_balls) for _ in range(num_positions)
        ])
    
    def forward(self, h: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of 6 tensors, each [batch, 33] probabilities."""
        return [F.softmax(head(h), dim=-1) for head in self.heads]


class BlueHead(nn.Module):
    """Task-specific head for blue ball prediction."""
    
    def __init__(self, hidden_dim: int, num_balls: int = 16):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_balls)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.fc(h), dim=-1)


class SumHead(nn.Module):
    """Task-specific head for sum prediction (regression)."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h).squeeze(-1)


class MultiTaskLotteryModel(nn.Module):
    """
    Multi-task learning model for lottery prediction.
    
    Shares representations across red, blue, and sum prediction tasks.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim)
        self.red_head = RedHead(hidden_dim)
        self.blue_head = BlueHead(hidden_dim)
        self.sum_head = SumHead(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (red_probs, blue_probs, sum_pred)
        """
        h = self.encoder(x)
        red_probs = self.red_head(h)
        blue_probs = self.blue_head(h)
        sum_pred = self.sum_head(h)
        return red_probs, blue_probs, sum_pred


def multitask_loss(
    red_probs: List[torch.Tensor],
    blue_probs: torch.Tensor,
    sum_pred: torch.Tensor,
    red_targets: torch.Tensor,
    blue_targets: torch.Tensor,
    sum_targets: torch.Tensor,
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.1)
) -> torch.Tensor:
    """
    Multi-task loss with task-specific weights.
    
    Args:
        weights: (red_weight, blue_weight, sum_weight)
    """
    # Red loss (cross-entropy for each position)
    red_loss = 0
    for i, probs in enumerate(red_probs):
        targets = red_targets[:, i].long() - 1  # Convert to 0-indexed
        red_loss += F.cross_entropy(probs, targets)
    red_loss /= len(red_probs)
    
    # Blue loss
    blue_loss = F.cross_entropy(blue_probs, blue_targets.long() - 1)
    
    # Sum loss (MSE)
    sum_loss = F.mse_loss(sum_pred, sum_targets)
    
    return weights[0] * red_loss + weights[1] * blue_loss + weights[2] * sum_loss


class MultiTaskPredictor(BasePredictor):
    """Multi-task predictor implementing BasePredictor interface."""
    
    def __init__(self, config: Any = None):
        self.cfg = config or {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = getattr(config, 'window', 10) if config else 10
        self.hidden_dim = getattr(config, 'hidden_dim', 128) if config else 128
        self.epochs = getattr(config, 'epochs', 50) if config else 50
    
    def train(self, df: pd.DataFrame) -> None:
        """Train multi-task model."""
        logger.info("Training Multi-Task Model...")
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        
        # Build features from features module
        from lottery.features import build_features
        features_df = build_features(df)
        
        # Select numeric feature columns
        feature_cols = [c for c in features_df.columns if c not in red_cols + ["blue", "issue", "draw_date"]]
        
        if not feature_cols:
            feature_cols = ["ac", "span", "odd_ratio", "big_ratio"]
        
        X = features_df[feature_cols].fillna(0).to_numpy(dtype=float)
        
        # Targets
        red_targets = df[red_cols].to_numpy(dtype=float)
        blue_targets = df["blue"].to_numpy(dtype=float)
        sum_targets = red_targets.sum(axis=1) + blue_targets
        
        # To tensors
        X_t = torch.FloatTensor(X).to(self.device)
        red_t = torch.FloatTensor(red_targets).to(self.device)
        blue_t = torch.FloatTensor(blue_targets).to(self.device)
        sum_t = torch.FloatTensor(sum_targets).to(self.device)
        
        input_dim = X.shape[1]
        self.model = MultiTaskLotteryModel(input_dim, self.hidden_dim).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            red_probs, blue_probs, sum_pred = self.model(X_t)
            loss = multitask_loss(red_probs, blue_probs, sum_pred, red_t, blue_t, sum_t)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"MultiTask Epoch {epoch}: Loss {loss.item():.4f}")
        
        self.feature_cols = feature_cols
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate multi-task predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        from lottery.features import build_features
        features_df = build_features(df)
        
        X = features_df[self.feature_cols].fillna(0).to_numpy(dtype=float)
        X_t = torch.FloatTensor(X[-1:]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            red_probs, blue_probs, sum_pred = self.model(X_t)
        
        # Get top predictions
        red_pred = {}
        for i, probs in enumerate(red_probs):
            top_idx = probs[0].argmax().item() + 1
            red_pred[i + 1] = [(top_idx, probs[0].max().item())]
        
        blue_top = blue_probs[0].argmax().item() + 1
        blue_prob = blue_probs[0].max().item()
        
        return {
            "red": red_pred,
            "blue": [(blue_top, blue_prob)],
            "sum_pred": sum_pred[0].item()
        }
    
    def save(self, save_dir: str) -> None:
        if self.model:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': self.model.state_dict(),
                'feature_cols': self.feature_cols
            }, Path(save_dir) / "multitask.pt")
    
    def load(self, save_dir: str) -> bool:
        path = Path(save_dir) / "multitask.pt"
        if path.exists():
            data = torch.load(path, weights_only=False)
            self.feature_cols = data['feature_cols']
            self.model = MultiTaskLotteryModel(len(self.feature_cols), self.hidden_dim).to(self.device)
            self.model.load_state_dict(data['model'])
            return True
        return False
