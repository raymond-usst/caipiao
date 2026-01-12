"""Meta-Learning and Regime Detection for adaptive prediction.

Detects distributional shifts and adapts model accordingly.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque

from lottery.engine.predictor import BasePredictor
from lottery.utils.logger import logger


class RegimeDetector:
    """
    Detects regime changes using statistical tests.
    
    Uses exponentially weighted statistics and CUSUM-like detection.
    """
    
    def __init__(self, window: int = 50, threshold: float = 2.0, alpha: float = 0.1):
        """
        Args:
            window: Baseline window size
            threshold: Z-score threshold for regime change
            alpha: Exponential decay for running statistics
        """
        self.window = window
        self.threshold = threshold
        self.alpha = alpha
        
        self.running_mean = None
        self.running_var = None
        self.baseline_mean = None
        self.baseline_var = None
        self.history = deque(maxlen=window)
        self.regime_changes = []
    
    def update(self, value: float) -> Tuple[bool, float]:
        """
        Update detector with new value.
        
        Args:
            value: New observation
            
        Returns:
            (is_regime_change, z_score)
        """
        self.history.append(value)
        
        # Initialize baseline
        if len(self.history) < self.window:
            return False, 0.0
        
        if self.baseline_mean is None:
            self.baseline_mean = np.mean(list(self.history))
            self.baseline_var = np.var(list(self.history)) + 1e-6
            self.running_mean = self.baseline_mean
            self.running_var = self.baseline_var
            return False, 0.0
        
        # Update running statistics
        self.running_mean = self.alpha * value + (1 - self.alpha) * self.running_mean
        diff_sq = (value - self.running_mean) ** 2
        self.running_var = self.alpha * diff_sq + (1 - self.alpha) * self.running_var
        
        # Compute z-score relative to baseline
        z_score = abs(self.running_mean - self.baseline_mean) / np.sqrt(self.baseline_var)
        
        is_change = z_score > self.threshold
        
        if is_change:
            # Reset baseline to current regime
            self.baseline_mean = self.running_mean
            self.baseline_var = self.running_var
            self.regime_changes.append(len(self.regime_changes))
        
        return is_change, z_score
    
    def get_regime_count(self) -> int:
        return len(self.regime_changes)


class MAMLBase(nn.Module):
    """Simple base model for MAML-style meta-learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))


class MetaLearningPredictor(BasePredictor):
    """
    Meta-learning predictor with regime detection.
    
    Combines online learning with fast adaptation to regime changes.
    """
    
    def __init__(self, config: Any = None):
        self.cfg = config or {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regime_detector = RegimeDetector(window=50, threshold=2.0)
        
        self.inner_lr = 0.01  # Fast adaptation learning rate
        self.outer_lr = 0.001  # Meta learning rate
        self.adapt_steps = 5  # Fast adaptation steps
        
        self.recent_data = deque(maxlen=100)
    
    def train(self, df: pd.DataFrame) -> None:
        """Initial training on historical data."""
        logger.info("Training Meta-Learning model...")
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols + ["blue"]].to_numpy(dtype=float)
        
        # Normalize
        data[:, :6] = data[:, :6] / 33.0
        data[:, 6] = data[:, 6] / 16.0
        
        # Store for online updates
        for row in data:
            self.recent_data.append(row)
        
        X = torch.FloatTensor(data[:-1]).to(self.device)
        y = torch.FloatTensor(data[1:]).to(self.device)
        
        self.model = MAMLBase(input_dim=7, hidden_dim=32, output_dim=7).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        
        # Initial training
        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
        
        logger.info("Meta-Learning initial training complete")
    
    def _fast_adapt(self, support_X: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """Create adapted model from support set."""
        # Clone model for task-specific adaptation
        adapted_model = MAMLBase(7, 32, 7).to(self.device)
        adapted_model.load_state_dict(self.model.state_dict())
        
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        adapted_model.train()
        for _ in range(self.adapt_steps):
            optimizer.zero_grad()
            pred = adapted_model(support_X)
            loss = F.mse_loss(pred, support_y)
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions with regime-aware adaptation."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        last_row = df[red_cols + ["blue"]].iloc[-1].to_numpy(dtype=float)
        
        # Check for regime change
        row_sum = last_row[:6].sum()
        is_change, z_score = self.regime_detector.update(row_sum)
        
        if is_change:
            logger.info(f"Regime change detected (z={z_score:.2f}), adapting model...")
        
        # Normalize
        last_row_norm = last_row.copy()
        last_row_norm[:6] = last_row_norm[:6] / 33.0
        last_row_norm[6] = last_row_norm[6] / 16.0
        
        # Update recent data
        self.recent_data.append(last_row_norm)
        
        # Fast adaptation using recent data
        if len(self.recent_data) > 10:
            recent = np.array(list(self.recent_data))
            support_X = torch.FloatTensor(recent[:-1]).to(self.device)
            support_y = torch.FloatTensor(recent[1:]).to(self.device)
            
            adapted_model = self._fast_adapt(support_X, support_y)
        else:
            adapted_model = self.model
        
        # Predict
        adapted_model.eval()
        X = torch.FloatTensor(last_row_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = adapted_model(X)[0].cpu().numpy()
        
        # Denormalize
        red_pred = np.clip(np.round(pred[:6] * 33), 1, 33).astype(int)
        blue_pred = int(np.clip(np.round(pred[6] * 16), 1, 16))
        
        return {
            "red": {i+1: [(int(red_pred[i]), 1.0)] for i in range(6)},
            "blue": [(blue_pred, 1.0)],
            "sum_pred": float(sum(red_pred) + blue_pred),
            "regime_info": {
                "is_change": is_change,
                "z_score": z_score,
                "n_regimes": self.regime_detector.get_regime_count()
            }
        }
    
    def save(self, save_dir: str) -> None:
        if self.model:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), Path(save_dir) / "meta.pt")
    
    def load(self, save_dir: str) -> bool:
        path = Path(save_dir) / "meta.pt"
        if path.exists():
            self.model = MAMLBase(7, 32, 7).to(self.device)
            self.model.load_state_dict(torch.load(path, weights_only=True))
            return True
        return False


# PBT Adapter
from lottery.pbt import Member, ModelAdapter
from typing import Tuple

class MetaModelAdapter(ModelAdapter):
    """PBT adapter for Meta-Learning model."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
    
    def train_step(self, member: Member, dataset: pd.DataFrame, steps: int) -> Tuple[Any, float]:
        cfg = member.config
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = dataset[red_cols + ["blue"]].to_numpy(dtype=float)
        data[:, :6] = data[:, :6] / 33.0
        data[:, 6] = data[:, 6] / 16.0
        
        X = torch.FloatTensor(data[:-1]).to(self.device)
        y = torch.FloatTensor(data[1:]).to(self.device)
        
        hidden_dim = cfg.get("hidden_dim", 32)
        model = MAMLBase(7, hidden_dim, 7).to(self.device)
        if member.model_state is not None:
            model.load_state_dict(member.model_state)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("outer_lr", 1e-3))
        
        model.train()
        for _ in range(steps):
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
        
        return model.state_dict(), -loss.item()
    
    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        return 0.0
    
    def perturb_config(self, config: Any) -> Any:
        new_cfg = config.copy()
        new_cfg["hidden_dim"] = max(16, config.get("hidden_dim", 32) + np.random.randint(-8, 9))
        new_cfg["outer_lr"] = np.clip(config.get("outer_lr", 1e-3) * np.random.choice([0.8, 1.0, 1.25]), 1e-5, 0.1)
        new_cfg["inner_lr"] = np.clip(config.get("inner_lr", 0.01) * np.random.choice([0.8, 1.0, 1.25]), 1e-4, 0.5)
        return new_cfg
    
    def save(self, member: Member, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(member.model_state, path / "meta_pbt.pt")
    
    def load(self, path: Path) -> Member:
        state = torch.load(path / "meta_pbt.pt", weights_only=True)
        return Member(config={}, model_state=state)

