"""Bayesian Neural Network for epistemic uncertainty quantification.

Uses weight distributions instead of point estimates.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from lottery.engine.predictor import BasePredictor
from lottery.utils.logger import logger


class BayesianLinear(nn.Module):
    """
    Bayesian Linear layer with learnable weight distributions.
    
    Uses local reparameterization trick for efficient sampling.
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight mean and log-variance
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.full((out_features, in_features), -5.0))
        
        # Bias mean and log-variance
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_var = nn.Parameter(torch.full((out_features,), -5.0))
        
        # Prior
        self.prior_std = prior_std
        
        # Initialize
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out')
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass with optional sampling.
        
        Args:
            x: Input tensor
            sample: If True, sample weights; else use mean
        """
        if sample and self.training:
            # Sample weights
            weight_std = torch.exp(0.5 * self.weight_log_var)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.exp(0.5 * self.bias_log_var)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from posterior to prior."""
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)
        
        prior_var = self.prior_std ** 2
        
        # KL for weights
        kl_weight = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_var) / prior_var
            - 1
            - self.weight_log_var
            + np.log(prior_var)
        )
        
        # KL for bias
        kl_bias = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_var) / prior_var
            - 1
            - self.bias_log_var
            + np.log(prior_var)
        )
        
        return kl_weight + kl_bias


class LotteryBNN(nn.Module):
    """Bayesian Neural Network for lottery prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.out = BayesianLinear(hidden_dim, 7)  # 6 reds + 1 blue
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = F.relu(self.fc1(x, sample))
        h = F.relu(self.fc2(h, sample))
        return torch.sigmoid(self.out(h, sample))
    
    def kl_divergence(self) -> torch.Tensor:
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.out.kl_divergence()
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with uncertainty estimates."""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_samples, batch, 7]
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


class BNNPredictor(BasePredictor):
    """BNN-based predictor with uncertainty quantification."""
    
    def __init__(self, config: Any = None):
        self.cfg = config or {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = getattr(config, 'window', 10) if config else 10
        self.hidden_dim = getattr(config, 'hidden_dim', 64) if config else 64
        self.epochs = getattr(config, 'epochs', 100) if config else 100
    
    def train(self, df: pd.DataFrame) -> None:
        """Train BNN."""
        logger.info("Training BNN...")
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols + ["blue"]].to_numpy(dtype=float)
        
        # Normalize
        data[:, :6] = data[:, :6] / 33.0
        data[:, 6] = data[:, 6] / 16.0
        
        X = torch.FloatTensor(data[:-1]).to(self.device)
        y = torch.FloatTensor(data[1:]).to(self.device)
        
        input_dim = 7
        self.model = LotteryBNN(input_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        n_samples = len(X)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            pred = self.model(X)
            recon_loss = F.mse_loss(pred, y)
            kl_loss = self.model.kl_divergence() / n_samples
            
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"BNN Epoch {epoch}: Loss {loss.item():.4f}")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions with uncertainty."""
        if self.model is None:
            raise RuntimeError("BNN not trained")
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        last_row = df[red_cols + ["blue"]].iloc[-1].to_numpy(dtype=float)
        
        # Normalize
        last_row[:6] = last_row[:6] / 33.0
        last_row[6] = last_row[6] / 16.0
        
        X = torch.FloatTensor(last_row).unsqueeze(0).to(self.device)
        
        mean, std = self.model.predict_with_uncertainty(X, n_samples=50)
        mean = mean[0].cpu().numpy()
        std = std[0].cpu().numpy()
        
        # Denormalize
        red_pred = np.clip(np.round(mean[:6] * 33), 1, 33).astype(int)
        blue_pred = int(np.clip(np.round(mean[6] * 16), 1, 16))
        
        red_uncertainty = std[:6] * 33
        blue_uncertainty = std[6] * 16
        
        return {
            "red": {i+1: [(int(red_pred[i]), 1.0)] for i in range(6)},
            "blue": [(blue_pred, 1.0)],
            "sum_pred": float(sum(red_pred) + blue_pred),
            "uncertainty": {
                "red": red_uncertainty.tolist(),
                "blue": float(blue_uncertainty)
            }
        }
    
    def save(self, save_dir: str) -> None:
        if self.model:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), Path(save_dir) / "bnn.pt")
    
    def load(self, save_dir: str) -> bool:
        path = Path(save_dir) / "bnn.pt"
        if path.exists():
            self.model = LotteryBNN(7, self.hidden_dim).to(self.device)
            self.model.load_state_dict(torch.load(path, weights_only=True))
            return True
        return False


# PBT Adapter
from lottery.pbt import Member, ModelAdapter
from typing import Tuple

class BNNModelAdapter(ModelAdapter):
    """PBT adapter for Bayesian Neural Network."""
    
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
        
        hidden_dim = cfg.get("hidden_dim", 64)
        model = LotteryBNN(7, hidden_dim).to(self.device)
        if member.model_state is not None:
            model.load_state_dict(member.model_state)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
        
        model.train()
        for _ in range(steps):
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, y) + 0.01 * model.kl_divergence() / len(X)
            loss.backward()
            optimizer.step()
        
        return model.state_dict(), -loss.item()
    
    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        return 0.0
    
    def perturb_config(self, config: Any) -> Any:
        new_cfg = config.copy()
        new_cfg["hidden_dim"] = max(16, config.get("hidden_dim", 64) + np.random.randint(-16, 17))
        new_cfg["lr"] = np.clip(config.get("lr", 1e-3) * np.random.choice([0.8, 1.0, 1.25]), 1e-5, 0.1)
        return new_cfg
    
    def save(self, member: Member, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(member.model_state, path / "bnn_pbt.pt")
    
    def load(self, path: Path) -> Member:
        state = torch.load(path / "bnn_pbt.pt", weights_only=True)
        return Member(config={}, model_state=state)

