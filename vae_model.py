"""Variational Autoencoder (VAE) for generative lottery modeling.

Learns the latent distribution of lottery outcomes and can generate
novel lottery-like sequences.
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
from lottery.config import load_config
from lottery.utils.logger import logger


class Encoder(nn.Module):
    """Encoder network: maps input to latent distribution parameters."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network: maps latent vector to reconstructed output."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc_out(h))  # Normalized output [0, 1]


class LotteryVAE(nn.Module):
    """
    Variational Autoencoder for lottery data.
    
    Input: [red1, red2, ..., red6, blue] normalized to [0, 1]
    Output: Reconstructed lottery numbers
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through stochastic layer."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def sample(self, n_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """Sample new lottery numbers from the latent space."""
        z = torch.randn(n_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples


def vae_loss(recon: torch.Tensor, x: torch.Tensor, 
             mu: torch.Tensor, logvar: torch.Tensor, 
             beta: float = 1.0) -> torch.Tensor:
    """VAE loss = Reconstruction loss + KL divergence."""
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


class VAEPredictor(BasePredictor):
    """VAE-based predictor implementing BasePredictor interface."""
    
    def __init__(self, config: Any = None):
        self.cfg = config or {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = getattr(config, 'hidden_dim', 64) if config else 64
        self.latent_dim = getattr(config, 'latent_dim', 16) if config else 16
        self.epochs = getattr(config, 'epochs', 100) if config else 100
    
    def train(self, df: pd.DataFrame) -> None:
        """Train VAE on historical lottery data."""
        logger.info("Training VAE...")
        
        # Prepare data: normalize to [0, 1]
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols + ["blue"]].to_numpy(dtype=float)
        data[:, :6] = data[:, :6] / 33.0  # Red balls
        data[:, 6] = data[:, 6] / 16.0    # Blue ball
        
        X = torch.FloatTensor(data).to(self.device)
        
        self.model = LotteryVAE(
            input_dim=7,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon, mu, logvar = self.model(X)
            loss = vae_loss(recon, X, mu, logvar)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"VAE Epoch {epoch}: Loss {loss.item():.4f}")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions by sampling from VAE."""
        if self.model is None:
            raise RuntimeError("VAE not trained")
        
        self.model.eval()
        with torch.no_grad():
            # Sample multiple and take mode/mean
            samples = self.model.sample(100, self.device).cpu().numpy()
        
        # Denormalize
        samples[:, :6] = samples[:, :6] * 33
        samples[:, 6] = samples[:, 6] * 16
        
        # Round and clip to valid ranges
        red_samples = np.clip(np.round(samples[:, :6]), 1, 33).astype(int)
        blue_samples = np.clip(np.round(samples[:, 6]), 1, 16).astype(int)
        
        # Find most common predictions
        from collections import Counter
        red_pred = []
        for i in range(6):
            counts = Counter(red_samples[:, i])
            red_pred.append(counts.most_common(1)[0][0])
        
        blue_counts = Counter(blue_samples)
        blue_pred = blue_counts.most_common(1)[0][0]
        
        return {
            "red": {i+1: [(red_pred[i], 1.0)] for i in range(6)},
            "blue": [(blue_pred, 1.0)],
            "sum_pred": float(sum(red_pred) + blue_pred),
            "sampled": True
        }
    
    def save(self, save_dir: str) -> None:
        if self.model:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), Path(save_dir) / "vae.pt")
    
    def load(self, save_dir: str) -> bool:
        path = Path(save_dir) / "vae.pt"
        if path.exists():
            self.model = LotteryVAE(7, self.hidden_dim, self.latent_dim).to(self.device)
            self.model.load_state_dict(torch.load(path, weights_only=True))
            return True
        return False
