"""Echo State Network (ESN) for chaotic time series prediction.

Reservoir computing approach designed for complex temporal dynamics.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from lottery.engine.predictor import BasePredictor
from lottery.utils.logger import logger


class EchoStateNetwork:
    """
    Echo State Network for time series prediction.
    
    Uses a fixed random reservoir with trained output weights.
    Particularly suited for chaotic dynamics with positive Lyapunov exponent.
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 500,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.1,
        leak_rate: float = 0.3,
        sparsity: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize ESN.
        
        Args:
            input_dim: Number of input features
            reservoir_size: Number of reservoir neurons
            spectral_radius: Controls memory/chaos trade-off
            input_scaling: Scaling of input weights
            leak_rate: Leaky integration constant
            sparsity: Sparsity of reservoir connections
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        
        # Input weights
        self.W_in = (np.random.rand(reservoir_size, input_dim) - 0.5) * 2 * input_scaling
        
        # Reservoir weights (sparse)
        W = np.random.rand(reservoir_size, reservoir_size) - 0.5
        mask = np.random.rand(reservoir_size, reservoir_size) > sparsity
        W[mask] = 0
        
        # Scale to spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 0:
            W = W * (spectral_radius / rho)
        self.W = W
        
        # Output weights (to be trained)
        self.W_out = None
        
        # Reservoir state
        self.state = np.zeros(reservoir_size)
    
    def _update(self, x: np.ndarray) -> np.ndarray:
        """Update reservoir state with input x."""
        pre_activation = np.dot(self.W_in, x) + np.dot(self.W, self.state)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre_activation)
        return self.state
    
    def fit(self, X: np.ndarray, y: np.ndarray, reg: float = 1e-6) -> 'EchoStateNetwork':
        """
        Train output weights using ridge regression.
        
        Args:
            X: Input sequences [n_samples, input_dim]
            y: Targets [n_samples, output_dim]
            reg: Regularization coefficient
        
        Returns:
            self
        """
        n_samples = len(X)
        
        # Collect reservoir states
        states = np.zeros((n_samples, self.reservoir_size))
        self.state = np.zeros(self.reservoir_size)
        
        for i in range(n_samples):
            states[i] = self._update(X[i])
        
        # Ridge regression for output weights
        # W_out = y^T * states * (states^T * states + reg * I)^-1
        R = np.dot(states.T, states) + reg * np.eye(self.reservoir_size)
        P = np.dot(states.T, y)
        self.W_out = np.linalg.solve(R, P)
        
        return self
    
    def predict(self, X: np.ndarray, reset_state: bool = False) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input sequences [n_samples, input_dim]
            reset_state: Whether to reset reservoir state
            
        Returns:
            Predictions [n_samples, output_dim]
        """
        if self.W_out is None:
            raise RuntimeError("Model not trained")
        
        if reset_state:
            self.state = np.zeros(self.reservoir_size)
        
        predictions = []
        for x in X:
            state = self._update(x)
            pred = np.dot(state, self.W_out)
            predictions.append(pred)
        
        return np.array(predictions)


class ESNPredictor(BasePredictor):
    """ESN-based predictor implementing BasePredictor interface."""
    
    def __init__(self, config: Any = None):
        self.cfg = config or {}
        self.esn = None
        self.window = getattr(config, 'window', 20) if config else 20
        self.reservoir_size = getattr(config, 'reservoir_size', 500) if config else 500
    
    def train(self, df: pd.DataFrame) -> None:
        """Train ESN on historical data."""
        logger.info("Training ESN...")
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols + ["blue"]].to_numpy(dtype=float)
        
        # Normalize
        data[:, :6] = data[:, :6] / 33.0
        data[:, 6] = data[:, 6] / 16.0
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i:i+self.window].flatten())
            y.append(data[i+self.window])
        
        X = np.array(X)
        y = np.array(y)
        
        input_dim = X.shape[1]
        self.esn = EchoStateNetwork(
            input_dim=input_dim,
            reservoir_size=self.reservoir_size,
            spectral_radius=0.95,
            leak_rate=0.3
        )
        self.esn.fit(X, y)
        logger.info("ESN training complete")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate ESN predictions."""
        if self.esn is None:
            raise RuntimeError("ESN not trained")
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols + ["blue"]].to_numpy(dtype=float)
        
        # Normalize
        data[:, :6] = data[:, :6] / 33.0
        data[:, 6] = data[:, 6] / 16.0
        
        # Use last window
        X = data[-self.window:].flatten().reshape(1, -1)
        pred = self.esn.predict(X, reset_state=False)[0]
        
        # Denormalize
        red_pred = np.clip(np.round(pred[:6] * 33), 1, 33).astype(int)
        blue_pred = int(np.clip(np.round(pred[6] * 16), 1, 16))
        
        return {
            "red": {i+1: [(int(red_pred[i]), 1.0)] for i in range(6)},
            "blue": [(blue_pred, 1.0)],
            "sum_pred": float(sum(red_pred) + blue_pred)
        }
    
    def save(self, save_dir: str) -> None:
        if self.esn:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            np.savez(
                Path(save_dir) / "esn.npz",
                W_in=self.esn.W_in,
                W=self.esn.W,
                W_out=self.esn.W_out
            )
    
    def load(self, save_dir: str) -> bool:
        path = Path(save_dir) / "esn.npz"
        if path.exists():
            data = np.load(path)
            input_dim = data['W_in'].shape[1]
            reservoir_size = data['W_in'].shape[0]
            self.esn = EchoStateNetwork(input_dim, reservoir_size)
            self.esn.W_in = data['W_in']
            self.esn.W = data['W']
            self.esn.W_out = data['W_out']
            return True
        return False


# PBT Adapter
from lottery.pbt import Member, ModelAdapter
from typing import Tuple

class ESNModelAdapter(ModelAdapter):
    """PBT adapter for Echo State Network."""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def train_step(self, member: Member, dataset: pd.DataFrame, steps: int) -> Tuple[Any, float]:
        cfg = member.config
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = dataset[red_cols + ["blue"]].to_numpy(dtype=float)
        data[:, :6] = data[:, :6] / 33.0
        data[:, 6] = data[:, 6] / 16.0
        
        X, y = [], []
        for i in range(len(data) - self.window):
            X.append(data[i:i+self.window].flatten())
            y.append(data[i+self.window])
        X, y = np.array(X), np.array(y)
        
        esn = EchoStateNetwork(
            input_dim=X.shape[1],
            reservoir_size=cfg.get("reservoir_size", 500),
            spectral_radius=cfg.get("spectral_radius", 0.95),
            leak_rate=cfg.get("leak_rate", 0.3)
        )
        esn.fit(X, y, reg=cfg.get("reg", 1e-6))
        
        # Evaluate
        preds = esn.predict(X[-50:])
        mse = np.mean((preds - y[-50:]) ** 2)
        
        state = {"W_in": esn.W_in, "W": esn.W, "W_out": esn.W_out}
        return state, -mse  # Negative MSE as fitness
    
    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        return 0.0  # Use train_step fitness
    
    def perturb_config(self, config: Any) -> Any:
        new_cfg = config.copy()
        new_cfg["reservoir_size"] = max(100, config.get("reservoir_size", 500) + np.random.randint(-100, 101))
        new_cfg["spectral_radius"] = np.clip(config.get("spectral_radius", 0.95) + np.random.uniform(-0.05, 0.05), 0.5, 0.99)
        new_cfg["leak_rate"] = np.clip(config.get("leak_rate", 0.3) + np.random.uniform(-0.1, 0.1), 0.1, 0.9)
        return new_cfg
    
    def save(self, member: Member, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "esn_pbt.npz", **member.model_state)
    
    def load(self, path: Path) -> Member:
        data = np.load(path / "esn_pbt.npz")
        return Member(config={}, model_state=dict(data))

