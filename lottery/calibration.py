"""Probabilistic Calibration for model predictions.

Provides Temperature Scaling and Platt Scaling for post-hoc probability calibration.
"""
from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize


class TemperatureScaling:
    """
    Temperature Scaling for probability calibration.
    
    Divides logits by a learned temperature T > 0 before softmax.
    Useful for neural network outputs.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.fitted = False
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> 'TemperatureScaling':
        """
        Fit temperature on validation set.
        
        Args:
            logits: Pre-softmax logits, shape (n_samples,) or (n_samples, n_classes)
            labels: True labels, shape (n_samples,)
            
        Returns:
            self
        """
        def nll_loss(T):
            """Negative log likelihood with temperature."""
            scaled = logits / T
            if scaled.ndim == 1:
                # Binary case
                probs = 1.0 / (1.0 + np.exp(-scaled))
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            else:
                # Multi-class
                exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
                probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        
        result = minimize(nll_loss, x0=1.0, method='Nelder-Mead',
                         options={'maxiter': 100})
        self.temperature = max(result.x[0], 0.01)  # Ensure T > 0
        self.fitted = True
        return self
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        Note: For proper calibration, input should be logits, then apply softmax.
        This version approximates by converting probs -> logits -> scale -> softmax.
        
        Args:
            probs: Probability values [0, 1]
            
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            return probs
        
        # Convert to logits (inverse sigmoid)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        
        # Scale
        scaled_logits = logits / self.temperature
        
        # Back to probabilities
        return 1.0 / (1.0 + np.exp(-scaled_logits))


class PlattScaling:
    """
    Platt Scaling for probability calibration.
    
    Fits a logistic regression on the model's outputs.
    Works well for SVM-style classifiers.
    """
    
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', max_iter=200)
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """
        Fit Platt scaling on validation set.
        
        Args:
            scores: Model scores/logits, shape (n_samples,)
            labels: True labels, shape (n_samples,)
            
        Returns:
            self
        """
        # Reshape for sklearn
        X = scores.reshape(-1, 1)
        self.model.fit(X, labels)
        self.fitted = True
        return self
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to scores.
        
        Args:
            scores: Model scores/logits
            
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            return scores
        
        X = scores.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


def calibrate_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
    method: str = "temperature"
) -> Tuple[np.ndarray, object]:
    """
    Convenience function to calibrate predictions.
    
    Args:
        probs: Raw probabilities from model
        labels: True labels for calibration
        method: 'temperature' or 'platt'
        
    Returns:
        (calibrated_probs, calibrator)
    """
    if method == "temperature":
        calibrator = TemperatureScaling()
        # Convert probs to logits for fitting
        probs_clip = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs_clip / (1 - probs_clip))
        calibrator.fit(logits, labels)
        calibrated = calibrator.calibrate(probs)
    elif method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(probs, labels)
        calibrated = calibrator.calibrate(probs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return calibrated, calibrator


def compute_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities
        labels: True binary labels
        n_bins: Number of bins for histogram
        
    Returns:
        ECE score (lower is better)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.sum() * np.abs(bin_acc - bin_conf)
    
    return ece / len(probs)
