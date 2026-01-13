"""Proper Scoring Rules for probabilistic prediction evaluation.

Implements Brier Score, Log Loss, and CRPS for calibrated uncertainty evaluation.
"""
from typing import Union, List
import numpy as np


def brier_score(probs: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Brier Score for binary classification.
    
    Brier Score = mean((probs - actuals)^2)
    Lower is better (0 is perfect).
    
    Args:
        probs: Predicted probabilities [0, 1]
        actuals: Binary outcomes (0 or 1)
        
    Returns:
        Brier score
    """
    probs = np.asarray(probs)
    actuals = np.asarray(actuals)
    return float(np.mean((probs - actuals) ** 2))


def brier_skill_score(probs: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Brier Skill Score (improvement over climatology).
    
    BSS = 1 - BS / BS_clim
    where BS_clim is the Brier score of always predicting the base rate.
    
    Returns:
        BSS in (-inf, 1], where 1 is perfect and 0 is no skill
    """
    bs = brier_score(probs, actuals)
    base_rate = np.mean(actuals)
    bs_clim = base_rate * (1 - base_rate)
    
    if bs_clim == 0:
        return 0.0
    
    return 1 - bs / bs_clim


def log_loss(probs: np.ndarray, actuals: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute Log Loss (cross-entropy) for binary classification.
    
    LogLoss = -mean(y * log(p) + (1-y) * log(1-p))
    Lower is better.
    
    Args:
        probs: Predicted probabilities [0, 1]
        actuals: Binary outcomes (0 or 1)
        eps: Small value to avoid log(0)
        
    Returns:
        Log loss
    """
    probs = np.clip(np.asarray(probs), eps, 1 - eps)
    actuals = np.asarray(actuals)
    
    return float(-np.mean(
        actuals * np.log(probs) + (1 - actuals) * np.log(1 - probs)
    ))


def crps(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Continuous Ranked Probability Score for regression.
    
    Simplified CRPS for point forecasts with Gaussian assumption.
    
    For true CRPS with ensemble forecasts, use:
    CRPS = mean(|forecast - actual|)
    
    Args:
        forecasts: Point predictions
        actuals: Actual values
        
    Returns:
        CRPS score (lower is better)
    """
    forecasts = np.asarray(forecasts)
    actuals = np.asarray(actuals)
    return float(np.mean(np.abs(forecasts - actuals)))


def crps_ensemble(ensemble: np.ndarray, actual: float) -> float:
    """
    Compute CRPS for an ensemble forecast.
    
    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    
    Args:
        ensemble: Array of ensemble predictions
        actual: Single actual value
        
    Returns:
        CRPS score
    """
    ensemble = np.asarray(ensemble)
    n = len(ensemble)
    
    # E[|X - y|]
    term1 = np.mean(np.abs(ensemble - actual))
    
    # E[|X - X'|] using pairwise differences
    pairwise = np.abs(ensemble[:, None] - ensemble[None, :])
    term2 = 0.5 * np.sum(pairwise) / (n * n)
    
    return float(term1 - term2)


def hit_rate(predictions: List[int], actuals: List[int]) -> float:
    """
    Simple hit rate for lottery predictions.
    
    Args:
        predictions: List of predicted numbers
        actuals: List of actual numbers
        
    Returns:
        Proportion of correct predictions
    """
    pred_set = set(predictions)
    actual_set = set(actuals)
    
    if len(actual_set) == 0:
        return 0.0
    
    return len(pred_set & actual_set) / len(actual_set)


def evaluate_prediction(
    red_pred: List[int],
    blue_pred: int,
    red_actual: List[int],
    blue_actual: int
) -> dict:
    """
    Comprehensive evaluation of a lottery prediction.
    
    Returns:
        dict with various metrics
    """
    red_hits = len(set(red_pred) & set(red_actual))
    blue_hit = 1 if blue_pred == blue_actual else 0
    
    return {
        "red_hits": red_hits,
        "blue_hit": blue_hit,
        "total_hits": red_hits + blue_hit,
        "red_hit_rate": red_hits / 6,
        "perfect_red": red_hits == 6,
        "any_hit": red_hits > 0 or blue_hit > 0
    }
