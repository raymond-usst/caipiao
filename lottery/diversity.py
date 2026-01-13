"""Ensemble diversity metrics for model blending optimization."""
from typing import List, Dict, Callable, Any
import numpy as np
import pandas as pd
from itertools import combinations


def pairwise_disagreement(
    predictions: List[Dict],
    key: str = "blue"
) -> float:
    """
    Calculate pairwise disagreement rate between model predictions.
    
    Higher disagreement = more diverse ensemble.
    
    Args:
        predictions: List of prediction dicts from different models
        key: Which prediction key to compare ('blue', 'sum_pred', etc.)
        
    Returns:
        Disagreement rate [0, 1]
    """
    if len(predictions) < 2:
        return 0.0
    
    # Extract values
    values = []
    for pred in predictions:
        val = pred.get(key)
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], (list, tuple)):
                values.append(val[0][0])  # Top prediction
            else:
                values.append(val[0])
        elif isinstance(val, (int, float)):
            values.append(val)
        else:
            values.append(None)
    
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    if len(valid_values) < 2:
        return 0.0
    
    # Count disagreements
    n_pairs = 0
    n_disagree = 0
    for v1, v2 in combinations(valid_values, 2):
        n_pairs += 1
        if v1 != v2:
            n_disagree += 1
    
    return n_disagree / n_pairs if n_pairs > 0 else 0.0


def correlation_diversity(
    predictors: List[Callable[[pd.DataFrame], Dict]],
    df: pd.DataFrame,
    n_samples: int = 50
) -> Dict[str, Any]:
    """
    Measure correlation-based diversity among ensemble members.
    
    Args:
        predictors: List of predictor functions
        df: Historical data for generating predictions
        n_samples: Number of samples to use
        
    Returns:
        dict with correlation matrix and diversity score
    """
    if len(predictors) < 2:
        return {"diversity_score": 0.0, "message": "Need at least 2 predictors"}
    
    # Sample predictions at different points
    sample_indices = np.linspace(20, len(df) - 1, min(n_samples, len(df) - 20), dtype=int)
    
    # Collect sum predictions for each model
    model_preds = {i: [] for i in range(len(predictors))}
    
    for idx in sample_indices:
        hist = df.iloc[:idx+1]
        for i, pred_fn in enumerate(predictors):
            try:
                pred = pred_fn(hist)
                sum_val = pred.get("sum_pred", 0)
                if sum_val is None:
                    sum_val = 0
                model_preds[i].append(float(sum_val))
            except Exception:
                model_preds[i].append(0)
    
    # Build prediction matrix
    pred_matrix = np.array([model_preds[i] for i in range(len(predictors))])
    
    # Calculate correlation matrix
    if pred_matrix.shape[1] > 1:
        corr_matrix = np.corrcoef(pred_matrix)
        # Handle NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)
    else:
        corr_matrix = np.eye(len(predictors))
    
    # Diversity score: 1 - mean absolute correlation (excluding diagonal)
    mask = ~np.eye(len(predictors), dtype=bool)
    mean_corr = np.abs(corr_matrix[mask]).mean() if mask.any() else 0.0
    diversity_score = 1.0 - mean_corr
    
    return {
        "correlation_matrix": corr_matrix.tolist(),
        "diversity_score": float(diversity_score),
        "n_models": len(predictors),
        "n_samples": len(sample_indices)
    }


def optimal_ensemble_weights(
    predictors: List[Callable[[pd.DataFrame], Dict]],
    df: pd.DataFrame,
    target_key: str = "blue"
) -> List[float]:
    """
    Calculate optimal ensemble weights based on diversity and performance.
    
    This is a simplified version; in practice, use cross-validation.
    
    Args:
        predictors: List of predictor functions
        df: Historical data
        target_key: Target to optimize for
        
    Returns:
        List of weights (sum to 1)
    """
    n = len(predictors)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    
    # Equal weights as baseline (could be optimized with more data)
    return [1.0 / n] * n
