"""Monte Carlo simulation for prediction confidence intervals."""
from typing import Dict, List, Tuple, Callable, Any
import numpy as np
import pandas as pd


def sample_predictions(
    predictor_fn: Callable[[pd.DataFrame], Dict],
    df: pd.DataFrame,
    n_samples: int = 1000,
    noise_std: float = 0.05
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation by adding noise to predictions.
    
    Args:
        predictor_fn: Prediction function that returns dict with 'sum_pred', 'blue'
        df: Historical data DataFrame
        n_samples: Number of Monte Carlo samples
        noise_std: Standard deviation of Gaussian noise to add
        
    Returns:
        dict with confidence intervals for predictions
    """
    sum_samples = []
    blue_samples = []
    
    # Get base prediction
    base_pred = predictor_fn(df)
    base_sum = base_pred.get("sum_pred", 100.0)
    
    # Extract base blue probability
    blue_probs = base_pred.get("blue", [])
    if isinstance(blue_probs, list) and len(blue_probs) > 0:
        if isinstance(blue_probs[0], (list, tuple)):
            base_blue = blue_probs[0][0]
        else:
            base_blue = blue_probs[0]
    else:
        base_blue = 8  # Default mid-range
    
    # Run Monte Carlo
    for _ in range(n_samples):
        # Add noise to sum prediction
        noisy_sum = base_sum + np.random.normal(0, base_sum * noise_std)
        sum_samples.append(noisy_sum)
        
        # Add noise to blue (discrete, so sample from distribution)
        noisy_blue = int(np.clip(base_blue + np.random.normal(0, 2), 1, 16))
        blue_samples.append(noisy_blue)
    
    sum_samples = np.array(sum_samples)
    blue_samples = np.array(blue_samples)
    
    return {
        "sum": {
            "mean": float(np.mean(sum_samples)),
            "std": float(np.std(sum_samples)),
            "ci_95": (float(np.percentile(sum_samples, 2.5)), float(np.percentile(sum_samples, 97.5))),
            "ci_80": (float(np.percentile(sum_samples, 10)), float(np.percentile(sum_samples, 90))),
        },
        "blue": {
            "mode": int(np.bincount(blue_samples).argmax()),
            "mean": float(np.mean(blue_samples)),
            "ci_95": (int(np.percentile(blue_samples, 2.5)), int(np.percentile(blue_samples, 97.5))),
            "distribution": dict(zip(*np.unique(blue_samples, return_counts=True)))
        },
        "n_samples": n_samples
    }


def bootstrap_prediction_variance(
    predictor_fn: Callable[[pd.DataFrame], Dict],
    df: pd.DataFrame,
    n_bootstrap: int = 100
) -> Dict[str, float]:
    """
    Estimate prediction variance using bootstrap resampling of historical data.
    
    Args:
        predictor_fn: Prediction function
        df: Historical data
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        dict with variance estimates
    """
    sum_preds = []
    blue_preds = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_df = df.sample(frac=1.0, replace=True)
        pred = predictor_fn(sample_df)
        
        if "sum_pred" in pred:
            sum_preds.append(pred["sum_pred"])
        
        blue = pred.get("blue", [])
        if isinstance(blue, list) and len(blue) > 0:
            if isinstance(blue[0], (list, tuple)):
                blue_preds.append(blue[0][0])
            else:
                blue_preds.append(blue[0])
    
    return {
        "sum_variance": float(np.var(sum_preds)) if sum_preds else 0.0,
        "sum_std": float(np.std(sum_preds)) if sum_preds else 0.0,
        "blue_variance": float(np.var(blue_preds)) if blue_preds else 0.0,
        "n_bootstrap": n_bootstrap
    }
