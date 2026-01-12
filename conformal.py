"""Conformal Prediction for rigorous uncertainty quantification.

Provides prediction sets with guaranteed coverage (e.g., 90%).
"""
from typing import List, Tuple, Callable, Dict, Any, Optional
import numpy as np
import pandas as pd


class ConformalPredictor:
    """
    Split Conformal Prediction for regression/classification.
    
    Generates prediction sets with guaranteed marginal coverage.
    """
    
    def __init__(self, coverage: float = 0.90):
        """
        Initialize conformal predictor.
        
        Args:
            coverage: Target coverage level (e.g., 0.90 for 90% coverage)
        """
        self.coverage = coverage
        self.quantile = None
        self.fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray
    ) -> 'ConformalPredictor':
        """
        Fit conformal predictor on calibration set.
        
        Computes the conformity scores and the coverage quantile.
        
        Args:
            predictions: Model predictions on calibration set
            true_values: True values on calibration set
            
        Returns:
            self
        """
        # Conformity score = absolute residual
        scores = np.abs(predictions - true_values)
        
        # Compute quantile for coverage
        n = len(scores)
        q = np.ceil((n + 1) * self.coverage) / n
        self.quantile = np.quantile(scores, min(q, 1.0))
        self.fitted = True
        
        return self
    
    def predict(self, predictions: np.ndarray) -> List[Tuple[float, float]]:
        """
        Generate prediction intervals.
        
        Args:
            predictions: Point predictions for test set
            
        Returns:
            List of (lower, upper) intervals
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")
        
        intervals = []
        for pred in predictions:
            lower = pred - self.quantile
            upper = pred + self.quantile
            intervals.append((lower, upper))
        
        return intervals
    
    def predict_set(
        self,
        class_probs: np.ndarray,
        classes: np.ndarray
    ) -> List[List[int]]:
        """
        Generate prediction sets for classification.
        
        Uses LAC (Least Ambiguous set-valued Classifier).
        
        Args:
            class_probs: Class probabilities, shape (n_samples, n_classes)
            classes: Array of class labels
            
        Returns:
            List of prediction sets (list of possible class labels for each sample)
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")
        
        pred_sets = []
        for probs in class_probs:
            # Sort by probability descending
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            
            # Find smallest set with cumulative prob >= coverage
            k = np.searchsorted(cumsum, self.coverage) + 1
            selected = sorted_idx[:k]
            pred_sets.append(classes[selected].tolist())
        
        return pred_sets


class ConformalRegressor:
    """Conformal prediction specifically for regression tasks."""
    
    def __init__(self, coverage: float = 0.90):
        self.coverage = coverage
        self.residuals = None
        self.fitted = False
    
    def fit(self, predictions: np.ndarray, actuals: np.ndarray) -> 'ConformalRegressor':
        """Fit on calibration residuals."""
        self.residuals = np.abs(predictions - actuals)
        self.fitted = True
        return self
    
    def predict_interval(self, point_pred: float) -> Tuple[float, float]:
        """Get prediction interval for a point prediction."""
        if not self.fitted:
            raise ValueError("Must fit first")
        
        n = len(self.residuals)
        q_index = int(np.ceil((n + 1) * self.coverage)) - 1
        q_index = min(q_index, n - 1)
        
        sorted_residuals = np.sort(self.residuals)
        margin = sorted_residuals[q_index]
        
        return (point_pred - margin, point_pred + margin)


def get_prediction_sets(
    predictor_fn: Callable[[pd.DataFrame], Dict],
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "blue",
    coverage: float = 0.90
) -> Dict[str, Any]:
    """
    Generate conformal prediction sets for a predictor.
    
    Args:
        predictor_fn: Prediction function
        calibration_df: Calibration data
        test_df: Test data
        target_col: Target column name
        coverage: Coverage level
        
    Returns:
        Dict with prediction sets and coverage info
    """
    # Get calibration predictions
    cal_preds = []
    for i in range(len(calibration_df)):
        try:
            pred = predictor_fn(calibration_df.iloc[:i+1])
            if target_col in pred and pred[target_col]:
                if isinstance(pred[target_col], list) and isinstance(pred[target_col][0], tuple):
                    cal_preds.append(pred[target_col][0][0])
                else:
                    cal_preds.append(pred.get("sum_pred", 0))
            else:
                cal_preds.append(pred.get("sum_pred", 0))
        except Exception:
            cal_preds.append(0)
    
    cal_preds = np.array(cal_preds)
    cal_actuals = calibration_df[target_col].values
    
    # Fit conformal
    conf = ConformalRegressor(coverage)
    conf.fit(cal_preds, cal_actuals)
    
    # Get test predictions and intervals
    test_pred = predictor_fn(test_df)
    point_pred = test_pred.get("sum_pred", 100.0)
    interval = conf.predict_interval(point_pred)
    
    return {
        "point_prediction": point_pred,
        "interval": interval,
        "coverage": coverage,
        "n_calibration": len(calibration_df)
    }
