"""Time-Series Cross-Validation with purged walk-forward validation.

Prevents lookahead bias in backtesting.
"""
from typing import Iterator, Tuple, List, Callable, Dict, Any
import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """
    Time-series cross-validator with gap to prevent data leakage.
    
    Unlike sklearn's TimeSeriesSplit, this includes a purge gap between
    train and test sets to avoid lookahead bias.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 50,
        gap: int = 1,
        min_train_size: int = 100
    ):
        """
        Initialize purged time-series splitter.
        
        Args:
            n_splits: Number of CV folds
            test_size: Number of samples in each test set
            gap: Number of samples between train and test (purge zone)
            min_train_size: Minimum samples required for training
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.min_train_size = min_train_size
    
    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.
        
        Args:
            X: Data array or DataFrame
            
        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate starting positions for each fold
        total_needed = self.min_train_size + self.gap + self.test_size
        if n_samples < total_needed:
            raise ValueError(f"Not enough samples: {n_samples} < {total_needed}")
        
        available = n_samples - self.min_train_size - self.gap - self.test_size
        step = available // max(self.n_splits - 1, 1)
        
        for i in range(self.n_splits):
            train_end = self.min_train_size + i * step
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self) -> int:
        return self.n_splits


class ExpandingWindowSplit:
    """Expanding window cross-validation."""
    
    def __init__(
        self,
        initial_train_size: int = 100,
        test_size: int = 10,
        step: int = 10,
        gap: int = 0
    ):
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step = step
        self.gap = gap
    
    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding window splits."""
        n_samples = len(X)
        train_end = self.initial_train_size
        
        while train_end + self.gap + self.test_size <= n_samples:
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            train_end += self.step


def walk_forward_backtest(
    df: pd.DataFrame,
    predictor_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    train_fn: Callable[[pd.DataFrame], None],
    cv: PurgedTimeSeriesSplit = None,
    retrain_every: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform walk-forward backtesting.
    
    Args:
        df: Full dataset
        predictor_fn: Function that makes predictions given historical data
        train_fn: Function that trains the model
        cv: Cross-validator (defaults to PurgedTimeSeriesSplit)
        retrain_every: Retrain model every N folds
        
    Returns:
        List of evaluation results for each test sample
    """
    if cv is None:
        cv = PurgedTimeSeriesSplit(n_splits=10, test_size=10, gap=1)
    
    results = []
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df.to_numpy())):
        # Train on training data
        if fold_idx % retrain_every == 0:
            train_data = df.iloc[train_idx]
            train_fn(train_data)
        
        # Predict and evaluate on test data
        for i in test_idx:
            hist_data = df.iloc[:i]
            try:
                pred = predictor_fn(hist_data)
            except Exception as e:
                continue
            
            actual_row = df.iloc[i]
            actual_red = actual_row[red_cols].tolist()
            actual_blue = int(actual_row["blue"])
            
            # Extract predictions
            pred_red = []
            for j in range(1, 7):
                if "red" in pred and j in pred["red"]:
                    pred_red.append(pred["red"][j][0][0])
                else:
                    pred_red.append(0)
            
            pred_blue = pred.get("blue", [(0, 0)])[0][0] if pred.get("blue") else 0
            
            # Compute hits
            red_hits = len(set(pred_red) & set(actual_red))
            blue_hit = 1 if pred_blue == actual_blue else 0
            
            results.append({
                "fold": fold_idx,
                "index": i,
                "red_hits": red_hits,
                "blue_hit": blue_hit,
                "total_hits": red_hits + blue_hit
            })
    
    return results


def compute_backtest_summary(results: List[Dict]) -> Dict[str, float]:
    """Summarize backtest results."""
    if not results:
        return {}
    
    total = len(results)
    red_hits = sum(r["red_hits"] for r in results)
    blue_hits = sum(r["blue_hit"] for r in results)
    
    return {
        "n_predictions": total,
        "avg_red_hits": red_hits / total,
        "avg_blue_hits": blue_hits / total,
        "avg_total_hits": (red_hits + blue_hits) / total,
        "red_hit_rate": red_hits / (total * 6),
        "blue_hit_rate": blue_hits / total,
        "any_hit_rate": sum(1 for r in results if r["total_hits"] > 0) / total
    }
