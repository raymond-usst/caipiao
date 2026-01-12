from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path

class BasePredictor(ABC):
    """Abstract base class for all lottery predictors."""

    def __init__(self, config: Any):
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Train the model using the provided dataframe."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict the next draw's values using the provided historical dataframe.
        
        Returns:
            Dict containing at least:
            - 'red': Dict[pos, List[Tuple[num, prob]]]
            - 'blue': List[Tuple[num, prob]]
            - 'sum_pred': float (optional)
        """
        pass

    @abstractmethod
    def save(self, save_dir: str) -> None:
        """Save the model to the specified directory."""
        pass

    @abstractmethod
    def load(self, save_dir: str) -> bool:
        """
        Load the model from the specified directory.
        Returns True if successful, False otherwise.
        """
        pass
