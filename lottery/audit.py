"""Prediction Audit Trail for logging and tracking predictions."""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import hashlib


class AuditLogger:
    """Logger for tracking all predictions with timestamps."""
    
    def __init__(self, log_file: str = "predictions.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(
        self,
        model_name: str,
        prediction: Dict[str, Any],
        input_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a prediction with timestamp.
        
        Args:
            model_name: Name of the model making the prediction
            prediction: Prediction output dict
            input_hash: Optional hash of input data for reproducibility
            metadata: Optional additional metadata
            
        Returns:
            Prediction ID
        """
        pred_id = self._generate_id()
        
        entry = {
            "id": pred_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prediction": self._serialize_prediction(prediction),
            "input_hash": input_hash,
            "metadata": metadata or {}
        }
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        return pred_id
    
    def get_prediction_history(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve prediction history with optional filters.
        
        Args:
            model_name: Filter by model name
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of entries to return
            
        Returns:
            List of prediction entries
        """
        if not self.log_file.exists():
            return []
        
        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Apply filters
                    if model_name and entry.get("model") != model_name:
                        continue
                    
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue
                    
                    entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return entries[-limit:]
    
    def get_latest_prediction(self, model_name: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent prediction."""
        history = self.get_prediction_history(model_name=model_name, limit=1)
        return history[-1] if history else None
    
    def compute_input_hash(self, df) -> str:
        """Compute hash of input DataFrame for reproducibility."""
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            data_str = df.to_json()
        else:
            data_str = str(df)
        return hashlib.md5(data_str.encode()).hexdigest()[:12]
    
    def _generate_id(self) -> str:
        """Generate unique prediction ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _serialize_prediction(self, pred: Dict) -> Dict:
        """Serialize prediction for JSON storage."""
        result = {}
        for k, v in pred.items():
            if hasattr(v, 'tolist'):  # numpy array
                result[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                result[k] = v
            else:
                result[k] = str(v)
        return result


# Global audit logger instance
_audit_logger = None

def get_audit_logger(log_file: str = "predictions.jsonl") -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_file)
    return _audit_logger


def log_prediction(
    model_name: str,
    prediction: Dict[str, Any],
    input_hash: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to log a prediction."""
    return get_audit_logger().log_prediction(model_name, prediction, input_hash, metadata)
