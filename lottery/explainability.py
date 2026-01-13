"""SHAP-based model explainability for CatBoost predictions."""
from typing import Optional, Any
import numpy as np
import pandas as pd

def explain_catboost(model: Any, X: pd.DataFrame, max_samples: int = 100) -> dict:
    """
    Generate SHAP explanations for a CatBoost model.
    
    Args:
        model: Trained CatBoost model
        X: Feature DataFrame
        max_samples: Maximum samples to explain (for performance)
        
    Returns:
        dict with 'shap_values', 'feature_importance', 'top_features'
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed. Run: pip install shap"}
    
    # Limit samples for performance
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate feature importance (mean absolute SHAP value)
    if isinstance(shap_values, list):
        # Multi-class: average across classes
        shap_vals = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
    else:
        shap_vals = np.abs(shap_values).mean(axis=0)
    
    feature_importance = dict(zip(X.columns, shap_vals))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "top_features": sorted_features[:10],
        "sample_size": len(X_sample)
    }


def get_feature_importance_summary(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Return feature importance as a sorted DataFrame."""
    result = explain_catboost(model, X)
    if "error" in result:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame(
        list(result["feature_importance"].items()),
        columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=False)
    
    return importance_df
