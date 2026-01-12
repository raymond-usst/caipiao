import numpy as np
from typing import List, Tuple

def calc_top_k_accuracy(y_true: List[int], y_preds: List[List[Tuple[int, float]]], k: int = 3) -> float:
    """
    Calculate top-k accuracy.
    y_true: list of integer targets
    y_preds: list of prediction lists, where each prediction list contains (value, probability) tuples
    """
    hits = 0
    total = len(y_true)
    if total == 0: return 0.0

    for true_val, pred_list in zip(y_true, y_preds):
        # Extract top k numbers
        top_k_nums = [n for n, p in pred_list[:k]]
        if true_val in top_k_nums:
            hits += 1
            
    return hits / total

def calc_nll(y_true: List[int], y_preds: List[List[Tuple[int, float]]]) -> float:
    """
    Calculate Negative Log Likelihood.
    """
    nll_sum = 0.0
    total = len(y_true)
    if total == 0: return 0.0
    
    epsilon = 1e-15
    for true_val, pred_list in zip(y_true, y_preds):
        # Find probability assigned to true_val
        prob = epsilon
        for n, p in pred_list:
            if n == true_val:
                prob = max(p, epsilon)
                break
        nll_sum -= np.log(prob)
        
    return nll_sum / total
