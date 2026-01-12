import pandas as pd
from typing import Dict, Any, Type, List
from lottery.engine.predictor import BasePredictor
from lottery.utils.logger import logger
from lottery.utils.metrics import calc_top_k_accuracy, calc_nll

class RollingBacktester:
    def __init__(self, predictor_cls: Type[BasePredictor], config: Any, retrain_interval: int = 1):
        self.predictor_cls = predictor_cls
        self.config = config
        self.retrain_interval = retrain_interval
        # Force fresh training for backtest correctness
        if hasattr(self.config, 'fresh'):
             self.config.fresh = True
        if hasattr(self.config, 'resume'):
             self.config.resume = False

    def run(self, df: pd.DataFrame, start_ratio: float = 0.8) -> Dict[str, float]:
        """
        Run rolling backtest.
        start_ratio: proportion of data to use for initial training.
        """
        n = len(df)
        start_idx = int(n * start_ratio)
        if start_idx >= n:
            raise ValueError("Start index exceeds dataframe length")
            
        logger.info(f"[Backtest] Starting rolling backtest from index {start_idx}/{n} (Interval={self.retrain_interval})...")
        
        preds_blue: List[List[Any]] = []
        truth_blue: List[int] = []
        
        predictor = None
        
        for i in range(start_idx, n):
            # Train window: 0 to i
            train_df = df.iloc[:i]
            # Test sample: i
            test_row = df.iloc[i]
            
            # Retrain if needed
            if (i - start_idx) % self.retrain_interval == 0 or predictor is None:
                # logger.debug(f"[Backtest] Retraining at {i}...")
                predictor = self.predictor_cls(self.config)
                # Suppress training logs to avoid spam
                predictor.train(train_df)
                
            # Predict
            # Note: predict() expects a dataframe history to predict NEXT step.
            # train_df contains history up to i-1 (inclusive of i-1 target? No, iloc[:i] excludes i which is target)
            # Actually, standard logic in this codebase seems to be:
            # Training on df[:N] learns to predict N+1? 
            # Let's verify standard interface. 
            # In simple terms: input history -> predict next.
            # So passing train_df (which ends at i-1) should predict i.
            
            try:
                # Some predictors might need 'fresh' instance or might update state. 
                # Re-instantiating `predictor_cls` above handles freshness.
                res = predictor.predict(train_df)
                
                # Blue Analysis
                blue_res = res.get("blue", []) # List[(num, prob)]
                preds_blue.append(blue_res)
                truth_blue.append(int(test_row["blue"]))
                
            except Exception as e:
                logger.warning(f"[Backtest] Prediction failed at {i}: {e}")
        
        # Calculate Metrics
        acc_top1 = calc_top_k_accuracy(truth_blue, preds_blue, k=1)
        acc_top3 = calc_top_k_accuracy(truth_blue, preds_blue, k=3)
        nll = calc_nll(truth_blue, preds_blue)
        
        logger.info(f"[Backtest] Completed. Samples: {len(truth_blue)}")
        logger.info(f"[Backtest] Blue Top1: {acc_top1:.4f}")
        logger.info(f"[Backtest] Blue Top3: {acc_top3:.4f}")
        logger.info(f"[Backtest] Blue NLL:  {nll:.4f}")
        
        return {
            "samples": len(truth_blue),
            "blue_top1": acc_top1,
            "blue_top3": acc_top3,
            "blue_nll": nll
        }
