import pandas as pd
from pathlib import Path
import hashlib
from typing import Callable, Optional
from lottery.utils.logger import logger

class FeatureStore:
    def __init__(self, cache_dir: str = "cache/features"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, df: pd.DataFrame, extra: str = "") -> str:
        """
        Generate a unique key based on:
        1. Last issue in the dataframe (data freshness).
        2. DataFrame length.
        3. Extra identifier (function name, args).
        """
        if "issue" in df.columns:
            last_issue = str(df["issue"].iloc[-1])
        else:
            last_issue = "no_issue"
        
        data_sig = f"{last_issue}_{len(df)}"
        key_str = f"{data_sig}_{extra}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def load_or_compute(
        self, 
        df: pd.DataFrame, 
        compute_fn: Callable[[pd.DataFrame], pd.DataFrame], 
        suffix: str = "feats"
    ) -> pd.DataFrame:
        """
        Load features from cache if available and fresh; otherwise compute and save.
        """
        key = self._get_cache_key(df, extra=suffix)
        cache_path = self.cache_dir / f"{key}_{suffix}.parquet"
        
        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                # Verify length matches (simple integrity check)
                if len(cached_df) == len(df):
                    logger.debug(f"[FeatureStore] Hit cache: {cache_path.name}")
                    return cached_df
            except Exception as e:
                logger.warning(f"[FeatureStore] Read cache failed: {e}")
        
        logger.debug(f"[FeatureStore] Computing features ({suffix})...")
        res_df = compute_fn(df)
        
        try:
            res_df.to_parquet(cache_path, index=False)
            logger.debug(f"[FeatureStore] Saved cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"[FeatureStore] Save cache failed: {e}")
            
        return res_df
