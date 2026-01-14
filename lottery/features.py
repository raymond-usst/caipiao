from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict
from functools import lru_cache

import numpy as np
import pandas as pd

LARGE_PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], dtype=int)

# Numba JIT for omission calculation
try:
    from numba import jit
    
    @jit(nopython=True)
    def _compute_omission_numba(reds_np: np.ndarray, blue_vals: np.ndarray):
        """Numba-accelerated omission calculation."""
        n = len(reds_np)
        omit_mean = np.empty(n, dtype=np.float64)
        omit_max = np.empty(n, dtype=np.int64)
        omit_blue = np.empty(n, dtype=np.int64)
        
        red_last = np.full(33, -1, dtype=np.int64)
        blue_last = np.full(16, -1, dtype=np.int64)
        
        for i in range(n):
            # Update Red
            red_last += 1
            for j in range(6):
                red_last[reds_np[i, j] - 1] = 0
            omit_mean[i] = red_last.mean()
            omit_max[i] = red_last.max()
            
            # Update Blue
            b = blue_vals[i]
            blue_last += 1
            blue_last[b - 1] = 0
            omit_blue[i] = blue_last[b - 1]
        
        return omit_mean, omit_max, omit_blue
    
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _lunar_parts(series: pd.Series) -> pd.DataFrame:
    """农历月、日（依赖 lunardate；若不可用则返回0占位）"""
    try:
        from lunardate import LunarDate
    except Exception:
        return pd.DataFrame({"lunar_month": 0, "lunar_day": 0}, index=series.index)

    @lru_cache(maxsize=4096)
    def _solar_to_lunar(y: int, m: int, d: int) -> tuple[int, int]:
        try:
            ld = LunarDate.fromSolarDate(y, m, d)
            return ld.month, ld.day
        except Exception:
            return 0, 0

    dts = pd.to_datetime(series)
    months = np.empty(len(dts), dtype=int)
    days = np.empty(len(dts), dtype=int)
    for i, dt in enumerate(dts):
        months[i], days[i] = _solar_to_lunar(dt.year, dt.month, dt.day)

    return pd.DataFrame({"lunar_month": months, "lunar_day": days}, index=series.index)


def _ac_values(reds_all: np.ndarray) -> np.ndarray:
    """向量化计算 AC（两两差分平均绝对值）。"""
    # reds_all: [N, 6]
    diffs = np.abs(reds_all[:, :, None] - reds_all[:, None, :])  # [N,6,6]
    iu = np.triu_indices(6, k=1)
    pairwise = diffs[:, iu[0], iu[1]]  # [N, 15]
    return pairwise.mean(axis=1)


import polars as pl

from lottery.feature_store import FeatureStore

_store = FeatureStore()

def _compute_features_internal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features using Polars for performance.
    """
    # Convert to Polars
    # Ensure columns are correct types
    pldf = pl.from_pandas(df)
    
    # 1. Macro Features using Polars Expressions
    # reds: slice columns red1..red6
    # Note: Polars doesn't have direct row-wise array ops like numpy, so we use list/struct or multi-col expressions.
    # But for simple row-wise sums/max/min, we can use `pl.max_horizontal`, etc.
    
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    
    # Pre-calc some lists/arrays if easier, but let's stick to Polars exprs where efficient.
    
    # AC Value: Mean absolute difference of all pairs.
    # N(N-1)/2 = 15 pairs. Hard to vectorize in pure Polars Expr without custom map.
    # For AC, we might stick to numpy or use map_batches if complex.
    # Span: max - min
    # Sum Tail: (sum(reds) + blue) % 10
    # Odd Ratio: count(odd) / 6
    # Prime Ratio: count(prime) / 6
    # Big Ratio: count(>16) / 6
    
    # Define Expressions
    exprs = []
    
    # Span
    exprs.append((pl.max_horizontal(red_cols) - pl.min_horizontal(red_cols)).alias("span"))
    
    # Sum Tail
    red_sum = pl.sum_horizontal(red_cols)
    exprs.append(((red_sum + pl.col("blue")) % 10).alias("sum_tail"))
    
    # Count conditions horizontal
    # Odd: x % 2 == 1
    is_odd = sum(pl.col(c) % 2 for c in red_cols)
    exprs.append((is_odd / 6.0).alias("odd_ratio"))
    
    # Big: x > 16
    is_big = sum((pl.col(c) > 16).cast(pl.Int32) for c in red_cols)
    exprs.append((is_big / 6.0).alias("big_ratio"))
    
    # Prime: 
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    is_prime = sum(pl.col(c).is_in(primes).cast(pl.Int32) for c in red_cols)
    exprs.append((is_prime / 6.0).alias("prime_ratio"))
    
    # Execute simpleExprs
    res_pl = pldf.with_columns(exprs)
    
    # AC computation (easiest in numpy for now due to diff pairs)
    # Convert reds to numpy
    reds_np = res_pl.select(red_cols).to_numpy()
    ac_vals = _ac_values(reds_np)
    res_pl = res_pl.with_columns(pl.Series("ac", ac_vals))
    
    # Omission (use Numba if available for speed)
    blue_vals = res_pl["blue"].to_numpy().astype(np.int64)
    reds_np_int = reds_np.astype(np.int64)
    
    if HAS_NUMBA:
        omit_mean, omit_max, omit_blue = _compute_omission_numba(reds_np_int, blue_vals)
    else:
        # Fallback: pure Python loop
        n = len(df)
        omit_mean = np.empty(n, dtype=float)
        omit_max = np.empty(n, dtype=int)
        omit_blue = np.empty(n, dtype=int)
        red_last = np.full(33, -1, dtype=int)
        blue_last = np.full(16, -1, dtype=int)
        
        for i in range(n):
            red_last += 1
            for j in range(6):
                red_last[reds_np[i, j] - 1] = 0
            omit_mean[i] = red_last.mean()
            omit_max[i] = red_last.max()
            
            b = blue_vals[i]
            blue_last += 1
            blue_last[b - 1] = 0
            omit_blue[i] = blue_last[b - 1]
        
    res_pl = res_pl.with_columns([
        pl.Series("omit_red_mean", omit_mean),
        pl.Series("omit_red_max", omit_max),
        pl.Series("omit_blue", omit_blue)
    ])
    
    # Weekday & Lunar
    # Reuse existing funcs or move to Polars datetime
    # Polars datetime: dt.weekday() returns 1-7 (Mon-Sun) or 0-6?
    # Polars: Monday=1, Sunday=7. Pandas: Mon=0, Sun=6.
    # We need to check 'draw_date'.
    
    if "draw_date" in pldf.columns:
        # Check if it's a string type (Utf8 or String)
        dtype = pldf["draw_date"].dtype
        if dtype == pl.Utf8 or dtype == pl.String or str(dtype).startswith("String") or str(dtype).startswith("Utf8"):
            dts = pldf["draw_date"].str.to_datetime("%Y-%m-%d", strict=False)
        else:
            # Already datetime or similar - use directly
            dts = pldf["draw_date"]
             
        wd = dts.dt.weekday() - 1 # Polars 1..7 -> 0..6 to match Pandas
        
        # Sin/Cos
        wd_sin = (2 * np.pi * wd / 7).sin()
        wd_cos = (2 * np.pi * wd / 7).cos()
        
        res_pl = res_pl.with_columns([
            wd.alias("weekday"),
            wd_sin.alias("weekday_sin"),
            wd_cos.alias("weekday_cos")
        ])
        
        # Lunar (fallback to pandas apply if needed, complex lib)
        # Just convert 'draw_date' back to pandas series for _lunar_parts
        # Or keep using the existing _lunar_parts on the original df["draw_date"]
        # Since _lunar_parts returns a DF, we can convert to Polars and hstack.
        lunar_pd = _lunar_parts(df["draw_date"])
        lunar_pl = pl.from_pandas(lunar_pd)
        res_pl = res_pl.hstack(lunar_pl)
        
    else:
        # Defaults
        res_pl = res_pl.with_columns([
            pl.lit(0).alias("weekday"),
            pl.lit(0.0).alias("weekday_sin"),
            pl.lit(1.0).alias("weekday_cos"),
            pl.lit(0).alias("lunar_month"),
            pl.lit(0).alias("lunar_day"),
        ])
    
    # Add rolling chaos features (Lyapunov exponent, Correlation dimension)
    # Computed on sum series with a rolling window
    sum_series = (res_pl.select(["red1", "red2", "red3", "red4", "red5", "red6"])
                  .sum_horizontal().to_numpy().astype(float))
    
    chaos_window = 30  # Rolling window for chaos metrics
    lyap_vals = np.full(len(df), np.nan)
    corr_dim_vals = np.full(len(df), np.nan)
    
    for i in range(chaos_window, len(df)):
        window_data = sum_series[i - chaos_window : i]
        try:
            from lottery.analyzer import lyapunov_exponent, correlation_dimension
            lyap = lyapunov_exponent(window_data)
            lyap_vals[i] = lyap if lyap is not None else 0.0
            cdim = correlation_dimension(window_data)
            corr_dim_vals[i] = cdim if cdim is not None else 0.0
        except Exception:
            lyap_vals[i] = 0.0
            corr_dim_vals[i] = 0.0
    
    # Fill NaN for first chaos_window rows with 0
    lyap_vals = np.nan_to_num(lyap_vals, nan=0.0)
    corr_dim_vals = np.nan_to_num(corr_dim_vals, nan=0.0)
    
    res_pl = res_pl.with_columns([
        pl.Series("lyapunov_exp", lyap_vals),
        pl.Series("corr_dim", corr_dim_vals)
    ])
        
    # Return as Pandas DataFrame because the rest of the app expects it
    # Drop non-numeric columns that might cause issues downstream
    out_df = res_pl.to_pandas()
    drop_cols = ["draw_date", "issue"]
    out_df.drop(columns=[c for c in drop_cols if c in out_df.columns], inplace=True)
    return out_df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造宏观特征（带缓存）。
    """
    out_df = _store.load_or_compute(df, _compute_features_internal, suffix="macro_polars")
    # Ensure non-numeric columns are dropped even if loaded from old cache
    drop_cols = ["draw_date", "issue"]
    out_df.drop(columns=[c for c in drop_cols if c in out_df.columns], inplace=True)
    return out_df


