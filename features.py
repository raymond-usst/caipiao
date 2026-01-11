from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict
from functools import lru_cache

import numpy as np
import pandas as pd

LARGE_PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], dtype=int)


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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造宏观特征：
      - ac, sum_tail, span
      - odd_ratio, prime_ratio, big_ratio
      - omit_red_mean/max, omit_blue
      - weekday(0-6), weekday_sin/cos
      - lunar_month, lunar_day
    """
    res = pd.DataFrame(index=df.index)
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    reds_all = data[:, :6]
    blue_all = data[:, 6]

    # 向量化宏观特征
    res["ac"] = _ac_values(reds_all)
    res["span"] = reds_all.max(axis=1) - reds_all.min(axis=1)
    res["sum_tail"] = (reds_all.sum(axis=1) + blue_all) % 10
    res["odd_ratio"] = (reds_all % 2).sum(axis=1) / 6.0
    res["prime_ratio"] = np.isin(reds_all, LARGE_PRIMES).sum(axis=1) / 6.0
    res["big_ratio"] = (reds_all > 16).sum(axis=1) / 6.0

    # 遗漏：简化为当前期的红球均值遗漏、最大遗漏；蓝球当前遗漏
    # 计算红球遗漏（逐期累积，但用 numpy 数组避免 DataFrame 逐行循环）
    red_last = np.full(33, -1, dtype=int)  # index 0 对应号码 1
    omit_mean = np.empty(len(df), dtype=float)
    omit_max = np.empty(len(df), dtype=int)
    for i, reds in enumerate(reds_all):
        red_last += 1
        red_last[reds - 1] = 0
        omit_mean[i] = red_last.mean()
        omit_max[i] = red_last.max()
    res["omit_red_mean"] = omit_mean
    res["omit_red_max"] = omit_max

    # 蓝球遗漏
    blue_last = np.full(16, -1, dtype=int)
    omit_blue = np.empty(len(df), dtype=int)
    for i, b in enumerate(blue_all):
        blue_last += 1
        blue_last[b - 1] = 0
        omit_blue[i] = blue_last[b - 1]
    res["omit_blue"] = omit_blue

    # 星期特征
    if "draw_date" in df.columns:
        dts = pd.to_datetime(df["draw_date"])
        weekday = dts.dt.weekday
    else:
        weekday = pd.Series([0] * len(df), index=df.index)
    res["weekday"] = weekday
    res["weekday_sin"] = np.sin(2 * np.pi * res["weekday"] / 7)
    res["weekday_cos"] = np.cos(2 * np.pi * res["weekday"] / 7)

    # 农历
    if "draw_date" in df.columns:
        lunar_df = _lunar_parts(df["draw_date"])
        res = pd.concat([res, lunar_df], axis=1)
    else:
        res["lunar_month"] = 0
        res["lunar_day"] = 0

    return res.reset_index(drop=True)

