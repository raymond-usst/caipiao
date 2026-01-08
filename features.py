from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

LARGE_PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


def _ac_value(reds: np.ndarray) -> float:
    """AC值（算术复杂度），简化为两两差分平均绝对值。"""
    reds = np.sort(reds.astype(int))
    diffs = []
    for i in range(len(reds)):
        for j in range(i + 1, len(reds)):
            diffs.append(abs(reds[j] - reds[i]))
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def _parity_ratio(reds: np.ndarray) -> float:
    odd = (reds % 2).sum()
    return odd / len(reds)


def _prime_ratio(reds: np.ndarray) -> float:
    primes = sum(1 for r in reds if r in LARGE_PRIMES)
    return primes / len(reds)


def _big_ratio(reds: np.ndarray) -> float:
    big = sum(1 for r in reds if r > 16)
    return big / len(reds)


def _lunar_parts(series: pd.Series) -> pd.DataFrame:
    """农历月、日（依赖 lunardate；若不可用则返回0占位）"""
    try:
        from lunardate import LunarDate
    except Exception:
        return pd.DataFrame({"lunar_month": 0, "lunar_day": 0}, index=series.index)
    months = []
    days = []
    for dt in pd.to_datetime(series):
        try:
            ld = LunarDate.fromSolarDate(dt.year, dt.month, dt.day)
            months.append(ld.month)
            days.append(ld.day)
        except Exception:
            months.append(0)
            days.append(0)
    return pd.DataFrame({"lunar_month": months, "lunar_day": days}, index=series.index)


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

    ac_list = []
    span_list = []
    sum_tail = []
    odd_ratio = []
    prime_ratio = []
    big_ratio = []
    for reds in reds_all:
        ac_list.append(_ac_value(reds))
        span_list.append(int(reds.max() - reds.min()))
        s = int(reds.sum() + reds[-1])
        sum_tail.append(s % 10)
        odd_ratio.append(_parity_ratio(reds))
        prime_ratio.append(_prime_ratio(reds))
        big_ratio.append(_big_ratio(reds))

    res["ac"] = ac_list
    res["span"] = span_list
    res["sum_tail"] = sum_tail
    res["odd_ratio"] = odd_ratio
    res["prime_ratio"] = prime_ratio
    res["big_ratio"] = big_ratio

    # 遗漏：简化为当前期的红球均值遗漏、最大遗漏；蓝球当前遗漏
    # 计算红球遗漏
    red_omit = {n: -1 for n in range(1, 34)}
    omit_mean = []
    omit_max = []
    for reds in reds_all:
        for n in red_omit:
            red_omit[n] += 1
        for r in reds:
            red_omit[r] = 0
        vals = list(red_omit.values())
        omit_mean.append(float(np.mean(vals)))
        omit_max.append(int(np.max(vals)))
    res["omit_red_mean"] = omit_mean
    res["omit_red_max"] = omit_max

    # 蓝球遗漏
    blue_omit = {n: -1 for n in range(1, 17)}
    omit_blue = []
    for b in blue_all:
        for n in blue_omit:
            blue_omit[n] += 1
        blue_omit[b] = 0
        omit_blue.append(int(blue_omit[b]))
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

