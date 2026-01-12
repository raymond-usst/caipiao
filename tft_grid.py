# -*- coding: utf-8 -*-
"""
TFT 贝叶斯调参网格控制器：在不同重评 epochs / TPE 试次组合上运行 bayes_optimize_tft，
选出全局最优。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

import pandas as pd

from .tft_model import bayes_optimize_tft


def bayes_optimize_tft_grid(
    df: pd.DataFrame,
    recent: int = 400,
    n_iter: int = 40,
    random_state: int = 42,
    heavy_epochs_grid: List[int] | None = None,
    tpe_trials_grid: List[int] | None = None,
) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    在多个 (heavy_epochs, tpe_trials) 组合上运行 bayes_optimize_tft，返回全局最优。
    - heavy_epochs_grid: 重评/正式评估的重训练 epochs 选择，默认 [12, 14, 16]
    - tpe_trials_grid: TPE 试次数选择，默认 [60, 80, 90]
    返回: (best_params, best_loss, all_results) 其中 all_results 记录各组合的最优 loss。
    """
    if heavy_epochs_grid is None:
        heavy_epochs_grid = [12, 14, 16]
    if tpe_trials_grid is None:
        tpe_trials_grid = [60, 80, 90]

    results: Dict[str, float] = {}
    best_params: Dict[str, Any] | None = None
    best_loss = float("inf")

    for he in heavy_epochs_grid:
        for tt in tpe_trials_grid:
            key = f"he{he}_tt{tt}"
            try:
                params, loss = bayes_optimize_tft(
                    df=df,
                    recent=recent,
                    n_iter=n_iter,
                    random_state=random_state,
                    heavy_epochs=he,
                    tpe_trials=tt,
                )
                results[key] = loss
                if loss < best_loss:
                    best_loss = loss
                    best_params = params
            except Exception:
                results[key] = float("inf")
    if best_params is None:
        raise RuntimeError("TFT 网格搜索未找到可行解")
    return best_params, best_loss, results


def bayes_optimize_tft_random(
    df: pd.DataFrame,
    recent: int = 400,
    n_iter: int = 40,
    random_state: int = 42,
    heavy_epochs_range: tuple[int, int] = (12, 18),
    tpe_trials_range: tuple[int, int] = (50, 90),
    samples: int = 6,
) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
    """
    随机抽取 (heavy_epochs, tpe_trials) 组合，运行 bayes_optimize_tft，返回全局最优。
    - samples: 抽样组合数量（默认 6），可结合 grid 结果做补充探索。
    """
    rng = np.random.default_rng(random_state)
    results: Dict[str, float] = {}
    best_params: Dict[str, Any] | None = None
    best_loss = float("inf")
    for i in range(samples):
        he = int(rng.integers(heavy_epochs_range[0], heavy_epochs_range[1] + 1))
        tt = int(rng.integers(tpe_trials_range[0], tpe_trials_range[1] + 1))
        key = f"he{he}_tt{tt}"
        try:
            params, loss = bayes_optimize_tft(
                df=df,
                recent=recent,
                n_iter=n_iter,
                random_state=random_state + i * 13,
                heavy_epochs=he,
                tpe_trials=tt,
            )
            results[key] = loss
            if loss < best_loss:
                best_loss = loss
                best_params = params
        except Exception:
            results[key] = float("inf")
    if best_params is None:
        raise RuntimeError("TFT 随机搜索未找到可行解")
    return best_params, best_loss, results

