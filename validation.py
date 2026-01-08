from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


@dataclass
class CVResult:
    fold: int
    train_end: int
    test_end: int
    red_hit: float
    blue_hit: float


def rolling_cv(
    df: pd.DataFrame,
    trainer: Callable[[pd.DataFrame], object],
    predictor: Callable[[object, pd.DataFrame], Dict],
    train_size: int = 300,
    test_size: int = 20,
    step: int = 20,
) -> List[CVResult]:
    """
    简单滚动验证：
    - 每折用 train_size 条训练，后接 test_size 条作为测试。
    - 预测每一期的 Top1，统计红球位置命中率平均、蓝球命中率。
    """
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    results: List[CVResult] = []
    total = len(df)
    fold = 0
    for start in range(0, total - train_size - test_size + 1, step):
        train_df = df.iloc[start : start + train_size]
        test_df = df.iloc[start + train_size : start + train_size + test_size]
        if len(train_df) < train_size or len(test_df) == 0:
            continue
        model = trainer(train_df)
        red_hits = []
        blue_hits = []
        for i in range(len(test_df)):
            hist = pd.concat([train_df, test_df.iloc[:i]])
            preds = predictor(model, hist)
            target = test_df.iloc[i][cols].to_numpy(dtype=int)
            # red: average of 6 positions top1
            red_correct = 0
            for pos in range(1, 7):
                top1 = preds["red"][pos][0][0]
                if top1 == target[pos - 1]:
                    red_correct += 1
            red_hits.append(red_correct / 6.0)
            blue_top1 = preds["blue"][0][0]
            blue_hits.append(1.0 if blue_top1 == target[6] else 0.0)
        results.append(
            CVResult(
                fold=fold,
                train_end=start + train_size,
                test_end=start + train_size + len(test_df),
                red_hit=float(np.mean(red_hits)) if red_hits else 0.0,
                blue_hit=float(np.mean(blue_hits)) if blue_hits else 0.0,
            )
        )
        fold += 1
    return results


def rolling_cv_generic(
    df: pd.DataFrame,
    trainer: Callable[[pd.DataFrame], object],
    predictor: Callable[[object, pd.DataFrame], Dict],
    train_size: int,
    test_size: int,
    step: int,
) -> List[CVResult]:
    return rolling_cv(df, trainer, predictor, train_size=train_size, test_size=test_size, step=step)

