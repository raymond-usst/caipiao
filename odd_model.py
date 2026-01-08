from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier

from .features import build_features

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@dataclass
class OddConfig:
    window: int = 10
    iterations: int = 200
    depth: int = 6
    learning_rate: float = 0.1
    topk: int = 3


def _prepare_dataset(df: pd.DataFrame, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """使用前 window 期展开 + 组合哈希 + 手工特征预测下一期红球奇数个数(0-6)。"""
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    feats_arr = build_features(df).to_numpy(dtype=float)
    X_list = []
    y = []
    for i in range(window, len(data)):
        hist = data[i - window : i].reshape(-1)
        last_reds = data[i - 1, :6]
        combo_id = hash(tuple(sorted(int(x) for x in last_reds)))
        feat_vec = feats_arr[i]
        X_list.append(np.concatenate([hist, np.array([combo_id], dtype=int), feat_vec]))
        odd_cnt = int((data[i, :6] % 2).sum())
        y.append(odd_cnt)
    return np.stack(X_list), np.array(y)


def _odd_ckpt_path(save_dir: Path, cfg: OddConfig) -> Path:
    base = f"odd_w{cfg.window}_it{cfg.iterations}_d{cfg.depth}_lr{cfg.learning_rate}"
    return save_dir / f"{base}.cbm"


def train_odd_model(
    df: pd.DataFrame,
    cfg: OddConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
) -> CatBoostClassifier:
    X, y = _prepare_dataset(df, window=cfg.window)
    if len(y) < 10:
        raise ValueError("样本不足，无法训练奇偶比模型")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    ckpt = None
    if save_dir:
        ckpt = _odd_ckpt_path(Path(save_dir), cfg)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if resume and not fresh and ckpt.exists():
            try:
                m = CatBoostClassifier().load_model(str(ckpt))
                print(f"[odd] 检测到已保存模型，直接加载 {ckpt}")
                return m
            except Exception:
                pass
    model = CatBoostClassifier(
        iterations=cfg.iterations,
        depth=cfg.depth,
        learning_rate=cfg.learning_rate,
        loss_function="MultiClass",
        verbose=100,
        task_type=device,
    )
    model.fit(X, y)
    if ckpt:
        model.save_model(str(ckpt))
        print(f"[odd] 模型已保存到 {ckpt}")
    return model


def predict_odd(model: CatBoostClassifier, df: pd.DataFrame, cfg: OddConfig) -> Dict:
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    feats_arr = build_features(df).to_numpy(dtype=float)
    if len(data) < cfg.window:
        raise ValueError("样本不足，无法预测奇偶比")
    hist = data[-cfg.window :].reshape(1, -1)
    last_reds = data[-1, :6]
    combo_id = hash(tuple(sorted(int(x) for x in last_reds)))
    feat_vec = feats_arr[-1:].astype(float)
    X = np.concatenate([hist, np.array([[combo_id]], dtype=int), feat_vec], axis=1)
    proba = model.predict_proba(X)[0]
    top_idx = np.argsort(proba)[::-1][: cfg.topk]
    odds = [(int(i), float(proba[i])) for i in top_idx]
    return {"odd_pred": odds[0][0], "odd_probs": odds}


def bayes_optimize_odd(
    df: pd.DataFrame,
    window: int = 10,
    n_iter: int = 8,
    random_state: int = 42,
):
    """简化贝叶斯调参，对 depth/learning_rate/iterations 搜索。"""
    try:
        from skopt import BayesSearchCV
        from skopt.space import Integer, Real
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行奇偶比贝叶斯调参") from e

    X, y = _prepare_dataset(df, window=window)
    if len(y) < 30:
        raise ValueError("样本不足，无法进行奇偶比贝叶斯调参")
    model = CatBoostClassifier(
        loss_function="MultiClass",
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        verbose=0,
    )
    search = BayesSearchCV(
        model,
        {
            "depth": Integer(4, 8),
            "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
            "iterations": Integer(80, 400),
        },
        n_iter=n_iter,
        cv=3,
        random_state=random_state,
        verbose=0,
    )
    search.fit(X, y)
    best = search.best_params_
    best_score = float(search.best_score_)
    return best, best_score

