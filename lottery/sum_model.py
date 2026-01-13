from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from .features import build_features

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@dataclass
class SumStdConfig:
    window: int = 10
    iterations: int = 200
    depth: int = 6
    learning_rate: float = 0.05


def _prepare_dataset(df: pd.DataFrame, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """构造训练集：预测下一期和值的标准差（以滑窗内和值标准差为目标）。"""
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    feats_arr = build_features(df).to_numpy(dtype=float)
    X_list = []
    y_list = []
    sums = data.sum(axis=1)
    for i in range(window, len(data)):
        hist = data[i - window : i].reshape(-1)
        last_reds = data[i - 1, :6]
        combo_id = hash(tuple(sorted(int(x) for x in last_reds)))
        feat_vec = feats_arr[i]
        X_list.append(np.concatenate([hist, np.array([combo_id], dtype=int), feat_vec]))
        y_list.append(float(np.std(sums[i - window : i])))
    return np.stack(X_list), np.array(y_list)


def _sum_ckpt_path(save_dir: Path, cfg: SumStdConfig) -> Path:
    base = f"sumstd_w{cfg.window}_it{cfg.iterations}_d{cfg.depth}_lr{cfg.learning_rate}"
    return save_dir / f"{base}.cbm"


def train_sumstd_model(
    df: pd.DataFrame,
    cfg: SumStdConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
) -> CatBoostRegressor:
    X, y = _prepare_dataset(df, window=cfg.window)
    if len(y) < 10:
        raise ValueError("样本不足，无法训练和值标准差模型")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    ckpt = None
    if save_dir:
        ckpt = _sum_ckpt_path(Path(save_dir), cfg)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if resume and not fresh and ckpt.exists():
            try:
                m = CatBoostRegressor().load_model(str(ckpt))
                print(f"[sumstd] 检测到已保存模型，直接加载 {ckpt}")
                return m
            except Exception:
                pass
    model = CatBoostRegressor(
        iterations=cfg.iterations,
        depth=cfg.depth,
        learning_rate=cfg.learning_rate,
        loss_function="RMSE",
        verbose=100,
        task_type=device,
    )
    model.fit(X, y)
    if ckpt:
        model.save_model(str(ckpt))
        print(f"[sumstd] 模型已保存到 {ckpt}")
    return model


def predict_sumstd(model: CatBoostRegressor, df: pd.DataFrame, cfg: SumStdConfig) -> float:
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    feats_arr = build_features(df).to_numpy(dtype=float)
    if len(data) < cfg.window:
        raise ValueError("样本不足，无法预测和值标准差")
    hist = data[-cfg.window :].reshape(1, -1)
    last_reds = data[-1, :6]
    combo_id = hash(tuple(sorted(int(x) for x in last_reds)))
    feat_vec = feats_arr[-1:].astype(float)
    X = np.concatenate([hist, np.array([[combo_id]], dtype=int), feat_vec], axis=1)
    pred = float(model.predict(X)[0])
    return max(pred, 1e-3)


def bayes_optimize_sumstd(
    df: pd.DataFrame,
    window: int = 10,
    n_iter: int = 8,
    random_state: int = 42,
):
    """贝叶斯调参：depth/learning_rate/iterations。"""
    try:
        from skopt import BayesSearchCV
        from skopt.space import Integer, Real
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行和值标准差贝叶斯调参") from e

    X, y = _prepare_dataset(df, window=window)
    if len(y) < 30:
        raise ValueError("样本不足，无法进行和值标准差贝叶斯调参")
    model = CatBoostRegressor(
        loss_function="RMSE",
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


def random_optimize_sumstd(
    df: pd.DataFrame,
    window: int = 10,
    samples: int = 20,
    random_state: int = 42,
) -> Tuple[Dict, float]:
    """
    随机搜索和值标准差模型超参，适用于不想用贝叶斯或噪声场景。
    评估：顺序切分 80/20 做一次验证，指标用负 RMSE（越大越好）。
    """
    X, y = _prepare_dataset(df, window=window)
    if len(y) < 50:
        raise ValueError("样本不足，无法进行和值标准差随机调参")
    split = int(len(y) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(random_state)
    best = None
    best_score = -1e9
    for _ in range(samples):
        depth = int(rng.integers(4, 9))
        lr = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.3))))
        it = int(rng.integers(80, 401))
        model = CatBoostRegressor(
            iterations=it,
            depth=depth,
            learning_rate=lr,
            loss_function="RMSE",
            verbose=False,
            task_type="GPU" if torch.cuda.is_available() else "CPU",
        )
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            rmse = float(np.sqrt(((pred - y_val) ** 2).mean()))
            score = -rmse
        except Exception:
            score = -1e9
        if score > best_score:
            best_score = score
            best = {"depth": depth, "learning_rate": lr, "iterations": it}
    if best is None:
        raise RuntimeError("和值标准差随机搜索未找到可行解")
    return best, best_score
