from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import torch
import os
from pathlib import Path
from typing import Dict, Tuple


def _prepare_dataset(df: pd.DataFrame, window: int = 10):
    """
    将历史开奖转换为监督学习样本。
    特征：前 window 期的 6 红 + 1 蓝 展开为平铺向量。
    目标：下一期的各位置红球(red1-6)与蓝球。
    """
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    X_list = []
    y_red = {i: [] for i in range(6)}
    y_blue = []
    for i in range(window, len(data)):
        window_slice = data[i - window : i].reshape(-1)
        X_list.append(window_slice)
        for p in range(6):
            y_red[p].append(data[i][p])
        y_blue.append(data[i][6])
    X = np.stack(X_list)
    y_blue = np.array(y_blue)
    y_red = {k: np.array(v) for k, v in y_red.items()}
    return X, y_red, y_blue


def _cat_model_paths(save_dir: Path, window: int, iterations: int, depth: int, learning_rate: float):
    base = f"cat_w{window}_iter{iterations}_d{depth}_lr{learning_rate}"
    red_paths = {p: save_dir / f"{base}_red{p}.cbm" for p in range(6)}
    blue_path = save_dir / f"{base}_blue.cbm"
    return red_paths, blue_path


def bayes_optimize_catboost(
    df: pd.DataFrame,
    window: int = 10,
    n_iter: int = 15,
    cv_splits: int = 3,
    save_dir: str | None = "models",
) -> Tuple[CatBoostClassifier, Dict, float]:
    """
    使用贝叶斯优化（skopt）对 CatBoost 蓝球模型进行调参。
    仅优化蓝球多分类，搜索 depth / learning_rate / iterations。
    """
    try:
        from skopt import BayesSearchCV
        from skopt.space import Integer, Real
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 与 scikit-learn 以运行贝叶斯调参") from e

    X, _, y_blue = _prepare_dataset(df, window=window)
    if len(np.unique(y_blue)) < 2 or len(y_blue) < 20:
        raise ValueError("样本或类别过少，无法进行贝叶斯调参")

    params = {
        "loss_function": "MultiClass",
        "verbose": False,
        "random_seed": 42,
    }
    if torch.cuda.is_available():
        params["task_type"] = "GPU"
        params["devices"] = "0"

    estimator = CatBoostClassifier(**params)
    search_spaces = {
        "depth": Integer(4, 10),
        "learning_rate": Real(0.02, 0.3, prior="log-uniform"),
        "iterations": Integer(80, 400),
    }
    # 折数保护：不超过样本数
    n_splits = max(2, min(cv_splits, len(y_blue)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    opt = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=1,
        random_state=42,
        verbose=0,
    )
    opt.fit(X, y_blue)
    best_est: CatBoostClassifier = opt.best_estimator_
    best_params = opt.best_params_
    best_score = float(opt.best_score_)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_path = save_path / f"cat_bayes_w{window}.cbm"
        best_est.save_model(model_path, format="cbm")
        print(f"[bayes-cat] 最优模型已保存到 {model_path}")

    return best_est, best_params, best_score


def train_models(
    df: pd.DataFrame,
    window: int = 10,
    iterations: int = 300,
    depth: int = 6,
    learning_rate: float = 0.1,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    """
    训练 6 个位置红球模型 + 1 个蓝球模型（CatBoost 多分类）。
    支持持久化与加载：
      - save_dir: 模型保存目录（None 则不保存/加载）
      - resume: 尝试加载已存在模型
      - fresh: 强制重训并覆盖
    """
    X, y_red, y_blue = _prepare_dataset(df, window=window)
    models_red = {}
    # 参数基础
    params = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "loss_function": "MultiClass",
        "verbose": 100,
        "random_seed": 42,
    }
    # 优先尝试 GPU，若不可用则自动回退 CPU
    if torch.cuda.is_available():
        params["task_type"] = "GPU"
        params["devices"] = "0"

    # 尝试加载已保存模型（仅当 save_dir 指定且 resume 且非 fresh）
    save_path = None
    red_paths = blue_path = None
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        red_paths, blue_path = _cat_model_paths(save_path, window, iterations, depth, learning_rate)
        if resume and not fresh:
            try:
                loaded_red = {}
                for p, pth in red_paths.items():
                    model = CatBoostClassifier()
                    model.load_model(pth)
                    loaded_red[p] = model
                blue_model_loaded = CatBoostClassifier()
                blue_model_loaded.load_model(blue_path)
                print("[catboost] 检测到已保存模型，直接加载复用")
                return {"red": loaded_red, "blue": blue_model_loaded, "window": window}
            except Exception:
                print("[catboost] 加载已保存模型失败，将重新训练并覆盖")

    # 构造 9:1 训练/验证，用于 loss 改进早停；若验证类缺失则回退无早停
    n = len(X)
    val_size = max(1, int(n * 0.1))
    perm = np.random.permutation(n)
    X_perm = X[perm]
    y_red_perm = {k: v[perm] for k, v in y_red.items()}
    y_blue_perm = y_blue[perm]

    if n - val_size > 0:
        X_train = X_perm[: n - val_size]
        X_val = X_perm[n - val_size :]
        y_train = {k: v[: n - val_size] for k, v in y_red_perm.items()}
        y_val = {k: v[n - val_size :] for k, v in y_red_perm.items()}
        y_train_blue = y_blue_perm[: n - val_size]
        y_val_blue = y_blue_perm[n - val_size :]
        # 检查验证集标签是否包含训练集中不存在的类别
        train_classes = {k: set(v.tolist()) for k, v in y_train.items()}
        val_ok = all(set(y_val[k].tolist()).issubset(train_classes[k]) for k in y_val)
    else:
        val_ok = False
        X_train, X_val = X_perm, None
        y_train = y_red_perm
        y_val = None
        y_train_blue = y_blue_perm
        y_val_blue = None

    if not val_ok:
        # 回退为全量训练，不使用验证集，避免类别缺失或长度不匹配
        X_train, X_val = X_perm, None
        y_train = y_red_perm
        y_val = None
        y_train_blue = y_blue_perm
        y_val_blue = None

    # 根据验证集可用性设置早停
    if val_ok:
        params["use_best_model"] = True
        params["od_type"] = "Iter"
        params["od_wait"] = max(50, iterations // 5)
    else:
        params["use_best_model"] = False
        params["od_type"] = None

    for p in range(6):
        model = CatBoostClassifier(**params)
        if val_ok:
            model.fit(X_train, y_train[p], eval_set=(X_val, y_val[p]))
        else:
            model.fit(X_train, y_train[p])
        models_red[p] = model

    blue_model = CatBoostClassifier(**params)
    if val_ok:
        blue_model.fit(X_train, y_train_blue, eval_set=(X_val, y_val_blue))
    else:
        blue_model.fit(X_train, y_blue)
    # 保存
    if save_path is not None:
        for p, m in models_red.items():
            m.save_model(red_paths[p], format="cbm")
        blue_model.save_model(blue_path, format="cbm")

    return {"red": models_red, "blue": blue_model, "window": window}


def predict_next(models: dict, df: pd.DataFrame, top_k: int = 3):
    """
    使用训练好的模型对最新 window 窗口做预测。
    返回：每个位置 top_k 概率最高的号码，以及蓝球 top_k。
    """
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    window = models["window"]
    data = df[cols].to_numpy(dtype=int)
    if len(data) < window:
        raise ValueError("样本不足，无法进行预测")
    x = data[-window:].reshape(1, -1)

    red_preds = {}
    for p, model in models["red"].items():
        proba = model.predict_proba(x)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        red_preds[p + 1] = [(int(i), float(proba[i])) for i in top_idx]

    blue_model = models["blue"]
    proba_b = blue_model.predict_proba(x)[0]
    top_idx_b = np.argsort(proba_b)[::-1][:top_k]
    blue_preds = [(int(i), float(proba_b[i])) for i in top_idx_b]
    return {"red": red_preds, "blue": blue_preds}

