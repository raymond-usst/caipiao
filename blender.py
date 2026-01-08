from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler


def _build_base_preds(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
) -> pd.DataFrame:
    """
    对每期生成基础模型预测特征。
    predictor 返回示例：
      {
        "red": {pos: [(num, prob), ...]},
        "blue": [(num, prob)],
        "sum_pred": float
      }
    """
    rows = []
    for i in range(len(df)):
        hist = df.iloc[: i + 1]
        preds_merge: Dict[str, float] = {}
        for j, pred_fn in enumerate(base_predictors):
            try:
                preds = pred_fn(hist)
            except Exception:
                continue
            # 蓝球 top1（兼容多种返回格式：list[(num,prob)] / list[num] / float/int / blue_pred）
            blue_val = preds.get("blue")
            if blue_val is None and "blue_pred" in preds:
                blue_val = preds["blue_pred"]
            blue_num = None
            blue_prob = None
            if isinstance(blue_val, (list, tuple)):
                if len(blue_val) > 0:
                    first = blue_val[0]
                    if isinstance(first, (list, tuple)) and len(first) >= 2:
                        blue_num, blue_prob = first[0], first[1]
                    elif isinstance(first, (int, float)):
                        blue_num, blue_prob = first, 1.0
            elif isinstance(blue_val, (int, float)):
                blue_num, blue_prob = blue_val, 1.0
            if blue_num is not None:
                preds_merge[f"m{j}_blue_num"] = blue_num
                preds_merge[f"m{j}_blue_prob"] = blue_prob if blue_prob is not None else 1.0

            # 和值
            if preds.get("sum_pred") is not None:
                preds_merge[f"m{j}_sum_pred"] = float(preds["sum_pred"])
            # 红球各位置 top1
            if preds.get("red"):
                for pos, arr in preds["red"].items():
                    if arr:
                        preds_merge[f"m{j}_r{pos}_num"] = arr[0][0]
                        preds_merge[f"m{j}_r{pos}_prob"] = arr[0][1]
        if preds_merge:
            rows.append({**{"idx": i, "issue": df.iloc[i]["issue"]}, **preds_merge})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 缺失特征填 0，避免后续模型因 NaN 报错
    return out.fillna(0.0)


@dataclass
class BlendConfig:
    train_size: int = 300
    test_size: int = 20
    step: int = 20
    alpha: float = 0.5
    l1_ratio: float = 0.2


def rolling_blend_blue(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    cfg: BlendConfig,
) -> Tuple[float, List[Dict]]:
    """
    针对蓝球命中做滚动融合。
    - 构造特征：各模型蓝球 top1 号码与概率。
    - 标签：蓝球真实命中（one-vs-rest 简化为 top1 命中）。
    """
    results: List[Dict] = []
    total = len(df)
    fold = 0
    acc_list = []
    for start in range(0, total - cfg.train_size - cfg.test_size + 1, cfg.step):
        train_df = df.iloc[start : start + cfg.train_size]
        test_df = df.iloc[start + cfg.train_size : start + cfg.train_size + cfg.test_size]
        feat_train = _build_base_preds(train_df, base_predictors)
        feat_test = _build_base_preds(pd.concat([train_df, test_df]), base_predictors)
        if feat_train.empty or feat_test.empty:
            continue
        # 对齐到训练长度
        feat_test = feat_test.iloc[len(feat_train) :]

        y_train = (train_df["blue"].iloc[-len(feat_train) :].to_numpy())
        y_test = (test_df["blue"].to_numpy())

        X_train = feat_train.drop(columns=["idx", "issue"])
        X_test = feat_test.drop(columns=["idx", "issue"])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=500, multi_class="auto")
        clf.fit(X_train_scaled, y_train)
        pred_top1 = clf.predict(X_test_scaled)
        acc = float((pred_top1 == y_test).mean())
        acc_list.append(acc)
        results.append({"fold": fold, "acc": acc})
        fold += 1
    avg_acc = float(np.mean(acc_list)) if acc_list else 0.0
    return avg_acc, results


def rolling_blend_sum(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    cfg: BlendConfig,
) -> Tuple[float, List[Dict]]:
    """和值回归融合，返回平均 MAE。"""
    results: List[Dict] = []
    total = len(df)
    fold = 0
    maes = []
    for start in range(0, total - cfg.train_size - cfg.test_size + 1, cfg.step):
        train_df = df.iloc[start : start + cfg.train_size]
        test_df = df.iloc[start + cfg.train_size : start + cfg.train_size + cfg.test_size]
        feat_train = _build_base_preds(train_df, base_predictors)
        feat_test = _build_base_preds(pd.concat([train_df, test_df]), base_predictors)
        if feat_train.empty or feat_test.empty:
            continue
        feat_test = feat_test.iloc[len(feat_train) :]
        y_train = train_df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1).to_numpy()[-len(feat_train) :]
        y_test = test_df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1).to_numpy()
        X_train = feat_train.drop(columns=["idx", "issue"])
        X_test = feat_test.drop(columns=["idx", "issue"])
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        reg = ElasticNet(alpha=cfg.alpha, l1_ratio=cfg.l1_ratio, max_iter=1000)
        reg.fit(X_train_s, y_train)
        pred = reg.predict(X_test_s)
        mae = float(np.mean(np.abs(pred - y_test)))
        maes.append(mae)
        results.append({"fold": fold, "mae": mae})
        fold += 1
    avg_mae = float(np.mean(maes)) if maes else float("inf")
    return avg_mae, results


def rolling_blend_red(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    cfg: BlendConfig,
) -> Tuple[float, List[Dict]]:
    """红球位置分类融合，返回平均位置Top1命中。"""
    total = len(df)
    fold = 0
    accs = []
    results = []
    for start in range(0, total - cfg.train_size - cfg.test_size + 1, cfg.step):
        train_df = df.iloc[start : start + cfg.train_size]
        test_df = df.iloc[start + cfg.train_size : start + cfg.train_size + cfg.test_size]
        feat_train = _build_base_preds(train_df, base_predictors)
        feat_test = _build_base_preds(pd.concat([train_df, test_df]), base_predictors)
        if feat_train.empty or feat_test.empty:
            continue
        feat_test = feat_test.iloc[len(feat_train) :]
        pos_acc = []
        for pos in range(1, 7):
            cols_pos = [c for c in feat_train.columns if f"_r{pos}_" in c]
            if not cols_pos:
                continue
            X_train = feat_train[cols_pos]
            X_test = feat_test[cols_pos]
            y_train = train_df[f"red{pos}"].iloc[-len(feat_train) :].to_numpy()
            y_test = test_df[f"red{pos}"].to_numpy()
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            clf = LogisticRegression(max_iter=500, multi_class="auto")
            clf.fit(X_train_s, y_train)
            pred = clf.predict(X_test_s)
            pos_acc.append(float((pred == y_test).mean()))
        if pos_acc:
            fold_acc = float(np.mean(pos_acc))
            accs.append(fold_acc)
            results.append({"fold": fold, "acc": fold_acc})
        fold += 1
    avg_acc = float(np.mean(accs)) if accs else 0.0
    return avg_acc, results


def blend_blue_latest(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
) -> Tuple[int, float]:
    """
    在全量上训练蓝球融合模型并输出最新期的融合 Top1。
    """
    feat = _build_base_preds(df, base_predictors)
    if feat.empty:
        raise ValueError("基础预测为空，无法融合")
    y = df["blue"].iloc[-len(feat) :].to_numpy()
    X = feat.drop(columns=["idx", "issue"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500, multi_class="auto")
    clf.fit(X_scaled, y)
    last_feat = X_scaled[-1].reshape(1, -1)
    proba = clf.predict_proba(last_feat)[0]
    top_idx = int(np.argmax(proba))
    return top_idx + 1, float(proba[top_idx])


def blend_sum_latest(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
) -> float:
    feat = _build_base_preds(df, base_predictors)
    if feat.empty:
        raise ValueError("基础预测为空，无法融合")
    y = df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1).iloc[-len(feat) :].to_numpy()
    X = feat.drop(columns=["idx", "issue"])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reg = ElasticNet(alpha=0.5, l1_ratio=0.2, max_iter=1000)
    reg.fit(Xs, y)
    pred = float(reg.predict(Xs[-1].reshape(1, -1))[0])
    return pred


def blend_red_latest(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
) -> Dict[int, int]:
    feat = _build_base_preds(df, base_predictors)
    if feat.empty:
        raise ValueError("基础预测为空，无法融合")
    res: Dict[int, int] = {}
    for pos in range(1, 7):
        cols_pos = [c for c in feat.columns if f"_r{pos}_" in c]
        if not cols_pos:
            continue
        X = feat[cols_pos]
        y = df[f"red{pos}"].iloc[-len(feat) :].to_numpy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=500, multi_class="auto")
        clf.fit(Xs, y)
        proba = clf.predict_proba(Xs[-1].reshape(1, -1))[0]
        top_idx = int(np.argmax(proba))
        res[pos] = top_idx + 1
    return res

