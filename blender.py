from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import warnings
import itertools
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from xgboost import XGBClassifier

# 静默 sklearn 关于 multi_class 弃用的重复警告
warnings.filterwarnings("ignore", message=".*multi_class.*deprecated.*LogisticRegression.*")


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
            # 蓝球概率：优先 blue_probs，其次 blue（list/tuple），再次 blue_pred 单值
            blue_probs = preds.get("blue_probs")
            if blue_probs is None:
                blue_val = preds.get("blue")
                if blue_val is None and "blue_pred" in preds:
                    blue_val = preds["blue_pred"]
                if isinstance(blue_val, (list, tuple)) and len(blue_val) > 0:
                    if isinstance(blue_val[0], (list, tuple)) and len(blue_val[0]) >= 2:
                        blue_probs = blue_val
                    elif isinstance(blue_val[0], (int, float)):
                        blue_probs = [(blue_val[0], 1.0)]
                elif isinstance(blue_val, (int, float)):
                    blue_probs = [(blue_val, 1.0)]
            if blue_probs:
                preds_merge[f"m{j}_blue_num"] = blue_probs[0][0]
                preds_merge[f"m{j}_blue_prob"] = float(blue_probs[0][1])

            # 和值
            if preds.get("sum_pred") is not None:
                preds_merge[f"m{j}_sum_pred"] = float(preds["sum_pred"])
            # 红球各位置 top1
            if preds.get("red"):
                for pos, arr in preds["red"].items():
                    if arr:
                        preds_merge[f"m{j}_r{pos}_num"] = arr[0][0]
                        preds_merge[f"m{j}_r{pos}_prob"] = arr[0][1]
            elif preds.get("red_probs"):
                red_probs = preds["red_probs"]
                if red_probs and isinstance(red_probs, list):
                    top_num, top_p = red_probs[0]
                    for pos in range(1, 7):
                        preds_merge[f"m{j}_r{pos}_num"] = int(top_num)
                        preds_merge[f"m{j}_r{pos}_prob"] = float(top_p)
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

        clf = LogisticRegression(max_iter=500)
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
            clf = LogisticRegression(max_iter=500)
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
    clf = LogisticRegression(max_iter=500)
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
        clf = LogisticRegression(max_iter=500)
        clf.fit(Xs, y)
        proba = clf.predict_proba(Xs[-1].reshape(1, -1))[0]
        top_idx = int(np.argmax(proba))
        res[pos] = top_idx + 1
    return res


def _aggregate_probs(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    聚合各基础模型的红/蓝概率为全局分布。
    优先 red_probs/blue_probs，其次位置 red，再次单值 fallback。
    """
    red_scores = np.zeros(34)
    blue_scores = np.zeros(17)
    for pred_fn in base_predictors:
        try:
            preds = pred_fn(df)
        except Exception:
            continue
        red_probs = preds.get("red_probs")
        if red_probs:
            for n, p in red_probs:
                if 1 <= int(n) <= 33:
                    red_scores[int(n)] += float(p)
        elif preds.get("red"):
            for _, arr in preds["red"].items():
                if arr:
                    for n, p in arr:
                        if 1 <= int(n) <= 33:
                            red_scores[int(n)] += float(p)
        blue_probs = preds.get("blue_probs")
        if not blue_probs:
            blue_val = preds.get("blue")
            if blue_val is None and "blue_pred" in preds:
                blue_val = preds["blue_pred"]
            if isinstance(blue_val, (list, tuple)) and len(blue_val) > 0:
                if isinstance(blue_val[0], (list, tuple)) and len(blue_val[0]) >= 2:
                    blue_probs = blue_val
                elif isinstance(blue_val[0], (int, float)):
                    blue_probs = [(blue_val[0], 1.0)]
            elif isinstance(blue_val, (int, float)):
                blue_probs = [(blue_val, 1.0)]
        if blue_probs:
            for n, p in blue_probs:
                if 1 <= int(n) <= 16:
                    blue_scores[int(n)] += float(p)
    # 归一化
    if red_scores.sum() <= 0:
        red_scores[1:] = 1.0 / 33
    if blue_scores.sum() <= 0:
        blue_scores[1:] = 1.0 / 16
    red_probs_out = [(i, float(red_scores[i] / red_scores.sum())) for i in range(1, 34)]
    blue_probs_out = [(i, float(blue_scores[i] / blue_scores.sum())) for i in range(1, 17)]
    red_probs_out.sort(key=lambda x: x[1], reverse=True)
    blue_probs_out.sort(key=lambda x: x[1], reverse=True)
    return red_probs_out, blue_probs_out


def _prob_vector(pairs: List[Tuple[int, float]], size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=float)
    for n, p in pairs:
        idx = int(n) - 1
        if 0 <= idx < size:
            vec[idx] = float(p)
    s = vec.sum()
    if s > 0:
        vec = vec / s
    else:
        vec = np.ones(size, dtype=float) / size
    return vec


def _stack_features(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    target: str = "blue",
    pos: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 stacking 训练特征：
      - 使用历史数据 hist = df[:i] 预测第 i 条，避免信息泄露。
      - features: 拼接各模型的概率向量（蓝16维或红33维）。
    """
    feats = []
    labels = []
    for i in range(1, len(df)):
        hist = df.iloc[:i]
        tgt_row = df.iloc[i]
        pred_vecs = []
        for pred_fn in base_predictors:
            try:
                preds = pred_fn(hist)
            except Exception:
                # 缺失则用均匀分布
                pred_vecs.append(np.ones(16 if target == "blue" else 33) / (16 if target == "blue" else 33))
                continue
            if target == "blue":
                blue_probs = preds.get("blue_probs")
                if not blue_probs:
                    blue_val = preds.get("blue")
                    if blue_val is None and "blue_pred" in preds:
                        blue_val = preds["blue_pred"]
                    if isinstance(blue_val, (list, tuple)) and len(blue_val) > 0:
                        if isinstance(blue_val[0], (list, tuple)) and len(blue_val[0]) >= 2:
                            blue_probs = blue_val
                        elif isinstance(blue_val[0], (int, float)):
                            blue_probs = [(blue_val[0], 1.0)]
                    elif isinstance(blue_val, (int, float)):
                        blue_probs = [(blue_val, 1.0)]
                vec = _prob_vector(blue_probs or [], 16)
                pred_vecs.append(vec)
            else:
                red_probs = None
                if preds.get("red_probs"):
                    red_probs = preds["red_probs"]
                elif preds.get("red") and pos is not None and preds["red"].get(pos):
                    red_probs = preds["red"][pos]
                vec = _prob_vector(red_probs or [], 33)
                pred_vecs.append(vec)
        feats.append(np.concatenate(pred_vecs))
        if target == "blue":
            labels.append(int(tgt_row["blue"]) - 1)
        else:
            labels.append(int(tgt_row[f"red{pos}"]) - 1)
    if not feats:
        return np.empty((0, 0)), np.array([])
    return np.stack(feats), np.array(labels)


def _fit_meta_classifier(X: np.ndarray, y: np.ndarray, bayes: bool = False, n_iter: int = 8):
    if X.size == 0 or y.size == 0:
        raise ValueError("stacking 训练样本为空")
    # train/val 划分；若类别极端稀疏则不使用 stratify
    _, counts_full = np.unique(y, return_counts=True)
    use_stratify = counts_full.min() >= 2
    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if use_stratify else None
    )
    # 使用训练集拟合编码器，保证标签连续且与模型类别一致；验证集中未见标签直接过滤
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    mask_val = np.isin(y_val_raw, le.classes_)
    X_val = X_val[mask_val]
    y_val_raw = y_val_raw[mask_val]
    y_val = le.transform(y_val_raw)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1.0,
        verbosity=0,
    )
    best_model = base_model
    n_iter = max(int(n_iter), 10)  # bayes 至少 10 次迭代
    # 动态选择 CV：若最小类别不足 2，则使用 KFold，否则 StratifiedKFold
    _, counts_full = np.unique(y_train, return_counts=True)
    min_class = counts_full.min()
    if min_class >= 2 and len(y_train) >= 3:
        n_splits = min(3, int(min_class))
        cv_obj = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = min(3, len(y_train))
        n_splits = max(2, n_splits)
        cv_obj = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if bayes:
        try:
            from skopt import BayesSearchCV
            from skopt.space import Integer, Real

            search = BayesSearchCV(
                estimator=base_model,
                search_spaces={
                    "n_estimators": Integer(120, 400),
                    "max_depth": Integer(3, 8),
                    "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
                    "subsample": Real(0.6, 1.0),
                    "colsample_bytree": Real(0.6, 1.0),
                    "min_child_weight": Real(0.5, 5.0),
                },
                n_iter=n_iter,
                cv=cv_obj,
                random_state=42,
                verbose=0,
            )
            search.fit(X_train_s, y_train)
            best_model = search.best_estimator_
        except Exception:
            # 兜底随机采样，避免调参失败直接退化
            rng = np.random.default_rng(42)
            best_score = 1e9
            best_model = None
            for _ in range(n_iter):
                params = {
                    "n_estimators": int(rng.integers(120, 401)),
                    "max_depth": int(rng.integers(3, 9)),
                    "learning_rate": float(np.exp(rng.uniform(np.log(1e-3), np.log(0.3)))),
                    "subsample": float(rng.uniform(0.6, 1.0)),
                    "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                    "min_child_weight": float(rng.uniform(0.5, 5.0)),
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "verbosity": 0,
                }
                model_try = XGBClassifier(**params)
                model_try.fit(X_train_s, y_train)
                score = float(model_try.score(X_val_s, y_val))
                if score < best_score:
                    best_score = score
                    best_model = model_try
            if best_model is None:
                best_model = base_model
    best_model.fit(X_train_s, y_train)
    return scaler, best_model, le


def train_stacking_blue(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    bayes: bool = False,
    n_iter: int = 6,
):
    X, y = _stack_features(df, base_predictors, target="blue")
    scaler, model, le = _fit_meta_classifier(X, y, bayes=bayes, n_iter=n_iter)
    return scaler, model, le


def predict_stacking_blue(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    scaler,
    model,
    le: LabelEncoder,
) -> Tuple[int, float]:
    # 使用全量历史生成特征预测下一期
    pred_vecs = []
    for pred_fn in base_predictors:
        try:
            preds = pred_fn(df)
        except Exception:
            pred_vecs.append(np.ones(16) / 16)
            continue
        vec = _prob_vector(preds.get("blue_probs") or [], 16)
        pred_vecs.append(vec)
    X = np.concatenate(pred_vecs).reshape(1, -1)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[0]
    classes = le.classes_
    top_idx = int(np.argmax(proba))
    top_class = int(classes[top_idx])
    return top_class + 1, float(proba[top_idx])


def train_stacking_red(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    bayes: bool = False,
    n_iter: int = 6,
):
    scalers = {}
    models = {}
    label_encoders = {}
    for pos in range(1, 7):
        X, y = _stack_features(df, base_predictors, target="red", pos=pos)
        if X.size == 0:
            continue
        scaler, model, le = _fit_meta_classifier(X, y, bayes=bayes, n_iter=n_iter)
        scalers[pos] = scaler
        models[pos] = model
        label_encoders[pos] = le
    return scalers, models, label_encoders


def predict_stacking_red(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    scalers: Dict[int, StandardScaler],
    models: Dict[int, LogisticRegression],
    label_encoders: Dict[int, LabelEncoder],
    topk: int = 1,
) -> Dict[int, List[Tuple[int, float]]]:
    res: Dict[int, List[Tuple[int, float]]] = {}
    for pos, model in models.items():
        vecs = []
        for pred_fn in base_predictors:
            try:
                preds = pred_fn(df)
            except Exception:
                vecs.append(np.ones(33) / 33)
                continue
            rp = None
            if preds.get("red_probs"):
                rp = preds["red_probs"]
            elif preds.get("red") and preds["red"].get(pos):
                rp = preds["red"][pos]
            vecs.append(_prob_vector(rp or [], 33))
        X = np.concatenate(vecs).reshape(1, -1)
        scaler = scalers[pos]
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)[0]
        classes = label_encoders[pos].classes_
        top_idx = np.argsort(proba)[::-1][:topk]
        res[pos] = [(int(classes[i] + 1), float(proba[i])) for i in top_idx]
    return res


def generate_duplex_combos(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    top_red: int = 10,
    top_blue: int = 5,
    sum_range: Tuple[int, int] | None = None,
    odd_count: int | None = None,
    max_combos: int = 30,
) -> List[Dict]:
    """
    基于聚合概率生成符合约束的复式组合。
    约束：
      - top_red: 从红球概率前N中选6
      - top_blue: 从蓝球概率前M中任选1
      - sum_range: (min, max) 可选
      - odd_count: 指定红球奇数个数，可选
    """
    red_probs, blue_probs = _aggregate_probs(df, base_predictors)
    red_cands = [n for n, _ in red_probs[:top_red]]
    blue_cands = blue_probs[:top_blue]

    combos = []
    for reds in itertools.combinations(red_cands, 6):
        if odd_count is not None:
            if sum(r % 2 for r in reds) != odd_count:
                continue
        s = sum(reds)
        if sum_range and not (sum_range[0] <= s <= sum_range[1]):
            continue
        score_r = 0.0
        for r in reds:
            # 使用 log 概率求和避免下溢
            p = dict(red_probs).get(r, 1e-9)
            score_r += math.log(p + 1e-9)
        for b, pb in blue_cands:
            score = score_r + math.log(pb + 1e-9)
            combos.append(
                {
                    "reds": sorted(reds),
                    "blue": int(b),
                    "score": score,
                    "sum": s,
                    "odd": sum(r % 2 for r in reds),
                }
            )
    combos.sort(key=lambda x: x["score"], reverse=True)
    return combos[:max_combos]


def kill_numbers(
    df: pd.DataFrame,
    base_predictors: List[Callable[[pd.DataFrame], Dict]],
    red_thresh: float = 1e-4,  # 0.01%
    blue_thresh: float = 1e-4,
) -> Dict[str, List[int]]:
    """
    输出概率极低的红/蓝号码列表，供过滤。
    """
    red_probs, blue_probs = _aggregate_probs(df, base_predictors)
    red_kill = [n for n, p in red_probs if p < red_thresh]
    blue_kill = [n for n, p in blue_probs if p < blue_thresh]
    return {"red_kill": red_kill, "blue_kill": blue_kill}

