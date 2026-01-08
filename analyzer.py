from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree
from itertools import combinations


def load_dataframe(conn, recent: Optional[int] = None) -> pd.DataFrame:
    base_sql = """
        SELECT
            issue, draw_date,
            red1, red2, red3, red4, red5, red6,
            blue
        FROM draws
        ORDER BY draw_date DESC
    """
    if recent:
        base_sql += " LIMIT ?"
        df = pd.read_sql(base_sql, conn, params=(recent,))
    else:
        df = pd.read_sql(base_sql, conn)
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    return df.sort_values("draw_date")


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def frequency_stats(df: pd.DataFrame, alpha: float = 0.5) -> Dict[str, Dict[int, float]]:
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    reds = df[red_cols].to_numpy(dtype=int)
    blues = df["blue"].to_numpy(dtype=int)

    red_counts = np.bincount(reds.ravel(), minlength=34)[1:]  # 1..33
    blue_counts = np.bincount(blues, minlength=17)[1:]  # 1..16

    red_probs = (red_counts + alpha) / (red_counts.sum() + alpha * 33)
    blue_probs = (blue_counts + alpha) / (blue_counts.sum() + alpha * 16)

    return {
        "red": {i + 1: float(p) for i, p in enumerate(red_probs)},
        "blue": {i + 1: float(p) for i, p in enumerate(blue_probs)},
    }


def hot_cold(prob_map: Dict[int, float], top_k: int = 6) -> Tuple[List[int], List[int]]:
    sorted_items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    hot = [n for n, _ in sorted_items[:top_k]]
    cold = [n for n, _ in sorted_items[-top_k:]]
    return hot, cold


def moving_entropy(df: pd.DataFrame, window: int = 30) -> List[Tuple[str, float]]:
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    flattened = df[red_cols].to_numpy(dtype=int).ravel()
    ent_list: List[Tuple[str, float]] = []
    for i in range(window, len(flattened) + 1):
        segment = flattened[i - window : i]
        counts = np.bincount(segment, minlength=34)[1:]
        probs = counts / counts.sum()
        ent_list.append((str(df["draw_date"].iloc[(i - 1) // 6].date()), _entropy(probs)))
    return ent_list


def _embed(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n_vectors = len(series) - (dim - 1) * tau
    if n_vectors <= 1:
        return np.empty((0, dim))
    return np.stack([series[i : i + dim * tau : tau] for i in range(n_vectors)], axis=0)


def lyapunov_exponent(series: Sequence[float], dim: int = 6, tau: int = 1, max_t: int = 6) -> Optional[float]:
    arr = np.asarray(series, dtype=float)
    X = _embed(arr, dim, tau)
    n = len(X)
    if n < 5:
        return None

    log_divergences = []
    for i in range(n):
        # 排除时间上过近的邻居
        idx = np.arange(n)
        mask = np.abs(idx - i) > dim
        candidates = idx[mask]
        if len(candidates) == 0:
            continue

        dists = np.linalg.norm(X[candidates] - X[i], axis=1)
        j = candidates[np.argmin(dists)]
        d0 = dists[np.argmin(dists)]
        if d0 < 1e-9:
            continue

        local_logs = []
        for k in range(1, max_t + 1):
            if i + k >= n or j + k >= n:
                break
            dk = np.linalg.norm(X[i + k] - X[j + k])
            local_logs.append(math.log(dk + 1e-12) - math.log(d0))
        if len(local_logs) >= 2:
            steps = np.arange(1, len(local_logs) + 1)
            slope, _ = np.polyfit(steps, local_logs, 1)
            log_divergences.append(slope)

    if not log_divergences:
        return None
    return float(np.mean(log_divergences))


def correlation_dimension(
    series: Sequence[float],
    dim: int = 3,
    tau: int = 1,
    r_values: Optional[Sequence[float]] = None,
    max_points: int = 1200,
) -> Optional[float]:
    """简化版 GP 相关维计算：对嵌入点对距离做 log-log 拟合估计维度."""
    arr = np.asarray(series, dtype=float)
    X = _embed(arr, dim, tau)
    if len(X) < 20:
        return None
    if max_points and len(X) > max_points:
        X = X[:max_points]
    # 距离矩阵（上三角）
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    dists = dists[np.triu_indices(len(X), k=1)]
    dists = dists[dists > 0]
    if len(dists) == 0:
        return None
    dists.sort()
    if r_values is None:
        # 取 10 个对数均匀间隔的 r
        r_min, r_max = np.percentile(dists, [5, 80])
        r_values = np.logspace(math.log10(r_min), math.log10(r_max), num=10)
    C = []
    for r in r_values:
        C.append((dists < r).mean())
    log_r = np.log(r_values)
    log_C = np.log(C)
    # 取中段做线性拟合
    mid = slice(len(r_values) // 4, -len(r_values) // 4 or None)
    if mid.start >= len(log_r) or mid.stop is None or mid.stop <= mid.start:
        return None
    slope, _ = np.polyfit(log_r[mid], log_C[mid], 1)
    return float(slope)


def false_nearest_neighbors(
    series: Sequence[float],
    max_dim: int = 6,
    tau: int = 1,
    rtol: float = 15.0,
    atol: float = 2.0,
    max_points: int = 1200,
) -> Dict[int, float]:
    """Kennel 方法近似：返回各维度的虚假最近邻比例."""
    arr = np.asarray(series, dtype=float)
    result: Dict[int, float] = {}
    std_all = arr.std()
    for m in range(1, max_dim):
        X_m = _embed(arr, m, tau)
        X_m1 = _embed(arr, m + 1, tau)
        if len(X_m) < 10 or len(X_m1) < 10:
            break
        if max_points and len(X_m) > max_points:
            X_m = X_m[:max_points]
            X_m1 = X_m1[:max_points]
        # 最近邻
        dists = np.linalg.norm(X_m[:, None, :] - X_m[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        nn_idx = np.argmin(dists, axis=1)
        d_m = dists[np.arange(len(X_m)), nn_idx]
        # 扩展一维后的距离
        d_m1 = np.linalg.norm(X_m1 - X_m1[nn_idx], axis=1)
        false_mask = (d_m > 0) & (((d_m1 - d_m) / d_m > rtol) | (d_m1 / (std_all + 1e-9) > atol))
        result[m] = float(false_mask.mean())
    return result


def suggest_numbers(prob_red: Dict[int, float], prob_blue: Dict[int, float]) -> Tuple[List[int], int]:
    hot_red, _ = hot_cold(prob_red, top_k=6)
    blue_pick = max(prob_blue.items(), key=lambda kv: kv[1])[0]
    return sorted(hot_red), blue_pick


def basic_stats(df: pd.DataFrame) -> Dict[str, float]:
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    red_vals = df[red_cols].to_numpy(dtype=float).ravel()
    blue_vals = df["blue"].to_numpy(dtype=float)
    sums = df[red_cols + ["blue"]].sum(axis=1).to_numpy(dtype=float)

    return {
        "red_mean": float(red_vals.mean()),
        "red_std": float(red_vals.std(ddof=1)),
        "blue_mean": float(blue_vals.mean()),
        "blue_std": float(blue_vals.std(ddof=1)),
        "sum_mean": float(sums.mean()),
        "sum_std": float(sums.std(ddof=1)),
    }


def chi_square_uniform(counts: np.ndarray) -> Dict[str, float]:
    total = counts.sum()
    k = len(counts)
    if total == 0 or k == 0:
        return {"stat": float("nan"), "dof": 0}
    expected = total / k
    stat = float(((counts - expected) ** 2 / expected).sum())
    return {"stat": stat, "dof": k - 1}


def runs_test(sequence: Sequence[float]) -> Optional[Dict[str, float]]:
    seq = np.asarray(sequence, dtype=float)
    if len(seq) < 10:
        return None
    median = np.median(seq)
    bits = (seq >= median).astype(int)
    n1 = int(bits.sum())
    n0 = len(bits) - n1
    if n1 == 0 or n0 == 0:
        return None
    runs = 1 + np.sum(bits[1:] != bits[:-1])
    mu = 1 + (2 * n1 * n0) / (n1 + n0)
    sigma_sq = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
    if sigma_sq <= 0:
        return None
    z = (runs - mu) / math.sqrt(sigma_sq)
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"runs": runs, "z": float(z), "p": float(p)}


def autocorrelation(series: Sequence[float], max_lag: int = 10) -> Dict[int, float]:
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < 5:
        return {}
    result: Dict[int, float] = {}
    for lag in range(1, min(max_lag, n - 1) + 1):
        a = arr[:-lag]
        b = arr[lag:]
        if a.std() == 0 or b.std() == 0:
            result[lag] = float("nan")
            continue
        result[lag] = float(np.corrcoef(a, b)[0, 1])
    return result


def omission_stats(df: pd.DataFrame) -> Dict[str, Dict[int, Dict[str, int]]]:
    """计算当前遗漏与历史最大遗漏."""
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    red_missing: Dict[int, Dict[str, int]] = {}
    blue_missing: Dict[int, Dict[str, int]] = {}

    # 按时间顺序
    rows = df[red_cols + ["blue"]].to_numpy(dtype=int)

    def compute(range_end: int, idx_func):
        stats: Dict[int, Dict[str, int]] = {n: {"current": 0, "max": 0} for n in range(1, range_end + 1)}
        for row in rows:
            present_set = idx_func(row)
            for n in range(1, range_end + 1):
                if n in present_set:
                    stats[n]["current"] = 0
                else:
                    stats[n]["current"] += 1
                    stats[n]["max"] = max(stats[n]["max"], stats[n]["current"])
        return stats

    red_missing = compute(33, lambda r: set(r[:6]))
    blue_missing = compute(16, lambda r: {r[6]})
    return {"red": red_missing, "blue": blue_missing}


def omission_periodicity(df: pd.DataFrame, cv_threshold: float = 0.35, ratio_threshold: float = 1.8) -> Dict[str, Dict[str, List[Dict]]]:
    """分析遗漏间隔的近似周期性，并返回所有号码的间隔统计."""
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    rows = df[red_cols + ["blue"]].to_numpy(dtype=int)

    def gaps_for(num_range: int, idx_func):
        all_stats = []
        periodic = []
        for n in range(1, num_range + 1):
            positions = [i for i, row in enumerate(rows) if n in idx_func(row)]
            if len(positions) < 5:
                continue
            gaps = np.diff(positions)
            if len(gaps) < 3:
                continue
            mean_gap = gaps.mean()
            std_gap = gaps.std(ddof=1)
            cv = std_gap / mean_gap if mean_gap > 0 else float("inf")
            gap_ratio = gaps.max() / gaps.min() if gaps.min() > 0 else float("inf")
            item = {
                "num": n,
                "count": len(positions),
                "mean_gap": float(mean_gap),
                "std_gap": float(std_gap),
                "cv": float(cv),
                "min_gap": int(gaps.min()),
                "max_gap": int(gaps.max()),
            }
            all_stats.append(item)
            if np.isfinite(cv) and cv <= cv_threshold and gap_ratio <= ratio_threshold:
                periodic.append(item)
        all_stats.sort(key=lambda x: (x["cv"], x["mean_gap"]))
        periodic.sort(key=lambda x: (x["cv"], x["mean_gap"]))
        return {"all": all_stats, "periodic": periodic}

    red_periodic = gaps_for(33, lambda r: set(r[:6]))
    blue_periodic = gaps_for(16, lambda r: {r[6]})
    return {"red": red_periodic, "blue": blue_periodic}


def phase_space(series: Sequence[float], dim: int = 3, tau: int = 1, max_points: int = 1000) -> pd.DataFrame:
    """相空间重构，返回前 max_points 个重构向量."""
    arr = np.asarray(series, dtype=float)
    X = _embed(arr, dim, tau)
    if len(X) == 0:
        return pd.DataFrame()
    if max_points and len(X) > max_points:
        X = X[:max_points]
    cols = [f"x{i+1}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


def _embed_limited(series: Sequence[float], dim: int, tau: int, sample: int) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if len(arr) > sample:
        arr = arr[-sample:]
    return _embed(arr, dim, tau)


def correlation_dimension(series: Sequence[float], max_dim: int = 6, tau: int = 1, sample: int = 2000) -> Optional[Dict[int, float]]:
    """简化的 Grassberger-Procaccia：多尺度相关和斜率估算."""
    result = {}
    eps_quantiles = [0.05, 0.1, 0.2, 0.3]
    for m in range(2, max_dim + 1):
        X = _embed_limited(series, m, tau, sample)
        if len(X) < 10:
            continue
        dists = pdist(X)
        logs_eps = []
        logs_c = []
        for q in eps_quantiles:
            eps = np.quantile(dists, q)
            if eps <= 0:
                continue
            c = (dists < eps).mean()
            if c <= 0:
                continue
            logs_eps.append(math.log(eps))
            logs_c.append(math.log(c))
        if len(logs_eps) >= 2:
            slope, _ = np.polyfit(logs_eps, logs_c, 1)
            result[m] = float(slope)
    return result or None


def false_nearest_neighbors(series: Sequence[float], max_dim: int = 6, tau: int = 1, r_tol: float = 10.0, a_tol: float = 2.0, sample: int = 2000) -> Optional[Dict[int, float]]:
    """简化 FNN：计算各维度的假邻居比例."""
    result = {}
    arr = np.asarray(series, dtype=float)
    if len(arr) > sample:
        arr = arr[-sample:]
    std_all = arr.std()
    if std_all == 0:
        return None
    for m in range(1, max_dim):
        X_m = _embed(arr, m, tau)
        X_m1 = _embed(arr, m + 1, tau)
        n = min(len(X_m), len(X_m1))
        if n < 10:
            continue
        X_m = X_m[:n]
        X_m1 = X_m1[:n]
        tree = cKDTree(X_m)
        dists, idx = tree.query(X_m, k=2)
        d_m = dists[:, 1]
        nbr = idx[:, 1]
        extra = np.abs(X_m1[:, -1] - X_m1[nbr, -1])
        fnn_mask = (np.sqrt(d_m**2 + extra**2) / (d_m + 1e-12) > r_tol) | ((extra / std_all) > a_tol)
        result[m] = float(fnn_mask.mean() * 100)
    return result or None


def recurrence_metrics(series: Sequence[float], dim: int = 3, tau: int = 1, eps_quantile: float = 0.1, sample: int = 800) -> Optional[Dict[str, float]]:
    """计算复现率与确定性(DET)的简化指标."""
    X = _embed_limited(series, dim, tau, sample)
    n = len(X)
    if n < 20:
        return None
    D = cdist(X, X)
    eps = np.quantile(D, eps_quantile)
    if eps <= 0:
        return None
    R = (D <= eps).astype(np.int8)
    np.fill_diagonal(R, 0)
    rr = R.mean()

    # 计算对角线长度>=2的复现点占比（DET）
    det_points = 0
    for offset in range(-n + 2, n - 1):
        diag = np.diag(R, k=offset)
        if len(diag) < 2:
            continue
        run = 0
        for v in diag:
            if v:
                run += 1
            elif run:
                if run >= 2:
                    det_points += run
                run = 0
        if run >= 2:
            det_points += run
    total_points = R.sum()
    det = det_points / total_points if total_points > 0 else 0.0
    return {"rr": float(rr), "det": float(det), "eps": float(eps)}


# ------------------ 关联规则（Apriori） ------------------
def _freq_itemsets(transactions: List[set], min_support: float, max_len: int) -> Dict[frozenset, float]:
    n = len(transactions)
    if n == 0:
        return {}

    def support(cand: frozenset) -> float:
        cnt = sum(1 for t in transactions if cand.issubset(t))
        return cnt / n

    # 初始化 1-项集
    items = {frozenset([x]) for t in transactions for x in t}
    freq: Dict[frozenset, float] = {}
    level = {i for i in items if support(i) >= min_support}
    freq.update({i: support(i) for i in level})

    k = 2
    while level and k <= max_len:
        # 连接生成候选
        candidates = set()
        level_list = list(level)
        for i in range(len(level_list)):
            for j in range(i + 1, len(level_list)):
                u = level_list[i] | level_list[j]
                if len(u) == k:
                    candidates.add(u)
        # 剪枝：所有子集必须频繁
        pruned = set()
        for c in candidates:
            if all(frozenset(sub) in freq for sub in combinations(c, k - 1)):
                pruned.add(c)
        level = {c for c in pruned if support(c) >= min_support}
        freq.update({i: support(i) for i in level})
        k += 1
    return freq


def apriori_rules(
    transactions: List[set],
    min_support: float = 0.01,
    min_conf: float = 0.2,
    max_len: int = 3,
) -> List[Dict]:
    """
    简单 Apriori 规则挖掘，返回按提升度排序的规则。
    """
    freq = _freq_itemsets(transactions, min_support, max_len)
    rules = []
    for itemset, sup in freq.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent in map(frozenset, combinations(items, r)):
                consequent = itemset - antecedent
                sup_a = freq.get(antecedent)
                sup_c = freq.get(consequent)
                if not sup_a or not sup_c:
                    continue
                conf = sup / sup_a
                lift = conf / sup_c
                if conf >= min_conf:
                    rules.append(
                        {
                            "antecedent": sorted(antecedent),
                            "consequent": sorted(consequent),
                            "support": sup,
                            "confidence": conf,
                            "lift": lift,
                        }
                    )
    rules.sort(key=lambda x: x["lift"], reverse=True)
    return rules


def significant_extremes(counts: np.ndarray, threshold: float = 2.0) -> Dict[str, List[int]]:
    """基于计数的标准化残差标记显著热/冷号."""
    total = counts.sum()
    k = len(counts)
    if total == 0 or k == 0:
        return {"hot": [], "cold": []}
    expected = total / k
    std = math.sqrt(expected * (1 - 1 / k)) if expected > 0 else 0.0
    if std <= 0:
        return {"hot": [], "cold": []}
    z_scores = (counts - expected) / std
    hot = [i + 1 for i, z in enumerate(z_scores) if z >= threshold]
    cold = [i + 1 for i, z in enumerate(z_scores) if z <= -threshold]
    return {"hot": hot, "cold": cold}


def analyze(df: pd.DataFrame, entropy_window: int = 60) -> Dict:
    probs = frequency_stats(df)
    red_hot, red_cold = hot_cold(probs["red"], top_k=6)
    blue_hot, blue_cold = hot_cold(probs["blue"], top_k=3)

    sums = df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1).to_numpy()
    lle = lyapunov_exponent(sums)

    entropy_curve = moving_entropy(df, window=min(entropy_window, len(df) * 6))
    entropy_recent = entropy_curve[-1][1] if entropy_curve else None

    red_choice, blue_choice = suggest_numbers(probs["red"], probs["blue"])

    # 统计与假设检验
    stats_basic = basic_stats(df)
    red_counts = np.bincount(df[["red1", "red2", "red3", "red4", "red5", "red6"]].to_numpy(dtype=int).ravel(), minlength=34)[1:]
    blue_counts = np.bincount(df["blue"].to_numpy(dtype=int), minlength=17)[1:]
    chi_red = chi_square_uniform(red_counts)
    chi_blue = chi_square_uniform(blue_counts)
    runs = runs_test(df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1).to_numpy(dtype=float))
    sig_red = significant_extremes(red_counts)
    sig_blue = significant_extremes(blue_counts)
    autocorr = autocorrelation(sums)
    omissions = omission_stats(df)
    periodic = omission_periodicity(df)
    corr_dim = correlation_dimension(sums)
    fnn = false_nearest_neighbors(sums)
    recur = recurrence_metrics(sums)
    # 关联规则：将红球与蓝球统一处理，其中蓝球以字符串前缀区分
    transactions = [set([f"R{n}" for n in row[:6]] + [f"B{row[6]}"]) for row in df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].to_numpy(dtype=int)]
    rules = apriori_rules(transactions, min_support=0.01, min_conf=0.2, max_len=3)

    return {
        "count": int(len(df)),
        "latest_issue": df["issue"].iloc[-1],
        "probs_red": probs["red"],
        "probs_blue": probs["blue"],
        "hot_red": red_hot,
        "cold_red": red_cold,
        "hot_blue": blue_hot,
        "cold_blue": blue_cold,
        "lyapunov": lle,
        "entropy_recent": entropy_recent,
        "suggestion": {"reds": red_choice, "blue": blue_choice},
        "basic_stats": stats_basic,
        "chi_square": {"red": chi_red, "blue": chi_blue},
        "runs_test_sum": runs,
        "significant": {"red": sig_red, "blue": sig_blue},
        "autocorr": autocorr,
        "omission": omissions,
        "periodic": periodic,
        "chaos": {"corr_dim": corr_dim, "fnn": fnn, "recur": recur},
        "rules": rules,
    }

