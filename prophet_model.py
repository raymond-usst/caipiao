from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import pickle
import os
from pathlib import Path

# 避免 Prophet/cmdstan 在受限环境下因并行或临时目录权限报错
os.environ.setdefault("PROPHET_DISABLE_STAN_MULTIPROCESSING", "1")
os.environ.setdefault("CMDSTANPY_FORCE_SERIAL", "1")
os.environ.setdefault("PROPHET_DONT_LOG", "1")
os.environ.setdefault("CMDSTANPY_VERBOSE", "0")
_cmdstan_tmp = Path(".cmdstan_tmp")
_cmdstan_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CMDSTANPY_TMPDIR", str(_cmdstan_tmp.resolve()))

import pandas as pd
from prophet import Prophet
import numpy as np
import logging
from .features import build_features

FEAT_COLS = [
    "ac",
    "span",
    "sum_tail",
    "odd_ratio",
    "prime_ratio",
    "big_ratio",
    "omit_red_mean",
    "omit_red_max",
    "omit_blue",
    "weekday",
    "weekday_sin",
    "weekday_cos",
    "lunar_month",
    "lunar_day",
]

# 静默 prophet/cmdstanpy INFO
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").propagate = False
logging.getLogger("prophet").setLevel(logging.ERROR)

def _build_series(df: pd.DataFrame, col: str, uid: str, feats: pd.DataFrame) -> pd.DataFrame:
    s = df[[col, "draw_date"]].copy()
    s = s.rename(columns={col: "y", "draw_date": "ds"})
    s["ds"] = pd.to_datetime(s["ds"])
    s["unique_id"] = uid
    for c in FEAT_COLS:
        s[c] = feats[c].to_numpy()
    return s[["unique_id", "ds", "y", *FEAT_COLS]]


@dataclass
class ProphetConfig:
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0


def _prophet_ckpt_path(save_dir: Path, cfg: ProphetConfig) -> Path:
    base = f"prophet_y{int(cfg.yearly_seasonality)}_w{int(cfg.weekly_seasonality)}_d{int(cfg.daily_seasonality)}_cps{cfg.changepoint_prior_scale}_sps{cfg.seasonality_prior_scale}"
    return save_dir / f"{base}.pkl"


def _eval_prophet_mae(df: pd.DataFrame, cfg: ProphetConfig, recent: int = 200) -> float:
    df_recent = df.tail(max(recent, 60))
    if len(df_recent) < 40:
        raise ValueError("样本不足，无法评估 Prophet")
    feats = build_features(df_recent)
    # 留出最后1步做验证
    train_df = df_recent.iloc[:-1]
    val_df = df_recent.iloc[-1:]
    mae = 0.0
    count = 0
    # 使用单链优化，避免 cmdstanpy optimize 参数不支持 chains/cores/verbose
    fit_kwargs = {"iter": 800, "seed": 42}
    for uid, col in [("sum", "sum_val"), ("blue", "blue")]:
        if col == "sum_val":
            train_series = _build_series(
                train_df.assign(sum_val=train_df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1)),
                "sum_val",
                "sum",
                feats.iloc[:-1],
            )
            val_series = _build_series(
                val_df.assign(sum_val=val_df[["red1", "red2", "red3", "red4", "red5", "blue"]].sum(axis=1)),
                "sum_val",
                "sum",
                feats.iloc[-1:],
            )
        else:
            train_series = _build_series(train_df, "blue", "blue", feats.iloc[:-1])
            val_series = _build_series(val_df, "blue", "blue", feats.iloc[-1:])
        m = Prophet(
            yearly_seasonality=cfg.yearly_seasonality,
            weekly_seasonality=cfg.weekly_seasonality,
            daily_seasonality=cfg.daily_seasonality,
            changepoint_prior_scale=cfg.changepoint_prior_scale,
            seasonality_prior_scale=cfg.seasonality_prior_scale,
        )
        for c in FEAT_COLS:
            m.add_regressor(c)
        m.fit(train_series[["ds", "y", *FEAT_COLS]], **fit_kwargs)
        future_df = val_series[["ds", *FEAT_COLS]]
        yhat = m.predict(future_df)["yhat"].to_numpy()
        truth = val_series["y"].to_numpy()
        if len(yhat) == len(truth):
            mae += float(np.abs(yhat - truth).mean())
            count += 1
    if count == 0:
        return 1e3
    return mae / count


def bayes_optimize_prophet(
    df: pd.DataFrame,
    recent: int = 200,
    n_iter: int = 15,
    random_state: int = 42,
):
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行 Prophet 贝叶斯调参") from e

    space = [
        Real(0.005, 0.5, prior="log-uniform", name="cps"),
        Real(1.0, 30.0, prior="log-uniform", name="sps"),
    ]

    def objective(params):
        cps, sps = params
        cfg = ProphetConfig(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
        )
        try:
            target_df = df if recent <= 0 else df.tail(max(recent, 60))
            mae = _eval_prophet_mae(target_df, cfg, recent=recent if recent > 0 else len(target_df))
        except Exception:
            return 1e3
        return mae

    n_calls = max(int(n_iter), 12)  # skopt 要求至少 10 次迭代，略提升下限
    n_coarse = max(6, n_calls // 2)
    n_fine = max(6, n_calls - n_coarse)

    def _run_search(space, calls):
        try:
            res = gp_minimize(
                objective,
                space,
                n_calls=calls,
                random_state=random_state,
                verbose=False,
            )
            return res.x, float(res.fun)
        except Exception:
            rng = np.random.default_rng(random_state)
            loc_best = None
            loc_loss = 1e9
            for _ in range(calls):
                c = float(np.exp(rng.uniform(np.log(space[0].low), np.log(space[0].high))))
                s = float(np.exp(rng.uniform(np.log(space[1].low), np.log(space[1].high))))
                cfg = ProphetConfig(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=c,
                    seasonality_prior_scale=s,
                )
                try:
                    target_df = df if recent <= 0 else df.tail(max(recent, 60))
                    mae = _eval_prophet_mae(target_df, cfg, recent=recent if recent > 0 else len(target_df))
                except Exception:
                    mae = 1e3
                if mae < loc_loss:
                    loc_loss = mae
                    loc_best = [c, s]
            return loc_best, loc_loss

    best, best_mae = _run_search(space, n_coarse)

    if best is not None:
        b_cps, b_sps = best
        space_fine = [
            Real(max(0.001, b_cps / 3), min(0.5, b_cps * 3), prior="log-uniform", name="cps"),
            Real(max(1.0, b_sps / 3), min(30.0, b_sps * 3), prior="log-uniform", name="sps"),
        ]
        fine_res, fine_mae = _run_search(space_fine, n_fine)
        if fine_res is not None and fine_mae < best_mae:
            best = fine_res
            best_mae = fine_mae
    cps, sps = best
    best_params = {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps}
    return best_params, best_mae


def train_prophet(
    df: pd.DataFrame,
    cfg: ProphetConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    fit_kwargs = {"iter": 1000, "seed": 42}
    feats = build_features(df)
    sum_df = _build_series(df.assign(sum_val=df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1)), "sum_val", "sum", feats)
    blue_df = _build_series(df, "blue", "blue", feats)

    ckpt_path = None
    if save_dir is not None:
        ckpt_path = _prophet_ckpt_path(Path(save_dir), cfg)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and not fresh and ckpt_path.exists():
            try:
                with open(ckpt_path, "rb") as f:
                    models = pickle.load(f)
                print(f"[Prophet] 检测到已保存模型，直接加载 {ckpt_path}")
                return models, cfg
            except Exception as e:
                print(f"[Prophet] 加载已保存模型失败，将重新训练: {e}")

    models: Dict[str, Prophet] = {}
    for uid, s_df in [("sum", sum_df), ("blue", blue_df)]:
        m = Prophet(
            yearly_seasonality=cfg.yearly_seasonality,
            weekly_seasonality=cfg.weekly_seasonality,
            daily_seasonality=cfg.daily_seasonality,
            changepoint_prior_scale=cfg.changepoint_prior_scale,
            seasonality_prior_scale=cfg.seasonality_prior_scale,
        )
        for c in FEAT_COLS:
            m.add_regressor(c)
        m.fit(s_df[["ds", "y", *FEAT_COLS]], **fit_kwargs)
        models[uid] = m
    if ckpt_path is not None:
        with open(ckpt_path, "wb") as f:
            pickle.dump(models, f)
        print(f"[Prophet] 模型已保存到 {ckpt_path}")
    return models, cfg


def predict_prophet(models: Dict[str, Prophet], df: pd.DataFrame) -> Dict[str, float]:
    future = {}
    probs = {}
    feats = build_features(df)
    last_feats = feats.iloc[-1]
    reg_vals = {c: [float(last_feats[c])] for c in FEAT_COLS}
    for uid, m in models.items():
        last_ds = pd.to_datetime(df["draw_date"].iloc[-1])
        future_df = pd.DataFrame({"ds": [last_ds + pd.Timedelta(days=1)], **reg_vals})
        yhat = float(m.predict(future_df)["yhat"].iloc[0])
        future[uid] = yhat
        if uid == "blue":
            classes = np.arange(1, 17, dtype=float)
            logits = -((classes - yhat) / 3.0) ** 2
            logits = logits - logits.max()
            p = np.exp(logits)
            p = p / p.sum()
            probs["blue_probs"] = [(int(c), float(pp)) for c, pp in zip(classes.astype(int), p)]
        if uid == "sum":
            reds = df[["red1", "red2", "red3", "red4", "red5", "red6"]].to_numpy(dtype=int).ravel()
            counts = np.bincount(reds, minlength=34)[1:]
            logits_r = counts.astype(float)
            logits_r = logits_r - logits_r.max()
            pr = np.exp(logits_r)
            pr = pr / pr.sum()
            classes_r = np.arange(1, 34, dtype=int)
            probs["red_probs"] = [(int(c), float(pp)) for c, pp in zip(classes_r, pr)]
    return {**future, **probs}

