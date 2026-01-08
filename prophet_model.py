from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import pickle

import pandas as pd
from prophet import Prophet
import numpy as np


def _build_series(df: pd.DataFrame, col: str, uid: str) -> pd.DataFrame:
    s = df[[col, "draw_date"]].copy()
    s = s.rename(columns={col: "y", "draw_date": "ds"})
    s["ds"] = pd.to_datetime(s["ds"])
    s["unique_id"] = uid
    return s[["unique_id", "ds", "y"]]


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
    # 留出最后1步做验证
    train_df = df_recent.iloc[:-1]
    val_df = df_recent.iloc[-1:]
    models: Dict[str, Prophet] = {}
    mae = 0.0
    count = 0
    for uid, col in [("sum", "sum_val"), ("blue", "blue")]:
        if col == "sum_val":
            train_series = _build_series(train_df.assign(sum_val=train_df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1)), "sum_val", "sum")
            val_series = _build_series(val_df.assign(sum_val=val_df[["red1", "red2", "red3", "red4", "red5", "blue"]].sum(axis=1)), "sum_val", "sum")
        else:
            train_series = _build_series(train_df, "blue", "blue")
            val_series = _build_series(val_df, "blue", "blue")
        m = Prophet(
            yearly_seasonality=cfg.yearly_seasonality,
            weekly_seasonality=cfg.weekly_seasonality,
            daily_seasonality=cfg.daily_seasonality,
            changepoint_prior_scale=cfg.changepoint_prior_scale,
            seasonality_prior_scale=cfg.seasonality_prior_scale,
        )
        m.fit(train_series[["ds", "y"]])
        future_df = pd.DataFrame({"ds": val_series["ds"]})
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
    n_iter: int = 6,
    random_state: int = 42,
):
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行 Prophet 贝叶斯调参") from e

    space = [
        Real(0.01, 0.5, prior="log-uniform", name="cps"),
        Real(1.0, 20.0, prior="log-uniform", name="sps"),
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

    res = gp_minimize(
        objective,
        space,
        n_calls=n_iter,
        random_state=random_state,
        verbose=False,
    )
    cps, sps = res.x
    best_mae = float(res.fun)
    best_params = {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps}
    return best_params, best_mae


def train_prophet(
    df: pd.DataFrame,
    cfg: ProphetConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    sum_df = _build_series(df.assign(sum_val=df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1)), "sum_val", "sum")
    blue_df = _build_series(df, "blue", "blue")

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
        m.fit(s_df[["ds", "y"]])
        models[uid] = m
    if ckpt_path is not None:
        with open(ckpt_path, "wb") as f:
            pickle.dump(models, f)
        print(f"[Prophet] 模型已保存到 {ckpt_path}")
    return models, cfg


def predict_prophet(models: Dict[str, Prophet], df: pd.DataFrame) -> Dict[str, float]:
    future = {}
    for uid, m in models.items():
        last_ds = pd.to_datetime(df["draw_date"].iloc[-1])
        future_df = pd.DataFrame({"ds": [last_ds + pd.Timedelta(days=1)]})
        yhat = float(m.predict(future_df)["yhat"].iloc[0])
        future[uid] = yhat
    return future

