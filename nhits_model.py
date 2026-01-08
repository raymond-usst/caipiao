from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pickle

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("LIGHTNING_PROGRESS_BAR", "0")

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS


def _build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    # 将和值作为单变量时间序列；同时输出蓝球序列，便于双通道建模（此处简单分两条序列）
    sums = df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1).to_numpy()
    blue = df["blue"].to_numpy()
    dates = pd.to_datetime(df["draw_date"])
    ts_sum = pd.DataFrame({"unique_id": "sum", "ds": dates, "y": sums})
    ts_blue = pd.DataFrame({"unique_id": "blue", "ds": dates, "y": blue})
    return pd.concat([ts_sum, ts_blue], axis=0)


@dataclass
class NHitsConfig:
    input_size: int = 60
    h: int = 1
    # 兼容旧参数（训练中不再使用）
    n_layers: int = 2
    n_blocks: int = 1
    n_harmonics: int = 2
    n_polynomials: int = 1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_steps: int = 200
    batch_size: int = 32
    valid_size: float = 0.1  # 若 < h，会自动调至 >= h
    topk: int = 3
    early_stop_patience_steps: int = 5  # 更敏感的基于 val loss 的早停


def _nhits_ckpt_path(save_dir: Path, cfg: NHitsConfig) -> Path:
    base = f"nhits_in{cfg.input_size}_h{cfg.h}_ms{cfg.max_steps}_bs{cfg.batch_size}_lr{cfg.learning_rate}"
    return save_dir / f"{base}.pkl"


def _eval_nhits_loss(df: pd.DataFrame, cfg: NHitsConfig, max_train: int = 150) -> float:
    df_recent = df.tail(max_train)
    ts = _build_timeseries(df_recent)
    val_size = max(cfg.valid_size, cfg.h, 2)
    model = NHITS(
        input_size=cfg.input_size,
        h=cfg.h,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        batch_size=cfg.batch_size,
        random_seed=42,
        early_stop_patience_steps=cfg.early_stop_patience_steps,
    )
    if hasattr(model, "trainer_kwargs"):
        model.trainer_kwargs["logger"] = False
        model.trainer_kwargs["enable_progress_bar"] = False
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(ts, val_size=val_size)
    fcst = nf.predict(ts)
    # 计算sum和blue的简单MAE
    mae = 0.0
    count = 0
    for uid in ["sum", "blue"]:
        truth = ts[ts["unique_id"] == uid]["y"].iloc[-cfg.h :].to_numpy()
        pred = fcst[fcst["unique_id"] == uid]["NHITS"].iloc[-cfg.h :].to_numpy()
        if len(truth) == len(pred):
            mae += float(np.abs(truth - pred).mean())
            count += 1
    if count == 0:
        return 1e3
    return mae / count


def bayes_optimize_nhits(
    df: pd.DataFrame,
    recent: int = 300,
    n_iter: int = 6,
    random_state: int = 42,
):
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real, Categorical
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行 N-HiTS 贝叶斯调参") from e

    df_recent = df if recent <= 0 else df.tail(max(recent, 120))
    if len(df_recent) < 80:
        raise ValueError("样本不足，无法进行 N-HiTS 贝叶斯调参")

    space = [
        Integer(30, min(150, len(df_recent) - 5), name="input_size"),
        Real(1e-4, 5e-3, prior="log-uniform", name="learning_rate"),
        Integer(30, 150, name="max_steps"),
        Categorical([16, 32], name="batch_size"),
    ]

    def objective(params):
        inp, lr, ms, bs = params
        cfg = NHitsConfig(
            input_size=inp,
            h=1,
            learning_rate=lr,
            max_steps=ms,
            batch_size=bs,
            valid_size=0.1,
            early_stop_patience_steps=3,
        )
        try:
            loss = _eval_nhits_loss(df_recent, cfg, max_train=recent)
        except Exception:
            return 1e3
        return loss

    res = gp_minimize(
        objective,
        space,
        n_calls=n_iter,
        random_state=random_state,
        verbose=False,
    )
    best = res.x
    best_loss = float(res.fun)
    best_params = {
        "input_size": best[0],
        "learning_rate": best[1],
        "max_steps": best[2],
        "batch_size": best[3],
    }
    return best_params, best_loss


def train_nhits(
    df: pd.DataFrame,
    cfg: NHitsConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    ts = _build_timeseries(df)
    val_size = max(cfg.valid_size, cfg.h)
    ckpt_path = None
    if save_dir is not None:
        ckpt_path = _nhits_ckpt_path(Path(save_dir), cfg)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and not fresh and ckpt_path.exists():
            try:
                with open(ckpt_path, "rb") as f:
                    nf = pickle.load(f)
                print(f"[N-HiTS] 检测到已保存模型，直接加载 {ckpt_path}")
                return nf, cfg
            except Exception as e:
                print(f"[N-HiTS] 加载已保存模型失败，将重新训练: {e}")

    models = [
        NHITS(
            input_size=cfg.input_size,
            h=cfg.h,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            batch_size=cfg.batch_size,
            random_seed=42,
            early_stop_patience_steps=cfg.early_stop_patience_steps,
        )
    ]
    # 关闭 tensorboard logger 与进度条，避免冗余输出
    for m in models:
        if hasattr(m, "trainer_kwargs"):
            m.trainer_kwargs["logger"] = False
            m.trainer_kwargs["enable_progress_bar"] = False
            m.trainer_kwargs["enable_model_summary"] = False  # 屏蔽重复的模型摘要提示
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(ts, val_size=val_size)
    if ckpt_path is not None:
        with open(ckpt_path, "wb") as f:
            pickle.dump(nf, f)
        print(f"[N-HiTS] 模型已保存到 {ckpt_path}")
    return nf, cfg


def predict_nhits(nf: NeuralForecast, cfg: NHitsConfig, df: pd.DataFrame) -> Dict:
    ts = _build_timeseries(df)
    fcst = nf.predict(ts)
    # 取最后一条预测
    sum_pred = float(fcst[fcst["unique_id"] == "sum"]["NHITS"].iloc[-1])
    blue_pred = float(fcst[fcst["unique_id"] == "blue"]["NHITS"].iloc[-1])
    # 将和值预测转为红球 Top3 粗略建议：回退到已训练概率不可行，给出和值区间
    return {"sum_pred": sum_pred, "blue_pred": blue_pred}

