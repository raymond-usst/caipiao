from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import pickle

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("LIGHTNING_PROGRESS_BAR", "0")

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet


def _build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """构造和值与蓝球的单变量时间序列."""
    sums = df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1)
    blue = df["blue"]
    dates = pd.to_datetime(df["draw_date"])
    ts_sum = pd.DataFrame({"unique_id": "sum", "ds": dates, "y": sums})
    ts_blue = pd.DataFrame({"unique_id": "blue", "ds": dates, "y": blue})
    return pd.concat([ts_sum, ts_blue], axis=0)


@dataclass
class TimesNetConfig:
    input_size: int = 120
    h: int = 1
    hidden_size: int = 64
    top_k: int = 5
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_steps: int = 300
    batch_size: int = 32
    valid_size: float = 0.1
    early_stop_patience_steps: int = 5  # 更敏感的基于 val loss 的早停


def _timesnet_ckpt_path(save_dir: Path, cfg: TimesNetConfig) -> Path:
    base = f"timesnet_in{cfg.input_size}_h{cfg.h}_hid{cfg.hidden_size}_tk{cfg.top_k}_dr{cfg.dropout}_ms{cfg.max_steps}_lr{cfg.learning_rate}"
    return save_dir / f"{base}.pkl"


def _eval_timesnet_loss(df: pd.DataFrame, cfg: TimesNetConfig, max_train: int = 200) -> float:
    df_recent = df.tail(max_train)
    ts = _build_timeseries(df_recent)
    val_size = max(cfg.valid_size, cfg.h, 2)
    model = TimesNet(
        input_size=cfg.input_size,
        h=cfg.h,
        hidden_size=cfg.hidden_size,
        top_k=cfg.top_k,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        batch_size=cfg.batch_size,
        random_seed=42,
        early_stop_patience_steps=cfg.early_stop_patience_steps,
    )
    if hasattr(model, "trainer_kwargs"):
        model.trainer_kwargs["logger"] = False
        model.trainer_kwargs["enable_progress_bar"] = False
        model.trainer_kwargs["enable_model_summary"] = False  # 屏蔽重复的模型摘要提示
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(ts, val_size=val_size)
    fcst = nf.predict(ts)
    mae = 0.0
    count = 0
    for uid in ["sum", "blue"]:
        truth = ts[ts["unique_id"] == uid]["y"].iloc[-cfg.h :].to_numpy()
        pred = fcst[fcst["unique_id"] == uid]["TimesNet"].iloc[-cfg.h :].to_numpy()
        if len(truth) == len(pred):
            mae += float((abs(truth - pred)).mean())
            count += 1
    if count == 0:
        return 1e3
    return mae / count


def bayes_optimize_timesnet(
    df: pd.DataFrame,
    recent: int = 400,
    n_iter: int = 6,
    random_state: int = 42,
):
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real, Categorical
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行 TimesNet 贝叶斯调参") from e

    df_recent = df if recent <= 0 else df.tail(max(recent, 150))
    if len(df_recent) < 100:
        raise ValueError("样本不足，无法进行 TimesNet 贝叶斯调参")

    space = [
        Integer(40, min(200, len(df_recent) - 5), name="input_size"),
        Integer(32, 128, name="hidden_size"),
        Integer(2, 8, name="top_k"),
        Real(0.0, 0.3, name="dropout"),
        Real(1e-4, 5e-3, prior="log-uniform", name="learning_rate"),
        Integer(50, 200, name="max_steps"),
        Categorical([16, 32], name="batch_size"),
    ]

    def objective(params):
        inp, hid, tk, dr, lr, ms, bs = params
        cfg = TimesNetConfig(
            input_size=inp,
            h=1,
            hidden_size=hid,
            top_k=tk,
            dropout=dr,
            learning_rate=lr,
            max_steps=ms,
            batch_size=bs,
            valid_size=0.1,
            early_stop_patience_steps=3,
        )
        try:
            loss = _eval_timesnet_loss(df_recent, cfg, max_train=recent)
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
        "hidden_size": best[1],
        "top_k": best[2],
        "dropout": best[3],
        "learning_rate": best[4],
        "max_steps": best[5],
        "batch_size": best[6],
    }
    return best_params, best_loss


def train_timesnet(
    df: pd.DataFrame,
    cfg: TimesNetConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    ts = _build_timeseries(df)
    val_size = max(cfg.valid_size, cfg.h)
    ckpt_path = None
    if save_dir is not None:
        ckpt_path = _timesnet_ckpt_path(Path(save_dir), cfg)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and not fresh and ckpt_path.exists():
            try:
                with open(ckpt_path, "rb") as f:
                    nf = pickle.load(f)
                print(f"[TimesNet] 检测到已保存模型，直接加载 {ckpt_path}")
                return nf, cfg
            except Exception as e:
                print(f"[TimesNet] 加载已保存模型失败，将重新训练: {e}")

    model = TimesNet(
        input_size=cfg.input_size,
        h=cfg.h,
        hidden_size=cfg.hidden_size,
        top_k=cfg.top_k,
        dropout=cfg.dropout,
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
    if ckpt_path is not None:
        with open(ckpt_path, "wb") as f:
            pickle.dump(nf, f)
        print(f"[TimesNet] 模型已保存到 {ckpt_path}")
    return nf, cfg


def predict_timesnet(nf: NeuralForecast, df: pd.DataFrame) -> Dict[str, float]:
    ts = _build_timeseries(df)
    fcst = nf.predict(ts)
    sum_pred = float(fcst[fcst["unique_id"] == "sum"]["TimesNet"].iloc[-1])
    blue_pred = float(fcst[fcst["unique_id"] == "blue"]["TimesNet"].iloc[-1])
    return {"sum_pred": sum_pred, "blue_pred": blue_pred}

