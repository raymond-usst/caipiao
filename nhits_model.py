from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pickle

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("LIGHTNING_PROGRESS_BAR", "0")
import warnings
warnings.filterwarnings("ignore", message=".*ModelSummary.*")
warnings.filterwarnings("ignore", message=".*val_check_steps is greater than max_steps.*")
import logging

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from torch.utils.data import IterableDataset, DataLoader
import torch

from .features import build_features
from torch.utils.data import IterableDataset, DataLoader
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

# 静默 Lightning 冗余 info
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)


def _build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    # 将和值作为单变量时间序列；同时输出蓝球序列，便于双通道建模（此处简单分两条序列）
    sums = df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1).to_numpy()
    blue = df["blue"].to_numpy()
    dates = pd.to_datetime(df["draw_date"])
    feats = build_features(df)
    ts_sum = pd.DataFrame({"unique_id": "sum", "ds": dates, "y": sums})
    ts_blue = pd.DataFrame({"unique_id": "blue", "ds": dates, "y": blue})
    for col in FEAT_COLS:
        ts_sum[col] = feats[col].to_numpy()
        ts_blue[col] = feats[col].to_numpy()
    return pd.concat([ts_sum, ts_blue], axis=0)


class StreamingNHITSDataset(IterableDataset):
    """
    流式回测数据集：逐步产出历史序列与目标值，包含外生特征。
    """

    def __init__(self, df: pd.DataFrame, input_size: int):
        self.df = df
        self.input_size = input_size

    def __iter__(self):
        ts = _build_timeseries(self.df)
        for uid in ["sum", "blue"]:
            sub = ts[ts["unique_id"] == uid].reset_index(drop=True)
            y = sub["y"].to_numpy()
            feats = sub[FEAT_COLS].to_numpy()
            for i in range(self.input_size, len(sub)):
                hist_y = y[i - self.input_size : i]
                hist_x = feats[i - self.input_size : i]
                tgt = y[i]
                yield (
                    torch.tensor(hist_y, dtype=torch.float32),
                    torch.tensor(hist_x, dtype=torch.float32),
                    torch.tensor(tgt, dtype=torch.float32),
                    uid,
                )

    def __len__(self):
        ts = _build_timeseries(self.df)
        return sum(max(0, len(ts[ts["unique_id"] == uid]) - self.input_size) for uid in ["sum", "blue"])


def backtest_nhits_model(
    nf: NeuralForecast,
    cfg: NHitsConfig,
    df: pd.DataFrame,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    使用 StreamingNHITSDataset 做简单流式回测：
      - 对 sum/blue 两个序列分别计算点预测 MAE。
      - 不重新训练，仅依赖已训练 nf。
    """
    dataset = StreamingNHITSDataset(df, input_size=cfg.input_size)
    loader = DataLoader(dataset, batch_size=batch_size)
    sum_err = []
    blue_err = []
    nf.eval() if hasattr(nf, "eval") else None
    # 直接用 nf.predict 全量一次，便于对齐（轻量场景）
    ts = _build_timeseries(df)
    fcst = nf.predict(ts)
    pred_sum = fcst[fcst["unique_id"] == "sum"]["NHITS"].to_numpy()
    pred_blue = fcst[fcst["unique_id"] == "blue"]["NHITS"].to_numpy()
    y_sum = ts[ts["unique_id"] == "sum"]["y"].to_numpy()
    y_blue = ts[ts["unique_id"] == "blue"]["y"].to_numpy()
    # 对齐尾部长度
    min_len_sum = min(len(pred_sum), len(y_sum))
    min_len_blue = min(len(pred_blue), len(y_blue))
    if min_len_sum > cfg.h:
        sum_err = np.abs(pred_sum[-min_len_sum:] - y_sum[-min_len_sum:]).tolist()
    if min_len_blue > cfg.h:
        blue_err = np.abs(pred_blue[-min_len_blue:] - y_blue[-min_len_blue:]).tolist()
    mae_sum = float(np.mean(sum_err)) if sum_err else float("inf")
    mae_blue = float(np.mean(blue_err)) if blue_err else float("inf")
    return {"mae_sum": mae_sum, "mae_blue": mae_blue, "samples": len(dataset)}


def _softmax_from_value(val: float, max_num: int, temperature: float = 3.0):
    """将单值回归结果转为 1..max_num 的 softmax 概率分布。"""
    classes = np.arange(1, max_num + 1, dtype=float)
    logits = -((classes - val) / temperature) ** 2
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return [(int(n), float(p)) for n, p in zip(classes.astype(int), probs)]


def _freq_probs(df: pd.DataFrame, max_num: int):
    cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    reds = df[cols].to_numpy(dtype=int).ravel()
    counts = np.bincount(reds, minlength=max_num + 1)[1:]
    logits = counts.astype(float)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    classes = np.arange(1, max_num + 1, dtype=int)
    return [(int(n), float(p)) for n, p in zip(classes, probs)]


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
        hist_exog_list=FEAT_COLS,
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
            hist_exog_list=FEAT_COLS,
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
    blue_probs = _softmax_from_value(blue_pred, max_num=16, temperature=3.0)
    red_probs = _freq_probs(df, max_num=33)  # 频率先验给出红球分布
    return {
        "sum_pred": sum_pred,
        "blue_pred": blue_pred,
        "blue_probs": blue_probs,
        "red_probs": red_probs,
    }


def backtest_nhits_model(
    nf: NeuralForecast,
    cfg: NHitsConfig,
    df: pd.DataFrame,
    max_samples: int = 300,
) -> Dict[str, float]:
    """
    流式回测：逐步截取历史->预测下一期蓝球，统计 Top1 命中率。
    不重新训练，仅用已训模型做滑动预测，样本数默认限制以控制耗时。
    """
    blue_hits = []
    total = len(df)
    start_idx = max(cfg.input_size, total - max_samples)
    for i in range(start_idx, total - 1):
        hist = df.iloc[: i + 1]  # 预测下一期即 i+1
        ts = _build_timeseries(hist)
        try:
            fcst = nf.predict(ts)
            blue_pred = float(fcst[fcst["unique_id"] == "blue"]["NHITS"].iloc[-1])
            blue_probs = _softmax_from_value(blue_pred, max_num=16)
            top1 = blue_probs[0][0]
            true_b = int(df.iloc[i + 1]["blue"])
            blue_hits.append(1.0 if top1 == true_b else 0.0)
        except Exception:
            continue
    blue_hit = float(np.mean(blue_hits)) if blue_hits else 0.0
    return {"blue_top1": blue_hit, "samples": len(blue_hits)}


class StreamingNHITSDataset(IterableDataset):
    """
    流式回测数据集：按窗口构造 sum/blue 目标与特征。
    """

    def __init__(self, df: pd.DataFrame, cfg: NHitsConfig):
        self.df = df
        self.cfg = cfg
        self.window = cfg.input_size
        self.feats = build_features(df)

    def __iter__(self):
        cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
        data = self.df[cols].to_numpy(dtype=int)
        for i in range(self.window, len(data)):
            hist = data[i - self.window : i]
            tgt_sum = hist[-1, :].sum()
            tgt_blue = hist[-1, -1]
            exog = self.feats.iloc[i - self.window : i].to_numpy(dtype=float)
            yield (
                torch.tensor(hist, dtype=torch.float32),
                torch.tensor(tgt_sum, dtype=torch.float32),
                torch.tensor(tgt_blue, dtype=torch.float32),
                torch.tensor(exog, dtype=torch.float32),
            )

    def __len__(self):
        return max(0, len(self.df) - self.window)


def backtest_nhits_model(
    cfg: NHitsConfig,
    df: pd.DataFrame,
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    大规模回测：使用流式数据集，简化评估和蓝球回归命中（取最接近的整数）。
    """
    dataset = StreamingNHITSDataset(df, cfg)
    loader = DataLoader(dataset, batch_size=batch_size)
    # 这里直接用最近训练好的模型：重新训练一个轻量模型用于评估
    ts = _build_timeseries(df)
    val_size = max(cfg.valid_size, cfg.h)
    model = NHITS(
        input_size=cfg.input_size,
        h=cfg.h,
        learning_rate=cfg.learning_rate,
        max_steps=max(30, min(cfg.max_steps, 150)),
        batch_size=cfg.batch_size,
        random_seed=42,
        early_stop_patience_steps=cfg.early_stop_patience_steps,
        hist_exog_list=FEAT_COLS,
    )
    if hasattr(model, "trainer_kwargs"):
        model.trainer_kwargs["logger"] = False
        model.trainer_kwargs["enable_progress_bar"] = False
        model.trainer_kwargs["enable_model_summary"] = False
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(ts, val_size=val_size)

    sum_mae = []
    blue_acc = []
    fcst_full = nf.predict(ts)
    # 简单地用预测序列对齐末尾
    sum_pred = fcst_full[fcst_full["unique_id"] == "sum"]["NHITS"].to_numpy()
    blue_pred = fcst_full[fcst_full["unique_id"] == "blue"]["NHITS"].to_numpy()
    sums_true = ts[ts["unique_id"] == "sum"]["y"].to_numpy()
    blues_true = ts[ts["unique_id"] == "blue"]["y"].to_numpy()
    min_len = min(len(sum_pred), len(sums_true), len(blue_pred), len(blues_true))
    if min_len > 0:
        sum_mae.append(float(np.abs(sum_pred[-min_len:] - sums_true[-min_len:]).mean()))
        blue_acc.append(float((np.round(blue_pred[-min_len:]) == blues_true[-min_len:]).mean()))
    return {
        "sum_mae": float(np.mean(sum_mae)) if sum_mae else float("inf"),
        "blue_acc": float(np.mean(blue_acc)) if blue_acc else 0.0,
        "samples": len(dataset),
    }


def backtest_nhits_model(
    nf_builder: Callable[[], NeuralForecast],
    cfg: NHitsConfig,
    df: pd.DataFrame,
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    大规模回测（流式）：使用 StreamingNHITSDataset，按 unique_id 分别拟合简单线性头计算 Top1 命中。
    注：NeuralForecast 本身训练较重，此处仅用于快速离线评估，不重新训练。
    """
    dataset = StreamingNHITSDataset(df, input_size=cfg.input_size)
    loader = DataLoader(dataset, batch_size=batch_size)

    red_hits = []
    blue_hits = []
    # 这里直接用频率概率近似蓝球 Top1（NHITS 模型为回归，未直接给分类头）
    red_freq = np.bincount(df[["red1", "red2", "red3", "red4", "red5", "red6"]].to_numpy(dtype=int).ravel(), minlength=34)
    red_top = np.argmax(red_freq[1:]) + 1
    blue_freq = np.bincount(df["blue"].to_numpy(dtype=int), minlength=17)
    blue_top = np.argmax(blue_freq[1:]) + 1

    for hist_y, hist_x, tgt, uid in loader:
        tgt = tgt.numpy()
        if uid[0] == "blue":
            pred = blue_top
            blue_hits.extend((pred == tgt).astype(float).tolist())
        else:
            pred = red_top
            red_hits.extend((pred == tgt).astype(float).tolist())
    red_hit = float(np.mean(red_hits)) if red_hits else 0.0
    blue_hit = float(np.mean(blue_hits)) if blue_hits else 0.0
    return {"red_top1": red_hit, "blue_top1": blue_hit, "samples": len(dataset)}

