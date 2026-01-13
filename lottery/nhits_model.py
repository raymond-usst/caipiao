from __future__ import annotations

from datetime import datetime
from lottery.utils.logger import logger
from lottery.engine.predictor import BasePredictor

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
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
from torch.utils.data import IterableDataset

from .features import build_features
from .pbt import ModelAdapter, Member
import copy
import random
from typing import Any, Tuple

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
    max_samples: int | None = None,
) -> Dict[str, float]:
    """
    轻量回测：使用已训练 N-HiTS 计算和值/蓝球 MAE，并估计蓝球 Top1 命中率。
    可选截断最近 max_samples 期，避免全量预测过慢。
    """
    nf.eval() if hasattr(nf, "eval") else None
    hist_df = df if not max_samples or max_samples <= 0 else df.tail(max_samples + cfg.input_size)
    ts = _build_timeseries(hist_df)
    fcst = nf.predict(ts)

    pred_sum = fcst[fcst["unique_id"] == "sum"]["NHITS"].to_numpy()
    pred_blue = fcst[fcst["unique_id"] == "blue"]["NHITS"].to_numpy()
    y_sum = ts[ts["unique_id"] == "sum"]["y"].to_numpy()
    y_blue = ts[ts["unique_id"] == "blue"]["y"].iloc[-len(pred_blue):].to_numpy()
    
    # Alignment: pred_blue is forecast for [T-h+1 ... T] or similar horizon
    # Ensure we compare same length from end of array
    min_len_sum = min(len(pred_sum), len(y_sum))
    # Align from the end
    p_sum_aligned = pred_sum[-min_len_sum:]
    y_sum_aligned = y_sum[-min_len_sum:]
    
    min_len_blue = min(len(pred_blue), len(y_blue))
    p_blue_aligned = pred_blue[-min_len_blue:]
    y_blue_aligned = y_blue[-min_len_blue:]

    mae_sum = float(np.mean(np.abs(p_sum_aligned - y_sum_aligned))) if min_len_sum else float("inf")
    mae_blue = float(np.mean(np.abs(p_blue_aligned - y_blue_aligned))) if min_len_blue else float("inf")


    blue_hits = []
    for pred, true in zip(pred_blue[-min_len_blue:], y_blue[-min_len_blue:]):
        probs = _softmax_from_value(float(pred), max_num=16, temperature=3.0)
        top1 = max(probs, key=lambda x: x[1])[0]
        blue_hits.append(1.0 if top1 == int(round(true)) else 0.0)
    blue_top1 = float(np.mean(blue_hits)) if blue_hits else 0.0

    return {"mae_sum": mae_sum, "mae_blue": mae_blue, "blue_top1": blue_top1, "samples": min_len_blue}


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


def _eval_nhits_loss(df: pd.DataFrame, cfg: NHitsConfig, max_train: int = 150, trials: int = 2) -> float:
    """
    稳健评估：多次子采样取中位数，失败返回较大惩罚但不直接 1000。
    """
    losses = []
    df_recent = df.tail(max_train)
    for _ in range(max(1, trials)):
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
        try:
            nf = NeuralForecast(models=[model], freq="D")
            nf.fit(ts, val_size=val_size)
            fcst = nf.predict(ts)
            mae = 0.0
            count = 0
            for uid in ["sum", "blue"]:
                truth = ts[ts["unique_id"] == uid]["y"].iloc[-cfg.h :].to_numpy()
                pred = fcst[fcst["unique_id"] == uid]["NHITS"].iloc[-cfg.h :].to_numpy()
                if len(truth) == len(pred):
                    mae += float(np.abs(truth - pred).mean())
                    count += 1
            if count == 0:
                losses.append(1e3)
            else:
                losses.append(mae / count)
        except Exception:
            losses.append(500.0)
    return float(np.median(losses))


def bayes_optimize_nhits(
    df: pd.DataFrame,
    recent: int = 300,
    n_iter: int = 15,
    random_state: int = 42,
):
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real, Categorical
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行 N-HiTS 贝叶斯调参") from e

    # 动态样本：至少 150 条，最多 600 条，避免过长评估波动
    if recent <= 0:
        df_recent = df.tail(min(len(df), 600))
    else:
        df_recent = df.tail(max(150, min(recent, 600)))
    if len(df_recent) < 80:
        raise ValueError("样本不足，无法进行 N-HiTS 贝叶斯调参")

    space_coarse = [
        Integer(60, min(180, len(df_recent) - 5), name="input_size"),
        Real(1e-4, 3e-3, prior="log-uniform", name="learning_rate"),
        Integer(40, 150, name="max_steps"),
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
            early_stop_patience_steps=2,
        )
        losses = []
        fails = 0
        for _ in range(3):  # 稳健评估取中位数
            try:
                eval_cfg = cfg
                eval_cfg.max_steps = min(cfg.max_steps, 100)
                loss = _eval_nhits_loss(df_recent, eval_cfg, max_train=len(df_recent), trials=2)
            except Exception:
                loss = 1e3
                fails += 1
            losses.append(loss)
        losses = sorted(losses)
        med = losses[len(losses) // 2]
        p_fail = fails / 3
        return p_fail * 1000 + (1 - p_fail) * med

    d = 4
    n_calls = max(10, min(40, int(4 * d), int(n_iter)))  # 动态 n_calls
    n_coarse = max(6, n_calls // 3)
    n_mid = max(6, n_calls // 3)
    n_fine = max(6, n_calls - n_coarse - n_mid)

    def _run_search(space, calls, seed):
        try:
            res = gp_minimize(
                objective,
                space,
                n_calls=calls,
                random_state=seed,
                verbose=False,
            )
            return res.x, float(res.fun)
        except Exception:
            rng = np.random.default_rng(seed)
            loc_best = None
            loc_loss = 1e9
            for _ in range(calls):
                inp = rng.integers(space[0].low, space[0].high + 1)
                lr = float(np.exp(rng.uniform(np.log(space[1].low), np.log(space[1].high))))
                ms = rng.integers(space[2].low, space[2].high + 1)
                bs = int(rng.choice(space[3].categories))
                cfg = NHitsConfig(
                    input_size=int(inp),
                    h=1,
                    learning_rate=lr,
                    max_steps=int(ms),
                    batch_size=bs,
                    valid_size=0.1,
                    early_stop_patience_steps=2,
                )
                try:
                    eval_cfg = cfg
                    eval_cfg.max_steps = min(cfg.max_steps, 100)
                    loss = _eval_nhits_loss(df_recent, eval_cfg, max_train=len(df_recent), trials=2)
                except Exception:
                    loss = 1e3
                if loss < loc_loss:
                    loc_loss = loss
                    loc_best = [inp, lr, ms, bs]
            return loc_best, loc_loss

    best, best_loss = _run_search(space_coarse, n_coarse, seed=random_state)

    # Mid stage：围绕 coarse 最优自适应收缩
    if best is not None:
        b_inp, b_lr, b_ms, b_bs = best
        space_mid = [
            Integer(max(50, int(b_inp) - 40), min(int(b_inp) + 40, len(df_recent) - 5), name="input_size"),
            Real(max(1e-5, b_lr / 3), min(3e-3, b_lr * 3), prior="log-uniform", name="learning_rate"),
            Integer(max(30, int(b_ms) - 40), min(180, int(b_ms) + 40), name="max_steps"),
            Categorical([b_bs, 16, 32]),
        ]
        mid_res, mid_loss = _run_search(space_mid, n_mid, seed=random_state + 1)
        if mid_res is not None and mid_loss < best_loss:
            best, best_loss = mid_res, mid_loss

    # Fine stage：进一步收缩
    if best is not None:
        b_inp, b_lr, b_ms, b_bs = best
        space_fine = [
            Integer(max(40, int(b_inp) - 25), min(int(b_inp) + 25, len(df_recent) - 5), name="input_size"),
            Real(max(1e-5, b_lr / 2), min(3e-3, b_lr * 2), prior="log-uniform", name="learning_rate"),
            Integer(max(30, int(b_ms) - 25), min(180, int(b_ms) + 25), name="max_steps"),
            Categorical([b_bs, 16, 32]),
        ]
        fine_res, fine_loss = _run_search(space_fine, n_fine, seed=random_state + 2)
        if fine_res is not None and fine_loss < best_loss:
            best, best_loss = fine_res, fine_loss

    if best is None or best_loss >= 900:
        best = [
            max(60, len(df_recent) * 2 // 3),
            1e-3,
            100,
            16,
        ]
        best_loss = float(1e3)

    # 若仍无有效解，回退经验参数
    if best is None or best_loss >= 900:
        best = [
            max(60, len(df_recent) * 2 // 3),
            1e-3,
            100,
            16,
        ]
        best_loss = float(1e3)

    best_params = {
        "input_size": int(best[0]),
        "learning_rate": float(best[1]),
        "max_steps": int(best[2]),
        "batch_size": int(best[3]),
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


class NHitsModelAdapter(ModelAdapter):
    def train_step(self, member: Member, dataset: pd.DataFrame, steps: int) -> Tuple[Any, float]:
        cfg = member.config
        ts = _build_timeseries(dataset)
        val_size = max(cfg.valid_size, cfg.h)
        
        if member.model_state is None:
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
             nf = NeuralForecast(models=models, freq="D")
        else:
             nf = member.model_state
             
        # Increment max_steps for the model(s) inside
        for m in nf.models:
             m.max_steps += 50
        
        try:
            nf.fit(ts, val_size=val_size)
            
            # Predict only for validation horizon
            # To predict the last 'h' steps of 'ts' (which we treat as validation here)
            # we should provide input up to T-h.
            # But TS DataFrame is stacked.
            # Let's filter by date?
            dates = ts["ds"].unique()
            dates.sort()
            val_dates = dates[-cfg.h:]
            cutoff_date = val_dates[0]
            
            # Input for prediction: all data BEFORE cutoff_date
            ts_in = ts[ts["ds"] < cutoff_date]
            
            fcst = nf.predict(ts_in)
            mae = 0.0
            count = 0
            for uid in ["sum", "blue"]:
                 truth = ts[(ts["unique_id"] == uid) & (ts["ds"].isin(val_dates))]["y"].to_numpy()
                 # fcst 'ds' align with val_dates?
                 # fcst starts from last date of ts_in + 1.
                 # ts_in ends at T-h. So fcst is T-h+1 ... T.
                 # which matches val_dates.
                 pred = fcst[fcst["unique_id"] == uid]["NHITS"].to_numpy()
                 
                 # Ensure length match
                 min_len = min(len(truth), len(pred))
                 if min_len > 0:
                     mae += float(np.abs(truth[:min_len] - pred[:min_len]).mean())
                     count += 1
            loss = mae / count if count > 0 else 1000.0
        except Exception as e:
            print(f"NHits Train Step Error: {e}")
            import traceback
            traceback.print_exc()
            return nf, 0.0 # Return 0 loss to avoid crash, or high loss?
        
        return nf, -loss
        
        return nf, -loss 

    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        nf = member.model_state
        if nf is None: return 0.0
        # Check if fitted (handle potential training failure gracefully)
        if not getattr(nf, "fitted", False):
            # Try to infer from private attribute if 'fitted' property missing?
            # Or just return 0.0
            # Check if models inside are fitted?
            # Assuming if fit() succeeded, fitted is True.
            return 0.0
            
        cfg = member.config
        
        try:
            res = backtest_nhits_model(nf, cfg, dataset, max_samples=60)
            score = res["blue_top1"]
        except Exception as e:
            print(f"NHits Eval Error: {e}")
            return 0.0
        return score

    def perturb_config(self, config: NHitsConfig) -> NHitsConfig:
        new_cfg = copy.deepcopy(config)
        factors = [0.8, 1.2]
        if random.random() < 0.3:
            new_cfg.learning_rate *= random.choice(factors)
        return new_cfg

    def save(self, member: Member, path: Path) -> None:
        if member.model_state:
            with open(path, "wb") as f:
                pickle.dump(member.model_state, f)

    def load(self, path: Path) -> Member:
        pass


class NHitsPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = NHitsConfig(
            input_size=config.input_size,
            h=1,
            n_layers=config.n_layers,
            n_blocks=config.n_blocks,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            batch_size=32,
            valid_size=0.1
        )
        self.model = None

    def train(self, df: pd.DataFrame) -> None:
        logger.info(f"Training NHits (size={self.cfg.input_size})...")
        self.model, self.cfg = train_nhits(
            df,
            self.cfg,
            save_dir="models/NHits",
            resume=self.cfg.resume,
            fresh=self.cfg.fresh
        )

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("NHits model not trained or loaded")
        return predict_nhits(self.model, self.cfg, df)
    
    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / f"nhits_in{self.cfg.input_size}_{self.cfg.n_layers}l_{self.cfg.max_steps}s.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"NHits model saved to {model_path}")

    def load(self, save_dir: str) -> bool:
        path = Path(save_dir)
        if not path.exists(): return False
        
        model_path = path / f"nhits_in{self.cfg.input_size}_{self.cfg.n_layers}l_{self.cfg.max_steps}s.pkl"
        if not model_path.exists(): return False
        
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"NHits model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load NHits model: {e}")
            return False




