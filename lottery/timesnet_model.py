from __future__ import annotations

from datetime import datetime
from lottery.utils.logger import logger
from lottery.engine.predictor import BasePredictor

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import pickle
import numpy as np

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("LIGHTNING_PROGRESS_BAR", "0")
import warnings
warnings.filterwarnings("ignore", message=".*ModelSummary.*")
warnings.filterwarnings("ignore", message=".*val_check_steps is greater than max_steps.*")
import logging

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
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
    """构造和值与蓝球的单变量时间序列，附加历史特征作为 hist_exog。"""
    sums = df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1)
    blue = df["blue"]
    dates = pd.to_datetime(df["draw_date"])
    feats = build_features(df)
    ts_sum = pd.DataFrame({"unique_id": "sum", "ds": dates, "y": sums})
    ts_blue = pd.DataFrame({"unique_id": "blue", "ds": dates, "y": blue})
    for col in FEAT_COLS:
        ts_sum[col] = feats[col].to_numpy()
        ts_blue[col] = feats[col].to_numpy()
    return pd.concat([ts_sum, ts_blue], axis=0)


class StreamingTimesNetDataset(IterableDataset):
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


def _softmax_from_value(val: float, max_num: int, temperature: float = 3.0):
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
    resume: bool = True
    fresh: bool = False


def _timesnet_ckpt_path(save_dir: Path, cfg: TimesNetConfig) -> Path:
    base = f"timesnet_in{cfg.input_size}_h{cfg.h}_hid{cfg.hidden_size}_tk{cfg.top_k}_dr{cfg.dropout}_ms{cfg.max_steps}_lr{cfg.learning_rate}"
    return save_dir / f"{base}.pkl"


def _eval_timesnet_loss(df: pd.DataFrame, cfg: TimesNetConfig, max_train: int = 200, trials: int = 2) -> float:
    """
    稳健评估：多次子采样取中位数；hidden_size 自动偶数化。
    """
    df_recent = df.tail(max_train)
    losses = []

    def _even_cfg(c: TimesNetConfig) -> TimesNetConfig:
        hid = c.hidden_size if c.hidden_size % 2 == 0 else c.hidden_size + 1
        return TimesNetConfig(
            input_size=c.input_size,
            h=c.h,
            hidden_size=hid,
            top_k=c.top_k,
            dropout=c.dropout,
            learning_rate=c.learning_rate,
            max_steps=c.max_steps,
            batch_size=c.batch_size,
            valid_size=c.valid_size,
            early_stop_patience_steps=c.early_stop_patience_steps,
        )

    cfg = _even_cfg(cfg)

    for _ in range(max(1, trials)):
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
            hist_exog_list=FEAT_COLS,
        )
        if hasattr(model, "trainer_kwargs"):
            model.trainer_kwargs["logger"] = False
            model.trainer_kwargs["enable_progress_bar"] = False
            model.trainer_kwargs["enable_model_summary"] = False  # 屏蔽重复的模型摘要提示
        try:
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
            losses.append(mae / count if count else 1e3)
        except Exception:
            losses.append(500.0)
    return float(np.median(losses))


def bayes_optimize_timesnet(
    df: pd.DataFrame,
    recent: int = 400,
    n_iter: int = 40,
    random_state: int = 42,
):
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real, Categorical
        from skopt.sampler import Sobol
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行 TimesNet 贝叶斯调参") from e

    # 动态样本：固定在 220~320，减少长序列波动
    if recent <= 0:
        df_recent = df.tail(min(len(df), 320))
    else:
        df_recent = df.tail(max(220, min(recent, 320)))
    if len(df_recent) < 100:
        raise ValueError("样本不足，无法进行 TimesNet 贝叶斯调参")

    # 离散化先验，降低搜索难度
    space = [
        Categorical([120, 140, 160, 180, min(200, len(df_recent) - 5)], name="input_size"),
        Categorical([72, 96, 112], name="hidden_size"),
        Categorical([3, 4], name="top_k"),
        Real(0.05, 0.15, name="dropout"),
        Real(3e-4, 1.2e-3, prior="log-uniform", name="learning_rate"),
        Integer(60, 110, name="max_steps"),
        Categorical([16, 32], name="batch_size"),
    ]

    fail_stats = {"fails": 0, "total": 0}

    def _adjust_hidden(hid: int) -> int:
        return int(hid if hid % 2 == 0 else hid + 1)

    def _eval_with_mode(params, light: bool = False) -> float:
        """light 模式用更短步数/更小窗口，降低单次失败惩罚；heavy 模式为正式评估。
        引入 trimmed mean + std 惩罚，爆炸/失败软罚后重试一次。"""
        inp, hid, tk, dr, lr, ms, bs = params
        hid = _adjust_hidden(hid)
        tk = max(1, min(tk, hid // 4 if hid // 4 > 0 else 1))
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
            early_stop_patience_steps=2,
        )
        losses = []
        fails = 0
        trials = 3 if light else 4
        penalty = 200.0 if light else 220.0
        for _ in range(trials):
            try:
                eval_cfg = cfg
                eval_cfg.max_steps = min(cfg.max_steps, 70 if light else 110)
                max_train = min(len(df_recent), 260 if light else 340)
                loss = _eval_timesnet_loss(df_recent, eval_cfg, max_train=max_train)
            except Exception:
                # 爆炸/失败软罚；再追加一次重试机会
                try:
                    eval_cfg = cfg
                    eval_cfg.max_steps = min(cfg.max_steps, 60 if light else 90)
                    loss = _eval_timesnet_loss(df_recent, eval_cfg, max_train=min(len(df_recent), 220 if light else 280))
                except Exception:
                    loss = penalty
                fails += 1
            losses.append(loss)
        fail_stats["fails"] += fails
        fail_stats["total"] += trials
        if len(losses) >= 3:
            arr = np.sort(losses)
            trimmed = arr[1:-1] if len(arr) > 3 else arr
            med = float(np.median(trimmed))
            std_part = float(np.std(trimmed))
            smooth_loss = med + 0.3 * std_part
        else:
            smooth_loss = float(np.median(losses))
        p_fail = fails / trials
        return p_fail * penalty + (1 - p_fail) * smooth_loss

    # 优先使用 Optuna TPE，多保真：轻评估筛选，再对前若干进行重评
    try:
        import optuna
        from optuna.study import StudyDirection

        def _optuna_objective(trial: "optuna.Trial"):
            fail_ratio = fail_stats["fails"] / fail_stats["total"] if fail_stats["total"] > 0 else 0.0
            lr_low_eff = 2e-4 if fail_ratio >= 0.5 else space[4].low
            lr_high_eff = 1.5e-3 if fail_ratio >= 0.5 else space[4].high
            dr_high_eff = 0.12 if fail_ratio >= 0.5 else space[3].high

            inp = trial.suggest_categorical("input_size", list(space[0].categories))
            hid = _adjust_hidden(trial.suggest_categorical("hidden_size", list(space[1].categories)))
            tk = trial.suggest_categorical("top_k", list(space[2].categories))
            dr = trial.suggest_float("dropout", space[3].low, space[3].high)
            dr = float(np.clip(dr, space[3].low, dr_high_eff))
            lr = trial.suggest_float("learning_rate", space[4].low, space[4].high, log=True)
            lr = float(np.clip(lr, lr_low_eff, lr_high_eff))
            ms = trial.suggest_int("max_steps", space[5].low, space[5].high)
            bs = trial.suggest_categorical("batch_size", list(space[6].categories))
            return _eval_with_mode([inp, hid, tk, dr, lr, ms, bs], light=True)

        study = optuna.create_study(direction=StudyDirection.MINIMIZE, sampler=optuna.samplers.TPESampler(seed=random_state))
        tpe_trials = max(40, min(70, int(n_iter)))  # 再增试次，争取更好初筛
        study.optimize(_optuna_objective, n_trials=tpe_trials, show_progress_bar=False)

        # 取前3-8名做重评 + 多起点重启
        topk = min(8, len(study.trials))
        sorted_trials = sorted(study.trials, key=lambda t: t.value)[:topk]
        best = None
        best_loss = 1e9
        seeds = [random_state + i * 101 for i in range(topk)]
        # 连续失败放宽标志
        fail_counts = 0
        def _maybe_relax_bounds():
            nonlocal space
            # 放宽 lr 下界到 2e-4，上界到 1.5e-3；dropout 上界降到 0.12
            space = [
                space[0],
                space[1],
                space[2],
                Real(0.05, 0.12, name="dropout"),
                Real(2e-4, 1.5e-3, prior="log-uniform", name="learning_rate"),
                space[5],
                space[6],
            ]

        for t, sd in zip(sorted_trials, seeds):
            p = t.params
            cand = [
                int(p["input_size"]),
                _adjust_hidden(int(p["hidden_size"])),
                int(p["top_k"]),
                float(p["dropout"]),
                float(p["learning_rate"]),
                int(p["max_steps"]),
                int(p["batch_size"]),
            ]
            # 根据失败率动态放宽/收紧 lr 和 dropout
            fail_ratio = fail_stats["fails"] / fail_stats["total"] if fail_stats["total"] > 0 else 0.0
            lr_low_eff = 2e-4 if fail_ratio >= 0.5 else space[4].low
            lr_high_eff = 1.5e-3 if fail_ratio >= 0.5 else space[4].high
            dr_high_eff = 0.12 if fail_ratio >= 0.5 else space[3].high
            cand[3] = float(np.clip(cand[3], space[3].low, dr_high_eff))
            cand[4] = float(np.clip(cand[4], lr_low_eff, lr_high_eff))
            # 重评
            loss = _eval_with_mode(cand, light=False)
            if loss < best_loss:
                best_loss = loss
                best = cand
            else:
                fail_counts += 1
                if fail_counts >= max(2, topk // 2):
                    _maybe_relax_bounds()
            # 多起点局部扰动子代
            rng = np.random.default_rng(sd)
            for _ in range(12):
                loc_inp = int(rng.choice(space[0].categories))
                loc_hid = _adjust_hidden(int(rng.choice(space[1].categories)))
                loc_tk = int(rng.choice(space[2].categories))
                loc_dr = float(np.clip(rng.normal(cand[3], 0.02), space[3].low, dr_high_eff))
                loc_lr = float(np.clip(rng.normal(cand[4], cand[4] * 0.25), lr_low_eff, lr_high_eff))
                loc_ms = int(np.clip(rng.integers(cand[5] - 10, cand[5] + 11), space[5].low, space[5].high))
                loc_bs = int(rng.choice(space[6].categories))
                loc_cand = [loc_inp, loc_hid, loc_tk, loc_dr, loc_lr, loc_ms, loc_bs]
                loc_loss = _eval_with_mode(loc_cand, light=False)
                if loc_loss < best_loss:
                    best_loss = loc_loss
                    best = loc_cand
        hid_final = _adjust_hidden(int(best[1]))
        tk_final = int(max(1, min(int(best[2]), hid_final // 4 if hid_final // 4 > 0 else 1)))
        best_params = {
            "input_size": int(best[0]),
            "hidden_size": hid_final,
            "top_k": tk_final,
            "dropout": float(best[3]),
            "learning_rate": float(best[4]),
            "max_steps": int(best[5]),
            "batch_size": int(best[6]),
        }
        return best_params, best_loss
    except Exception:
        # 如果 Optuna 不可用或失败，退回 skopt 流程
        pass

    n_calls = max(30, min(40, int(n_iter)))  # 默认 30~36，可扩到 40
    n_coarse = max(10, int(n_calls * 0.4))
    n_mid = max(10, int(n_calls * 0.35))
    n_fine = max(10, n_calls - n_coarse - n_mid)

    def _run_search(space, calls, seed):
        try:
            res = gp_minimize(
                objective,
                space,
                n_calls=calls,
                random_state=seed,
                verbose=False,
            )
            return res.x, float(res.fun), None
        except Exception:
            rng = np.random.default_rng(seed)
            loc_best = None
            loc_loss = 1e9
            losses_log: list[float] = []
            samples = None  # 混合 Categorical，Sobol 难以适配，直接用随机采样
            for idx in range(calls):
                dim = space[0]
                inp = int(rng.choice(dim.categories)) if hasattr(dim, "categories") else int(rng.integers(dim.low, dim.high + 1))
                dim = space[1]
                hid = int(rng.choice(dim.categories)) if hasattr(dim, "categories") else int(rng.integers(dim.low, dim.high + 1))
                dim = space[2]
                tk = int(rng.choice(dim.categories)) if hasattr(dim, "categories") else int(rng.integers(dim.low, dim.high + 1))
                dim = space[3]
                dr = float(rng.uniform(dim.low, dim.high))
                dim = space[4]
                lr = float(np.exp(rng.uniform(np.log(dim.low), np.log(dim.high))))
                dim = space[5]
                ms = int(rng.integers(dim.low, dim.high + 1))
                bs = int(rng.choice(space[6].categories))
                hid = _adjust_hidden(int(hid))
                tk = max(1, min(tk, hid // 4 if hid // 4 > 0 else 1))
                cfg = TimesNetConfig(
                    input_size=int(inp),
                    h=1,
                    hidden_size=int(hid),
                    top_k=int(tk),
                    dropout=dr,
                    learning_rate=lr,
                    max_steps=int(ms),
                    batch_size=bs,
                    valid_size=0.1,
                    early_stop_patience_steps=2,
                )
                try:
                    eval_cfg = cfg
                    eval_cfg.max_steps = min(cfg.max_steps, 80)
                    loss = _eval_timesnet_loss(df_recent, eval_cfg, max_train=min(len(df_recent), 300))
                except Exception:
                    loss = 500.0
                losses_log.append(loss)
                if loss < loc_loss:
                    loc_loss = loss
                    loc_best = [inp, hid, tk, dr, lr, ms, bs]
            return loc_best, loc_loss, losses_log

    best, best_loss, coarse_log = _run_search(space, n_coarse, seed=random_state)
    if coarse_log and len(coarse_log) >= 3:
        tail = coarse_log[-max(3, len(coarse_log) // 3):]
        if np.std(tail) > 50:
            extra_calls = max(4, int(n_coarse * 0.2))
            extra_res, extra_loss, _ = _run_search(space, extra_calls, seed=random_state + 99)
            if extra_res is not None and extra_loss < best_loss:
                best, best_loss = extra_res, extra_loss

    # Mid stage
    if best is not None:
        b_inp, b_hid, b_tk, b_dr, b_lr, b_ms, b_bs = best
        b_hid = _adjust_hidden(int(b_hid))
        space_mid = [
            Integer(max(80, int(b_inp) - 30), min(int(b_inp) + 30, len(df_recent) - 5), name="input_size"),
            Integer(max(60, int(b_hid) - 30), min(140, int(b_hid) + 30), name="hidden_size"),
            Integer(max(2, int(b_tk) - 1), min(4, int(b_tk) + 1), name="top_k"),
            Real(max(0.04, b_dr / 1.5), min(0.15, b_dr * 1.5), name="dropout"),
            Real(max(3e-4, b_lr / 1.5), min(1.2e-3, b_lr * 1.5), prior="log-uniform", name="learning_rate"),
            Integer(max(60, int(b_ms) - 25), min(120, int(b_ms) + 25), name="max_steps"),
            Categorical([b_bs, 16, 32]),
        ]
        mid_res, mid_loss, mid_log = _run_search(space_mid, n_mid, seed=random_state + 1)
        if mid_res is not None and mid_loss < best_loss:
            best, best_loss = mid_res, mid_loss
        if mid_log and len(mid_log) >= 3:
            tail = mid_log[-max(3, len(mid_log) // 3):]
            if np.std(tail) > 40:
                extra_calls = max(4, int(n_mid * 0.2))
                extra_res, extra_loss, _ = _run_search(space_mid, extra_calls, seed=random_state + 199)
                if extra_res is not None and extra_loss < best_loss:
                    best, best_loss = extra_res, extra_loss

    # Fine stage
    if best is not None:
        b_inp, b_hid, b_tk, b_dr, b_lr, b_ms, b_bs = best
        b_hid = _adjust_hidden(int(b_hid))
        space_fine = [
            Integer(max(85, int(b_inp) - 20), min(int(b_inp) + 20, len(df_recent) - 5), name="input_size"),
            Integer(max(60, int(b_hid) - 20), min(140, int(b_hid) + 20), name="hidden_size"),
            Integer(max(2, int(b_tk) - 1), min(4, int(b_tk) + 1), name="top_k"),
            Real(max(0.045, b_dr / 1.3), min(0.14, b_dr * 1.3), name="dropout"),
            Real(max(3e-4, b_lr / 1.3), min(1.1e-3, b_lr * 1.3), prior="log-uniform", name="learning_rate"),
            Integer(max(60, int(b_ms) - 20), min(110, int(b_ms) + 20), name="max_steps"),
            Categorical([b_bs, 16, 32]),
        ]
        fine_res, fine_loss, _ = _run_search(space_fine, n_fine, seed=random_state + 2)
        if fine_res is not None and fine_loss < best_loss:
            best, best_loss = fine_res, fine_loss

    # Local refine around current best
    def _local_refine(current_best, current_loss, seed):
        if current_best is None:
            return current_best, current_loss
        rng = np.random.default_rng(seed)
        inp, hid, tk, dr, lr, ms, bs = current_best
        loc_best = current_best
        loc_loss = current_loss
        inp_choices = list(space[0].categories) if hasattr(space[0], "categories") else None
        hid_choices = list(space[1].categories) if hasattr(space[1], "categories") else None
        tk_choices = list(space[2].categories) if hasattr(space[2], "categories") else None
        dr_low, dr_high = (space[3].low, space[3].high) if hasattr(space[3], "low") else (0.05, 0.15)
        lr_low, lr_high = (space[4].low, space[4].high) if hasattr(space[4], "low") else (3e-4, 1.2e-3)
        for _ in range(8):
            cand_inp = int(rng.choice(inp_choices)) if inp_choices else int(np.clip(rng.integers(inp - 15, inp + 16), 90, 200))
            cand_hid = _adjust_hidden(int(rng.choice(hid_choices))) if hid_choices else _adjust_hidden(int(np.clip(rng.integers(hid - 10, hid + 11), 64, 140)))
            cand_tk = int(rng.choice(tk_choices)) if tk_choices else int(np.clip(rng.integers(max(2, tk - 1), tk + 2), 2, 4))
            cand_dr = float(np.clip(rng.normal(dr, 0.02), dr_low, min(dr_high, 0.18)))
            cand_lr = float(np.clip(rng.normal(lr, lr * 0.25), lr_low, 1.5e-3))
            cand_ms = int(np.clip(rng.integers(ms - 15, ms + 16), 60, 120))
            cand_bs = int(bs)
            cfg = TimesNetConfig(
                input_size=cand_inp,
                h=1,
                hidden_size=cand_hid,
                top_k=max(1, min(cand_tk, cand_hid // 4 if cand_hid // 4 > 0 else 1)),
                dropout=cand_dr,
                learning_rate=cand_lr,
                max_steps=cand_ms,
                batch_size=cand_bs,
                valid_size=0.1,
                early_stop_patience_steps=2,
            )
            try:
                eval_cfg = cfg
                eval_cfg.max_steps = min(cfg.max_steps, 100)
                loss = _eval_timesnet_loss(df_recent, eval_cfg, max_train=min(len(df_recent), 320))
            except Exception:
                loss = 250.0
            if loss < loc_loss:
                loc_loss = loss
                loc_best = [
                    cand_inp,
                    cand_hid,
                    cand_tk,
                    cand_dr,
                    cand_lr,
                    cand_ms,
                    cand_bs,
                ]
        return loc_best, loc_loss

    best, best_loss = _local_refine(best, best_loss, seed=random_state + 333)

    # 兜底：若仍高loss，回退保守参数
    if best is None or best_loss >= 400:
        best = [
            min(140, max(100, len(df_recent) // 2)),
            _adjust_hidden(96),
            3,
            0.08,
            6e-4,
            90,
            16,
        ]
        best_loss = float(best_loss if best_loss is not None else 500.0)

    hid_final = _adjust_hidden(int(best[1]))
    tk_final = int(max(1, min(int(best[2]), hid_final // 4 if hid_final // 4 > 0 else 1)))
    best_params = {
        "input_size": int(best[0]),
        "hidden_size": hid_final,
        "top_k": tk_final,
        "dropout": float(best[3]),
        "learning_rate": float(best[4]),
        "max_steps": int(best[5]),
        "batch_size": int(best[6]),
    }
    return best_params, best_loss


def train_timesnet(
    df: pd.DataFrame,
    cfg: TimesNetConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    # 确保 hidden_size 为偶数，避免位置编码 shape 错误
    if cfg.hidden_size % 2 != 0:
        cfg = TimesNetConfig(
            input_size=cfg.input_size,
            h=cfg.h,
            hidden_size=cfg.hidden_size + 1,
            top_k=cfg.top_k,
            dropout=cfg.dropout,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            batch_size=cfg.batch_size,
            valid_size=cfg.valid_size,
            early_stop_patience_steps=cfg.early_stop_patience_steps,
        )
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


def random_optimize_timesnet(
    df: pd.DataFrame,
    recent: int = 400,
    samples: int = 60,
    random_state: int = 42,
):
    """
    重型随机搜索 TimesNet 超参，适合作为 TPE/贝叶斯不稳定时的兜底。
    空间与当前 TPE 一致，评估采用稳健多次中位 + 软罚。
    """
    rng = np.random.default_rng(random_state)
    df_recent = df.tail(max(220, min(recent if recent > 0 else len(df), 500)))
    if len(df_recent) < 120:
        raise ValueError("样本不足，无法进行 TimesNet 随机调参")

    def _adjust_hidden(hid: int) -> int:
        return int(hid if hid % 2 == 0 else hid + 1)

    def _sample_one():
        inp = int(rng.integers(90, min(200, len(df_recent) - 5) + 1))
        hid = int(rng.integers(64, 121))
        tk = int(rng.choice([3, 4]))
        dr = float(rng.uniform(0.05, 0.15))
        lr = float(np.exp(rng.uniform(np.log(3e-4), np.log(1.2e-3))))
        ms = int(rng.integers(60, 111))
        bs = int(rng.choice([16, 32]))
        hid = _adjust_hidden(hid)
        tk = max(1, min(tk, hid // 4 if hid // 4 > 0 else 1))
        return [inp, hid, tk, dr, lr, ms, bs]

    def _eval(params):
        inp, hid, tk, dr, lr, ms, bs = params
        cfg = TimesNetConfig(
            input_size=int(inp),
            h=1,
            hidden_size=int(hid),
            top_k=int(tk),
            dropout=float(dr),
            learning_rate=float(lr),
            max_steps=int(ms),
            batch_size=int(bs),
            valid_size=0.1,
            early_stop_patience_steps=2,
        )
        losses = []
        fails = 0
        for _ in range(3):
            try:
                eval_cfg = cfg
                eval_cfg.max_steps = min(cfg.max_steps, 100)
                loss = _eval_timesnet_loss(df_recent, eval_cfg, max_train=min(len(df_recent), 320))
            except Exception:
                loss = 300.0
                fails += 1
            losses.append(loss)
        med = float(np.median(losses))
        std_part = float(np.std(losses))
        smooth = med + 0.3 * std_part
        p_fail = fails / 3
        return p_fail * 300 + (1 - p_fail) * smooth

    best = None
    best_loss = 1e9
    for _ in range(samples):
        cand = _sample_one()
        loss = _eval(cand)
        if loss < best_loss:
            best_loss = loss
            best = cand
    if best is None:
        raise RuntimeError("TimesNet 随机搜索未找到可行解")
    return {
        "input_size": int(best[0]),
        "hidden_size": int(best[1]),
        "top_k": int(best[2]),
        "dropout": float(best[3]),
        "learning_rate": float(best[4]),
        "max_steps": int(best[5]),
        "batch_size": int(best[6]),
    }, float(best_loss)


    return {
        "sum_pred": sum_pred,
        "blue_pred": blue_pred,
        "blue_probs": blue_probs,
        "red_probs": red_probs,
    }


def predict_timesnet(nf: NeuralForecast, df: pd.DataFrame) -> Dict[str, float]:
    # TimesNet 需要一定的输入长度，若过短则返回默认值
    if len(df) < 10: 
        return {
            "sum_pred": 100.0,
            "blue_pred": 8.0,
            "blue_probs": [(8, 1.0)],
            "red_probs": [],
        }
    ts = _build_timeseries(df)
    fcst = nf.predict(ts)
    sum_pred = float(fcst[fcst["unique_id"] == "sum"]["TimesNet"].iloc[-1])
    blue_pred = float(fcst[fcst["unique_id"] == "blue"]["TimesNet"].iloc[-1])
    blue_probs = _softmax_from_value(blue_pred, max_num=16, temperature=3.0)
    red_probs = _freq_probs(df, max_num=33)
    return {
        "sum_pred": sum_pred,
        "blue_pred": blue_pred,
        "blue_probs": blue_probs,
        "red_probs": red_probs,
    }


def batch_predict_timesnet(nf: NeuralForecast, cfg: TimesNetConfig, df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Use cross_validation for batch predictions.
    Identical logic to batch_predict_nhits but for TimesNet.
    """
    ts = _build_timeseries(df)
    total_len = len(ts) // 2 
    n_windows = total_len - cfg.input_size
    if n_windows <= 0:
        return {}

    try:
        cv_df = nf.cross_validation(
            ts, 
            n_windows=n_windows, 
            step_size=1, 
            val_size=1, 
            refit=False
        )
    except Exception as e:
        msg = str(e)
        if "too short" in msg or "input size" in msg:
            pass
        else:
            print(f"[TimesNet] Batch Predict Warning: {e}")
        return {}
    
    blue_preds = cv_df[cv_df["unique_id"] == "blue"].set_index("ds")
    sum_preds = cv_df[cv_df["unique_id"] == "sum"].set_index("ds")
    
    results = {}
    dates = pd.to_datetime(df["draw_date"])
    
    for idx, date in enumerate(dates):
        if idx < cfg.input_size: continue
        
        if date not in blue_preds.index:
            continue
            
        try:
            b_p = float(blue_preds.loc[date]["TimesNet"])
            s_p = float(sum_preds.loc[date]["TimesNet"]) if date in sum_preds.index else 0.0
            
            blue_probs = _softmax_from_value(b_p, max_num=16, temperature=3.0)
            
            results[idx] = {
                "sum_pred": s_p,
                "blue_pred": b_p,
                "blue_probs": blue_probs,
            }
        except TypeError:
            continue
            
    return results


def backtest_timesnet_model(
    nf: NeuralForecast,
    cfg: TimesNetConfig,
    df: pd.DataFrame,
    max_samples: int = 300,
) -> Dict[str, float]:
    """
    流式回测：逐步截取历史->预测下一期蓝球，统计 Top1 命中率和 MAE。
    """
    blue_hits = []
    mae_blue_list = []
    total = len(df)
    start_idx = max(cfg.input_size, total - max_samples)
    for i in range(start_idx, total - 1):
        hist = df.iloc[: i + 1]
        ts = _build_timeseries(hist)
        try:
            fcst = nf.predict(ts)
            blue_pred = float(fcst[fcst["unique_id"] == "blue"]["TimesNet"].iloc[-1])
            # Use rounding for Top1 accuracy in regression
            top1 = int(round(blue_pred))
            true_b = int(df.iloc[i + 1]["blue"])
            blue_hits.append(1.0 if top1 == true_b else 0.0)
            mae_blue_list.append(abs(blue_pred - true_b))
        except Exception:
            continue
    blue_hit = float(np.mean(blue_hits)) if blue_hits else 0.0
    mae_blue = float(np.mean(mae_blue_list)) if mae_blue_list else 100.0
    
    score = 1.0 / (1.0 + mae_blue)
    return {"blue_top1": blue_hit, "mae_blue": mae_blue, "score": score, "samples": len(blue_hits)}


class TimesNetModelAdapter(ModelAdapter):
    def train_step(self, member: Member, dataset: pd.DataFrame, steps: int) -> Tuple[Any, float]:
        cfg = member.config
        ts = _build_timeseries(dataset)
        val_size = max(cfg.valid_size, cfg.h)
        
        if member.model_state is None:
             # Ensure hidden size is even
             hid = cfg.hidden_size if cfg.hidden_size % 2 == 0 else cfg.hidden_size + 1
             models = [
                TimesNet(
                    input_size=cfg.input_size,
                    h=cfg.h,
                    hidden_size=hid,
                    top_k=cfg.top_k,
                    dropout=cfg.dropout,
                    learning_rate=cfg.learning_rate,
                    max_steps=cfg.max_steps, 
                    batch_size=cfg.batch_size,
                    random_seed=42,
                    early_stop_patience_steps=cfg.early_stop_patience_steps,
                )
            ]
             nf = NeuralForecast(models=models, freq="D")
        else:
             nf = member.model_state
             
        for m in nf.models:
            m.max_steps += 50
            # Disable logger to prevent TensorBoard errors during PBT
            m.trainer_kwargs["logger"] = False
             
        try:
            nf.fit(ts, val_size=val_size)
            
            dates = ts["ds"].unique()
            dates = np.sort(dates)
            val_dates = dates[-cfg.h:]
            cutoff_date = val_dates[0]
            
            ts_in = ts[ts["ds"] < cutoff_date]
            
            fcst = nf.predict(ts_in)
            mae = 0.0
            count = 0
            for uid in ["sum", "blue"]:
                 truth = ts[(ts["unique_id"] == uid) & (ts["ds"].isin(val_dates))]["y"].to_numpy()
                 pred = fcst[fcst["unique_id"] == uid]["TimesNet"].to_numpy()
                 min_len = min(len(truth), len(pred))
                 if min_len > 0:
                     mae += float(np.abs(truth[:min_len] - pred[:min_len]).mean())
                     count += 1
            loss = mae / count if count > 0 else 1000.0
        except Exception as e:
            print(f"TimesNet Train Step Error: {e}")
            import traceback
            traceback.print_exc()
            return nf, 0.0
        
        return nf, -loss

    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        nf = member.model_state
        if nf is None: return 0.0
        
        # if not getattr(nf, "fitted", False):
        #     return 0.0
            
        cfg = member.config
        
        try:
            res = backtest_timesnet_model(nf, cfg, dataset, max_samples=60)
            score = res["score"]
        except Exception as e:
            print(f"TimesNet Eval Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        return score

    def perturb_config(self, config: TimesNetConfig) -> TimesNetConfig:
        new_cfg = copy.deepcopy(config)
        factors = [0.8, 1.2]
        if random.random() < 0.3:
            new_cfg.learning_rate *= random.choice(factors)
        if random.random() < 0.3:
             new_cfg.dropout = max(0.01, min(0.5, new_cfg.dropout * random.choice([0.9, 1.1])))
        return new_cfg

    def save(self, member: Member, path: Path) -> None:
        if member.model_state:
            with open(path, "wb") as f:
                pickle.dump(member.model_state, f)

    def load(self, path: Path) -> Member:
        pass


class TimesNetPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = TimesNetConfig(
            input_size=config.input_size,
            h=1,
            hidden_size=config.hidden_size,
            top_k=config.top_k,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            dropout=config.dropout,
            batch_size=32,
            valid_size=0.1,
            resume=getattr(config, 'resume', True),
            fresh=getattr(config, 'fresh', False),
        )
        self.model = None

    def train(self, df: pd.DataFrame) -> None:
        logger.info(f"Training TimesNet (size={self.cfg.input_size})...")
        self.model, self.cfg = train_timesnet(
            df,
            self.cfg,
            save_dir="models/TimesNet",
            resume=self.cfg.resume,
            fresh=self.cfg.fresh
        )

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("TimesNet model not trained or loaded")
        return predict_timesnet(self.model, df)
    
    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        # TimesNet also uses NeuralForecast object wrapper (pickle)
        model_path = path / f"timesnet_in{self.cfg.input_size}_h{self.cfg.hidden_size}_{self.cfg.max_steps}s.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"TimesNet model saved to {model_path}")

    def load(self, save_dir: str) -> bool:
        path = Path(save_dir)
        if not path.exists(): return False
        
        model_path = path / f"timesnet_in{self.cfg.input_size}_h{self.cfg.hidden_size}_{self.cfg.max_steps}s.pkl"
        if not model_path.exists(): return False
        
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"TimesNet model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load TimesNet model: {e}")
            return False


