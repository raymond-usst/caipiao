from __future__ import annotations

from datetime import datetime
from lottery.utils.logger import logger
from lottery.engine.predictor import BasePredictor

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import Adam
from .features import build_features
from .features import build_features
import torch.nn.functional as F
from .pbt import ModelAdapter, Member
import copy
import random

COMBO_VOCAB = 20000  # 红球组合哈希桶


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0]
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def combo_hash(reds: np.ndarray) -> int:
    """对排序后的红球组合做哈希映射，位置无关组合特征。"""
    import zlib
    reds_sorted = tuple(sorted(int(x) for x in reds))
    # 使用 adler32 保证确定性
    return zlib.adler32(str(reds_sorted).encode("utf-8")) % COMBO_VOCAB


def _build_feature_matrix(df: pd.DataFrame, freq_window: int = 50, entropy_window: int = 50) -> np.ndarray:
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    n = len(data)
    feat_list = []
    extra_feats = build_features(df).to_numpy(dtype=float)
    red_hist = np.zeros(34)  # 1..33
    blue_hist = np.zeros(17)  # 1..16
    red_counts_window = []
    blue_counts_window = []
    for i in range(n):
        row = data[i]
        reds = row[:6]
        blue = row[6]

        # 时间特征
        if "draw_date" in df.columns:
            dt = pd.to_datetime(df.iloc[i]["draw_date"])
            dow = dt.weekday()  # 0-6
            month = dt.month
        else:
            dow = 0
            month = 1
        dow_sin, dow_cos = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
        mon_sin, mon_cos = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)

        # 滑窗频率与熵（使用前 freq_window/entropy_window 期）
        if i > 0:
            red_counts_window.append(reds)
            blue_counts_window.append(blue)
        if len(red_counts_window) > freq_window:
            red_counts_window.pop(0)
        if len(blue_counts_window) > freq_window:
            blue_counts_window.pop(0)

        red_freq = 0.0
        blue_freq = 0.0
        if red_counts_window:
            rc = np.bincount(np.array(red_counts_window).ravel(), minlength=34)[1:]
            red_freq = float(rc.mean() / freq_window)
        if blue_counts_window:
            bc = np.bincount(np.array(blue_counts_window), minlength=17)[1:]
            blue_freq = float(bc.mean() / freq_window)

        # 熵
        red_entropy = 0.0
        if len(red_counts_window) >= max(5, entropy_window // 5):
            rc = np.bincount(np.array(red_counts_window).ravel(), minlength=34)[1:]
            red_entropy = _entropy(rc)

        reds_norm = reds / 33.0
        blue_norm = np.array([blue]) / 16.0
        s = row.sum() / (33 * 6 + 16)
        odd = (reds % 2).sum() / 6.0
        span = (reds.max() - reds.min()) / 33.0

        feat = np.concatenate(
            [
                reds_norm,
                blue_norm,
                [s, odd, span, red_freq, blue_freq, red_entropy, dow_sin, dow_cos, mon_sin, mon_cos],
                extra_feats[i],
            ]
        )
        feat_list.append(feat)
    return np.stack(feat_list)


class TFTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int = 20, freq_window: int = 50, entropy_window: int = 50):
        cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
        data = df[cols].to_numpy(dtype=int)
        feats = _build_feature_matrix(df, freq_window=freq_window, entropy_window=entropy_window)
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.combo_ids: List[int] = []
        for i in range(window, len(data)):
            hist = feats[i - window : i]
            tgt = data[i]
            combo_id = combo_hash(tgt[:6])
            self.samples.append((hist, tgt, combo_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        hist, tgt, combo_id = self.samples[idx]
        return (
            torch.tensor(hist, dtype=torch.float32),
            torch.tensor(tgt, dtype=torch.long),
            torch.tensor(combo_id, dtype=torch.long),
        )


def collate_tft(batch):
    xs, ys, combo = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(combo)


class StreamingTFTDataset(IterableDataset):
    """
    流式回测数据集，避免一次性展开所有样本。
    """

    def __init__(self, df: pd.DataFrame, window: int = 20, freq_window: int = 50, entropy_window: int = 50):
        self.df = df
        self.window = window
        self.freq_window = freq_window
        self.entropy_window = entropy_window

    def __iter__(self):
        cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
        data = self.df[cols].to_numpy(dtype=int)
        feats = _build_feature_matrix(self.df, freq_window=self.freq_window, entropy_window=self.entropy_window)
        for i in range(self.window, len(data)):
            hist = feats[i - self.window : i]
            tgt = data[i]
            combo_id = combo_hash(tgt[:6])
            yield (
                torch.tensor(hist, dtype=torch.float32),
                torch.tensor(tgt, dtype=torch.long),
                torch.tensor(combo_id, dtype=torch.long),
            )

    def __len__(self):
        return max(0, len(self.df) - self.window)


class GatedResidualNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.elu = nn.ELU()
        self.gate = nn.Linear(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.elu(self.fc1(x))
        y = self.dropout(self.fc2(y))
        skip = self.skip(x) if self.skip else x
        gate = torch.sigmoid(self.gate(y))
        return gate * y + (1 - gate) * skip


class SimpleTFT(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 3, ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.var_sel = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.combo_emb = nn.Embedding(COMBO_VOCAB, d_model)
        self.head_red = nn.ModuleList([nn.Linear(d_model, 33) for _ in range(6)])
        self.head_blue = nn.Linear(d_model, 16)
        # 多任务辅助头：和值回归、奇偶计数分类、跨度回归
        self.head_sum = nn.Linear(d_model, 1)
        self.head_odd = nn.Linear(d_model, 7)  # 0-6 奇数个数
        self.head_span = nn.Linear(d_model, 1)

    def forward(self, x, combo_ids=None):
        # x: [B, L, F]
        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.var_sel(h)
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)  # [B, d]
        if combo_ids is not None:
            pooled = pooled + self.combo_emb(combo_ids)
        reds = [head(pooled) for head in self.head_red]
        blue = self.head_blue(pooled)
        sum_pred = self.head_sum(pooled)
        odd_pred = self.head_odd(pooled)
        span_pred = self.head_span(pooled)
        return reds, blue, {"sum": sum_pred, "odd": odd_pred, "span": span_pred}


@dataclass
class TFTConfig:
    window: int = 20
    batch: int = 64
    epochs: int = 300
    patience: int = 8
    min_delta: float = 1e-3
    lr: float = 1e-3
    d_model: int = 128
    nhead: int = 4
    layers: int = 3
    ff: int = 256
    dropout: float = 0.1
    topk: int = 3
    freq_window: int = 50
    entropy_window: int = 50
    aux_sum_weight: float = 0.1
    aux_odd_weight: float = 0.1
    aux_span_weight: float = 0.1
    resume: bool = True
    fresh: bool = False


def _tft_ckpt_path(save_dir: Path, cfg: TFTConfig) -> Path:
    base = (
        f"tft_w{cfg.window}_dm{cfg.d_model}_h{cfg.nhead}_l{cfg.layers}_ff{cfg.ff}_dr{cfg.dropout}"
        f"_fw{cfg.freq_window}_ew{cfg.entropy_window}_lr{cfg.lr}"
    )
    return save_dir / f"{base}.pt"


def _quick_tft_loss(df: pd.DataFrame, cfg: "TFTConfig", max_epochs: int = 4) -> float:
    """
    用于贝叶斯优化的轻量评估：短周期训练返回最终loss。
    """
    dataset = TFTDataset(df, window=cfg.window, freq_window=cfg.freq_window, entropy_window=cfg.entropy_window)
    if len(dataset) < 10:
        raise ValueError("样本不足，无法评估TFT")
    loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True, collate_fn=collate_tft)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTFT(
        feature_dim=dataset[0][0].shape[-1],
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.layers,
        ff=cfg.ff,
        dropout=cfg.dropout,
    ).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    model.train()
    last_loss = 1e3
    for _ in range(max_epochs):
        total = 0.0
        steps = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            reds, blue, aux = model(x)
            loss = ce(blue, y[:, -1] - 1)
            for i, head_out in enumerate(reds):
                loss = loss + ce(head_out, y[:, i] - 1)
            sum_target = y[:, :7].sum(dim=1).float() / (33 * 6 + 16)
            odd_target = (y[:, :6] % 2).sum(dim=1)
            span_target = (y[:, :6].max(dim=1).values - y[:, :6].min(dim=1).values).float() / 33.0
            loss = (
                loss
                + cfg.aux_sum_weight * mse(aux["sum"].squeeze(-1), sum_target)
                + cfg.aux_odd_weight * ce(aux["odd"], odd_target)
                + cfg.aux_span_weight * mse(aux["span"].squeeze(-1), span_target)
            )
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        last_loss = total / max(1, steps)
    return float(last_loss)


def bayes_optimize_tft(
    df: pd.DataFrame,
    recent: int = 400,
    n_iter: int = 40,
    random_state: int = 42,
    heavy_epochs: int | None = None,
    tpe_trials: int | None = None,
):
    """
    使用 Optuna TPE + 多保真 + 稳健评估，对 TFT 做轻量超参搜索。
    - 离散/收缩空间，确保 d_model 可整除 nhead
    - 轻评估筛选，重评 + 局部扰动多起点，连续失败放宽 lr/dropout
    - trimmed mean + std 惩罚，失败软罚 + 重试
    """
    df_recent = df if recent <= 0 else df.tail(max(220, min(recent, 500)))
    if len(df_recent) < 120:
        raise ValueError("样本不足，无法进行 TFT 贝叶斯调参")

    # 离散化空间，降低搜索难度
    space = {
        "window": [10, 12, 14, 16, 18, 20, 22, 24],
        "d_model": [96, 128, 160, 192],
        "nhead": [2, 4],
        "layers": [2, 3, 4],
        "ff": [200, 240, 280, 320, 360, 400, 440, 480],
        "dropout": (0.05, 0.2),
        "lr": (3e-4, 1.2e-3),
        "batch": [32, 64],
    }

    def _adjust_dm(dm: int, nh: int) -> int:
        dm_adj = int(np.ceil(dm / nh) * nh)
        if dm_adj % nh != 0:
            dm_adj = (dm_adj // nh) * nh
        return max(min(dm_adj, max(space["d_model"])), min(space["d_model"]))

    fail_stats = {"fails": 0, "total": 0}

    def _eval_with_mode(params, light: bool = False, heavy_ep: int | None = None) -> float:
        """
        light：短步数、较小窗口、软罚；heavy：正式评估。
        采用 trimmed mean + 0.3*std，失败软罚，重试一次。
        """
        w, dm, nh, ly, ff, dr, lr, bs = params
        dm = _adjust_dm(dm, nh)
        cfg = TFTConfig(
            window=int(w),
            batch=int(bs),
            epochs=8 if light else (heavy_ep or 12),
            patience=3 if light else 6,
            lr=float(lr),
            d_model=int(dm),
            nhead=int(nh),
            layers=int(ly),
            ff=int(ff),
            dropout=float(dr),
            topk=1,
            freq_window=50,
            entropy_window=50,
        )
        trials = 3 if light else 4
        penalty = 190.0 if light else 210.0
        losses: list[float] = []
        fails = 0
        for _ in range(trials):
            try:
                loss = _quick_tft_loss(
                    df_recent,
                    cfg,
                    max_epochs=5 if light else (heavy_ep or 8),
                )
            except Exception:
                # 软罚并尝试降步数重试一次
                try:
                    loss = _quick_tft_loss(
                        df_recent,
                        cfg,
                        max_epochs=3 if light else (heavy_ep or 7),
                    )
                except Exception:
                    loss = penalty
                fails += 1
            losses.append(loss)
        fail_stats["fails"] += fails
        fail_stats["total"] += trials
        arr = np.sort(losses)
        trimmed = arr[1:-1] if len(arr) > 3 else arr
        med = float(np.median(trimmed))
        std_part = float(np.std(trimmed))
        smooth_loss = med + 0.3 * std_part
        p_fail = fails / trials
        return p_fail * penalty + (1 - p_fail) * smooth_loss

    try:
        import optuna
        from optuna.study import StudyDirection
    except Exception as e:
        raise ImportError("需要安装 optuna 以运行 TFT 贝叶斯调参") from e

    def _optuna_objective(trial: "optuna.Trial"):
        fail_ratio = fail_stats["fails"] / fail_stats["total"] if fail_stats["total"] > 0 else 0.0
        lr_low_eff = 2e-4 if fail_ratio >= 0.5 else space["lr"][0]
        lr_high_eff = 1.5e-3 if fail_ratio >= 0.5 else space["lr"][1]
        dr_high_eff = 0.12 if fail_ratio >= 0.5 else space["dropout"][1]

        w = trial.suggest_categorical("window", space["window"])
        dm = trial.suggest_categorical("d_model", space["d_model"])
        nh = trial.suggest_categorical("nhead", space["nhead"])
        ly = trial.suggest_categorical("layers", space["layers"])
        ff = trial.suggest_categorical("ff", space["ff"])
        dr = float(trial.suggest_float("dropout", space["dropout"][0], space["dropout"][1]))
        dr = float(np.clip(dr, space["dropout"][0], dr_high_eff))
        lr = float(trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True))
        lr = float(np.clip(lr, lr_low_eff, lr_high_eff))
        bs = trial.suggest_categorical("batch", space["batch"])
        return _eval_with_mode([w, dm, nh, ly, ff, dr, lr, bs], light=True)

    study = optuna.create_study(direction=StudyDirection.MINIMIZE, sampler=optuna.samplers.TPESampler(seed=random_state))
    tpe_trials_eff = tpe_trials if tpe_trials is not None else max(70, min(90, int(n_iter)))
    study.optimize(_optuna_objective, n_trials=tpe_trials_eff, show_progress_bar=False)

    # 重评与多起点局部扰动
    topk = min(10, len(study.trials))
    sorted_trials = sorted(study.trials, key=lambda t: t.value)[:topk]
    seeds = [random_state + i * 101 for i in range(topk)]
    best = None
    best_loss = 1e9

    def _maybe_relax_bounds():
        nonlocal space
        space["dropout"] = (space["dropout"][0], 0.12)
        space["lr"] = (2e-4, 1.5e-3)

    fail_counts = 0
    for t, sd in zip(sorted_trials, seeds):
        p = t.params
        cand = [
            int(p["window"]),
            int(p["d_model"]),
            int(p["nhead"]),
            int(p["layers"]),
            int(p["ff"]),
            float(p["dropout"]),
            float(p["lr"]),
            int(p["batch"]),
        ]
        # 连续失败放宽
        fail_ratio = fail_stats["fails"] / fail_stats["total"] if fail_stats["total"] > 0 else 0.0
        lr_low_eff = 2e-4 if fail_ratio >= 0.5 else space["lr"][0]
        lr_high_eff = 1.5e-3 if fail_ratio >= 0.5 else space["lr"][1]
        dr_high_eff = 0.12 if fail_ratio >= 0.5 else space["dropout"][1]
        cand[5] = float(np.clip(cand[5], space["dropout"][0], dr_high_eff))
        cand[6] = float(np.clip(cand[6], lr_low_eff, lr_high_eff))

        loss = _eval_with_mode(cand, light=False)
        if loss < best_loss:
            best_loss = loss
            best = cand
        else:
            fail_counts += 1
            if fail_counts >= max(2, topk // 2):
                _maybe_relax_bounds()
        rng = np.random.default_rng(sd)
        for _ in range(16):
            loc_inp = int(rng.choice(space["window"]))
            loc_dm = _adjust_dm(int(rng.choice(space["d_model"])), int(cand[2]))
            loc_tk = int(rng.choice(space["nhead"]))  # 实际 top_k 不适用 TFT，这里保持 nhead
            loc_ly = int(rng.choice(space["layers"]))
            loc_ff = int(rng.choice(space["ff"]))
            loc_dr = float(np.clip(rng.normal(cand[5], 0.02), space["dropout"][0], dr_high_eff))
            loc_lr = float(np.clip(rng.normal(cand[6], cand[6] * 0.25), lr_low_eff, lr_high_eff))
            loc_bs = int(rng.choice(space["batch"]))
            loc_cand = [loc_inp, loc_dm, cand[2], loc_ly, loc_ff, loc_dr, loc_lr, loc_bs]
            loc_loss = _eval_with_mode(loc_cand, light=False)
            if loc_loss < best_loss:
                best_loss = loc_loss
                best = loc_cand

    if best is None:
        raise RuntimeError("TFT 贝叶斯优化未找到可行解")

    best_params = {
        "window": int(best[0]),
        "d_model": int(_adjust_dm(int(best[1]), int(best[2]))),
        "nhead": int(best[2]),
        "layers": int(best[3]),
        "ff": int(best[4]),
        "dropout": float(best[5]),
        "lr": float(best[6]),
        "batch": int(best[7]),
    }
    return best_params, best_loss


def random_optimize_tft(
    df: pd.DataFrame,
    recent: int = 400,
    samples: int = 80,
    random_state: int = 42,
):
    """
    重型随机搜索 TFT 超参，作为 TPE/贝叶斯的兜底。
    空间与离散 TPE 一致，评估沿用稳健 trimmed mean + 软罚。
    """
    rng = np.random.default_rng(random_state)
    df_recent = df.tail(max(220, min(recent if recent > 0 else len(df), 500)))
    if len(df_recent) < 120:
        raise ValueError("样本不足，无法进行 TFT 随机调参")

    space = {
        "window": [10, 12, 14, 16, 18, 20, 22, 24],
        "d_model": [96, 128, 160, 192],
        "nhead": [2, 4],
        "layers": [2, 3, 4],
        "ff": [200, 240, 280, 320, 360, 400, 440, 480],
        "dropout": (0.05, 0.2),
        "lr": (3e-4, 1.2e-3),
        "batch": [32, 64],
    }

    def _adjust_dm(dm: int, nh: int) -> int:
        dm_adj = int(np.ceil(dm / nh) * nh)
        return max(min(dm_adj, max(space["d_model"])), min(space["d_model"]))

    def _eval(params):
        w, dm, nh, ly, ff, dr, lr, bs = params
        dm = _adjust_dm(dm, nh)
        cfg = TFTConfig(
            window=int(w),
            batch=int(bs),
            epochs=10,
            patience=5,
            lr=float(lr),
            d_model=int(dm),
            nhead=int(nh),
            layers=int(ly),
            ff=int(ff),
            dropout=float(dr),
            topk=1,
            freq_window=50,
            entropy_window=50,
        )
        trials = 3
        penalty = 210.0
        losses = []
        fails = 0
        for _ in range(trials):
            try:
                loss = _quick_tft_loss(df_recent, cfg, max_epochs=6)
            except Exception:
                try:
                    loss = _quick_tft_loss(df_recent, cfg, max_epochs=5)
                except Exception:
                    loss = penalty
                fails += 1
            losses.append(loss)
        arr = np.sort(losses)
        trimmed = arr[1:-1] if len(arr) > 3 else arr
        med = float(np.median(trimmed))
        std_part = float(np.std(trimmed))
        smooth = med + 0.3 * std_part
        p_fail = fails / trials
        return p_fail * penalty + (1 - p_fail) * smooth

    best = None
    best_loss = 1e9
    for _ in range(samples):
        cand = [
            int(rng.choice(space["window"])),
            int(rng.choice(space["d_model"])),
            int(rng.choice(space["nhead"])),
            int(rng.choice(space["layers"])),
            int(rng.choice(space["ff"])),
            float(rng.uniform(space["dropout"][0], space["dropout"][1])),
            float(np.exp(rng.uniform(np.log(space["lr"][0]), np.log(space["lr"][1])))),
            int(rng.choice(space["batch"])),
        ]
        loss = _eval(cand)
        if loss < best_loss:
            best_loss = loss
            best = cand
    if best is None:
        raise RuntimeError("TFT 随机搜索未找到可行解")
    best_params = {
        "window": int(best[0]),
        "d_model": int(_adjust_dm(int(best[1]), int(best[2]))),
        "nhead": int(best[2]),
        "layers": int(best[3]),
        "ff": int(best[4]),
        "dropout": float(best[5]),
        "lr": float(best[6]),
        "batch": int(best[7]),
    }
    return best_params, float(best_loss)


def train_tft(
    df: pd.DataFrame,
    cfg: TFTConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    if cfg.d_model % cfg.nhead != 0:
        raise ValueError(f"d_model 必须能被 nhead 整除，当前 d_model={cfg.d_model}, nhead={cfg.nhead}")
    dataset = TFTDataset(df, window=cfg.window, freq_window=cfg.freq_window, entropy_window=cfg.entropy_window)
    if len(dataset) < 10:
        raise ValueError("样本不足，无法训练 TFT")
    loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True, collate_fn=collate_tft)
    device = torch.device("cpu")  # Force CPU to avoid GPU precision issues with large loss
    model = SimpleTFT(
        feature_dim=dataset[0][0].shape[-1],
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.layers,
        ff=cfg.ff,
        dropout=cfg.dropout,
    ).to(device)

    ckpt_path = None
    if save_dir is not None:
        ckpt_path = _tft_ckpt_path(Path(save_dir), cfg)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and not fresh and ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                model.eval()
                print(f"[TFT] 检测到已保存模型，直接加载 {ckpt_path}")
                return model, cfg
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"[TFT] 模型特征维度不匹配，将重新训练")
                else:
                    print(f"[TFT] 加载已保存模型失败 ({e})，将重新训练")
            except Exception as e:
                print(f"[TFT] 加载已保存模型失败，将重新训练: {e}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, eps: float = 1e-7) -> torch.Tensor:
        # Clamp logits to prevent overflow in softmax
        logits = torch.clamp(logits, min=-50, max=50)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        # Clamp probabilities to prevent log(0)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=eps, max=1-eps)
        log_pt = torch.log(pt)
        loss = -((1 - pt) ** gamma) * log_pt
        return loss.mean()
    mse = nn.MSELoss()

    model.train()
    prev_loss = None
    deltas = []
    ep = 0
    # 基于近5轮 loss 差分的滑动均值收敛：avg(dl[-5:]) < 0.04 且已跑满 cfg.epochs
    while True:
        ep += 1
        total_loss = 0.0
        steps = 0
        for x, y, combo in loader:
            x, y, combo = x.to(device), y.to(device), combo.to(device)
            opt.zero_grad()
            reds, blue, aux = model(x, combo_ids=combo)
            loss = focal_loss(blue, y[:, -1] - 1)
            for i, head_out in enumerate(reds):
                loss = loss + focal_loss(head_out, y[:, i] - 1)
            sum_target = y[:, :7].sum(dim=1).float() / (33 * 6 + 16)
            odd_target = (y[:, :6] % 2).sum(dim=1)
            span_target = (y[:, :6].max(dim=1).values - y[:, :6].min(dim=1).values).float() / 33.0
            loss = (
                loss
                + cfg.aux_sum_weight * mse(aux["sum"].squeeze(-1), sum_target)
                + cfg.aux_odd_weight * focal_loss(aux["odd"], odd_target)
                + cfg.aux_span_weight * mse(aux["span"].squeeze(-1), span_target)
            )
            # Skip NaN loss to prevent training collapse
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            # Also clip weights to prevent explosion
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(-10, 10)
            total_loss += loss.item()
            steps += 1
        if steps == 0:
            print(f"[TFT] epoch {ep}, WARNING: all batches skipped due to NaN!")
            # Don't update prev_loss or deltas, just continue
            if ep > cfg.epochs * 2:  # Safety limit
                print("[TFT] Too many epochs with no progress, stopping")
                break
            continue
        avg_loss = total_loss / steps
        if prev_loss is not None:
            delta = abs(avg_loss - prev_loss)
            deltas.append(delta)
            if len(deltas) >= 5:
                mean_delta = sum(deltas[-5:]) / 5
                print(f"[TFT] epoch {ep}, loss={avg_loss:.4f}, mean_dl5={mean_delta:.4f}")
                if mean_delta < 0.04 and ep >= cfg.epochs:
                    print(f"[TFT] Early stop at epoch {ep}, mean_dl5={mean_delta:.4f}")
                    break
            else:
                print(f"[TFT] epoch {ep}, loss={avg_loss:.4f}, dl={delta:.4f}")
        else:
            print(f"[TFT] epoch {ep}, loss={avg_loss:.4f}")
        prev_loss = avg_loss

        if ep >= cfg.epochs:
            print(f"[TFT] Reached max epochs {cfg.epochs}")
            break
    model.eval()
    if ckpt_path is not None:
        torch.save(model.state_dict(), ckpt_path)
        print(f"[TFT] 模型已保存到 {ckpt_path}")
    return model, cfg


def predict_tft(model: SimpleTFT, cfg: TFTConfig, df: pd.DataFrame) -> Dict:
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    if len(data) < cfg.window:
        raise ValueError("样本不足，无法预测")
    feats = _build_feature_matrix(df, freq_window=cfg.freq_window, entropy_window=cfg.entropy_window)
    seq = feats[-cfg.window :]
    combo_id = combo_hash(data[-1, :6])
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    combo = torch.tensor([combo_id], dtype=torch.long)
    device = next(model.parameters()).device
    x, combo = x.to(device), combo.to(device)
    with torch.no_grad():
        reds, blue, _ = model(x, combo_ids=combo)
    red_preds = {}
    for i, head_out in enumerate(reds):
        probs = torch.softmax(head_out[0], dim=0).cpu().numpy()
        top_idx = np.argsort(probs)[::-1][: cfg.topk]
        red_preds[i + 1] = [(int(idx + 1), float(probs[idx])) for idx in top_idx]
    probs_b = torch.softmax(blue[0], dim=0).cpu().numpy()
    top_idx_b = np.argsort(probs_b)[::-1][: cfg.topk]
    blue_preds = [(int(idx + 1), float(probs_b[idx])) for idx in top_idx_b]
    return {"red": red_preds, "blue": blue_preds}


def batch_predict_tft(model: SimpleTFT, cfg: TFTConfig, df: pd.DataFrame) -> Dict[int, Dict]:
    """
    批量预测整个数据集（从 window 期开始）。
    返回：{row_idx: result_dict}
    优化：特征构建只做一次。
    """
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    window = cfg.window
    if len(data) <= window:
        return {}

    # 1. 一次性构建所有特征
    # feats_arr[i] features for row i
    # Prediction for target i uses data[i-window:i] and feats[i-window:i]
    # Actually wait. `_build_feature_matrix` is used as:
    # hist = feats[i - window : i]
    # Yes.
    all_feats = _build_feature_matrix(df, freq_window=cfg.freq_window, entropy_window=cfg.entropy_window)
    
    results = {}
    model.eval()
    device = next(model.parameters()).device
    
    indices = range(window, len(data))
    # Batch size usually larger for inference
    batch_size = cfg.batch * 2
    
    import torch
    
    def process_batch(idxs):
        batch_x = []
        batch_combo = []
        
        for i in idxs:
            # hist features for target i => [i-window : i]
            hist = all_feats[i - window : i]
            batch_x.append(hist)
            
            # combo based on LAST drawn red balls: seq[-1, :6]
            # seq is data[i-window:i]. Last one is data[i-1]
            last_reds = data[i-1, :6]
            batch_combo.append(combo_hash(last_reds))
            
        x_t = torch.tensor(np.stack(batch_x), dtype=torch.float32).to(device)
        c_t = torch.tensor(batch_combo, dtype=torch.long).to(device)
        
        with torch.no_grad():
            reds, blue, _ = model(x_t, combo_ids=c_t)
            
        probs_r_list = [torch.softmax(r, dim=1).cpu().numpy() for r in reds]
        probs_b = torch.softmax(blue, dim=1).cpu().numpy()
        
        for k, real_idx in enumerate(idxs):
            row_res = {"red": {}, "blue": []}
            # Red
            for p in range(6):
                probs = probs_r_list[p][k]
                top_idx = np.argsort(probs)[::-1][:cfg.topk]
                row_res["red"][p + 1] = [(int(idx + 1), float(probs[idx])) for idx in top_idx]
            # Blue
            b_p = probs_b[k]
            top_b = np.argsort(b_p)[::-1][:cfg.topk]
            row_res["blue"] = [(int(idx + 1), float(b_p[idx])) for idx in top_b]
            
            results[real_idx] = row_res

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        process_batch(chunk)
        
    return results


def backtest_tft_model(
    model: SimpleTFT,
    cfg: TFTConfig,
    df: pd.DataFrame,
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    大规模回测：使用 StreamingTFTDataset 流式遍历样本，计算 Top1 命中。
    """
    dataset = StreamingTFTDataset(df, window=cfg.window, freq_window=cfg.freq_window, entropy_window=cfg.entropy_window)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_tft)
    device = next(model.parameters()).device
    model.eval()
    red_hits = []
    blue_hits = []
    with torch.no_grad():
        for x, y, combo in loader:
            x, y, combo = x.to(device), y.to(device), combo.to(device)
            reds, blue, _ = model(x, combo_ids=combo)
            probs_b = torch.softmax(blue, dim=1)
            pred_b = torch.argmax(probs_b, dim=1) + 1
            blue_true = y[:, -1]
            blue_hits.extend((pred_b == blue_true).float().cpu().tolist())
            for i, head_out in enumerate(reds):
                probs_r = torch.softmax(head_out, dim=1)
                pred_r = torch.argmax(probs_r, dim=1) + 1
                red_true = y[:, i]
                red_hits.extend((pred_r == red_true).float().cpu().tolist())
    red_hit = float(np.mean(red_hits)) if red_hits else 0.0
    blue_hit = float(np.mean(blue_hits)) if blue_hits else 0.0
    blue_hit = float(np.mean(blue_hits)) if blue_hits else 0.0
    return {"red_top1": red_hit, "blue_top1": blue_hit, "samples": len(dataset)}


class TftModelAdapter(ModelAdapter):
    def train_step(self, member: Member, dataset: pd.DataFrame, steps: int) -> Tuple[Any, float]:
        cfg = member.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Init or load model
        if member.model_state is None:
             # Need to infer feature_dim from dataset?
             # Construct dataset just to check dim?
             temp_ds = TFTDataset(dataset, window=cfg.window, freq_window=cfg.freq_window, entropy_window=cfg.entropy_window)
             feat_dim = temp_ds[0][0].shape[-1]
             model = SimpleTFT(
                feature_dim=feat_dim,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_layers=cfg.layers,
                ff=cfg.ff,
                dropout=cfg.dropout,
            ).to(device)
        else:
            model = member.model_state.to(device)
            
        ds = TFTDataset(dataset, window=cfg.window, freq_window=cfg.freq_window, entropy_window=cfg.entropy_window)
        loader = DataLoader(ds, batch_size=cfg.batch, shuffle=True, collate_fn=collate_tft)
        opt = Adam(model.parameters(), lr=cfg.lr)
        mse = nn.MSELoss()
        
        def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
            log_probs = torch.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
            log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            loss = -((1 - pt) ** gamma) * log_pt
            return loss.mean()

        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for _ in range(steps):
             for x, y, combo in loader:
                x, y, combo = x.to(device), y.to(device), combo.to(device)
                opt.zero_grad()
                reds, blue, aux = model(x, combo_ids=combo)
                
                loss = focal_loss(blue, y[:, -1] - 1)
                for i, head_out in enumerate(reds):
                    loss = loss + focal_loss(head_out, y[:, i] - 1)
                
                sum_target = y[:, :7].sum(dim=1).float() / (33 * 6 + 16)
                odd_target = (y[:, :6] % 2).sum(dim=1)
                span_target = (y[:, :6].max(dim=1).values - y[:, :6].min(dim=1).values).float() / 33.0
                
                loss = (
                    loss
                    + cfg.aux_sum_weight * mse(aux["sum"].squeeze(-1), sum_target)
                    + cfg.aux_odd_weight * focal_loss(aux["odd"], odd_target)
                    + cfg.aux_span_weight * mse(aux["span"].squeeze(-1), span_target)
                )

                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
                
        avg_loss = total_loss / max(1, n_batches)
        return model.cpu(), avg_loss

    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        cfg = member.config
        model = member.model_state
        if model is None: return 0.0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        val_size = 40
        if len(dataset) > val_size + cfg.window:
             val_df = dataset.iloc[-val_size - cfg.window:]
        else:
             val_df = dataset # fallback
             
        res = backtest_tft_model(model, cfg, val_df, batch_size=cfg.batch)
        score = (res["red_top1"] + res["blue_top1"]) / 2
        model.cpu()
        return score

    def perturb_config(self, config: TFTConfig) -> TFTConfig:
        new_cfg = copy.deepcopy(config)
        factors = [0.8, 1.2]
        if random.random() < 0.3:
            new_cfg.lr *= random.choice(factors)
        if random.random() < 0.3:
            new_cfg.dropout = max(0.01, min(0.5, new_cfg.dropout * random.choice([0.9, 1.1])))
        if random.random() < 0.3:
             new_cfg.ff = int(new_cfg.ff * random.choice(factors))
        return new_cfg

    def save(self, member: Member, path: Path) -> None:
        if member.model_state:
            torch.save(member.model_state.state_dict(), path)

    def load(self, path: Path) -> Member:
        pass


class TFTPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = TFTConfig(
            window=config.window,
            batch=config.batch_size,
            epochs=config.epochs,
            lr=config.lr,
            d_model=config.d_model,
            nhead=config.nhead,
            layers=config.layers,
            ff=config.ff,
            dropout=config.dropout,
            freq_window=config.freq_window,
            entropy_window=config.entropy_window,
            resume=getattr(config, 'resume', True),
            fresh=getattr(config, 'fresh', False),
        )
        self.model = None

    def train(self, df: pd.DataFrame) -> None:
        logger.info(f"Training TFT (window={self.cfg.window})...")
        self.model, self.cfg = train_tft(
            df,
            self.cfg,
            save_dir="models/TFT",
            resume=self.cfg.resume,
            fresh=self.cfg.fresh
        )

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("TFT model not trained or loaded")
        # Ensure model is on correct device
        return predict_tft(self.model, self.cfg, df)
    
    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        ckpt_path = _tft_ckpt_path(path, self.cfg)
        if self.model:
            torch.save(self.model.state_dict(), ckpt_path)
            logger.info(f"TFT model saved to {ckpt_path}")

    def load(self, save_dir: str) -> bool:
        path = Path(save_dir)
        if not path.exists(): return False
        
        ckpt_path = _tft_ckpt_path(path, self.cfg)
        if not ckpt_path.exists(): return False
        try:
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             self.model = SimpleTFT(
                d_model=self.cfg.d_model,
                nhead=self.cfg.nhead,
                num_layers=self.cfg.layers,
                dim_feedforward=self.cfg.ff,
                dropout=self.cfg.dropout,
             ).to(device)
             
             state = torch.load(ckpt_path, map_location=device, weights_only=True)
             self.model.load_state_dict(state)
             self.model.eval()
             logger.info(f"TFT model loaded from {ckpt_path}")
             return True
        except Exception as e:
            logger.error(f"Failed to load TFT model: {e}")
            return False


