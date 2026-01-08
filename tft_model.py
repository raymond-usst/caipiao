from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0]
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _build_feature_matrix(df: pd.DataFrame, freq_window: int = 50, entropy_window: int = 50) -> np.ndarray:
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    n = len(data)
    feat_list = []
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
        for i in range(window, len(data)):
            hist = feats[i - window : i]
            tgt = data[i]
            self.samples.append((hist, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        hist, tgt = self.samples[idx]
        return torch.tensor(hist, dtype=torch.float32), torch.tensor(tgt, dtype=torch.long)


def collate_tft(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


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
        self.head_red = nn.ModuleList([nn.Linear(d_model, 33) for _ in range(6)])
        self.head_blue = nn.Linear(d_model, 16)
        # 多任务辅助头：和值回归、奇偶计数分类、跨度回归
        self.head_sum = nn.Linear(d_model, 1)
        self.head_odd = nn.Linear(d_model, 7)  # 0-6 奇数个数
        self.head_span = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, L, F]
        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.var_sel(h)
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)  # [B, d]
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
    n_iter: int = 6,
    random_state: int = 42,
):
    """
    使用 skopt 对 TFT 做轻量贝叶斯超参搜索，目标为短周期loss。
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real, Categorical
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行 TFT 贝叶斯调参") from e

    df_recent = df if recent <= 0 else df.tail(max(recent, 120))
    if len(df_recent) < 80:
        raise ValueError("样本不足，无法进行 TFT 贝叶斯调参")

    space = [
        Integer(8, 20, name="window"),
        Integer(64, 192, name="d_model"),
        Categorical([2, 4], name="nhead"),
        Integer(2, 4, name="layers"),
        Integer(128, 320, name="ff"),
        Real(0.0, 0.3, name="dropout"),
        Real(1e-4, 3e-3, prior="log-uniform", name="lr"),
        Categorical([32, 64], name="batch"),
    ]

    def objective(params):
        w, dm, nh, ly, ff, dr, lr, bs = params
        cfg = TFTConfig(
            window=w,
            batch=bs,
            epochs=6,
            lr=lr,
            d_model=dm,
            nhead=nh,
            layers=ly,
            ff=ff,
            dropout=dr,
            topk=1,
            freq_window=50,
            entropy_window=50,
        )
        try:
            loss = _quick_tft_loss(df_recent, cfg, max_epochs=4)
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
        "window": best[0],
        "d_model": best[1],
        "nhead": best[2],
        "layers": best[3],
        "ff": best[4],
        "dropout": best[5],
        "lr": best[6],
        "batch": best[7],
    }
    return best_params, best_loss


def train_tft(
    df: pd.DataFrame,
    cfg: TFTConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    dataset = TFTDataset(df, window=cfg.window, freq_window=cfg.freq_window, entropy_window=cfg.entropy_window)
    if len(dataset) < 10:
        raise ValueError("样本不足，无法训练 TFT")
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

    ckpt_path = None
    if save_dir is not None:
        ckpt_path = _tft_ckpt_path(Path(save_dir), cfg)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and not fresh and ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
                model.eval()
                print(f"[TFT] 检测到已保存模型，直接加载 {ckpt_path}")
                return model, cfg
            except Exception as e:
                print(f"[TFT] 加载已保存模型失败，将重新训练: {e}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
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
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            reds, blue, aux = model(x)
            loss = loss_fn(blue, y[:, -1] - 1)
            for i, head_out in enumerate(reds):
                loss = loss + loss_fn(head_out, y[:, i] - 1)
            sum_target = y[:, :7].sum(dim=1).float() / (33 * 6 + 16)
            odd_target = (y[:, :6] % 2).sum(dim=1)
            span_target = (y[:, :6].max(dim=1).values - y[:, :6].min(dim=1).values).float() / 33.0
            loss = (
                loss
                + cfg.aux_sum_weight * mse(aux["sum"].squeeze(-1), sum_target)
                + cfg.aux_odd_weight * loss_fn(aux["odd"], odd_target)
                + cfg.aux_span_weight * mse(aux["span"].squeeze(-1), span_target)
            )
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(1, steps)
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
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        reds, blue, _ = model(x)
    red_preds = {}
    for i, head_out in enumerate(reds):
        probs = torch.softmax(head_out[0], dim=0).cpu().numpy()
        top_idx = np.argsort(probs)[::-1][: cfg.topk]
        red_preds[i + 1] = [(int(idx + 1), float(probs[idx])) for idx in top_idx]
    probs_b = torch.softmax(blue[0], dim=0).cpu().numpy()
    top_idx_b = np.argsort(probs_b)[::-1][: cfg.topk]
    blue_preds = [(int(idx + 1), float(probs_b[idx])) for idx in top_idx_b]
    return {"red": red_preds, "blue": blue_preds}

