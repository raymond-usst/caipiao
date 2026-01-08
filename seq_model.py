from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import Adam
from .features import build_features
from .features import build_features


PAD_ID = 0
RED_OFFSET = 0
BLUE_OFFSET = 33  # reds: 1-33, blues will be offset to 34-49
VOCAB_SIZE = 33 + 16 + 1  # +1 for padding
COMBO_VOCAB = 20000  # 哈希桶，用于红球组合 embedding
FEAT_DIM = 14  # 手工特征维度


def encode_draw(row: np.ndarray) -> List[int]:
    # row: [red1..red6, blue]
    reds = row[:6].tolist()
    blue = int(row[6]) + BLUE_OFFSET
    return reds + [blue]


def combo_hash(reds: np.ndarray) -> int:
    """对排序后的红球组合做哈希，映射到固定桶."""
    reds_sorted = tuple(sorted(int(x) for x in reds))
    return hash(reds_sorted) % COMBO_VOCAB


class DrawDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int = 20):
        cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
        data = df[cols].to_numpy(dtype=int)
        feats_arr = build_features(df).to_numpy(dtype=float)
        self.window = window
        self.samples = []
        for i in range(window, len(data)):
            hist = data[i - window : i]
            target = data[i]
            src_tokens = [t for draw in hist for t in encode_draw(draw)]
            tgt_tokens = encode_draw(target)
            combo_id = combo_hash(target[:6])
            feat_vec = feats_arr[i]
            self.samples.append((src_tokens, tgt_tokens, combo_id, feat_vec))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, combo_id, feat_vec = self.samples[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            torch.tensor(combo_id, dtype=torch.long),
            torch.tensor(feat_vec, dtype=torch.float32),
        )


def collate_batch(batch):
    src_list, tgt_list, combo_list, feat_list = zip(*batch)
    max_len = max(len(x) for x in src_list)
    src_pad = []
    for src in src_list:
        pad_len = max_len - len(src)
        if pad_len > 0:
            src = torch.cat([src, torch.zeros(pad_len, dtype=torch.long)])
        src_pad.append(src)
    src_pad = torch.stack(src_pad)
    tgt = torch.stack(tgt_list)  # shape [B, 7]
    combo = torch.stack(combo_list)
    feats = torch.stack(feat_list)
    return src_pad, tgt, combo, feats


class StreamingBacktestDataset(IterableDataset):
    """
    适用于大规模回测的流式数据集，避免一次性构造所有样本。
    """

    def __init__(self, df: pd.DataFrame, window: int = 20):
        self.df = df
        self.window = window

    def __iter__(self):
        cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
        data = self.df[cols].to_numpy(dtype=int)
        feats_arr = build_features(self.df).to_numpy(dtype=float)
        for i in range(self.window, len(data)):
            hist = data[i - self.window : i]
            target = data[i]
            src_tokens = [t for draw in hist for t in encode_draw(draw)]
            tgt_tokens = encode_draw(target)
            combo_id = combo_hash(target[:6])
            feat_vec = feats_arr[i]
            yield (
                torch.tensor(src_tokens, dtype=torch.long),
                torch.tensor(tgt_tokens, dtype=torch.long),
                torch.tensor(combo_id, dtype=torch.long),
                torch.tensor(feat_vec, dtype=torch.float32),
            )

    def __len__(self):
        return max(0, len(self.df) - self.window)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SeqModel(nn.Module):
    def __init__(self, d_model: int = 96, nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 192, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.combo_emb = nn.Embedding(COMBO_VOCAB, d_model)
        self.feat_proj = nn.Linear(FEAT_DIM, d_model)
        self.head_red = nn.ModuleList([nn.Linear(d_model, 33) for _ in range(6)])
        self.head_blue = nn.Linear(d_model, 16)

    def forward(self, src: torch.Tensor, combo_ids: torch.Tensor | None = None, feats: torch.Tensor | None = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.embed(src)
        x = self.pos(x)
        h = self.encoder(x)  # [B, L, d]
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)  # [B, d]
        if combo_ids is not None:
            pooled = pooled + self.combo_emb(combo_ids)
        if feats is not None:
            pooled = pooled + self.feat_proj(feats)
        reds = [head(pooled) for head in self.head_red]
        blue = self.head_blue(pooled)
        return reds, blue


@dataclass
class TrainConfig:
    window: int = 20
    batch_size: int = 64
    epochs: int = 30
    patience: int = 8
    min_delta: float = 1e-3
    lr: float = 1e-3
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 3
    ff: int = 192
    dropout: float = 0.1
    topk: int = 3


def _seq_ckpt_path(save_dir: Path, cfg: TrainConfig) -> Path:
    base = f"seq_w{cfg.window}_dm{cfg.d_model}_h{cfg.nhead}_l{cfg.num_layers}_ff{cfg.ff}_dr{cfg.dropout}_lr{cfg.lr}"
    return save_dir / f"{base}.pt"


def _focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float | None = None) -> torch.Tensor:
    """
    简单 Focal Loss (multiclass): alpha 可为标量，gamma 默认 2.0
    """
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1 - pt) ** gamma) * log_pt
    if alpha is not None:
        loss = alpha * loss
    return loss.mean()


def _quick_seq_loss(df: pd.DataFrame, cfg: TrainConfig, max_epochs: int = 6) -> float:
    """
    用于贝叶斯优化的快速评估：短周期训练+返回最后一个epoch的平均loss。
    不保存模型，主要追求相对指标。
    """
    dataset = DrawDataset(df, window=cfg.window)
    if len(dataset) < 10:
        raise ValueError("样本不足，无法评估")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqModel(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.ff,
        dropout=cfg.dropout,
    ).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)
    model.train()
    last_loss = 1e3
    for _ in range(max_epochs):
        total = 0.0
        steps = 0
        for src, tgt, combo, feats in loader:
            src, tgt, combo, feats = src.to(device), tgt.to(device), combo.to(device), feats.to(device)
            opt.zero_grad()
            reds, blue = model(src, combo_ids=combo, feats=feats)
            loss = _focal_loss(blue, tgt[:, -1] - BLUE_OFFSET - 1)
            for i, head_out in enumerate(reds):
                loss = loss + _focal_loss(head_out, tgt[:, i] - 1)
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        last_loss = total / max(1, steps)
    return float(last_loss)


def bayes_optimize_seq(
    df: pd.DataFrame,
    recent: int = 400,
    n_iter: int = 8,
    random_state: int = 42,
):
    """
    使用 skopt 对 Transformer 序列模型做快速贝叶斯超参搜索。
    评估指标：短周期训练后的平均loss（越低越好）。
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real, Categorical
    except Exception as e:
        raise ImportError("需要安装 scikit-optimize 以运行 Transformer 贝叶斯调参") from e

    df_recent = df if recent <= 0 else df.tail(max(recent, 80))
    if len(df_recent) < 60:
        raise ValueError("样本不足，无法进行 Transformer 贝叶斯调参")

    space = [
        Integer(8, 16, name="window"),
        Integer(64, 160, name="d_model"),
        Categorical([2, 4, 6], name="nhead"),
        Integer(2, 4, name="num_layers"),
        Integer(128, 320, name="ff"),
        Real(0.0, 0.3, name="dropout"),
        Real(1e-4, 3e-3, prior="log-uniform", name="lr"),
        Categorical([32, 64], name="batch_size"),
    ]

    def objective(params):
        w, dm, nh, nl, ff, dr, lr, bs = params
        cfg = TrainConfig(
            window=w,
            batch_size=bs,
            epochs=6,
            lr=lr,
            d_model=dm,
            nhead=nh,
            num_layers=nl,
            ff=ff,
            dropout=dr,
            topk=1,
        )
        try:
            loss = _quick_seq_loss(df_recent, cfg, max_epochs=6)
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
        "num_layers": best[3],
        "ff": best[4],
        "dropout": best[5],
        "lr": best[6],
        "batch_size": best[7],
    }
    return best_params, best_loss


def train_seq_model(
    df: pd.DataFrame,
    cfg: TrainConfig,
    save_dir: str | None = "models",
    resume: bool = True,
    fresh: bool = False,
):
    dataset = DrawDataset(df, window=cfg.window)
    if len(dataset) < 10:
        raise ValueError("样本不足，无法训练序列模型")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqModel(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.ff,
        dropout=cfg.dropout,
    ).to(device)

    ckpt_path = None
    if save_dir is not None:
        ckpt_path = _seq_ckpt_path(Path(save_dir), cfg)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and not fresh and ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
                model.eval()
                print(f"[Transformer] 检测到已保存模型，直接加载 {ckpt_path}")
                return model, cfg
            except Exception as e:
                print(f"[Transformer] 加载已保存模型失败，将重新训练: {e}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    prev_loss = None
    deltas = []
    ep = 0
    # 基于近5轮 loss 差分的滑动均值收敛：avg(dl[-5:]) < 0.004 且已跑满 cfg.epochs
    while True:
        ep += 1
        total_loss = 0.0
        steps = 0
        for src, tgt, combo, feats in loader:
            src, tgt, combo, feats = src.to(device), tgt.to(device), combo.to(device), feats.to(device)
            opt.zero_grad()
            reds, blue = model(src, combo_ids=combo, feats=feats)
            loss = _focal_loss(blue, tgt[:, -1] - BLUE_OFFSET - 1)
            for i, head_out in enumerate(reds):
                loss = loss + _focal_loss(head_out, tgt[:, i] - 1)
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
                print(f"[Transformer] epoch {ep}, loss={avg_loss:.4f}, mean_dl5={mean_delta:.4f}")
                if mean_delta < 0.004 and ep >= cfg.epochs:
                    print(f"[Transformer] Early stop at epoch {ep}, mean_dl5={mean_delta:.4f}")
                    break
            else:
                print(f"[Transformer] epoch {ep}, loss={avg_loss:.4f}, dl={delta:.4f}")
        else:
            print(f"[Transformer] epoch {ep}, loss={avg_loss:.4f}")
        prev_loss = avg_loss
    model.eval()
    if ckpt_path is not None:
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Transformer] 模型已保存到 {ckpt_path}")
    return model, cfg


def predict_seq(model: SeqModel, cfg: TrainConfig, df: pd.DataFrame) -> Dict:
    cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    data = df[cols].to_numpy(dtype=int)
    if len(data) < cfg.window:
        raise ValueError("样本不足，无法预测")
    seq = data[-cfg.window :]
    combo_id = combo_hash(seq[-1, :6])
    tokens = [t for draw in seq for t in encode_draw(draw)]
    feats = build_features(df).iloc[-1].to_numpy(dtype=float)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    combo = torch.tensor([combo_id], dtype=torch.long)
    feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    device = next(model.parameters()).device
    x, combo, feats_t = x.to(device), combo.to(device), feats_t.to(device)
    with torch.no_grad():
        reds, blue = model(x, combo_ids=combo, feats=feats_t)
    red_preds = {}
    for i, head_out in enumerate(reds):
        probs = torch.softmax(head_out[0], dim=0).cpu().numpy()
        top_idx = np.argsort(probs)[::-1][: cfg.topk]
        red_preds[i + 1] = [(int(idx + 1), float(probs[idx])) for idx in top_idx]
    probs_b = torch.softmax(blue[0], dim=0).cpu().numpy()
    top_idx_b = np.argsort(probs_b)[::-1][: cfg.topk]
    blue_preds = [(int(idx + 1), float(probs_b[idx])) for idx in top_idx_b]
    return {"red": red_preds, "blue": blue_preds}


def backtest_seq_model(
    model: SeqModel,
    cfg: TrainConfig,
    df: pd.DataFrame,
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    大规模回测：使用 IterableDataset 流式遍历全部样本，计算 Top1 命中。
    """
    dataset = StreamingBacktestDataset(df, window=cfg.window)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
    device = next(model.parameters()).device
    model.eval()
    red_hits = []
    blue_hits = []
    with torch.no_grad():
        for src, tgt, combo, feats in loader:
            src, tgt, combo, feats = src.to(device), tgt.to(device), combo.to(device), feats.to(device)
            reds, blue = model(src, combo_ids=combo, feats=feats)
            # blue top1
            probs_b = torch.softmax(blue, dim=1)
            pred_b = torch.argmax(probs_b, dim=1) + 1
            blue_true = tgt[:, -1] - BLUE_OFFSET
            blue_hits.extend((pred_b == blue_true).float().cpu().tolist())
            # red top1 per position
            for i, head_out in enumerate(reds):
                probs_r = torch.softmax(head_out, dim=1)
                pred_r = torch.argmax(probs_r, dim=1) + 1
                red_true = tgt[:, i]
                red_hits.extend((pred_r == red_true).float().cpu().tolist())
    red_hit = float(np.mean(red_hits)) if red_hits else 0.0
    blue_hit = float(np.mean(blue_hits)) if blue_hits else 0.0
    return {"red_top1": red_hit, "blue_top1": blue_hit, "samples": len(dataset)}

