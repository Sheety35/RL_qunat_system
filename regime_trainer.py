"""
Phase 1 + Phase 2: LSTM Regime Classifier.

Phase 1 — Train:
  Supervised LSTM classifier on NIFTY 50 (2015–2022 train, 2023–2024 val).
  Target accuracy: > 70% on validation set.
  Saves: models/regime_classifier/best_model.pt  +  scaler.pkl

Phase 2 — Enrich:
  Runs inference on both NIFTY 50 and NIFTY BANK features parquets, stamps
  the regime and regime_max_prob columns (with 1-bar lag to prevent leakage),
  and re-saves the parquets.
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

FEATURES_DIR = Path("features")
MODELS_DIR   = Path("models") / "regime_classifier"

# ── LSTM input features (same 10 used by the training plan) ──────────────────
LSTM_FEATURES = [
    "close_return",        # 1-bar return
    "ema_spread_pct",      # (ema9-ema21)/close
    "price_vs_sma1500",    # distance from 20-day SMA
    "rsi14",               # RSI (0-100 raw; scaler normalises)
    "macd_pct",            # MACD / close
    "atr14_pct",           # ATR / close
    "bb_width",            # Bollinger band width
    "price_vs_vwap",       # distance from VWAP
    "sector_breadth_norm", # 0–1 breadth
    "volume_ratio",        # volume / 20-bar avg
]

SEQ_LEN     = 50   # 50 bars ≈ 4 h of 5-min data
BATCH_SIZE  = 512
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
LR          = 1e-3
MAX_EPOCHS  = 50
PATIENCE    = 7    # early stopping patience


# ── Model ─────────────────────────────────────────────────────────────────────

class RegimeLSTM(nn.Module):
    """LSTM Regime Classifier with softmax head (3 classes: Bear/Flat/Bull)."""

    def __init__(self, input_size: int = len(LSTM_FEATURES),
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # logits over last timestep


# ── Dataset ───────────────────────────────────────────────────────────────────

class RegimeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Label creation ────────────────────────────────────────────────────────────

def create_regime_labels(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based seed labels:
      ADX > 30 AND close > 20d SMA → Bull  (2)
      ADX > 30 AND close < 20d SMA → Bear  (0)
      ADX < 20                     → Flat  (1)
      20 ≤ ADX ≤ 30                → Flat  (1)  (ambiguous zone)
    """
    adx      = df["adx14"]
    vs_sma   = df.get("price_vs_sma1500", df["price_vs_sma20"])
    bull_cond = (adx > 30) & (vs_sma > 0)
    bear_cond = (adx > 30) & (vs_sma <= 0)
    labels    = pd.Series(1, index=df.index)   # default: Flat
    labels[bull_cond] = 2
    labels[bear_cond] = 0
    return labels


def make_sequences(df: pd.DataFrame, labels: pd.Series,
                   seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sequences. Each sample:
      X[i] = df[i:i+seq_len]  (seq_len × n_features)
      y[i] = labels[i+seq_len-1]   (regime at the END of the window)
    """
    feat_cols = [c for c in LSTM_FEATURES if c in df.columns]
    vals  = df[feat_cols].values.astype(np.float32)
    lbls  = labels.values.astype(np.int64)

    n  = len(vals) - seq_len
    X  = np.lib.stride_tricks.sliding_window_view(vals, (seq_len, len(feat_cols)))[: ,0, :, :]
    # shape: (n, seq_len, n_features)
    y  = lbls[seq_len:]
    return X[:n], y[:n]


# ── Training loop ─────────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(1) == targets).float().mean().item()


def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(y)
        total_acc  += _accuracy(logits, y) * len(y)
        n          += len(y)
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        total_acc  += _accuracy(logits, y) * len(y)
        n          += len(y)
    return total_loss / n, total_acc / n


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_regime(model: RegimeLSTM, df: pd.DataFrame,
                   scaler: StandardScaler,
                   device: torch.device,
                   seq_len: int = SEQ_LEN,
                   batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (regime_classes, max_probs) aligned to df index.
    First seq_len-1 bars will be NaN / 1 (Flat placeholder).
    """
    feat_cols   = [c for c in LSTM_FEATURES if c in df.columns]
    vals        = df[feat_cols].values.astype(np.float32)
    vals_scaled = scaler.transform(vals)

    n      = len(vals) - seq_len
    X_all  = np.lib.stride_tricks.sliding_window_view(
                 vals_scaled, (seq_len, len(feat_cols)))[: ,0, :, :]

    classes   = np.ones(len(df), dtype=np.int64)      # default Flat
    max_probs = np.full(len(df), 1/3, dtype=np.float32)

    model.eval()
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        Xb   = torch.tensor(X_all[start:end], dtype=torch.float32).to(device)
        lg   = model(Xb)
        prob = torch.softmax(lg, dim=1).cpu().numpy()
        classes[seq_len + start : seq_len + end]   = prob.argmax(1)
        max_probs[seq_len + start : seq_len + end] = prob.max(1)

    return classes, max_probs


# ── Main training function ────────────────────────────────────────────────────

def run_regime_training() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load features ──────────────────────────────────────────────────────
    parquet = FEATURES_DIR / "NIFTY_50_features.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} not found — run Phase 0 (feature_engineering.py) first.")

    print(f"Loading {parquet}...")
    df = pd.read_parquet(parquet)

    # ── Create labels ──────────────────────────────────────────────────────
    labels = create_regime_labels(df)
    label_counts = labels.value_counts().sort_index()
    print(f"Label distribution: Bear={label_counts.get(0,0):,}  "
          f"Flat={label_counts.get(1,0):,}  Bull={label_counts.get(2,0):,}")

    # ── Train / val split ──────────────────────────────────────────────────
    train_mask = df.index < "2023-01-01"
    val_mask   = (df.index >= "2023-01-01") & (df.index < "2025-01-01")

    train_df  = df[train_mask].copy()
    val_df    = df[val_mask].copy()
    train_lbl = labels[train_mask]
    val_lbl   = labels[val_mask]
    print(f"Train: {len(train_df):,} bars  |  Val: {len(val_df):,} bars")

    # ── Fill NaNs in LSTM features with forward-fill then 0 ───────────────
    feat_cols = [c for c in LSTM_FEATURES if c in df.columns]
    missing = [c for c in LSTM_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: missing LSTM features: {missing}")

    for d in (train_df, val_df):
        d[feat_cols] = d[feat_cols].ffill().bfill().fillna(0)

    # ── Fit scaler on training data ────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols].values)
    train_df[feat_cols] = scaler.transform(train_df[feat_cols].values)
    val_df[feat_cols]   = scaler.transform(val_df[feat_cols].values)

    # ── Build sequences ────────────────────────────────────────────────────
    print("Building training sequences...")
    X_train, y_train = make_sequences(train_df, train_lbl)
    print("Building validation sequences...")
    X_val, y_val     = make_sequences(val_df,   val_lbl)
    print(f"Train sequences: {len(y_train):,}  |  Val sequences: {len(y_val):,}")

    train_ds = RegimeDataset(X_train, y_train)
    val_ds   = RegimeDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # ── Build model ────────────────────────────────────────────────────────
    model     = RegimeLSTM(input_size=len(feat_cols)).to(device)
    criterion = nn.CrossEntropyLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_ckpt     = MODELS_DIR / "best_model.pt"

    print("\nEpoch  Train-Loss  Train-Acc  Val-Loss  Val-Acc")
    print("-" * 52)
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step(vl_loss)

        print(f"{epoch:5d}  {tr_loss:.4f}      {tr_acc:.3f}      "
              f"{vl_loss:.4f}    {vl_acc:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr  = 0
            torch.save({
                "model_state": model.state_dict(),
                "val_loss": vl_loss,
                "val_acc": vl_acc,
                "epoch": epoch,
                "feat_cols": feat_cols,
                "seq_len": SEQ_LEN,
            }, best_ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # ── Save scaler ────────────────────────────────────────────────────────
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved model → {best_ckpt}")
    print(f"Saved scaler → {scaler_path}")
    print(f"Best val accuracy: {best_val_loss:.4f} loss")

    # ── Phase 2: enrich both parquets with regime predictions ─────────────
    run_regime_enrichment()


def run_regime_enrichment() -> None:
    """
    Phase 2: load trained LSTM, stamp regime labels onto both parquets
    with a 1-bar lag (prevents lookahead leakage).
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = MODELS_DIR / "best_model.pt"
    scl_path  = MODELS_DIR / "scaler.pkl"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — train regime classifier first.")

    print("\nLoading regime model for enrichment...")
    ckpt    = torch.load(ckpt_path, map_location=device)
    model   = RegimeLSTM(input_size=len(ckpt["feat_cols"])).to(device)
    model.load_state_dict(ckpt["model_state"])

    with open(scl_path, "rb") as f:
        scaler = pickle.load(f)

    seq_len   = ckpt.get("seq_len", SEQ_LEN)
    feat_cols = ckpt["feat_cols"]

    for parquet_path in [
        FEATURES_DIR / "NIFTY_50_features.parquet",
        FEATURES_DIR / "NIFTY_BANK_features.parquet",
    ]:
        if not parquet_path.exists():
            print(f"  Skipping {parquet_path} (not found)")
            continue

        print(f"Enriching {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        # Fill missing LSTM features with 0
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0)

        classes, max_probs = predict_regime(model, df, scaler, device,
                                            seq_len=seq_len)

        # 1-bar lag to prevent leakage (regime at bar t uses data up to t-1)
        df["regime"]          = pd.Series(classes,   index=df.index).shift(1).fillna(1).astype(int)
        df["regime_max_prob"] = pd.Series(max_probs, index=df.index).shift(1).fillna(1/3)

        df.to_parquet(parquet_path)
        print(f"  Saved {parquet_path}  ({len(df):,} bars)")

    print("Phase 2 (regime enrichment) complete.")


if __name__ == "__main__":
    run_regime_training()
