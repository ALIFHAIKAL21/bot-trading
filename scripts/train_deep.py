"""High-performance PyTorch training script for Model C on 500,000+ bars using RTX 4050 GPU.

Architecture:
- RevIN (Reversible Instance Normalization) to remove financial distribution shift.
- Dilated 1D Residual CNN (multi-scale temporal receptive fields).
- Bidirectional GRU (2 layers) for sequence state dynamics.
- Multi-Head Self-Attention pooling.
- Multi-task heads: Direction P(long), forward 15m return, and 15m realized volatility.
- Mixed precision (fp16) via torch.amp.autocast for RTX 4050 Tensor Core acceleration.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

from src.models.deep_sequence_model import MultiTaskSequenceNetwork


class BulkSequenceDataset(Dataset):
    """Fast tensor dataset for sliding sequence windows on large financial time series."""

    def __init__(
        self,
        features: np.ndarray,
        labels_dir: np.ndarray,
        labels_ret: np.ndarray,
        labels_vol: np.ndarray,
        seq_len: int = 60,
        stride: int = 2,
    ):
        self.features = features.astype(np.float32)
        self.labels_dir = labels_dir.astype(np.float32)
        self.labels_ret = labels_ret.astype(np.float32)
        self.labels_vol = labels_vol.astype(np.float32)
        self.seq_len = seq_len
        # Valid sample indices respecting seq_len and stride
        self.indices = np.arange(seq_len, len(features) - 1, stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end_idx = self.indices[idx]
        start_idx = end_idx - self.seq_len
        x = self.features[start_idx:end_idx]
        y_dir = self.labels_dir[end_idx]
        y_ret = self.labels_ret[end_idx]
        y_vol = self.labels_vol[end_idx]
        return (
            torch.from_numpy(x),
            torch.tensor(y_dir, dtype=torch.float32),
            torch.tensor(y_ret, dtype=torch.float32),
            torch.tensor(y_vol, dtype=torch.float32),
        )


def compute_bulk_features(df: pd.DataFrame, horizon_bars: int = 15) -> Tuple[pd.DataFrame, List[str]]:
    """Compute high-speed vectorized technical and order-flow alpha indicators on 500k bars."""
    logger.info("Computing vectorized alpha features on bulk dataset...")
    t0 = time.time()
    res = df.copy()

    # 1. Multi-scale log returns
    for w in [1, 3, 5, 15, 30, 60]:
        res[f"ret_{w}"] = np.log(res["close"] / res["close"].shift(w))

    # 2. Parkinson volatility (High-Low)
    log_hl = np.log(res["high"] / res["low"])
    res["parkinson_vol_15"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (log_hl**2).rolling(15).mean())
    res["parkinson_vol_60"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (log_hl**2).rolling(60).mean())

    # 3. Realized volatility (rolling return std)
    res["realized_vol_15"] = res["ret_1"].rolling(15).std() * np.sqrt(15)
    res["realized_vol_60"] = res["ret_1"].rolling(60).std() * np.sqrt(60)

    # 4. RSI (14 bars)
    delta = res["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    res["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))) / 100.0  # normalized to [0, 1]

    # 5. MACD normalized by price
    ema12 = res["close"].ewm(span=12, adjust=False).mean()
    ema26 = res["close"].ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / res["close"]
    signal = macd.ewm(span=9, adjust=False).mean()
    res["macd_diff"] = macd - signal

    # 6. Order Flow & Volume Activity
    if "taker_buy_volume" in res.columns and "volume" in res.columns:
        res["taker_buy_ratio"] = res["taker_buy_volume"] / (res["volume"] + 1e-8)
    else:
        res["taker_buy_ratio"] = 0.5

    vol_ma60 = res["volume"].rolling(60).mean()
    vol_std60 = res["volume"].rolling(60).std()
    res["volume_zscore"] = (res["volume"] - vol_ma60) / (vol_std60 + 1e-8)

    # 7. Cyclical Time Encodings
    hour = res.index.hour
    dow = res.index.dayofweek
    res["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    res["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    res["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    res["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    # 8. Target Labels (strictly causally shifted forward)
    # Forward return over next `horizon_bars`
    fwd_ret = np.log(res["close"].shift(-horizon_bars) / res["close"])
    res["target_fwd_ret"] = fwd_ret
    res["target_binary_long"] = (fwd_ret > 0).astype(float)
    res["target_vol_fwd"] = res["ret_1"].rolling(horizon_bars).std().shift(-horizon_bars) * np.sqrt(horizon_bars)

    feature_cols = [
        "ret_1", "ret_3", "ret_5", "ret_15", "ret_30", "ret_60",
        "parkinson_vol_15", "parkinson_vol_60",
        "realized_vol_15", "realized_vol_60",
        "rsi_14", "macd_diff",
        "taker_buy_ratio", "volume_zscore",
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
    ]

    # Drop initial warm-up bars and terminal unlabeled bars
    clean_df = res.iloc[60:-horizon_bars].copy()
    clean_df[feature_cols] = clean_df[feature_cols].ffill().bfill().clip(-5.0, 5.0)

    logger.success(f"Computed {len(feature_cols)} features on {len(clean_df):,} bars in {time.time() - t0:.2f}s")
    return clean_df, feature_cols


def train_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    seq_len: int = 60,
    stride: int = 2,
    device_name: str = "auto",
    save_path: str = "models_store/model_c_best.pt",
    reports_dir: str = "reports",
) -> Dict:
    """Train MultiTaskSequenceNetwork on CUDA using PyTorch with AMP mixed precision."""
    if torch is None:
        logger.error("PyTorch is not yet installed in .venv. Please wait for PyTorch installation to complete.")
        sys.exit(1)

    # Resolve device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    logger.info(f"Using compute device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        logger.info(f"CUDA Hardware: {gpu_name} ({vram_mb:.0f} MB VRAM)")

    # 1. Purged Walk-Forward Split (80% Train, 20% Val)
    n = len(df)
    train_end = int(n * 0.80)
    embargo = 60  # 1 hour embargo to guarantee zero leakage

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end + embargo :]

    logger.info(f"Dataset Partition: Train = {len(df_train):,} bars | Val = {len(df_val):,} bars (Embargo = {embargo} bars)")

    # 2. Build Datasets
    X_tr = df_train[feature_cols].values
    y_dir_tr = df_train["target_binary_long"].values
    y_ret_tr = df_train["target_fwd_ret"].values
    y_vol_tr = df_train["target_vol_fwd"].fillna(0.01).values

    X_val = df_val[feature_cols].values
    y_dir_val = df_val["target_binary_long"].values
    y_ret_val = df_val["target_fwd_ret"].values
    y_vol_val = df_val["target_vol_fwd"].fillna(0.01).values

    train_ds = BulkSequenceDataset(X_tr, y_dir_tr, y_ret_tr, y_vol_tr, seq_len=seq_len, stride=stride)
    val_ds = BulkSequenceDataset(X_val, y_dir_val, y_ret_val, y_vol_val, seq_len=seq_len, stride=stride * 2)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0
    )

    logger.info(f"Sequences generated: {len(train_ds):,} training sequences | {len(val_ds):,} validation sequences")

    # 3. Initialize Model
    model = MultiTaskSequenceNetwork(
        in_features=len(feature_cols),
        d_model=64,
        n_layers=2,
        dropout=0.2,
    ).to(device)

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    bce_loss = nn.BCELoss()
    huber_loss = nn.HuberLoss(delta=0.01)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_brier": [], "epoch_times": []}

    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting GPU Fine-Tuning for {epochs} epochs (Batch Size={batch_size})...")

    total_start = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0

        for x_b, y_dir_b, y_ret_b, y_vol_b in train_loader:
            x_b = x_b.to(device, non_blocking=True)
            y_dir_b = y_dir_b.to(device, non_blocking=True)
            y_ret_b = y_ret_b.to(device, non_blocking=True)
            y_vol_b = y_vol_b.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                p_dir, p_ret, p_vol = model(x_b)
                l_dir = bce_loss(p_dir, y_dir_b)
                l_ret = huber_loss(p_ret, y_ret_b)
                l_vol = huber_loss(p_vol, y_vol_b)
                # Multi-task balance: 1.0 Direction + 10.0 Return + 5.0 Vol
                loss = l_dir + 10.0 * l_ret + 5.0 * l_vol

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(x_b)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds_dir = []
        val_trues_dir = []

        with torch.no_grad():
            for x_b, y_dir_b, y_ret_b, y_vol_b in val_loader:
                x_b = x_b.to(device, non_blocking=True)
                y_dir_b = y_dir_b.to(device, non_blocking=True)
                y_ret_b = y_ret_b.to(device, non_blocking=True)
                y_vol_b = y_vol_b.to(device, non_blocking=True)

                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    p_dir, p_ret, p_vol = model(x_b)
                    l_dir = bce_loss(p_dir, y_dir_b)
                    l_ret = huber_loss(p_ret, y_ret_b)
                    l_vol = huber_loss(p_vol, y_vol_b)
                    loss = l_dir + 10.0 * l_ret + 5.0 * l_vol

                val_loss += loss.item() * len(x_b)
                val_preds_dir.extend(p_dir.cpu().numpy().tolist())
                val_trues_dir.extend(y_dir_b.cpu().numpy().tolist())

        val_loss /= len(val_ds)
        val_auc = float(roc_auc_score(val_trues_dir, val_preds_dir))
        val_brier = float(brier_score_loss(val_trues_dir, val_preds_dir))
        ep_duration = time.time() - ep_start

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_auc"].append(round(val_auc, 4))
        history["val_brier"].append(round(val_brier, 4))
        history["epoch_times"].append(round(ep_duration, 2))

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | Val Brier: {val_brier:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "feature_cols": feature_cols,
                },
                save_p,
            )
            logger.success(f"-> Checkpoint saved: {save_p} (Best Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.warning(f"Early stopping triggered after {epoch} epochs.")
                break

    total_time = time.time() - total_start
    logger.success(f"Training completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    curves_path = rep_p / "model_c_500k_curves.json"
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved learning curves to {curves_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Deep Sequence Model on 500k dataset using GPU.")
    parser.add_argument("--data-file", type=str, default="data/cache/BTC_USDT_1m_bulk.parquet")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        logger.error(f"Bulk data file not found: {data_path}. Run scripts/download_bulk_data.py first.")
        sys.exit(1)

    logger.info(f"Loading bulk dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} total bars from Parquet.")

    features_cache_path = Path("data/cache/BTC_USDT_1m_features.parquet")
    if features_cache_path.exists():
        logger.info(f"Loading pre-computed features from {features_cache_path}...")
        clean_df = pd.read_parquet(features_cache_path)
        feature_cols = [
            "ret_1", "ret_3", "ret_5", "ret_15", "ret_30", "ret_60",
            "parkinson_vol_15", "parkinson_vol_60",
            "realized_vol_15", "realized_vol_60",
            "rsi_14", "macd_diff",
            "taker_buy_ratio", "volume_zscore",
            "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        ]
    else:
        clean_df, feature_cols = compute_bulk_features(df)
    train_model(
        df=clean_df,
        feature_cols=feature_cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        stride=args.stride,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
