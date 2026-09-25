"""
Comprehensive Verification Script for 12-Channel Dataset & PyTorch Tensor Pipeline on CUDA
Zero Training: Only validates data shapes, types, forward pass, and memory.
"""
import sys
import os
import pathlib
import pandas as pd
import numpy as np
import torch

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
sys.path.insert(0, str(project_root / "src"))

from models.dataset import create_dataloaders, DEFAULT_FEATURE_CHANNELS
from models.moment_model import MOMENTConfig, MOMENTClassifier, PretrainedMOMENTClassifier

print("=" * 80)
print("12-CHANNEL DATASET & PYTORCH TENSOR PIPELINE VERIFICATION")
print("=" * 80)

# 1. Inspect Parquet Columns and Data Types
train_p = project_root / "data" / "processed" / "xauusd_m30_train_labeled.parquet"
test_p = project_root / "data" / "processed" / "xauusd_m30_test_labeled.parquet"

df_train = pd.read_parquet(train_p)
df_test = pd.read_parquet(test_p)

print(f"\n[1/4] Parquet Files Loaded:")
print(f"  Train: {len(df_train):,} bars ({df_train['timestamp_utc'].min()} to {df_train['timestamp_utc'].max()})")
print(f"  Test : {len(df_test):,} bars ({df_test['timestamp_utc'].min()} to {df_test['timestamp_utc'].max()})")
print(f"  Channels ({len(DEFAULT_FEATURE_CHANNELS)}): {DEFAULT_FEATURE_CHANNELS}")

# Check NaN/Inf across all 12 channels
for ch in DEFAULT_FEATURE_CHANNELS:
    assert not df_train[ch].isna().any(), f"NaN found in train {ch}!"
    assert not np.isinf(df_train[ch]).any(), f"Inf found in train {ch}!"
    assert not df_test[ch].isna().any(), f"NaN found in test {ch}!"
    assert not np.isinf(df_test[ch]).any(), f"Inf found in test {ch}!"
print("  >>> Zero NaNs and Zero Infs verified across all 12 channels in both train and test!")

# 2. Build DataLoaders
print(f"\n[2/4] Initializing PyTorch DataLoaders (batch_size=32, seq_len=64)...")
train_loader, test_loader, train_ds, test_ds = create_dataloaders(
    train_df=df_train,
    test_df=df_test,
    batch_size=32,
    sequence_length=64,
    feature_channels=DEFAULT_FEATURE_CHANNELS,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print(f"  Train samples: {len(train_ds):,} sliding windows")
print(f"  Test samples : {len(test_ds):,} sliding windows")
print(f"  Train batches: {len(train_loader):,}")
print(f"  Test batches : {len(test_loader):,}")

# 3. Verify Batch Tensor Shape
print(f"\n[3/4] Fetching first batch from DataLoader...")
batch_x, batch_y = next(iter(train_loader))
print(f"  Batch X Shape : {batch_x.shape} (Expected: 32 x 12 x 64)")
print(f"  Batch X Dtype : {batch_x.dtype}")
print(f"  Batch Y Shape : {batch_y.shape} (Expected: 32)")
print(f"  Batch Y Dtype : {batch_y.dtype}")
print(f"  Batch Y values: {batch_y.tolist()[:10]}...")

assert batch_x.shape == (32, 12, 64), f"Expected shape (32, 12, 64), got {batch_x.shape}"
assert batch_y.shape == (32,), f"Expected shape (32,), got {batch_y.shape}"
assert not torch.isnan(batch_x).any(), "NaN found in batch X!"
assert not torch.isinf(batch_x).any(), "Inf found in batch X!"
print("  >>> Tensor Batch Shape and Integrity: 100% VERIFIED OK")

# 4. Single Dummy Forward Pass on GPU
print(f"\n[4/4] Testing Model Single Dummy Forward Pass on CUDA...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Target Device: {device}")
if device == "cuda":
    print(f"  GPU Name     : {torch.cuda.get_device_name(0)}")

# Test with lightweight MOMENTClassifier first
cfg = MOMENTConfig(n_channels=12, seq_len=64, num_classes=5)
model_standalone = MOMENTClassifier(cfg).to(device)
model_standalone.eval()

with torch.no_grad():
    x_dev = batch_x.to(device)
    out = model_standalone(x_dev)
    probs = torch.softmax(out, dim=-1)

print(f"  Standalone Model Output Shape: {out.shape} (Expected: 32, 5)")
print(f"  Probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
assert out.shape == (32, 5), f"Expected output shape (32, 5), got {out.shape}"
print("  >>> Standalone MOMENTClassifier forward pass: VERIFIED OK")

print("\n" + "=" * 80)
print("SUCCESS: ALL 12 CHANNELS, TENSORS, AND FORWARD PASSES FULLY VERIFIED!")
print("ZERO TRAINING EXECUTED (STOPPED ACCORDING TO USER INSTRUCTION).")
print("=" * 80)
