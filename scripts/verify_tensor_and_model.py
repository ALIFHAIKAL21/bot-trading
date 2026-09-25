"""
Dry-run verification of 15-channel PyTorch Dataset, DataLoader, and MOMENTClassifier.
DOES NOT TRAIN: Only performs forward pass verification and dimension checks.
"""
import sys, pathlib
import torch
import numpy as np
import pandas as pd

sys.path.append(r'c:\Ngoding\xau_deep_sniper')
from src.models.dataset import XAUTimeSeriesDataset, create_dataloaders
from src.models.moment_model import MOMENTConfig, MOMENTClassifier

data_path = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\data\processed\xauusd_m30_labeled_15ch.parquet')
print(f"Loading 15-channel dataset: {data_path}...")
df = pd.read_parquet(data_path)

print(f"\n1. Creating PyTorch Dataset...")
dataset = XAUTimeSeriesDataset(df, sequence_length=64)
print(f"  Dataset Length: {len(dataset)} samples")
print(f"  Sample 0 shape: x={dataset[0][0].shape}, y={dataset[0][1].item()}")

assert dataset[0][0].shape == (15, 64), f"Expected (15, 64), got {dataset[0][0].shape}"
print("  [PASS] Single sample shape is exactly (15 channels, 64 timesteps)!")

print(f"\n2. Creating DataLoader (Batch Size = 32)...")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
batch_x, batch_y = next(iter(loader))
print(f"  Batch x shape: {batch_x.shape}")
print(f"  Batch y shape: {batch_y.shape}")

assert batch_x.shape == (32, 15, 64), f"Expected (32, 15, 64), got {batch_x.shape}"
assert batch_y.shape == (32,), f"Expected (32,), got {batch_y.shape}"
print("  [PASS] Batch tensor dimensions verified perfectly!")

print(f"\n3. Initializing MOMENTClassifier with n_channels=15...")
cfg = MOMENTConfig(n_channels=15, seq_len=64, patch_len=8, patch_stride=8, d_model=1024, num_classes=5)
model = MOMENTClassifier(cfg)
summary = model.parameter_summary()
print(f"  Model Parameter Summary:")
for k, v in summary.items():
    print(f"    {k}: {v}")

print(f"\n4. Executing Forward Pass Dry-Run...")
model.eval()
with torch.no_grad():
    logits = model(batch_x)
    probs = model.predict_proba(batch_x)
    preds = model.predict(batch_x)

print(f"  Logits shape: {logits.shape}")
print(f"  Probs shape:  {probs.shape}")
print(f"  Preds shape:  {preds.shape}")

assert logits.shape == (32, 5), f"Expected (32, 5), got {logits.shape}"
assert probs.shape == (32, 5), f"Expected (32, 5), got {probs.shape}"
assert preds.shape == (32,), f"Expected (32,), got {preds.shape}"

# Check probabilities sum to 1
prob_sums = probs.sum(dim=-1)
assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Probabilities do not sum to 1!"
print("  [PASS] Model output shapes and softmax normalization are mathematically sound!")

print("\n=======================================================")
print("ALL SYSTEM VERIFICATIONS PASSED: 15-CHANNEL ARCHITECTURE IS 100% READY!")
print("=======================================================")
