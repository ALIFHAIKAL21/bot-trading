import sys, os, pathlib, time
import pandas as pd
import numpy as np
import torch

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
sys.path.insert(0, str(project_root / "src"))

from models.moment_model import PretrainedMOMENTClassifier
from models.dataset import XAUTimeSeriesDataset, DEFAULT_FEATURE_CHANNELS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Load test data
test_path = project_root / "data" / "processed" / "xauusd_m30_test_labeled.parquet"
df_test = pd.read_parquet(test_path)
df_test['timestamp_utc'] = pd.to_datetime(df_test['timestamp_utc'], utc=True)
print(f"Loaded {len(df_test):,} bars ({df_test['timestamp_utc'].min()} to {df_test['timestamp_utc'].max()})")

# 2. Load model checkpoint
ckpt_path = project_root / "checkpoints" / "best_moment_pretrained_lora.pt"
print(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

model = PretrainedMOMENTClassifier(num_classes=5, n_channels=12, lora_r=16, lora_alpha=32)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 3. Run Inference
dataset = XAUTimeSeriesDataset(df_test, sequence_length=64, feature_channels=DEFAULT_FEATURE_CHANNELS)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=(device == 'cuda'))

print(f"Running inference on {len(dataset):,} sliding windows...")
t0 = time.time()
all_probs = []
with torch.no_grad():
    for x_batch, _ in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x_batch)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

predictions = np.concatenate(all_probs, axis=0)
t1 = time.time()
print(f"Inference completed in {t1 - t0:.2f}s! Shape: {predictions.shape}")

# Save cached predictions to scratch
np.save(r'c:\Ngoding\bot_trading\scratch\test_predictions_2026.npy', predictions)
print("Saved predictions to scratch/test_predictions_2026.npy")

print(f"Mean P(HOLD): {predictions[:, 0].mean():.3f}")
print(f"Mean P(BUY_1R): {predictions[:, 1].mean():.3f}")
print(f"Mean P(BUY_2R): {predictions[:, 2].mean():.3f}")
print(f"Mean P(SELL_1R): {predictions[:, 3].mean():.3f}")
print(f"Mean P(SELL_2R): {predictions[:, 4].mean():.3f}")
