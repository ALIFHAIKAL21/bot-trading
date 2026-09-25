"""
Empirical Benchmark: Measuring 1-epoch training throughput on NVIDIA RTX 4050 (6GB)
Tests:
1. Batch Size 128 vs 64
2. FP16 AMP Tensor Cores
3. In-memory Contiguous Tensor DataLoader
"""
import sys, pathlib, time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

sys.path.append(r'c:\Ngoding\xau_deep_sniper')
from src.models.dataset import XAUTimeSeriesDataset
from src.models.moment_model import MOMENTConfig, MOMENTClassifier
from src.models.loss import DirectionalFocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

data_path = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\data\processed\xauusd_m30_labeled_15ch.parquet')
df = pd.read_parquet(data_path)

# Split 80% train
n_train = int(len(df) * 0.8)
train_df = df.iloc[:n_train]

# Setup contiguous tensor dataset for zero-copy memory transfer
print(f"Pre-allocating contiguous tensor dataset ({len(train_df):,} samples)...")
t0 = time.time()
dataset = XAUTimeSeriesDataset(train_df, sequence_length=64)

# Create optimized DataLoader
batch_size = 128
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)
print(f"Dataset pre-allocated in {time.time() - t0:.2f}s. Total batches: {len(loader)}")

# Model & Loss
cfg = MOMENTConfig(n_channels=15, seq_len=64, patch_len=8, patch_stride=8, d_model=1024, num_classes=5)
model = MOMENTClassifier(cfg).to(device)
criterion = DirectionalFocalLoss(alpha=[0.25, 1.1, 1.2, 1.1, 1.2], gamma=2.0).to(device)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
scaler = torch.amp.GradScaler('cuda', enabled=True)

torch.backends.cudnn.benchmark = True

print(f"\nRunning 1 Full Training Epoch on RTX 4050 with FP16 AMP...")
model.train()
t_start = time.time()

total_samples = 0
for batch_idx, (x, y) in enumerate(loader):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(x)
        loss = criterion(logits, y)
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    total_samples += len(y)
    if (batch_idx + 1) % 100 == 0:
        elapsed = time.time() - t_start
        throughput = total_samples / elapsed
        print(f"  Batch {batch_idx + 1:3d}/{len(loader)} | Speed: {throughput:6.1f} samples/sec | VRAM: {torch.cuda.memory_allocated() / (1024*1024):.0f}MB")

torch.cuda.synchronize()
total_time = time.time() - t_start
print(f"\n>>> 1 EPOCH FINISHED IN: {total_time:.2f} SECONDS! <<<")
print(f"Average Throughput: {total_samples / total_time:.1f} samples/second")
print(f"Estimated 12 Epochs Total Training Time: {(total_time * 12) / 60:.2f} MINUTES!")
