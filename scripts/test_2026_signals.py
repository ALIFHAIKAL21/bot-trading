import numpy as np
import pandas as pd
import pathlib

df_path = pathlib.Path(r"c:\Ngoding\xau_deep_sniper\data\processed\xauusd_m30_labeled_15ch.parquet")
preds_path = pathlib.Path(r"c:\Ngoding\xau_deep_sniper\checkpoints\predictions_15ch.npy")

df = pd.read_parquet(df_path)
preds = np.load(preds_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

mask = (df['timestamp_utc'] >= '2026-01-01') & (df['timestamp_utc'] <= '2026-08-31 23:59:59')
sub_df = df[mask]
seq_offset = 63

trading_days = sub_df['timestamp_utc'].dt.date.nunique()
print(f"Total 2026 Jan-Aug bars: {len(sub_df)}")
print(f"Total unique trading days: {trading_days}")
print(f"Start: {sub_df['timestamp_utc'].min()} | End: {sub_df['timestamp_utc'].max()}")

sub_indices = sub_df.index
pred_indices = sub_indices - seq_offset
valid_mask = (pred_indices >= 0) & (pred_indices < len(preds))
valid_preds = preds[pred_indices[valid_mask]]

# Check class predictions: 0: Neutral, 1: Strong Buy, 2: Scalp Buy, 3: Strong Sell, 4: Scalp Sell
trade_probs = valid_preds[:, 1:]
max_trade_probs = np.max(trade_probs, axis=1)
predicted_classes = np.argmax(valid_preds, axis=1)

print("\n--- Distribution of Max Trade Confidence (Classes 1-4) in Jan-Aug 2026 ---")
for t in [0.25, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45]:
    count = np.sum(max_trade_probs >= t)
    per_day = count / trading_days
    print(f"tau >= {t:.2f}: {count:5d} bars ({per_day:.2f} signals/day on average)")
