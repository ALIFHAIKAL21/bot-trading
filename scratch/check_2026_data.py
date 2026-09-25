import numpy as np
import pandas as pd

preds = np.load(r'c:\Ngoding\bot_trading\scratch\test_predictions_2026.npy')
df = pd.read_parquet(r'c:\Ngoding\xau_deep_sniper\data\processed\xauusd_m30_test_labeled.parquet')
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

print(f"Total DF: {len(df)}, Predictions: {len(preds)}")
print(f"Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")

mask_2026 = (df['timestamp_utc'] >= '2026-01-01') & (df['timestamp_utc'] <= '2026-08-31 23:59:59')
df_2026 = df[mask_2026]
print(f"2026 Jan-Aug bars: {len(df_2026)}")
for m, g in df_2026.groupby(df_2026['timestamp_utc'].dt.to_period('M')):
    print(f"  {m}: {len(g)} bars, min={g['timestamp_utc'].min().strftime('%Y-%m-%d')}, max={g['timestamp_utc'].max().strftime('%Y-%m-%d')}")
