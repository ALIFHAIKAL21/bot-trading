import sys, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

df_h4 = df.set_index('timestamp_utc').resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_h4['h4_ema200'] = df_h4['close'].ewm(span=200, adjust=False).mean()
df_h4['above_ema200'] = df_h4['close'] > df_h4['h4_ema200']

df_h4_lagged = df_h4[['h4_ema200', 'above_ema200']].shift(1)
df = df.merge(df_h4_lagged, on='timestamp_utc', how='left')
df['above_ema200'] = df['above_ema200'].ffill().fillna(False)

for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
    sub = df[df['timestamp_utc'].dt.year == yr]
    pct = sub['above_ema200'].mean() * 100
    print(f"Year {yr}: {pct:.1f}% time ABOVE H4 EMA200")
