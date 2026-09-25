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
df_h4['h4_atr'] = (df_h4['high'] - df_h4['low']).rolling(14).mean()
# Slope of 200 EMA over 20 H4 bars (normalized by ATR)
df_h4['h4_slope200'] = (df_h4['h4_ema200'] - df_h4['h4_ema200'].shift(20)) / (df_h4['h4_atr'] + 1e-8)

for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
    sub = df_h4[df_h4.index.year == yr]
    print(f"Year {yr}: H4 EMA200 Slope: Mean={sub['h4_slope200'].mean():.3f}, Min={sub['h4_slope200'].min():.3f}, Max={sub['h4_slope200'].max():.3f}, Pct > 0.15: {(sub['h4_slope200'] > 0.15).mean()*100:.1f}%")
