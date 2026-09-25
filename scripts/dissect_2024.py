import sys, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# H4 data
df_h4 = df.set_index('timestamp_utc').resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

df_h4['h4_ema20'] = df_h4['close'].ewm(span=20, adjust=False).mean()
df_h4['h4_ema50'] = df_h4['close'].ewm(span=50, adjust=False).mean()
df_h4['h4_ema200'] = df_h4['close'].ewm(span=200, adjust=False).mean()
df_h4['h4_atr'] = (df_h4['high'] - df_h4['low']).rolling(14).mean()
df_h4['h4_slope20'] = (df_h4['h4_ema20'] - df_h4['h4_ema20'].shift(3)) / (df_h4['h4_atr'] + 1e-8)
df_h4['h4_high_20d'] = df_h4['high'].rolling(120).max()
df_h4['h4_ath_proximity'] = (df_h4['close'] >= df_h4['h4_high_20d'] * 0.998).astype(int)

df_h4_lagged = df_h4[['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_high_20d', 'h4_ath_proximity']].shift(1)
df = df.merge(df_h4_lagged, on='timestamp_utc', how='left')
for col in ['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_high_20d', 'h4_ath_proximity']:
    df[col] = df[col].ffill().fillna(0)

# ATR
high, low, close = df["high"].values, df["low"].values, df["close"].values
tr = np.zeros(len(df))
tr[0] = high[0] - low[0]
for i in range(1, len(df)):
    tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
period = 14
atr = np.zeros(len(df))
atr[:period] = np.mean(tr[:period])
multiplier = 2.0 / (period + 1)
for i in range(period, len(df)):
    atr[i] = tr[i] * multiplier + atr[i - 1] * (1 - multiplier)

# Filter for 2024
df_2024 = df[(df['timestamp_utc'] >= '2024-01-01') & (df['timestamp_utc'] <= '2024-12-31')].copy()
seq_offset = 63
TAU = 0.355

signals_2024 = []
for idx, row in df_2024.iterrows():
    bar_idx = df.index.get_loc(idx)
    pred_idx = bar_idx - seq_offset
    if 0 <= pred_idx < len(preds):
        probs = preds[pred_idx]
        trade_probs = probs[1:]
        max_idx = int(np.argmax(trade_probs))
        action_class = max_idx + 1
        conf = float(trade_probs[max_idx])
        p_hold = float(probs[0])
        
        if conf >= TAU and conf > p_hold:
            d = "BUY" if action_class in [1, 2] else "SELL"
            ob = row.get("order_block_zone", 0.0)
            m_slp = row.get("ma_ribbon_slope", 0.0)
            ok = True
            if d == "BUY" and (ob < 0 or m_slp < -0.3): ok = False
            if d == "SELL" and (ob > 0 or m_slp > 0.3): ok = False
            
            if ok:
                signals_2024.append({
                    'timestamp': row['timestamp_utc'],
                    'month': row['timestamp_utc'].strftime('%Y-%m'),
                    'direction': d,
                    'close': row['close'],
                    'h4_slope': row['h4_slope20'],
                    'h4_e20': row['h4_ema20'],
                    'h4_e50': row['h4_ema50'],
                    'h4_e200': row['h4_ema200'],
                    'h4_ath': row['h4_ath_proximity'],
                    'confidence': conf,
                    'class': action_class
                })

sig_df = pd.DataFrame(signals_2024)
print(f"Total signals in 2024: {len(sig_df)}")
print("\nSignals by Month and Direction:")
print(sig_df.groupby(['month', 'direction']).size().unstack(fill_value=0))

print("\nAverage H4 Slope when SELL was generated in 2024:")
print(sig_df[sig_df['direction'] == 'SELL'].groupby('month')['h4_slope'].mean())

print("\nAverage Price vs H4 EMA50 when SELL was generated:")
sig_df['price_vs_ema50'] = sig_df['close'] - sig_df['h4_e50']
print(sig_df[sig_df['direction'] == 'SELL'].groupby('month')['price_vs_ema50'].mean())
