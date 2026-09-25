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

df_h4_lagged = df_h4[['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20']].shift(1)
df = df.merge(df_h4_lagged, on='timestamp_utc', how='left')
for col in ['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20']:
    df[col] = df[col].ffill().fillna(0)

# Check 2022 SELL signals where (close > h4_e50) and (h4_e20 > h4_e50) and (h4_slope > 0.05)
df_2022 = df[(df['timestamp_utc'] >= '2022-01-01') & (df['timestamp_utc'] <= '2022-12-31')].copy()
seq_offset = 63
TAU = 0.355

signals_2022 = []
for idx, row in df_2022.iterrows():
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
                c_p = row['close']
                e20 = row['h4_ema20']
                e50 = row['h4_ema50']
                slp = row['h4_slope20']
                naive_blocked = (d == "SELL") and (c_p > e50) and (e20 > e50) and (slp > 0.05)
                signals_2022.append({
                    'timestamp': row['timestamp_utc'],
                    'direction': d,
                    'close': c_p,
                    'h4_slope': slp,
                    'naive_blocked': naive_blocked,
                    'h4_e20': e20,
                    'h4_e50': e50
                })

s22 = pd.DataFrame(signals_2022)
print(f"2022 Total Valid Signals: {len(s22)}")
print(f"2022 Naive Blocked SELLs: {s22['naive_blocked'].sum()} out of {len(s22[s22['direction'] == 'SELL'])} total SELLs")
