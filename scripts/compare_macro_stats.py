import sys, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

df_h4 = df.set_index('timestamp_utc').resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

df_h4['h4_ema20'] = df_h4['close'].ewm(span=20, adjust=False).mean()
df_h4['h4_ema50'] = df_h4['close'].ewm(span=50, adjust=False).mean()
df_h4['h4_ema200'] = df_h4['close'].ewm(span=200, adjust=False).mean()
df_h4['h4_atr'] = (df_h4['high'] - df_h4['low']).rolling(14).mean()
df_h4['h4_slope20'] = (df_h4['h4_ema20'] - df_h4['h4_ema20'].shift(3)) / (df_h4['h4_atr'] + 1e-8)
df_h4['h4_high_60d'] = df_h4['high'].rolling(360).max() # 60 days
df_h4['h4_dist_to_60d_high'] = (df_h4['h4_high_60d'] - df_h4['close']) / (df_h4['h4_atr'] + 1e-8)

df_h4_lagged = df_h4[['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_high_60d', 'h4_dist_to_60d_high']].shift(1)
df = df.merge(df_h4_lagged, on='timestamp_utc', how='left')
for col in ['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_high_60d', 'h4_dist_to_60d_high']:
    df[col] = df[col].ffill().fillna(0)

# Check distribution of h4_slope across all 5 years for SELL signals
seq_offset = 63
TAU = 0.355

records = []
for idx in range(len(df)):
    pred_idx = idx - seq_offset
    if 0 <= pred_idx < len(preds):
        probs = preds[pred_idx]
        trade_probs = probs[1:]
        max_idx = int(np.argmax(trade_probs))
        action_class = max_idx + 1
        conf = float(trade_probs[max_idx])
        p_hold = float(probs[0])
        
        if conf >= TAU and conf > p_hold:
            row = df.iloc[idx]
            d = "BUY" if action_class in [1, 2] else "SELL"
            ob = row.get("order_block_zone", 0.0)
            m_slp = row.get("ma_ribbon_slope", 0.0)
            ok = True
            if d == "BUY" and (ob < 0 or m_slp < -0.3): ok = False
            if d == "SELL" and (ob > 0 or m_slp > 0.3): ok = False
            
            if ok:
                records.append({
                    'year': row['timestamp_utc'].year,
                    'direction': d,
                    'h4_slope': row['h4_slope20'],
                    'dist_to_60d_high': row['h4_dist_to_60d_high'],
                    'price_vs_e50': row['close'] - row['h4_ema50']
                })

rec_df = pd.DataFrame(records)
print("SELL signals H4 slope percentiles by Year:")
sells = rec_df[rec_df['direction'] == 'SELL']
for yr, grp in sells.groupby('year'):
    print(f"Year {yr} (N={len(grp)}): Mean={grp['h4_slope'].mean():.3f}, P50={grp['h4_slope'].median():.3f}, P75={grp['h4_slope'].quantile(0.75):.3f}, P90={grp['h4_slope'].quantile(0.90):.3f}, Max={grp['h4_slope'].max():.3f}")

print("\nDistance to 60-day High (in ATRs) when SELL triggered:")
for yr, grp in sells.groupby('year'):
    print(f"Year {yr}: Mean dist to high={grp['dist_to_60d_high'].mean():.2f} ATRs, P25={grp['dist_to_60d_high'].quantile(0.25):.2f}, Min={grp['dist_to_60d_high'].min():.2f}")
