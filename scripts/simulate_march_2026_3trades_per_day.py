"""
Ensure at least 3 trades per day for each day (Monday to Friday) in March 2026
"""

import sys, os, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

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

# Filter Week 1: 2-6 Maret 2026
st_date = "2026-03-02 00:00:00"
end_date = "2026-03-06 23:59:59"

sub_mask = (df['timestamp_utc'] >= st_date) & (df['timestamp_utc'] <= end_date)
week_indices = df[sub_mask].index

INITIAL_EQUITY = 10_000.0
BASE_RISK = 0.01  # 1% per trade
SPREAD_PRICE = 0.75 * 0.10
SLIPPAGE_PRICE = 0.3 * 0.10
COMMISSION = 3.50
CONTRACT_SIZE = 100.0
SL_ATR_MULT = 1.5
POS_A_PCT = 0.40
POS_B_PCT = 0.60
TP1_R = 1.0
TP2_R = 2.5
TRAIL_AFTER_R = 1.5
TRAIL_DIST_R = 0.8

def calc_friction(lot):
    return round(SPREAD_PRICE * lot * CONTRACT_SIZE + SLIPPAGE_PRICE * lot * CONTRACT_SIZE * 2 + COMMISSION * lot, 2)

days = sorted(df.loc[week_indices, 'timestamp_utc'].dt.strftime('%Y-%m-%d').unique())
seq_offset = 63

# Strategy: For each day in days, find the 3 best non-HOLD trade opportunities based on MOMENT predictions
# (1 morning/Asia session, 1 London session, 1 New York session) and execute bar-by-bar
equity = INITIAL_EQUITY
equity_curve = [equity]
trades = []
trade_counter = 0

for day_idx, day_str in enumerate(days):
    day_mask = (df['timestamp_utc'].dt.strftime('%Y-%m-%d') == day_str) & sub_mask
    day_indices = df[day_mask].index
    
    # Divide day into 3 sessions:
    # Sesi 1: 01:00 - 07:00 UTC (Asia/Early London)
    # Sesi 2: 08:30 - 13:30 UTC (London Morning)
    # Sesi 3: 14:00 - 20:00 UTC (New York)
    session_ranges = [
        (1, 7, "Sesi Pagi (Asia/London Awal)"),
        (8, 13, "Sesi Siang (London Morning)"),
        (14, 20, "Sesi Sore/Malam (New York)")
    ]
    
    for s_start, s_end, s_name in session_ranges:
        # Find best trade signal in this session
        best_bar_idx = None
        best_conf = -1.0
        best_action = None
        
        for b_idx in day_indices:
            row = df.iloc[b_idx]
            h = row['timestamp_utc'].hour
            m = row['timestamp_utc'].minute
            # skip london quarantine (07:00 - 08:30)
            if h == 7 or (h == 8 and m < 30):
                continue
            if s_start <= h <= s_end:
                p_idx = b_idx - seq_offset
                if p_idx >= 0 and p_idx < len(preds):
                    probs = preds[p_idx]
                    t_probs = probs[1:]
                    max_idx = int(np.argmax(t_probs))
                    conf = float(t_probs[max_idx])
                    if conf > best_conf:
                        best_conf = conf
                        best_bar_idx = b_idx
                        best_action = max_idx + 1

        if best_bar_idx is not None:
            # Execute trade from best_bar_idx
            entry_row = df.iloc[best_bar_idx]
            entry_ts = entry_row["timestamp_utc"]
            bar_close = entry_row["close"]
            d = "BUY" if best_action in [1, 2] else "SELL"
            
            atr_val = float(atr[best_bar_idx]) if best_bar_idx < len(atr) else 5.0
            sl_dist = round(atr_val * SL_ATR_MULT, 2)
            
            target_risk = equity * BASE_RISK
            raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
            lot_size = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 50.0))
            actual_risk = lot_size * sl_dist * CONTRACT_SIZE
            lot_a = max(0.01, round(lot_size * POS_A_PCT, 2))
            lot_b = max(0.01, round(lot_size - lot_a, 2))
            
            if d == "BUY":
                ep = round(bar_close + SLIPPAGE_PRICE, 2)
                sl_p = round(ep - sl_dist, 2)
                tp1_p = round(ep + (TP1_R * sl_dist), 2)
                tp2_p = round(ep + (TP2_R * sl_dist), 2)
            else:
                ep = round(bar_close - SLIPPAGE_PRICE, 2)
                sl_p = round(ep + sl_dist, 2)
                tp1_p = round(ep - (TP1_R * sl_dist), 2)
                tp2_p = round(ep - (TP2_R * sl_dist), 2)
                
            # Simulate forward until exit
            cur_sl_a = sl_p
            cur_sl_b = sl_p
            pos_a_open = True
            pos_b_open = True
            tp1_hit = False
            tp2_hit = False
            pnl_accum = 0.0
            exit_ts = None
            exit_price = None
            exit_reason = None
            
            # Forward simulation loop
            for f_idx in range(best_bar_idx + 1, min(best_bar_idx + 16, len(df))):
                f_row = df.iloc[f_idx]
                f_ts = f_row["timestamp_utc"]
                f_high = f_row["high"]
                f_low = f_row["low"]
                f_close = f_row["close"]
                f_dow = f_ts.dayofweek
                f_hour = f_ts.hour
                
                is_fri_close = (f_dow == 4 and f_hour >= 20)
                is_tb = (f_idx - best_bar_idx >= 12)
                
                if is_fri_close or is_tb:
                    exit_reason = "Tutup Jumat (20:00)" if is_fri_close else "Batas Waktu (6 Jam)"
                    exit_ts = f_ts
                    exit_price = f_close
                    if pos_a_open:
                        pnl_a = (exit_price - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * lot_a * CONTRACT_SIZE
                        pnl_accum += (pnl_a - calc_friction(lot_a))
                        pos_a_open = False
                    if pos_b_open:
                        pnl_b = (exit_price - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * lot_b * CONTRACT_SIZE
                        pnl_accum += (pnl_b - calc_friction(lot_b))
                        pos_b_open = False
                    break
                else:
                    if d == "BUY":
                        if not tp1_hit and f_low <= cur_sl_a:
                            pnl_g = (cur_sl_a - ep) * lot_size * CONTRACT_SIZE
                            pnl_accum = round(pnl_g - calc_friction(lot_size), 2)
                            exit_ts = f_ts
                            exit_price = cur_sl_a
                            exit_reason = "Stop Loss (-1.0R)"
                            pos_a_open = False
                            pos_b_open = False
                            break
                        else:
                            if pos_a_open and f_high >= tp1_p:
                                pnl_a = (tp1_p - ep) * lot_a * CONTRACT_SIZE
                                pnl_accum += (pnl_a - calc_friction(lot_a))
                                pos_a_open = False
                                tp1_hit = True
                                f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                                cur_sl_b = max(cur_sl_b, round(ep + f_p, 2))
                                
                            if pos_b_open:
                                if f_high >= tp2_p:
                                    pnl_b = (tp2_p - ep) * lot_b * CONTRACT_SIZE
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    pos_b_open = False
                                    exit_ts = f_ts
                                    exit_price = tp2_p
                                    exit_reason = "Take Profit (+2.5R)"
                                    break
                                elif f_low <= cur_sl_b:
                                    pnl_b = (cur_sl_b - ep) * lot_b * CONTRACT_SIZE
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    pos_b_open = False
                                    exit_ts = f_ts
                                    exit_price = cur_sl_b
                                    exit_reason = "Breakeven" if tp1_hit else "Stop Loss"
                                    break
                                elif tp1_hit:
                                    r_gain = (f_high - ep) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(f_high - (TRAIL_DIST_R * sl_dist), 2)
                                        cur_sl_b = max(cur_sl_b, trail_sl)
                    else: # SELL
                        if not tp1_hit and f_high >= cur_sl_a:
                            pnl_g = (ep - cur_sl_a) * lot_size * CONTRACT_SIZE
                            pnl_accum = round(pnl_g - calc_friction(lot_size), 2)
                            exit_ts = f_ts
                            exit_price = cur_sl_a
                            exit_reason = "Stop Loss (-1.0R)"
                            pos_a_open = False
                            pos_b_open = False
                            break
                        else:
                            if pos_a_open and f_low <= tp1_p:
                                pnl_a = (ep - tp1_p) * lot_a * CONTRACT_SIZE
                                pnl_accum += (pnl_a - calc_friction(lot_a))
                                pos_a_open = False
                                tp1_hit = True
                                f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                                cur_sl_b = min(cur_sl_b, round(ep - f_p, 2))
                                
                            if pos_b_open:
                                if f_low <= tp2_p:
                                    pnl_b = (ep - tp2_p) * lot_b * CONTRACT_SIZE
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    pos_b_open = False
                                    exit_ts = f_ts
                                    exit_price = tp2_p
                                    exit_reason = "Take Profit (+2.5R)"
                                    break
                                elif f_high >= cur_sl_b:
                                    pnl_b = (ep - cur_sl_b) * lot_b * CONTRACT_SIZE
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    pos_b_open = False
                                    exit_ts = f_ts
                                    exit_price = cur_sl_b
                                    exit_reason = "Breakeven" if tp1_hit else "Stop Loss"
                                    break
                                elif tp1_hit:
                                    r_gain = (ep - f_low) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(f_low + (TRAIL_DIST_R * sl_dist), 2)
                                        cur_sl_b = min(cur_sl_b, trail_sl)
            
            pnl_final = round(pnl_accum, 2)
            r_mult = round(pnl_final / actual_risk, 2)
            trades.append({
                'id': trade_counter + 1,
                'day': day_str,
                'session': s_name,
                'direction': d,
                'confidence': round(best_conf, 3),
                'entry_time': entry_ts.strftime('%H:%M'),
                'exit_time': exit_ts.strftime('%H:%M') if exit_ts else "-",
                'entry_price': ep,
                'exit_price': exit_price if exit_price else bar_close,
                'lot_size': lot_size,
                'pnl': pnl_final,
                'r_mult': r_mult,
                'win': 1 if pnl_final > 0 else 0,
                'exit_reason': exit_reason if exit_reason else "Selesai Sesi"
            })
            trade_counter += 1
            equity += pnl_final
            equity_curve.append(equity)

tdf = pd.DataFrame(trades)
print(f"\nHASIL SIMULASI TEPA 3 TRADE PER HARI (SENIN-JUMAT 2-6 MARET 2026):")
print(f"Total Trade: {len(tdf)}")
wins = tdf['win'].sum()
losses = len(tdf) - wins
wr = wins / len(tdf) * 100
net_pnl = equity - INITIAL_EQUITY
ret_pct = net_pnl / INITIAL_EQUITY * 100

eq_arr = np.array(equity_curve)
peak = np.maximum.accumulate(eq_arr)
max_dd = np.min((eq_arr - peak) / peak) * 100

print(f"Total Menang: {wins} | Kalah: {losses} | Akurasi (Win Rate): {wr:.1f}%")
print(f"Net PnL: ${net_pnl:+,.2f} ({ret_pct:+.2f}%) | Max DD: {max_dd:.2f}%")

print("\nREKAP PER HARI:")
for d_str, g in tdf.groupby('day'):
    w_cnt = g['win'].sum()
    print(f"  {d_str}: {len(g)} Trade | Menang: {w_cnt} | Kalah: {len(g)-w_cnt} | PnL: ${g['pnl'].sum():+7.2f}")

print("\nTABEL RINCIAN 15 TRADE:")
for _, r in tdf.iterrows():
    print(f"  Trade #{r['id']:2d} | {r['day']} | {r['session'][:9]} | {r['direction']:4s} | Entry: {r['entry_time']} (${r['entry_price']:,.2f}) | Exit: {r['exit_time']} (${r['exit_price']:,.2f}) | PnL: ${r['pnl']:+7.2f} ({r['r_mult']:+4.1f}R) | {r['exit_reason']}")
