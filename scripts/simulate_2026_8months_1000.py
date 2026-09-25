"""
Simulation of Jan - Aug 2026 with $1,000 Capital and 3 Trades per Day (Mon-Fri)
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

# Compute ATR(14)
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

st_date = "2026-01-01 00:00:00"
end_date = "2026-08-31 23:59:59"
sub_mask = (df['timestamp_utc'] >= st_date) & (df['timestamp_utc'] <= end_date)
week_indices = df[sub_mask].index

INITIAL_EQUITY = 1000.0  # Modal $1,000
SPREAD_PRICE = 0.75 * 0.10
SLIPPAGE_PRICE = 0.3 * 0.10
COMMISSION = 3.50
CONTRACT_SIZE = 100.0
SL_ATR_MULT = 1.5
TP1_R = 1.0
TP2_R = 2.5
TRAIL_AFTER_R = 1.5
TRAIL_DIST_R = 0.8

def calc_friction(lot):
    return round(SPREAD_PRICE * lot * CONTRACT_SIZE + SLIPPAGE_PRICE * lot * CONTRACT_SIZE * 2 + COMMISSION * lot, 2)

days = sorted(df.loc[week_indices, 'timestamp_utc'].dt.strftime('%Y-%m-%d').unique())
seq_offset = 63

print(f"Total Hari Pasar Buka (Jan - Agu 2026): {len(days)} hari")

# Test 2 mode lot sizing:
# Mode A: 0.02 Lot Fixed (0.01 Pos A + 0.01 Pos B, standar twin-order retail terkecil)
# Mode B: Dynamic Sizing 1% risk (minimum 0.02 lot, naik jika saldo berkembang)

def run_8months(mode="fixed_002"):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    trade_counter = 0

    session_ranges = [
        (1, 7, "Sesi Pagi (Asia)"),
        (8, 13, "Sesi Siang (London)"),
        (14, 20, "Sesi Sore (New York)")
    ]

    for day_str in days:
        day_mask = (df['timestamp_utc'].dt.strftime('%Y-%m-%d') == day_str) & sub_mask
        day_indices = df[day_mask].index
        
        # Cek apakah hari Senin-Jumat
        day_ts = df.loc[day_indices[0], 'timestamp_utc']
        if day_ts.dayofweek in [5, 6]:
            continue

        for s_start, s_end, s_name in session_ranges:
            best_bar_idx = None
            best_conf = -1.0
            best_action = None

            for b_idx in day_indices:
                row = df.iloc[b_idx]
                h = row['timestamp_utc'].hour
                m = row['timestamp_utc'].minute
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
                entry_row = df.iloc[best_bar_idx]
                entry_ts = entry_row["timestamp_utc"]
                bar_close = entry_row["close"]
                d = "BUY" if best_action in [1, 2] else "SELL"

                atr_val = float(atr[best_bar_idx]) if best_bar_idx < len(atr) else 5.0
                sl_dist = round(atr_val * SL_ATR_MULT, 2)

                if mode == "fixed_002":
                    lot_a = 0.01
                    lot_b = 0.01
                    total_lot = 0.02
                    actual_risk = total_lot * sl_dist * CONTRACT_SIZE
                elif mode == "fixed_001":
                    lot_a = 0.01
                    lot_b = 0.0
                    total_lot = 0.01
                    actual_risk = total_lot * sl_dist * CONTRACT_SIZE
                else: # dynamic compounding
                    target_risk = equity * 0.01
                    raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                    total_lot = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 5.0))
                    lot_a = max(0.01, round(total_lot * 0.5, 2))
                    lot_b = max(0.01, round(total_lot - lot_a, 2))
                    actual_risk = total_lot * sl_dist * CONTRACT_SIZE

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

                cur_sl_a = sl_p
                cur_sl_b = sl_p
                pos_a_open = True
                pos_b_open = True if lot_b > 0 else False
                tp1_hit = False
                tp2_hit = False
                pnl_accum = 0.0
                exit_ts = None
                exit_price = None
                exit_reason = None

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
                                pnl_g = (cur_sl_a - ep) * total_lot * CONTRACT_SIZE
                                pnl_accum = round(pnl_g - calc_friction(total_lot), 2)
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
                        else:  # SELL
                            if not tp1_hit and f_high >= cur_sl_a:
                                pnl_g = (ep - cur_sl_a) * total_lot * CONTRACT_SIZE
                                pnl_accum = round(pnl_g - calc_friction(total_lot), 2)
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
                m_str = entry_ts.strftime('%Y-%m')
                trades.append({
                    'id': trade_counter + 1,
                    'month': m_str,
                    'day': day_str,
                    'session': s_name,
                    'direction': d,
                    'entry_ts': entry_ts,
                    'pnl': pnl_final,
                    'r_mult': r_mult,
                    'win': 1 if pnl_final > 0 else 0,
                    'exit_reason': exit_reason if exit_reason else "Selesai Sesi"
                })
                trade_counter += 1
                equity += pnl_final
                equity_curve.append(equity)

    tdf = pd.DataFrame(trades)
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = np.min((eq_arr - peak) / peak) * 100
    return tdf, equity, eq_arr, max_dd

# Run Mode A (0.02 Lot)
print("Menjalankan Simulasi 8 Bulan Mode A (0.02 Lot Fixed)...")
tdf_a, eq_a, eq_curve_a, dd_a = run_8months(mode="fixed_002")

print(f"\nHASIL JANUARI - AGUSTUS 2026 (MODAL $1,000, 3 TRADE/HARI):")
print(f"Total Trade : {len(tdf_a)}")
wins_a = tdf_a['win'].sum()
losses_a = len(tdf_a) - wins_a
wr_a = wins_a / len(tdf_a) * 100
net_pnl_a = eq_a - INITIAL_EQUITY
ret_a = net_pnl_a / INITIAL_EQUITY * 100

print(f"Akurasi     : {wins_a} Menang / {losses_a} Kalah ({wr_a:.1f}% Win Rate)")
print(f"Net Profit  : ${net_pnl_a:+,.2f} ({ret_a:+.2f}%)")
print(f"Saldo Akhir : ${eq_a:,.2f}")
print(f"Max Drawdown: {dd_a:.2f}%")

# Rincian per Bulan
print("\nRINCIAN BULAN KE BULAN (JANUARI - AGUSTUS 2026):")
cur_e = 1000.0
for m in sorted(tdf_a['month'].unique()):
    sub = tdf_a[tdf_a['month'] == m]
    sw = sub['win'].sum()
    spnl = sub['pnl'].sum()
    spct = (spnl / cur_e) * 100
    cur_e += spnl
    print(f"  {m} | Trades: {len(sub):3d} | Menang: {sw:2d} ({sw/len(sub)*100:4.1f}%) | PnL: ${spnl:+7.2f} ({spct:+6.2f}%) | Saldo: ${cur_e:,.2f}")

# Also test Mode B (0.01 Lot Flat)
print("\nMenjalankan Simulasi 8 Bulan Mode B (0.01 Lot Flat)...")
tdf_b, eq_b, eq_curve_b, dd_b = run_8months(mode="fixed_001")
net_pnl_b = eq_b - INITIAL_EQUITY
ret_b = net_pnl_b / INITIAL_EQUITY * 100
print(f"Hasil Mode 0.01 Lot: Saldo Akhir=${eq_b:,.2f} | Net PnL=${net_pnl_b:+,.2f} ({ret_b:+.2f}%) | Max DD={dd_b:.2f}%")
