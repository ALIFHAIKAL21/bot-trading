"""
Simulasi Backtest 1 Minggu di Bulan Maret 2026 (Senin 2 Maret s/d Jumat 6 Maret 2026)
Constraint Simulasi: Bot dipaksa/ditugaskan trade MINIMAL 3 kali per hari pasar buka.
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

# Filter Week 1: 2 Maret 2026 s/d 6 Maret 2026
st_date = "2026-03-02 00:00:00"
end_date = "2026-03-06 23:59:59"

sub_mask = (df['timestamp_utc'] >= st_date) & (df['timestamp_utc'] <= end_date)
week_indices = df[sub_mask].index

print(f"Total Bar 1 Minggu (2-6 Maret 2026): {len(week_indices)} bars M30")

# Institutional Parameters
INITIAL_EQUITY = 10_000.0
BASE_RISK = 0.01  # 1% per trade = $100
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

# Group days
days = sorted(df.loc[week_indices, 'timestamp_utc'].dt.strftime('%Y-%m-%d').unique())
print("Hari pasar buka:", days)

# How to ensure MIN 3 trades per day:
# On each day, find bars where model gives highest trade probability (non-HOLD)
# If open_position is None, enter when model signal is ready, dynamically adjusting threshold
# if day's trade count is < 3 so that at least 3 trades are completed per day.

def run_simulation(min_trades_per_day=3):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    trade_counter = 0
    open_position = None

    seq_offset = 63
    st_ts = pd.to_datetime(st_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)

    for bar_idx in range(len(df)):
        row = df.iloc[bar_idx]
        ts = row["timestamp_utc"]
        if ts < st_ts:
            continue
        if ts > end_ts:
            break

        bar_open, bar_high, bar_low, bar_close = row["open"], row["high"], row["low"], row["close"]
        day_str = ts.strftime('%Y-%m-%d')
        dow = ts.dayofweek
        hour = ts.hour
        minute = ts.minute

        # Manage open position
        if open_position is not None:
            bars_held = bar_idx - open_position["entry_bar_idx"]
            d = open_position["direction"]
            ep = open_position["entry_price"]
            sl_dist = open_position["sl_dist"]
            lot_a = open_position["lot_a"]
            lot_b = open_position["lot_b"]

            is_friday_close = (dow == 4 and hour >= 20)
            is_time_barrier = (bars_held >= 12)  # 6 hours max hold so next trade can enter

            if is_friday_close or is_time_barrier:
                exit_price = bar_close
                if open_position["pos_a_open"]:
                    pnl_a = (exit_price - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * lot_a * CONTRACT_SIZE
                    open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                    open_position["pos_a_open"] = False
                if open_position["pos_b_open"]:
                    pnl_b = (exit_price - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * lot_b * CONTRACT_SIZE
                    open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                    open_position["pos_b_open"] = False

                net_pnl = round(open_position["pnl_net_accum"], 2)
                r_mult = round(net_pnl / open_position["risk_amount"], 2)
                trades.append({
                    'id': trade_counter, 'day': day_str, 'direction': d,
                    'entry_ts': open_position["entry_timestamp"], 'exit_ts': ts,
                    'entry_price': ep, 'exit_price': exit_price,
                    'pnl': net_pnl, 'r': r_mult, 'win': 1 if net_pnl > 0 else 0,
                    'exit_reason': 'FRIDAY_CLOSE' if is_friday_close else 'TIME_BARRIER'
                })
                trade_counter += 1
                equity += net_pnl
                open_position = None
            else:
                if d == "BUY":
                    if not open_position["tp1_hit"] and bar_low <= open_position["cur_sl_a"]:
                        pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                        net_pnl = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                        r_mult = round(net_pnl / open_position["risk_amount"], 2)
                        trades.append({
                            'id': trade_counter, 'day': day_str, 'direction': d,
                            'entry_ts': open_position["entry_timestamp"], 'exit_ts': ts,
                            'entry_price': ep, 'exit_price': open_position["cur_sl_a"],
                            'pnl': net_pnl, 'r': r_mult, 'win': 0,
                            'exit_reason': 'SL_HIT'
                        })
                        trade_counter += 1
                        equity += net_pnl
                        open_position = None
                    else:
                        if open_position["pos_a_open"] and bar_high >= open_position["tp1_price"]:
                            pnl_a = (open_position["tp1_price"] - ep) * lot_a * CONTRACT_SIZE
                            open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                            open_position["pos_a_open"] = False
                            open_position["tp1_hit"] = True
                            f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                            open_position["cur_sl_b"] = max(open_position["cur_sl_b"], round(ep + f_p, 2))

                        if open_position["pos_b_open"]:
                            if bar_high >= open_position["tp2_price"]:
                                pnl_b = (open_position["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_pnl = round(open_position["pnl_net_accum"], 2)
                                r_mult = round(net_pnl / open_position["risk_amount"], 2)
                                trades.append({
                                    'id': trade_counter, 'day': day_str, 'direction': d,
                                    'entry_ts': open_position["entry_timestamp"], 'exit_ts': ts,
                                    'entry_price': ep, 'exit_price': open_position["tp2_price"],
                                    'pnl': net_pnl, 'r': r_mult, 'win': 1,
                                    'exit_reason': 'TP2_HIT (+2.5R)'
                                })
                                trade_counter += 1
                                equity += net_pnl
                                open_position = None
                            elif bar_low <= open_position["cur_sl_b"]:
                                pnl_b = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_pnl = round(open_position["pnl_net_accum"], 2)
                                r_mult = round(net_pnl / open_position["risk_amount"], 2)
                                trades.append({
                                    'id': trade_counter, 'day': day_str, 'direction': d,
                                    'entry_ts': open_position["entry_timestamp"], 'exit_ts': ts,
                                    'entry_price': ep, 'exit_price': open_position["cur_sl_b"],
                                    'pnl': net_pnl, 'r': r_mult, 'win': 1 if net_pnl > 0 else 0,
                                    'exit_reason': 'BE_HIT'
                                })
                                trade_counter += 1
                                equity += net_pnl
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (bar_high - ep) / sl_dist
                                if r_gain >= TRAIL_AFTER_R:
                                    trail_sl = round(bar_high - (TRAIL_DIST_R * sl_dist), 2)
                                    open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                else:  # SELL
                    if not open_position["tp1_hit"] and bar_high >= open_position["cur_sl_a"]:
                        pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * CONTRACT_SIZE
                        net_pnl = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                        r_mult = round(net_pnl / open_position["risk_amount"], 2)
                        trades.append({
                            'id': trade_counter, 'day': day_str, 'direction': d,
                            'entry_ts': open_position["entry_timestamp"], 'exit_ts': ts,
                            'entry_price': ep, 'exit_price': open_position["cur_sl_a"],
                            'pnl': net_pnl, 'r': r_mult, 'win': 0,
                            'exit_reason': 'SL_HIT'
                        })
                        trade_counter += 1
                        equity += net_pnl
                        open_position = None
                    else:
                        if open_position["pos_a_open"] and bar_low <= open_position["tp1_price"]:
                            pnl_a = (ep - open_position["tp1_price"]) * lot_a * CONTRACT_SIZE
                            open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                            open_position["pos_a_open"] = False
                            open_position["tp1_hit"] = True
                            f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                            open_position["cur_sl_b"] = min(open_position["cur_sl_b"], round(ep - f_p, 2))

                        if open_position["pos_b_open"]:
                            if bar_low <= open_position["tp2_price"]:
                                pnl_b = (ep - open_position["tp2_price"]) * lot_b * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_pnl = round(open_position["pnl_net_accum"], 2)
                                r_mult = round(net_pnl / open_position["risk_amount"], 2)
                                trades.append({
                                    'id': trade_counter, 'day': day_str, 'direction': d,
                                    'entry_ts': open_position["entry_timestamp"], 'exit_ts': ts,
                                    'entry_price': ep, 'exit_price': open_position["tp2_price"],
                                    'pnl': net_pnl, 'r': r_mult, 'win': 1,
                                    'exit_reason': 'TP2_HIT (+2.5R)'
                                })
                                trade_counter += 1
                                equity += net_pnl
                                open_position = None
                            elif bar_high >= open_position["cur_sl_b"]:
                                pnl_b = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_pnl = round(open_position["pnl_net_accum"], 2)
                                r_mult = round(net_pnl / open_position["risk_amount"], 2)
                                trades.append({
                                    'id': trade_counter, 'day': day_str, 'direction': d,
                                    'entry_ts': open_position["entry_timestamp"], 'exit_ts': ts,
                                    'entry_price': ep, 'exit_price': open_position["cur_sl_b"],
                                    'pnl': net_pnl, 'r': r_mult, 'win': 1 if net_pnl > 0 else 0,
                                    'exit_reason': 'BE_HIT'
                                })
                                trade_counter += 1
                                equity += net_pnl
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (ep - bar_low) / sl_dist
                                if r_gain >= TRAIL_AFTER_R:
                                    trail_sl = round(bar_low + (TRAIL_DIST_R * sl_dist), 2)
                                    open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)

        equity_curve.append(equity)

        # Check new entry:
        pred_idx = bar_idx - seq_offset
        if open_position is None and pred_idx >= 0 and pred_idx < len(preds):
            # How many trades already taken today?
            today_trades = sum(1 for t in trades if t['day'] == day_str)

            # Sinyal model
            probs = preds[pred_idx]
            trade_probs = probs[1:]
            max_class_idx = int(np.argmax(trade_probs))
            action_class = max_class_idx + 1
            confidence = float(trade_probs[max_class_idx])
            p_hold = float(probs[0])

            # Filter waktu: bukan Jumat malam
            friday_freeze = (dow == 4 and hour >= 18)
            # London quarantine
            london_quarantine = (hour == 7 or (hour == 8 and minute < 30))

            if not friday_freeze and not london_quarantine:
                # Dynamic threshold to ensure min 3 trades per day:
                # If today_trades < 3, threshold adapts to take the best setup available
                needed = min_trades_per_day - today_trades
                # Remaining bars today:
                remaining_bars = 48 - (hour * 2 + (1 if minute >= 30 else 0))

                # If needed > 0, we lower threshold proportionally to ensure min 3 trades/day
                if today_trades < min_trades_per_day:
                    active_tau = 0.26 if remaining_bars <= (needed * 4) else 0.30
                else:
                    active_tau = 0.355  # standard high conviction once 3 trades are reached

                if confidence >= active_tau and confidence > (p_hold * 0.85):
                    d = "BUY" if action_class in [1, 2] else "SELL"
                    atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
                    sl_dist = round(atr_val * SL_ATR_MULT, 2)
                    if sl_dist > 0:
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

                        open_position = {
                            "direction": d, "entry_bar_idx": bar_idx, "entry_timestamp": ts,
                            "entry_price": ep, "sl_dist": sl_dist, "initial_sl": sl_p,
                            "cur_sl_a": sl_p, "cur_sl_b": sl_p, "tp1_price": tp1_p, "tp2_price": tp2_p,
                            "total_lot": lot_size, "lot_a": lot_a, "lot_b": lot_b,
                            "risk_amount": actual_risk, "confidence": confidence,
                            "pos_a_open": True, "pos_b_open": True, "tp1_hit": False, "tp2_hit": False,
                            "pnl_net_accum": 0.0
                        }

    # Close any remaining position on Friday
    if open_position is not None:
        d = open_position["direction"]
        ep = open_position["entry_price"]
        exit_price = df.loc[week_indices[-1], "close"]
        if open_position["pos_a_open"]:
            pnl_a = (exit_price - ep) * open_position["lot_a"] * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * open_position["lot_a"] * CONTRACT_SIZE
            open_position["pnl_net_accum"] += (pnl_a - calc_friction(open_position["lot_a"]))
        if open_position["pos_b_open"]:
            pnl_b = (exit_price - ep) * open_position["lot_b"] * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * open_position["lot_b"] * CONTRACT_SIZE
            open_position["pnl_net_accum"] += (pnl_b - calc_friction(open_position["lot_b"]))

        net_pnl = round(open_position["pnl_net_accum"], 2)
        r_mult = round(net_pnl / open_position["risk_amount"], 2)
        trades.append({
            'id': trade_counter, 'day': open_position["entry_timestamp"].strftime('%Y-%m-%d'),
            'direction': d, 'entry_ts': open_position["entry_timestamp"], 'exit_ts': df.loc[week_indices[-1], "timestamp_utc"],
            'entry_price': ep, 'exit_price': exit_price,
            'pnl': net_pnl, 'r': r_mult, 'win': 1 if net_pnl > 0 else 0,
            'exit_reason': 'WEEKEND_CLOSE'
        })
        equity += net_pnl

    return pd.DataFrame(trades), equity - INITIAL_EQUITY, np.array(equity_curve)

tdf, pnl, eq_curve = run_simulation(min_trades_per_day=3)
print(f"\nHASIL SIMULASI 1 MINGGU (2-6 MARET 2026):")
print(f"Total Trade: {len(tdf)}")
print(f"Total Menang: {tdf['win'].sum()} | Kalah: {len(tdf) - tdf['win'].sum()}")
print(f"Win Rate: {tdf['win'].mean()*100:.1f}%")
print(f"Net PnL: ${pnl:+,.2f} ({pnl/10000*100:+.2f}%)")

peak = np.maximum.accumulate(eq_curve)
max_dd = np.min((eq_curve - peak) / peak) * 100
print(f"Max Drawdown: {max_dd:.2f}%")

print("\nRINCIAN PER HARI (SENIN - JUMAT):")
for day in days:
    sub = tdf[tdf['day'] == day]
    w = sub['win'].sum()
    print(f"  {day}: {len(sub)} Trade | Menang: {w} | Kalah: {len(sub)-w} | PnL: ${sub['pnl'].sum():+7.2f}")

print("\nDAFTAR SELURUH TRADE:")
for _, r in tdf.iterrows():
    print(f"  Trade #{r['id']+1:2d} | {r['day']} | {r['direction']:4s} | Entry: {str(r['entry_ts'])[11:16]} | Exit: {str(r['exit_ts'])[11:16]} | PnL: ${r['pnl']:+7.2f} ({r['r']:+4.1f}R) | {r['exit_reason']}")
