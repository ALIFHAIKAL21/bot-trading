"""
Full 5-Year Evaluation of Optimized Tier 3 Adaptive Engine (Tau=0.355 + Asymmetric Payoff)
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

years = [2022, 2023, 2024, 2025, 2026]
n_bars = len(df)
seq_offset = 63
spread_price = 0.75 * 0.10
slippage_price = 0.3 * 0.10
commission = 3.50
contract_size = 100.0

def friction(lot):
    return round(spread_price * lot * contract_size + slippage_price * lot * contract_size * 2 + commission * lot, 2)

annual_results = []
all_months = []

for year in years:
    equity = 10_000.0
    equity_curve = [equity]
    trades = []
    open_position = None

    st_ts = pd.to_datetime(f"{year}-01-01", utc=True)
    end_ts = pd.to_datetime(f"{year}-08-31 23:59:59", utc=True)

    for bar_idx in range(n_bars):
        row = df.iloc[bar_idx]
        ts = row["timestamp_utc"]
        bar_open, bar_high, bar_low, bar_close = row["open"], row["high"], row["low"], row["close"]
        is_blackout = bool(row.get("is_news_blackout", False))
        is_eligible = bool(row.get("entry_eligible", True))

        in_date_range = (ts >= st_ts) and (ts <= end_ts)

        if open_position is not None:
            bars_held = bar_idx - open_position["entry_bar_idx"]
            d = open_position["direction"]
            ep = open_position["entry_price"]
            sl_dist = open_position["sl_dist"]
            lot_a = open_position["lot_a"]
            lot_b = open_position["lot_b"]
            dow = ts.dayofweek if hasattr(ts, 'dayofweek') else pd.Timestamp(ts).dayofweek
            hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour

            is_friday_close = (dow == 4 and hour >= 20) or dow in [5, 6]
            is_time_barrier = (bars_held >= 16)

            if is_friday_close or is_time_barrier:
                exit_price = bar_close
                if open_position["pos_a_open"]:
                    pnl_a_g = (exit_price - ep) * lot_a * contract_size if d == "BUY" else (ep - exit_price) * lot_a * contract_size
                    open_position["pnl_net_accum"] += (pnl_a_g - friction(lot_a))
                    open_position["pos_a_open"] = False
                if open_position["pos_b_open"]:
                    pnl_b_g = (exit_price - ep) * lot_b * contract_size if d == "BUY" else (ep - exit_price) * lot_b * contract_size
                    open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                    open_position["pos_b_open"] = False

                net_p = round(open_position["pnl_net_accum"], 2)
                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                trades.append({'month': m_str, 'pnl': net_p, 'win': 1 if net_p > 0 else 0, 'exit': 'TIME_OR_FRIDAY'})
                equity += net_p
                open_position = None

            else:
                if d == "BUY":
                    if not open_position["tp1_hit"] and bar_low <= open_position["cur_sl_a"]:
                        pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * contract_size
                        net_p = round(pnl_g - friction(open_position["total_lot"]), 2)
                        m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                        trades.append({'month': m_str, 'pnl': net_p, 'win': 0, 'exit': 'SL_HIT'})
                        equity += net_p
                        open_position = None
                    else:
                        if open_position["pos_a_open"] and bar_high >= open_position["tp1_price"]:
                            pnl_a_g = (open_position["tp1_price"] - ep) * lot_a * contract_size
                            open_position["pnl_net_accum"] += (pnl_a_g - friction(lot_a))
                            open_position["pos_a_open"] = False
                            open_position["tp1_hit"] = True
                            f_p = spread_price + (commission / contract_size)
                            open_position["cur_sl_b"] = max(open_position["cur_sl_b"], round(ep + f_p, 2))

                        if open_position["pos_b_open"]:
                            if bar_high >= open_position["tp2_price"]:
                                pnl_b_g = (open_position["tp2_price"] - ep) * lot_b * contract_size
                                open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                trades.append({'month': m_str, 'pnl': net_p, 'win': 1, 'exit': 'TP2_HIT'})
                                equity += net_p
                                open_position = None
                            elif bar_low <= open_position["cur_sl_b"]:
                                pnl_b_g = (open_position["cur_sl_b"] - ep) * lot_b * contract_size
                                open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                trades.append({'month': m_str, 'pnl': net_p, 'win': 1 if net_p > 0 else 0, 'exit': 'BE_HIT'})
                                equity += net_p
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (bar_high - ep) / sl_dist
                                if r_gain >= 1.5:
                                    trail_sl = round(bar_high - (0.8 * sl_dist), 2)
                                    open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                else:  # SELL
                    if not open_position["tp1_hit"] and bar_high >= open_position["cur_sl_a"]:
                        pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * contract_size
                        net_p = round(pnl_g - friction(open_position["total_lot"]), 2)
                        m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                        trades.append({'month': m_str, 'pnl': net_p, 'win': 0, 'exit': 'SL_HIT'})
                        equity += net_p
                        open_position = None
                    else:
                        if open_position["pos_a_open"] and bar_low <= open_position["tp1_price"]:
                            pnl_a_g = (ep - open_position["tp1_price"]) * lot_a * contract_size
                            open_position["pnl_net_accum"] += (pnl_a_g - friction(lot_a))
                            open_position["pos_a_open"] = False
                            open_position["tp1_hit"] = True
                            f_p = spread_price + (commission / contract_size)
                            open_position["cur_sl_b"] = min(open_position["cur_sl_b"], round(ep - f_p, 2))

                        if open_position["pos_b_open"]:
                            if bar_low <= open_position["tp2_price"]:
                                pnl_b_g = (ep - open_position["tp2_price"]) * lot_b * contract_size
                                open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                trades.append({'month': m_str, 'pnl': net_p, 'win': 1, 'exit': 'TP2_HIT'})
                                equity += net_p
                                open_position = None
                            elif bar_high >= open_position["cur_sl_b"]:
                                pnl_b_g = (ep - open_position["cur_sl_b"]) * lot_b * contract_size
                                open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                trades.append({'month': m_str, 'pnl': net_p, 'win': 1 if net_p > 0 else 0, 'exit': 'BE_HIT'})
                                equity += net_p
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (ep - bar_low) / sl_dist
                                if r_gain >= 1.5:
                                    trail_sl = round(bar_low + (0.8 * sl_dist), 2)
                                    open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)

        if in_date_range:
            equity_curve.append(equity)

        pred_idx = bar_idx - seq_offset
        if in_date_range and pred_idx >= 0 and pred_idx < len(preds) and open_position is None and is_eligible and not is_blackout:
            dow = ts.dayofweek if hasattr(ts, 'dayofweek') else pd.Timestamp(ts).dayofweek
            hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour
            minute = ts.minute if hasattr(ts, 'minute') else pd.Timestamp(ts).minute

            friday_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
            london_quarantine = (hour == 7 or (hour == 8 and minute < 30))

            if not friday_freeze and not london_quarantine:
                probs = preds[pred_idx]
                trade_probs = probs[1:]
                max_class_idx = int(np.argmax(trade_probs))
                action_class = max_class_idx + 1
                confidence = float(trade_probs[max_class_idx])
                p_hold = float(probs[0])

                if confidence >= 0.355 and confidence > p_hold:
                    d = "BUY" if action_class in [1, 2] else "SELL"
                    ob_zone = float(row.get("order_block_zone", 0.0))
                    ma_slope = float(row.get("ma_ribbon_slope", 0.0))

                    ok = True
                    if d == "BUY" and (ob_zone < 0 or ma_slope < -0.3): ok = False
                    if d == "SELL" and (ob_zone > 0 or ma_slope > 0.3): ok = False

                    if ok:
                        atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
                        sl_dist = round(atr_val * 1.5, 2)
                        if sl_dist > 0:
                            target_risk = equity * 0.01
                            raw_lot = target_risk / (sl_dist * contract_size)
                            lot_size = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 50.0))
                            actual_risk = lot_size * sl_dist * contract_size

                            lot_a = max(0.01, round(lot_size * 0.40, 2))
                            lot_b = max(0.01, round(lot_size - lot_a, 2))

                            if d == "BUY":
                                ep = round(bar_close + slippage_price, 2)
                                sl_p = round(ep - sl_dist, 2)
                                tp1_p = round(ep + (1.0 * sl_dist), 2)
                                tp2_p = round(ep + (2.5 * sl_dist), 2)
                            else:
                                ep = round(bar_close - slippage_price, 2)
                                sl_p = round(ep + sl_dist, 2)
                                tp1_p = round(ep - (1.0 * sl_dist), 2)
                                tp2_p = round(ep - (2.5 * sl_dist), 2)

                            open_position = {
                                "direction": d, "entry_bar_idx": bar_idx, "entry_timestamp": ts,
                                "entry_price": ep, "sl_dist": sl_dist, "initial_sl": sl_p,
                                "cur_sl_a": sl_p, "cur_sl_b": sl_p, "tp1_price": tp1_p, "tp2_price": tp2_p,
                                "total_lot": lot_size, "lot_a": lot_a, "lot_b": lot_b,
                                "risk_amount": actual_risk, "confidence": confidence,
                                "pos_a_open": True, "pos_b_open": True, "tp1_hit": False, "tp2_hit": False,
                                "pnl_net_accum": 0.0
                            }

    tdf = pd.DataFrame(trades)
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = np.min((eq_arr - peak) / peak) * 100 if len(eq_arr) > 0 else 0
    n_t = len(tdf)
    n_w = tdf['win'].sum() if n_t > 0 else 0
    wr = (n_w / n_t * 100) if n_t > 0 else 0
    net_pnl = equity - 10_000.0

    cur_eq = 10_000.0
    green_m = 0
    for m in [f"{year}-{i:02d}" for i in range(1, 9)]:
        sub = tdf[tdf['month'] == m] if n_t > 0 else pd.DataFrame()
        sub_n = len(sub)
        if sub_n > 0:
            sw = sub['win'].sum()
            spnl = sub['pnl'].sum()
            spct = spnl / cur_eq * 100
            cur_eq += spnl
            if spnl >= 0: green_m += 1
            all_months.append({
                'Year': year, 'Month': m, 'Trades': sub_n, 'Wins': sw,
                'WinRate': round(sw / sub_n * 100, 1), 'NetPnL': round(spnl, 2),
                'ProfitPct': round(spct, 2)
            })
        else:
            all_months.append({
                'Year': year, 'Month': m, 'Trades': 0, 'Wins': 0,
                'WinRate': 0.0, 'NetPnL': 0.0, 'ProfitPct': 0.0
            })

    annual_results.append({
        'Year': f"Jan - Agu {year}", 'Trades': n_t, 'Wins': n_w, 'Losses': n_t - n_w,
        'WinRatePct': round(wr, 1), 'NetPnL': round(net_pnl, 2),
        'ReturnPct': round(net_pnl / 100, 2), 'MaxDrawdownPct': round(max_dd, 2),
        'GreenMonths': f"{green_m}/8"
    })

print("\n=== OPTIMIZED AUDIT SCORECARD (TAU=0.355) ===")
summary_df = pd.DataFrame(annual_results)
print(summary_df.to_string(index=False))

total_pnl = sum(r['NetPnL'] for r in annual_results)
total_trades = sum(r['Trades'] for r in annual_results)
total_wins = sum(r['Wins'] for r in annual_results)
print(f"\nTOTAL 5 TAHUN: Trades={total_trades}, Wins={total_wins}, WR={total_wins/total_trades*100:.1f}%, NetPnL=+${total_pnl:,.2f} (+{total_pnl/10000*100:.2f}%)")

# Monthly details
mdf = pd.DataFrame(all_months)
print("\n=== MONTHLY BREAKDOWN SAMPLE (2025 & 2026) ===")
print(mdf[mdf['Year'].isin([2025, 2026])].to_string(index=False))
