"""
Test Macro Valuation Regime Filter across 5 Years (2022 - 2026)
"""

import sys, os, pathlib, math
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


def run_5years(macro_filter=True, pos_a_pct=0.40, tp1_r=1.0, tp2_r=2.5, tau=0.34):
    years = [2022, 2023, 2024, 2025, 2026]
    n_bars = len(df)
    seq_offset = 63
    spread_price = 0.75 * 0.10
    slippage_price = 0.3 * 0.10
    commission = 3.50
    contract_size = 100.0

    def friction(lot):
        return round(spread_price * lot * contract_size + slippage_price * lot * contract_size * 2 + commission * lot, 2)

    results = []

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
                    trades.append({'month': m_str, 'pnl': net_p, 'win': 1 if net_p > 0 else 0})
                    equity += net_p
                    open_position = None
                else:
                    if d == "BUY":
                        if not open_position["tp1_hit"] and bar_low <= open_position["cur_sl_a"]:
                            pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * contract_size
                            net_p = round(pnl_g - friction(open_position["total_lot"]), 2)
                            m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                            trades.append({'month': m_str, 'pnl': net_p, 'win': 0})
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
                                    trades.append({'month': m_str, 'pnl': net_p, 'win': 1})
                                    equity += net_p
                                    open_position = None
                                elif bar_low <= open_position["cur_sl_b"]:
                                    pnl_b_g = (open_position["cur_sl_b"] - ep) * lot_b * contract_size
                                    open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                                    open_position["pos_b_open"] = False
                                    net_p = round(open_position["pnl_net_accum"], 2)
                                    m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                    trades.append({'month': m_str, 'pnl': net_p, 'win': 1 if net_p > 0 else 0})
                                    equity += net_p
                                    open_position = None
                                elif open_position["tp1_hit"]:
                                    r_gain = (bar_high - ep) / sl_dist
                                    if r_gain >= 1.5:
                                        trail_sl = round(bar_high - (0.8 * sl_dist), 2)
                                        open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                    else:
                        if not open_position["tp1_hit"] and bar_high >= open_position["cur_sl_a"]:
                            pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * contract_size
                            net_p = round(pnl_g - friction(open_position["total_lot"]), 2)
                            m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                            trades.append({'month': m_str, 'pnl': net_p, 'win': 0})
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
                                    trades.append({'month': m_str, 'pnl': net_p, 'win': 1})
                                    equity += net_p
                                    open_position = None
                                elif bar_high >= open_position["cur_sl_b"]:
                                    pnl_b_g = (ep - open_position["cur_sl_b"]) * lot_b * contract_size
                                    open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                                    open_position["pos_b_open"] = False
                                    net_p = round(open_position["pnl_net_accum"], 2)
                                    m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                    trades.append({'month': m_str, 'pnl': net_p, 'win': 1 if net_p > 0 else 0})
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

                    if confidence >= tau and confidence > p_hold:
                        d = "BUY" if action_class in [1, 2] else "SELL"
                        ob_zone = float(row.get("order_block_zone", 0.0))
                        ma_slope = float(row.get("ma_ribbon_slope", 0.0))
                        val_regime = float(row.get("valuation_regime", 0.0))

                        ok = True
                        if d == "BUY" and (ob_zone < 0 or ma_slope < -0.3): ok = False
                        if d == "SELL" and (ob_zone > 0 or ma_slope > 0.3): ok = False

                        # Macro Regime Alignment
                        if macro_filter:
                            if d == "SELL" and val_regime > 0.4: ok = False
                            if d == "BUY" and val_regime < -0.4: ok = False

                        if ok:
                            atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
                            sl_dist = round(atr_val * 1.5, 2)
                            if sl_dist > 0:
                                lot_size = max(0.02, min(round(math.floor((equity * 0.01) / (sl_dist * contract_size) / 0.01) * 0.01, 2), 50.0))
                                actual_risk = lot_size * sl_dist * contract_size
                                lot_a = max(0.01, round(lot_size * pos_a_pct, 2))
                                lot_b = max(0.01, round(lot_size - lot_a, 2))

                                if d == "BUY":
                                    ep = round(bar_close + slippage_price, 2)
                                    sl_p = round(ep - sl_dist, 2)
                                    tp1_p = round(ep + (tp1_r * sl_dist), 2)
                                    tp2_p = round(ep + (tp2_r * sl_dist), 2)
                                else:
                                    ep = round(bar_close - slippage_price, 2)
                                    sl_p = round(ep + sl_dist, 2)
                                    tp1_p = round(ep - (tp1_r * sl_dist), 2)
                                    tp2_p = round(ep - (tp2_r * sl_dist), 2)

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

        gm = 0
        if n_t > 0:
            for m in [f"{year}-{i:02d}" for i in range(1, 9)]:
                mp = tdf[tdf['month'] == m]['pnl'].sum() if len(tdf[tdf['month'] == m]) > 0 else 0
                if mp >= 0: gm += 1

        results.append({
            'Year': year, 'Trades': n_t, 'Wins': n_w, 'WR': round(wr, 1),
            'NetPnL': round(net_pnl, 2), 'ReturnPct': round(net_pnl / 100, 2),
            'MaxDD': round(max_dd, 2), 'GreenMonths': f"{gm}/8"
        })

    return pd.DataFrame(results)


print("--- COMPARISON: BASELINE vs WITH MACRO REGIME ALIGNMENT ---")
print("\n[A] BASELINE (Tanpa Macro Regime Filter):")
df_base = run_5years(macro_filter=False)
print(df_base.to_string(index=False))

print("\n[B] DENGAN MACRO REGIME FILTER (Do not short in Bull Macro / Do not buy in Bear Macro):")
df_macro = run_5years(macro_filter=True)
print(df_macro.to_string(index=False))
