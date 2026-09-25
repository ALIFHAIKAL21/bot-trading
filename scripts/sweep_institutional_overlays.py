"""
Sweep Institutional Overlays on Jan - Aug 2026 and Multi-Year
"""

import sys, os, pathlib, math
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
sys.path.insert(0, str(project_root / "src"))
from validation.backtest_engine import RealisticBacktestEngine

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


def run_sim(
    pos_a_pct=0.50,
    tp1_r=1.2,
    tp2_r=2.5,
    monthly_loss_cap=0.018,  # max 1.8% loss in a calendar month
    exclude_asian=True,
    tau=0.34,
    start_date="2026-01-01",
    end_date="2026-08-31 23:59:59"
):
    n_bars = len(df)
    seq_offset = 63
    equity = 10_000.0
    equity_curve = [equity]
    trades = []
    trade_counter = 0
    open_position = None

    current_month = None
    month_start_equity = equity
    month_halted = False

    st_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)

    spread_price = 0.75 * 0.10
    slippage_price = 0.3 * 0.10
    commission = 3.50
    contract_size = 100.0

    def friction(lot):
        return round(spread_price * lot * contract_size + slippage_price * lot * contract_size * 2 + commission * lot, 2)

    for bar_idx in range(n_bars):
        row = df.iloc[bar_idx]
        ts = row["timestamp_utc"]
        bar_open, bar_high, bar_low, bar_close = row["open"], row["high"], row["low"], row["close"]
        is_blackout = bool(row.get("is_news_blackout", False))
        is_eligible = bool(row.get("entry_eligible", True))

        in_date_range = (ts >= st_ts) and (ts <= end_ts)

        # Monthly tracking & monthly loss cap
        m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
        if m_str != current_month:
            current_month = m_str
            month_start_equity = equity
            month_halted = False

        if monthly_loss_cap is not None and not month_halted and month_start_equity > 0:
            m_dd = (equity - month_start_equity) / month_start_equity
            if m_dd <= -monthly_loss_cap:
                month_halted = True

        # Manage open position
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
                r_m = round(net_p / open_position["risk_amount"], 3)
                trades.append({'month': m_str, 'pnl': net_p, 'r': r_m, 'win': 1 if net_p > 0 else 0, 'exit': 'FRIDAY' if is_friday_close else 'TB'})
                equity += net_p
                open_position = None

            else:
                if d == "BUY":
                    if not open_position["tp1_hit"] and bar_low <= open_position["cur_sl_a"]:
                        pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * contract_size
                        net_p = round(pnl_g - friction(open_position["total_lot"]), 2)
                        r_m = round(net_p / open_position["risk_amount"], 3)
                        trades.append({'month': m_str, 'pnl': net_p, 'r': r_m, 'win': 0, 'exit': 'SL_HIT'})
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
                                r_m = round(net_p / open_position["risk_amount"], 3)
                                trades.append({'month': m_str, 'pnl': net_p, 'r': r_m, 'win': 1, 'exit': 'TP2_HIT'})
                                equity += net_p
                                open_position = None
                            elif bar_low <= open_position["cur_sl_b"]:
                                pnl_b_g = (open_position["cur_sl_b"] - ep) * lot_b * contract_size
                                open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                r_m = round(net_p / open_position["risk_amount"], 3)
                                trades.append({'month': m_str, 'pnl': net_p, 'r': r_m, 'win': 1 if net_p > 0 else 0, 'exit': 'BE_HIT'})
                                equity += net_p
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (bar_high - ep) / sl_dist
                                if r_gain >= 1.5:
                                    trail_sl = round(bar_high - (0.75 * sl_dist), 2)
                                    open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                else:  # SELL
                    if not open_position["tp1_hit"] and bar_high >= open_position["cur_sl_a"]:
                        pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * contract_size
                        net_p = round(pnl_g - friction(open_position["total_lot"]), 2)
                        r_m = round(net_p / open_position["risk_amount"], 3)
                        trades.append({'month': m_str, 'pnl': net_p, 'r': r_m, 'win': 0, 'exit': 'SL_HIT'})
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
                                r_m = round(net_p / open_position["risk_amount"], 3)
                                trades.append({'month': m_str, 'pnl': net_p, 'r': r_m, 'win': 1, 'exit': 'TP2_HIT'})
                                equity += net_p
                                open_position = None
                            elif bar_high >= open_position["cur_sl_b"]:
                                pnl_b_g = (ep - open_position["cur_sl_b"]) * lot_b * contract_size
                                open_position["pnl_net_accum"] += (pnl_b_g - friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                r_m = round(net_p / open_position["risk_amount"], 3)
                                trades.append({'month': m_str, 'pnl': net_p, 'r': r_m, 'win': 1 if net_p > 0 else 0, 'exit': 'BE_HIT'})
                                equity += net_p
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (ep - bar_low) / sl_dist
                                if r_gain >= 1.5:
                                    trail_sl = round(bar_low + (0.75 * sl_dist), 2)
                                    open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)

        if in_date_range:
            equity_curve.append(equity)

        # New entry
        pred_idx = bar_idx - seq_offset
        if in_date_range and pred_idx >= 0 and pred_idx < len(preds) and open_position is None and is_eligible and not is_blackout and not month_halted:
            dow = ts.dayofweek if hasattr(ts, 'dayofweek') else pd.Timestamp(ts).dayofweek
            hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour
            minute = ts.minute if hasattr(ts, 'minute') else pd.Timestamp(ts).minute

            friday_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
            london_quarantine = (hour == 7 or (hour == 8 and minute < 30))
            asian_filter = exclude_asian and (hour < 7 or hour >= 18)

            if not friday_freeze and not london_quarantine and not asian_filter:
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
                    ok = True
                    if d == "BUY" and (ob_zone < 0 or ma_slope < -0.3): ok = False
                    if d == "SELL" and (ob_zone > 0 or ma_slope > 0.3): ok = False

                    if ok:
                        atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
                        sl_dist = round(atr_val * 1.5, 2)
                        if sl_dist > 0:
                            # Dynamic sizing: 1.0% base, or scale with confidence
                            risk_frac = 0.012 if confidence >= 0.38 else 0.009
                            target_risk = equity * risk_frac
                            raw_lot = target_risk / (sl_dist * contract_size)
                            lot_size = max(0.02, min(round(math.floor(raw_lot / 0.01) * 0.01, 2), 50.0))
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
    return tdf, equity - 10000.0, max_dd


# Run comparison on 2026
print("Testing configurations on 2026:")
test_configs = [
    ("Default (Baseline)", dict(pos_a_pct=0.40, tp1_r=1.0, tp2_r=2.5, monthly_loss_cap=None, exclude_asian=False)),
    ("Overlay 1: Monthly DD Cap 1.8%", dict(pos_a_pct=0.40, tp1_r=1.0, tp2_r=2.5, monthly_loss_cap=0.018, exclude_asian=False)),
    ("Overlay 2: Asian Filter + DD Cap 1.8%", dict(pos_a_pct=0.40, tp1_r=1.0, tp2_r=2.5, monthly_loss_cap=0.018, exclude_asian=True)),
    ("Overlay 3: Pos A 50% @ 1.2R + Uncapped Trail + DD Cap 1.8%", dict(pos_a_pct=0.50, tp1_r=1.2, tp2_r=3.0, monthly_loss_cap=0.018, exclude_asian=True)),
    ("Overlay 4: Pos A 60% @ 1.1R + Uncapped Trail + DD Cap 1.8%", dict(pos_a_pct=0.60, tp1_r=1.1, tp2_r=3.0, monthly_loss_cap=0.018, exclude_asian=True)),
]

for name, cfg in test_configs:
    tdf, pnl, max_dd = run_sim(**cfg)
    n_t = len(tdf)
    n_w = tdf['win'].sum() if n_t > 0 else 0
    wr = (n_w / n_t * 100) if n_t > 0 else 0
    # count green months
    gm = 0
    m_pnl_str = []
    if n_t > 0:
        for m in sorted(tdf['month'].unique()):
            mp = tdf[tdf['month'] == m]['pnl'].sum()
            if mp >= 0: gm += 1
            m_pnl_str.append(f"{m[-2:]}:{mp:+.0f}")
    print(f"\n{name}")
    print(f"  Trades: {n_t}, WR: {wr:.1f}%, Net PnL: ${pnl:+,.2f} ({pnl/10000*100:+.2f}%), Max DD: {max_dd:.2f}%, Green Months: {gm}/8")
    print(f"  Monthly: {' | '.join(m_pnl_str)}")
