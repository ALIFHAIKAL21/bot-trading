"""
Simulasi & Stress Test Modal $200 USD (Januari - Maret 2026)
Menguji berbagai skenario:
1. Akun Standar $200 USD dengan 3 trade/hari (apakah margin call di Januari?)
2. Akun Standar $200 USD dengan Sniper Filter (High Conviction Tau >= 0.40, HTF Filter, dll)
3. Akun Standar $200 USD dengan Tight SL (1.0 ATR) vs Normal SL (1.5 ATR)
4. Dynamic Micro Compounding (naik lot bertahap saat saldo tumbuh)
5. Akun Cent (20,000 Cent) untuk ketahanan institusional maksimal
"""

import sys, os, pathlib, json
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# ATR(14)
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

START_DATE = "2026-01-01 00:00:00"
END_DATE = "2026-03-31 23:59:59"
INITIAL_EQUITY = 200.0  # Modal $200 USD

sub_mask = (df['timestamp_utc'] >= START_DATE) & (df['timestamp_utc'] <= END_DATE)
sub_indices = df[sub_mask].index
days = sorted(df.loc[sub_indices, 'timestamp_utc'].dt.strftime('%Y-%m-%d').unique())
seq_offset = 63

def run_simulation(
    mode_name="test",
    selection_mode="3trades_daily", # "3trades_daily" or "sniper_tau"
    tau_threshold=0.34,
    sl_atr_mult=1.5,
    tp1_r=1.0,
    tp2_r=2.5,
    lot_mode="fixed_001", # "fixed_001", "fixed_002", "compounding_step", "cent_account"
    scratch_exit=False,
    structural_filter=False,
    friction_mult=1.0
):
    spread_price = 0.75 * 0.10 * friction_mult
    slippage_price = 0.3 * 0.10 * friction_mult
    commission = 3.50 * friction_mult
    contract_size = 100.0

    def calc_friction(lot):
        return round(spread_price * lot * contract_size + slippage_price * lot * contract_size * 2 + commission * lot, 2)

    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    trade_counter = 0
    margin_called = False
    min_equity = equity

    session_ranges = [
        (1, 7, "Sesi Pagi (Asia)"),
        (8, 13, "Sesi Siang (London)"),
        (14, 20, "Sesi Sore (New York)")
    ]

    for day_str in days:
        if margin_called:
            break

        day_mask = (df['timestamp_utc'].dt.strftime('%Y-%m-%d') == day_str) & sub_mask
        day_indices = df[day_mask].index
        day_ts = df.loc[day_indices[0], 'timestamp_utc']
        if day_ts.dayofweek in [5, 6]:
            continue

        selected_candidates = []

        if selection_mode == "3trades_daily":
            # Select best bar per session
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
                        if 0 <= p_idx < len(preds):
                            probs = preds[p_idx]
                            t_probs = probs[1:]
                            max_idx = int(np.argmax(t_probs))
                            conf = float(t_probs[max_idx])
                            if conf > best_conf:
                                best_conf = conf
                                best_bar_idx = b_idx
                                best_action = max_idx + 1

                if best_bar_idx is not None and best_conf >= tau_threshold:
                    selected_candidates.append((best_bar_idx, best_action, best_conf, s_name))

        elif selection_mode == "sniper_tau":
            # Event-driven: whenever confidence >= tau_threshold
            for b_idx in day_indices:
                row = df.iloc[b_idx]
                h = row['timestamp_utc'].hour
                m = row['timestamp_utc'].minute
                if h == 7 or (h == 8 and m < 30): # London quarantine
                    continue
                p_idx = b_idx - seq_offset
                if 0 <= p_idx < len(preds):
                    probs = preds[p_idx]
                    t_probs = probs[1:]
                    max_idx = int(np.argmax(t_probs))
                    conf = float(t_probs[max_idx])
                    p_hold = float(probs[0])
                    if conf >= tau_threshold and conf > p_hold:
                        action = max_idx + 1
                        d = "BUY" if action in [1, 2] else "SELL"
                        
                        if structural_filter:
                            ob_zone = float(row.get("order_block_zone", 0.0))
                            ma_slope = float(row.get("ma_ribbon_slope", 0.0))
                            if d == "BUY" and (ob_zone < 0 or ma_slope < -0.3):
                                continue
                            if d == "SELL" and (ob_zone > 0 or ma_slope > 0.3):
                                continue

                        s_name = "Asia" if h < 8 else ("London" if h < 14 else "NY")
                        selected_candidates.append((b_idx, action, conf, s_name))
                        if len(selected_candidates) >= 5: # max 5/day
                            break

        # Execute candidate trades
        for bar_idx, action, conf, s_name in selected_candidates:
            if equity <= 20.0:  # Margin Call threshold on $200 account
                margin_called = True
                break

            entry_row = df.iloc[bar_idx]
            entry_ts = entry_row["timestamp_utc"]
            bar_close = entry_row["close"]
            d = "BUY" if action in [1, 2] else "SELL"

            atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
            sl_dist = round(atr_val * sl_atr_mult, 2)

            # Determine lot size
            if lot_mode == "fixed_001":
                lot_a = 0.01
                lot_b = 0.0
                total_lot = 0.01
            elif lot_mode == "fixed_002":
                lot_a = 0.01
                lot_b = 0.01
                total_lot = 0.02
            elif lot_mode == "compounding_step":
                # Stepwise scaling:
                # $200 - $399 -> 0.01 lot
                # $400 - $699 -> 0.02 lot
                # $700 - $999 -> 0.03 lot
                # $1,000 - $1,499 -> 0.05 lot
                # $1,500+ -> 0.07 lot
                if equity < 400.0:
                    total_lot = 0.01
                elif equity < 700.0:
                    total_lot = 0.02
                elif equity < 1000.0:
                    total_lot = 0.03
                elif equity < 1500.0:
                    total_lot = 0.05
                else:
                    total_lot = min(round(equity / 30000.0, 2), 0.50)
                lot_a = max(0.01, round(total_lot * 0.5, 2)) if total_lot >= 0.02 else total_lot
                lot_b = round(total_lot - lot_a, 2) if total_lot >= 0.02 else 0.0
            elif lot_mode == "cent_proportional":
                # Proportional 1% risk simulating cent account where lot can be 0.001 standard equivalent
                target_risk = equity * 0.01
                raw_lot = target_risk / (sl_dist * contract_size)
                total_lot = max(0.005, round(raw_lot, 3))
                lot_a = round(total_lot * 0.5, 3)
                lot_b = round(total_lot - lot_a, 3)

            actual_risk = total_lot * sl_dist * contract_size

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

            cur_sl_a = sl_p
            cur_sl_b = sl_p
            pos_a_open = True
            pos_b_open = True if lot_b > 0 else False
            tp1_hit = False
            pnl_accum = 0.0
            exit_ts = None
            exit_reason = None

            for f_idx in range(bar_idx + 1, min(bar_idx + 16, len(df))):
                f_row = df.iloc[f_idx]
                f_ts = f_row["timestamp_utc"]
                f_high = f_row["high"]
                f_low = f_row["low"]
                f_close = f_row["close"]
                f_dow = f_ts.dayofweek
                f_hour = f_ts.hour

                is_fri_close = (f_dow == 4 and f_hour >= 20)
                is_tb = (f_idx - bar_idx >= 12)
                is_scratch = scratch_exit and (f_idx - bar_idx >= 5) and not tp1_hit

                if is_fri_close or is_tb or is_scratch:
                    exit_reason = "Tutup Jumat" if is_fri_close else ("Scratch Exit (Stagnan)" if is_scratch else "Time Barrier")
                    exit_ts = f_ts
                    exit_price = f_close
                    if pos_a_open:
                        pnl_a = (exit_price - ep) * lot_a * contract_size if d == "BUY" else (ep - exit_price) * lot_a * contract_size
                        pnl_accum += (pnl_a - calc_friction(lot_a))
                        pos_a_open = False
                    if pos_b_open:
                        pnl_b = (exit_price - ep) * lot_b * contract_size if d == "BUY" else (ep - exit_price) * lot_b * contract_size
                        pnl_accum += (pnl_b - calc_friction(lot_b))
                        pos_b_open = False
                    break
                else:
                    if d == "BUY":
                        if not tp1_hit and f_low <= cur_sl_a:
                            pnl_g = (cur_sl_a - ep) * total_lot * contract_size
                            pnl_accum = round(pnl_g - calc_friction(total_lot), 2)
                            exit_ts = f_ts
                            exit_reason = "Stop Loss (-1.0R)"
                            pos_a_open = False
                            pos_b_open = False
                            break
                        else:
                            if pos_a_open and f_high >= tp1_p:
                                pnl_a = (tp1_p - ep) * lot_a * contract_size
                                pnl_accum += (pnl_a - calc_friction(lot_a))
                                pos_a_open = False
                                tp1_hit = True
                                f_p = spread_price + (commission / contract_size)
                                cur_sl_b = max(cur_sl_b, round(ep + f_p, 2))

                            if pos_b_open:
                                if f_high >= tp2_p:
                                    pnl_b = (tp2_p - ep) * lot_b * contract_size
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    pos_b_open = False
                                    exit_ts = f_ts
                                    exit_reason = "Take Profit (+2.5R)"
                                    break
                                elif f_low <= cur_sl_b:
                                    pnl_b = (cur_sl_b - ep) * lot_b * contract_size
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    pos_b_open = False
                                    exit_ts = f_ts
                                    exit_reason = "Breakeven" if tp1_hit else "Stop Loss"
                                    break
                            elif lot_b == 0 and tp1_hit:
                                exit_ts = f_ts
                                exit_reason = "Take Profit (+1.0R)"
                                break
                    else:  # SELL
                        if not tp1_hit and f_high >= cur_sl_a:
                            pnl_g = (ep - cur_sl_a) * total_lot * contract_size
                            pnl_accum = round(pnl_g - calc_friction(total_lot), 2)
                            exit_ts = f_ts
                            exit_reason = "Stop Loss (-1.0R)"
                            pos_a_open = False
                            pos_b_open = False
                            break
                        else:
                            if pos_a_open and f_low <= tp1_p:
                                pnl_a = (ep - tp1_p) * lot_a * contract_size
                                pnl_accum += (pnl_a - calc_friction(lot_a))
                                pos_a_open = False
                                tp1_hit = True
                                f_p = spread_price + (commission / contract_size)
                                cur_sl_b = min(cur_sl_b, round(ep - f_p, 2))

                            if pos_b_open:
                                if f_low <= tp2_p:
                                    pnl_b = (ep - tp2_p) * lot_b * contract_size
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    pos_b_open = False
                                    exit_ts = f_ts
                                    exit_reason = "Take Profit (+2.5R)"
                                    break
                                elif f_high >= cur_sl_b:
                                    pnl_b = (ep - cur_sl_b) * lot_b * contract_size
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    pos_b_open = False
                                    exit_ts = f_ts
                                    exit_reason = "Breakeven" if tp1_hit else "Stop Loss"
                                    break
                            elif lot_b == 0 and tp1_hit:
                                exit_ts = f_ts
                                exit_reason = "Take Profit (+1.0R)"
                                break

            pnl_final = round(pnl_accum, 2)
            trades.append({
                'id': trade_counter + 1,
                'month': entry_ts.strftime('%Y-%m'),
                'day': day_str,
                'session': s_name,
                'direction': d,
                'entry_ts': entry_ts,
                'lot': total_lot,
                'pnl': pnl_final,
                'win': 1 if pnl_final > 0 else 0,
                'exit_reason': exit_reason if exit_reason else "Selesai Sesi"
            })
            trade_counter += 1
            equity += pnl_final
            equity_curve.append(equity)
            if equity < min_equity:
                min_equity = equity

    tdf = pd.DataFrame(trades)
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = np.min((eq_arr - peak) / peak) * 100 if len(eq_arr) > 0 else 0.0

    return {
        "name": mode_name,
        "final_equity": round(equity, 2),
        "min_equity": round(min_equity, 2),
        "net_pnl": round(equity - INITIAL_EQUITY, 2),
        "return_pct": round((equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100, 2),
        "max_dd_pct": round(max_dd, 2),
        "total_trades": len(tdf),
        "wins": int(tdf['win'].sum()) if len(tdf) > 0 else 0,
        "win_rate": round(tdf['win'].sum() / len(tdf) * 100, 2) if len(tdf) > 0 else 0.0,
        "margin_called": margin_called,
        "tdf": tdf,
        "equity_curve": eq_arr
    }

# RUN SUITE OF CONFIGURATIONS
experiments = [
    # 1. Brutal Reality: 0.02 Lot Twin-Order on $200 (Standard broker)
    ("1. 3T/Day Standar 0.02 Lot (Twin-Order)", "3trades_daily", 0.0, 1.5, 1.0, 2.5, "fixed_002", False, False, 1.0),
    
    # 2. Conservative Standard: 0.01 Lot Flat (Single Order, TP 1.0R)
    ("2. 3T/Day Standar 0.01 Lot Flat", "3trades_daily", 0.0, 1.5, 1.0, 2.5, "fixed_001", False, False, 1.0),
    
    # 3. 0.01 Lot Flat with Tight Stop (1.0 ATR)
    ("3. 3T/Day 0.01 Lot + Tight SL (1.0 ATR)", "3trades_daily", 0.0, 1.0, 1.0, 2.0, "fixed_001", False, False, 1.0),

    # 4. Sniper Tau Filter (Tau >= 0.40, only trade high conviction)
    ("4. Sniper Filter (Tau>=0.40) 0.01 Lot", "sniper_tau", 0.40, 1.5, 1.0, 2.5, "fixed_001", False, False, 1.0),

    # 5. Sniper Filter + Structural Gate (Order Block & Ribbon Slope)
    ("5. Sniper Filter (Tau>=0.38) + Structural OB/MA Gate", "sniper_tau", 0.38, 1.5, 1.0, 2.5, "fixed_001", False, True, 1.0),

    # 6. Stepwise Micro Compounding (Starts 0.01 Lot, scales up as profits accumulate)
    ("6. Sniper Stepwise Compounding (0.01 -> 0.02 -> 0.03)", "sniper_tau", 0.38, 1.5, 1.0, 2.5, "compounding_step", False, True, 1.0),

    # 7. Stress Test Friction: 2x Spread & Slippage Shock on Best Sniper
    ("7. STRESS TEST: 2x Spread & Slippage Shock", "sniper_tau", 0.38, 1.5, 1.0, 2.5, "compounding_step", False, True, 2.0),

    # 8. Akun Cent Equivalent (1% Proportional Risk)
    ("8. Akun Cent (Mikro Proportional 1% Risk)", "3trades_daily", 0.0, 1.5, 1.0, 2.5, "cent_proportional", False, False, 1.0),
]

results = []
print("Menjalankan Simulasi Stress Test Modal $200 USD (Q1 2026: Jan - Mar)...")
for exp in experiments:
    name, sel_m, tau, sl_m, tp1, tp2, lot_m, scr, struct, fric = exp
    res = run_simulation(
        mode_name=name,
        selection_mode=sel_m,
        tau_threshold=tau,
        sl_atr_mult=sl_m,
        tp1_r=tp1,
        tp2_r=tp2,
        lot_mode=lot_m,
        scratch_exit=scr,
        structural_filter=struct,
        friction_mult=fric
    )
    status_str = "MARGIN CALL / BLOWN" if res["margin_called"] else ("SURVIVED" if res["min_equity"] > 20 else "NEAR DEATH")
    print(f"\n[{res['name']}]")
    print(f"  Saldo Akhir: ${res['final_equity']:,.2f} | Net PnL: ${res['net_pnl']:+,.2f} ({res['return_pct']:+.1f}%)")
    print(f"  Min Saldo: ${res['min_equity']:,.2f} | Max DD: {res['max_dd_pct']:.1f}% | Status: {status_str}")
    print(f"  Total Trades: {res['total_trades']} | Win Rate: {res['win_rate']:.1f}% ({res['wins']} Wins)")
    
    # Monthly PnL
    tdf = res["tdf"]
    if len(tdf) > 0:
        for m in ['2026-01', '2026-02', '2026-03']:
            sub = tdf[tdf['month'] == m]
            spnl = sub['pnl'].sum() if len(sub) > 0 else 0.0
            print(f"    {m}: PnL ${spnl:+,.2f} ({len(sub)} trades)")
    
    results.append({
        "name": res["name"],
        "final_equity": res["final_equity"],
        "min_equity": res["min_equity"],
        "net_pnl": res["net_pnl"],
        "return_pct": res["return_pct"],
        "max_dd_pct": res["max_dd_pct"],
        "total_trades": res["total_trades"],
        "win_rate": res["win_rate"],
        "margin_called": res["margin_called"]
    })

# Save results
out_json = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\stress_test_modal_200_q1_2026.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSemua pengujian selesai! Hasil tersimpan di {out_json}")
