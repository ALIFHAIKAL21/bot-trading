"""
Deep Statistical Analysis: Monthly PnL, Profit Factor, Sharpe Ratio, and Drawdowns
Across all years (2021 - 2026) for tau = 0.380 and tau = 0.400
"""
import sys, pathlib, json
import numpy as np
import pandas as pd

from audit_15ch_model import (
    df, preds_15ch, atr, opens, highs, lows, closes, ts_arr, ob_zones, ma_slopes,
    dows, hours, minutes, day_ints, TAU, CONTRACT_SIZE, SL_ATR_MULT, BASE_RISK,
    INITIAL_EQUITY, POS_A_PCT, POS_B_PCT, TP1_R, TP2_R, TRAIL_AFTER_R, TRAIL_DIST_R,
    MAX_DAILY, SPREAD_PRICE, SLIPPAGE_PRICE, COMMISSION, calc_friction
)

def run_detailed_simulation(tau_val=0.380):
    seq_offset = 63
    n_bars = len(df)
    
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    open_position = None
    current_day = -1
    daily_count = 0
    
    for bar_idx in range(n_bars):
        ts = ts_arr[bar_idx]
        d_int = day_ints[bar_idx]
        
        if d_int != current_day:
            current_day = d_int
            daily_count = 0
            
        b_high = highs[bar_idx]
        b_low = lows[bar_idx]
        b_close = closes[bar_idx]
        dow = dows[bar_idx]
        hour = hours[bar_idx]
        minute = minutes[bar_idx]
        
        if open_position is not None:
            bars_held = bar_idx - open_position["entry_bar_idx"]
            d = open_position["direction"]
            ep = open_position["entry_price"]
            sl_dist = open_position["sl_dist"]
            lot_a = open_position["lot_a"]
            lot_b = open_position["lot_b"]
            
            is_fri_close = (dow == 4 and hour >= 20) or dow in [5, 6]
            is_tb = (bars_held >= 16)
            
            if is_fri_close or is_tb:
                exit_p = b_close
                pnl_a = (exit_p - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_a * CONTRACT_SIZE
                pnl_b = (exit_p - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_b * CONTRACT_SIZE
                tot = round((pnl_a - calc_friction(lot_a)) + (pnl_b - calc_friction(lot_b)), 2)
                trades.append({'bar': open_position['entry_bar_idx'], 'time': str(open_position['time'])[:16], 'year': int(str(open_position['time'])[:4]), 'month': str(open_position['time'])[:7], 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'TB/Fri'})
                equity += tot
                open_position = None
            else:
                if d == "BUY":
                    if not open_position["tp1_hit"] and b_low <= open_position["cur_sl_a"]:
                        pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                        tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                        trades.append({'bar': open_position['entry_bar_idx'], 'time': str(open_position['time'])[:16], 'year': int(str(open_position['time'])[:4]), 'month': str(open_position['time'])[:7], 'dir': d, 'win': 0, 'pnl': tot, 'exit': 'SL'})
                        equity += tot
                        open_position = None
                    else:
                        if open_position["pos_a_open"] and b_high >= open_position["tp1_price"]:
                            pnl_a = (open_position["tp1_price"] - ep) * lot_a * CONTRACT_SIZE
                            open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                            open_position["pos_a_open"] = False
                            open_position["tp1_hit"] = True
                            f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                            open_position["cur_sl_b"] = max(open_position["cur_sl_b"], round(ep + f_p, 2))
                        if open_position["pos_b_open"]:
                            if b_high >= open_position["tp2_price"]:
                                pnl_b = (open_position["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                                tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                trades.append({'bar': open_position['entry_bar_idx'], 'time': str(open_position['time'])[:16], 'year': int(str(open_position['time'])[:4]), 'month': str(open_position['time'])[:7], 'dir': d, 'win': 1, 'pnl': tot, 'exit': 'TP2'})
                                equity += tot
                                open_position = None
                            elif b_low <= open_position["cur_sl_b"]:
                                pnl_b = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                                tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                trades.append({'bar': open_position['entry_bar_idx'], 'time': str(open_position['time'])[:16], 'year': int(str(open_position['time'])[:4]), 'month': str(open_position['time'])[:7], 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'BE/Trail'})
                                equity += tot
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (b_high - ep) / sl_dist
                                if r_gain >= TRAIL_AFTER_R:
                                    trail_sl = round(b_high - (TRAIL_DIST_R * sl_dist), 2)
                                    open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                else: # SELL
                    if not open_position["tp1_hit"] and b_high >= open_position["cur_sl_a"]:
                        pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * CONTRACT_SIZE
                        tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                        trades.append({'bar': open_position['entry_bar_idx'], 'time': str(open_position['time'])[:16], 'year': int(str(open_position['time'])[:4]), 'month': str(open_position['time'])[:7], 'dir': d, 'win': 0, 'pnl': tot, 'exit': 'SL'})
                        equity += tot
                        open_position = None
                    else:
                        if open_position["pos_a_open"] and b_low <= open_position["tp1_price"]:
                            pnl_a = (ep - open_position["tp1_price"]) * lot_a * CONTRACT_SIZE
                            open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                            open_position["pos_a_open"] = False
                            open_position["tp1_hit"] = True
                            f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                            open_position["cur_sl_b"] = min(open_position["cur_sl_b"], round(ep - f_p, 2))
                        if open_position["pos_b_open"]:
                            if b_low <= open_position["tp2_price"]:
                                pnl_b = (ep - open_position["tp2_price"]) * lot_b * CONTRACT_SIZE
                                tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                trades.append({'bar': open_position['entry_bar_idx'], 'time': str(open_position['time'])[:16], 'year': int(str(open_position['time'])[:4]), 'month': str(open_position['time'])[:7], 'dir': d, 'win': 1, 'pnl': tot, 'exit': 'TP2'})
                                equity += tot
                                open_position = None
                            elif b_high >= open_position["cur_sl_b"]:
                                pnl_b = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                                tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                trades.append({'bar': open_position['entry_bar_idx'], 'time': str(open_position['time'])[:16], 'year': int(str(open_position['time'])[:4]), 'month': str(open_position['time'])[:7], 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'BE/Trail'})
                                equity += tot
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (ep - b_low) / sl_dist
                                if r_gain >= TRAIL_AFTER_R:
                                    trail_sl = round(b_low + (TRAIL_DIST_R * sl_dist), 2)
                                    open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)
        
        equity_curve.append(equity)
        pred_idx = bar_idx - seq_offset
        if pred_idx >= 0 and pred_idx < len(preds_15ch) and open_position is None and daily_count < MAX_DAILY:
            fri_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
            lon_quar = (hour == 7 or (hour == 8 and minute < 30))
            if not fri_freeze and not lon_quar:
                probs = preds_15ch[pred_idx]
                trade_probs = probs[1:]
                max_class_idx = int(np.argmax(trade_probs))
                action_class = max_class_idx + 1
                conf = float(trade_probs[max_class_idx])
                p_hold = float(probs[0])
                
                if conf >= tau_val and conf > p_hold:
                    d = "BUY" if action_class in [1, 2] else "SELL"
                    ob = ob_zones[bar_idx]
                    m_slp = ma_slopes[bar_idx]
                    
                    ok = True
                    if d == "BUY" and (ob < 0 or m_slp < -0.3): ok = False
                    if d == "SELL" and (ob > 0 or m_slp > 0.3): ok = False
                    
                    if ok:
                        atr_val = atr[bar_idx] if bar_idx < len(atr) else 5.0
                        sl_dist = round(atr_val * SL_ATR_MULT, 2)
                        if sl_dist > 0:
                            target_risk = equity * BASE_RISK
                            raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                            lot_size = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 50.0))
                            lot_a = max(0.01, round(lot_size * POS_A_PCT, 2))
                            lot_b = max(0.01, round(lot_size - lot_a, 2))
                            
                            if d == "BUY":
                                ep = round(b_close + SLIPPAGE_PRICE, 2)
                                sl_p = round(ep - sl_dist, 2)
                                tp1_p = round(ep + (TP1_R * sl_dist), 2)
                                tp2_p = round(ep + (TP2_R * sl_dist), 2)
                            else:
                                ep = round(b_close - SLIPPAGE_PRICE, 2)
                                sl_p = round(ep + sl_dist, 2)
                                tp1_p = round(ep - (TP1_R * sl_dist), 2)
                                tp2_p = round(ep - (TP2_R * sl_dist), 2)
                                
                            open_position = {
                                "direction": d, "entry_bar_idx": bar_idx, "time": ts,
                                "entry_price": ep, "sl_dist": sl_dist,
                                "cur_sl_a": sl_p, "cur_sl_b": sl_p, "tp1_price": tp1_p, "tp2_price": tp2_p,
                                "total_lot": lot_size, "lot_a": lot_a, "lot_b": lot_b,
                                "pos_a_open": True, "pos_b_open": True, "tp1_hit": False,
                                "pnl_net_accum": 0.0
                            }
                            daily_count += 1
                            
    tdf = pd.DataFrame(trades)
    return tdf

print("\n--- DEEP MONTHLY BREAKDOWN FOR TAU = 0.380 ---")
tdf_38 = run_detailed_simulation(tau_val=0.380)

for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
    sub = tdf_38[tdf_38['year'] == yr]
    print(f"\n================ YEAR {yr} (Total Trades: {len(sub)}) ================")
    for m, grp in sub.groupby('month'):
        pnl = grp['pnl'].sum()
        wr = grp['win'].mean() * 100
        n_b = len(grp[grp['dir'] == 'BUY'])
        n_s = len(grp[grp['dir'] == 'SELL'])
        print(f"  Month {m}: Trades={len(grp):2d} (B:{n_b:2d}, S:{n_s:2d}) | WR={wr:4.1f}% | Net PnL=${pnl:+8.2f}")
    
    # Calculate Profit Factor & Max DD for Year
    win_money = sub[sub['pnl'] > 0]['pnl'].sum()
    loss_money = abs(sub[sub['pnl'] < 0]['pnl'].sum())
    pf = (win_money / loss_money) if loss_money > 0 else 999.0
    print(f"  -> YEAR {yr} SUMMARY: Total PnL=${sub['pnl'].sum():+8.2f} | Profit Factor: {pf:.2f} | WR: {sub['win'].mean()*100:.1f}%")
