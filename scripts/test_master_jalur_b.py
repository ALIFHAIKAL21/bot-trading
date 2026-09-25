import sys, pathlib
import numpy as np
import pandas as pd

from test_jalur_b_proper import (
    df, preds, atr, opens, highs, lows, closes, ts_arr, ob_zones, ma_slopes,
    h4_slp20, h4_slp50, h4_aths, h4_e20s, h4_e50s, h4_e200s,
    dows, hours, minutes, day_ints, test_years, TAU, CONTRACT_SIZE,
    SL_ATR_MULT, BASE_RISK, INITIAL_EQUITY, POS_A_PCT, POS_B_PCT,
    TP1_R, TP2_R, TRAIL_AFTER_R, TRAIL_DIST_R, MAX_DAILY, SPREAD_PRICE, SLIPPAGE_PRICE, COMMISSION, calc_friction
)

def run_master_test(convert_parabolic=True, trail_runner=True, min_slope=0.15):
    seq_offset = 63
    n_bars = len(df)
    results = {}
    
    for y_cfg in test_years:
        year = y_cfg["year"]
        st = y_cfg["start_dt"]
        en = y_cfg["end_dt"]
        
        equity = INITIAL_EQUITY
        equity_curve = [equity]
        trades = []
        open_position = None
        current_day = -1
        daily_count = 0
        
        for bar_idx in range(n_bars):
            ts = ts_arr[bar_idx]
            in_range = (ts >= st) and (ts <= en)
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
                is_runner = open_position.get("is_runner", False)
                
                is_fri_close = (dow == 4 and hour >= 20) or dow in [5, 6]
                is_tb = (bars_held >= 24 if is_runner else bars_held >= 16)
                
                if is_fri_close or is_tb:
                    exit_p = b_close
                    pnl_a = (exit_p - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_a * CONTRACT_SIZE
                    pnl_b = (exit_p - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_b * CONTRACT_SIZE
                    tot = round((pnl_a - calc_friction(lot_a)) + (pnl_b - calc_friction(lot_b)), 2)
                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot, 'direction': d})
                    equity += tot
                    open_position = None
                else:
                    if d == "BUY":
                        if not open_position["tp1_hit"] and b_low <= open_position["cur_sl_a"]:
                            pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                            tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                            trades.append({'win': 0, 'pnl': tot, 'direction': d})
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
                                # If runner, allow TP2 to stretch to 4.0R!
                                target_tp2 = round(ep + (4.0 * sl_dist), 2) if is_runner else open_position["tp2_price"]
                                
                                if b_high >= target_tp2:
                                    pnl_b = (target_tp2 - ep) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1, 'pnl': tot, 'direction': d})
                                    equity += tot
                                    open_position = None
                                elif b_low <= open_position["cur_sl_b"]:
                                    pnl_b = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot, 'direction': d})
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
                            trades.append({'win': 0, 'pnl': tot, 'direction': d})
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
                                target_tp2 = round(ep - (4.0 * sl_dist), 2) if is_runner else open_position["tp2_price"]
                                
                                if b_low <= target_tp2:
                                    pnl_b = (ep - target_tp2) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1, 'pnl': tot, 'direction': d})
                                    equity += tot
                                    open_position = None
                                elif b_high >= open_position["cur_sl_b"]:
                                    pnl_b = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot, 'direction': d})
                                    equity += tot
                                    open_position = None
                                elif open_position["tp1_hit"]:
                                    r_gain = (ep - b_low) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(b_low + (TRAIL_DIST_R * sl_dist), 2)
                                        open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)
            
            if in_range:
                equity_curve.append(equity)
                
            pred_idx = bar_idx - seq_offset
            if in_range and pred_idx >= 0 and pred_idx < len(preds) and open_position is None and daily_count < MAX_DAILY:
                fri_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
                lon_quar = (hour == 7 or (hour == 8 and minute < 30))
                if not fri_freeze and not lon_quar:
                    probs = preds[pred_idx]
                    trade_probs = probs[1:]
                    max_class_idx = int(np.argmax(trade_probs))
                    action_class = max_class_idx + 1
                    conf = float(trade_probs[max_class_idx])
                    p_hold = float(probs[0])
                    
                    if conf >= TAU and conf > p_hold:
                        d = "BUY" if action_class in [1, 2] else "SELL"
                        ob = ob_zones[bar_idx]
                        m_slp = ma_slopes[bar_idx]
                        
                        ok = True
                        if d == "BUY" and (ob < 0 or m_slp < -0.3): ok = False
                        if d == "SELL" and (ob > 0 or m_slp > 0.3): ok = False
                        
                        if ok:
                            s20 = h4_slp20[bar_idx]
                            s50 = h4_slp50[bar_idx]
                            ath = bool(h4_aths[bar_idx])
                            c = closes[bar_idx]
                            e50 = h4_e50s[bar_idx]
                            
                            is_parabolic_bull = (s20 > min_slope) and ath
                            is_parabolic_bear = (s20 < -min_slope)
                            is_runner = False
                            
                            if is_parabolic_bull:
                                if d == "SELL":
                                    if convert_parabolic:
                                        d = "BUY" # Convert momentum to BUY!
                                        is_runner = True
                                else:
                                    is_runner = True
                            elif is_parabolic_bear:
                                if d == "BUY":
                                    if convert_parabolic:
                                        d = "SELL"
                                        is_runner = True
                                else:
                                    is_runner = True
                            elif (d == "BUY" and s50 > 0.1 and c > e50) or (d == "SELL" and s50 < -0.1 and c < e50):
                                is_runner = True
                                
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
                                    "direction": d, "entry_bar_idx": bar_idx,
                                    "entry_price": ep, "sl_dist": sl_dist,
                                    "cur_sl_a": sl_p, "cur_sl_b": sl_p, "tp1_price": tp1_p, "tp2_price": tp2_p,
                                    "total_lot": lot_size, "lot_a": lot_a, "lot_b": lot_b,
                                    "pos_a_open": True, "pos_b_open": True, "tp1_hit": False,
                                    "pnl_net_accum": 0.0, "is_runner": is_runner
                                }
                                daily_count += 1
                                
        tdf = pd.DataFrame(trades)
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        max_dd = np.min((eq_arr - peak) / peak) * 100 if len(eq_arr) > 0 else 0
        n_tr = len(tdf)
        n_w = tdf['win'].sum() if n_tr > 0 else 0
        wr = (n_w / n_tr * 100) if n_tr > 0 else 0.0
        ret = ((equity - INITIAL_EQUITY) / INITIAL_EQUITY) * 100
        results[year] = {"trades": n_tr, "ret": round(ret, 2), "max_dd": round(max_dd, 2), "wr": round(wr, 1)}
        
    return results

print("\n--- MASTER JALUR B: Slope > 0.15 ---")
res_15 = run_master_test(convert_parabolic=True, trail_runner=True, min_slope=0.15)
for y, v in res_15.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")

print("\n--- MASTER JALUR B: Slope > 0.20 ---")
res_20 = run_master_test(convert_parabolic=True, trail_runner=True, min_slope=0.20)
for y, v in res_20.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")

print("\n--- MASTER JALUR B: Slope > 0.10 ---")
res_10 = run_master_test(convert_parabolic=True, trail_runner=True, min_slope=0.10)
for y, v in res_10.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")
