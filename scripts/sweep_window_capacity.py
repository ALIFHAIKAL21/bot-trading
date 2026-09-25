"""
Test the Mathematical Tipping Point of Session Selectivity
How many high-quality trades can M30 realistically provide per day before alpha turns into noise?
"""
import sys, pathlib
import numpy as np
import pandas as pd

from fast_session_picker import df, preds, atr, opens, highs, lows, closes, ts_arr, dows, hours, minutes, day_ints, unique_days, day_to_indices, seq_offset, calc_friction, INITIAL_EQUITY, SPREAD_PRICE, SLIPPAGE_PRICE, COMMISSION, CONTRACT_SIZE, SL_ATR_MULT, TP1_R, TP2_R, TRAIL_AFTER_R, TRAIL_DIST_R

def run_n_window_test(n_windows=5, tau=0.32):
    # Divide 24h into n equal windows outside quarantine
    # Trading hours: 01:00 to 21:00 (20 hours total = 1200 minutes)
    total_mins = 20 * 60
    window_duration = total_mins // n_windows
    
    session_defs = []
    curr = 1 * 60 # start 01:00
    for w_i in range(n_windows):
        w_st = curr
        w_en = curr + window_duration
        session_defs.append((w_st, w_en, f"Win_{w_i+1}"))
        curr = w_en
        
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    
    for d_int in unique_days:
        day_idxs = day_to_indices[d_int]
        dow = dows[day_idxs[0]]
        if dow in [5, 6]:
            continue
            
        for w_st, w_en, s_name in session_defs:
            best_idx = None
            best_conf = -1.0
            best_act = None
            
            for b_idx in day_idxs:
                h = hours[b_idx]
                m = minutes[b_idx]
                if h == 7 or (h == 8 and m < 30):
                    continue
                if dow == 4 and h >= 18:
                    continue
                    
                t_val = h * 60 + m
                if w_st <= t_val <= w_en:
                    p_idx = b_idx - seq_offset
                    if 0 <= p_idx < len(preds):
                        probs = preds[p_idx]
                        t_probs = probs[1:]
                        max_i = int(np.argmax(t_probs))
                        conf = float(t_probs[max_i])
                        if conf > best_conf and conf >= tau:
                            best_conf = conf
                            best_idx = b_idx
                            best_act = max_i + 1
                            
            if best_idx is not None:
                d = "BUY" if best_act in [1, 2] else "SELL"
                atr_val = float(atr[best_idx]) if best_idx < len(atr) else 5.0
                sl_dist = round(atr_val * SL_ATR_MULT, 2)
                b_close = closes[best_idx]
                ts = ts_arr[best_idx]
                
                total_lot = 0.02
                lot_a = 0.01
                lot_b = 0.01
                
                ep = round(b_close + SLIPPAGE_PRICE, 2) if d == "BUY" else round(b_close - SLIPPAGE_PRICE, 2)
                sl_p = round(ep - sl_dist, 2) if d == "BUY" else round(ep + sl_dist, 2)
                tp1_p = round(ep + (TP1_R * sl_dist), 2) if d == "BUY" else round(ep - (TP1_R * sl_dist), 2)
                tp2_p = round(ep + (TP2_R * sl_dist), 2) if d == "BUY" else round(ep - (TP2_R * sl_dist), 2)
                
                cur_sl_a = sl_p
                cur_sl_b = sl_p
                pos_a_open = True
                pos_b_open = True
                tp1_hit = False
                pnl_accum = 0.0
                
                for f_idx in range(best_idx + 1, min(best_idx + 16, len(df))):
                    f_high = highs[f_idx]
                    f_low = lows[f_idx]
                    f_close = closes[f_idx]
                    f_dow = dows[f_idx]
                    f_hour = hours[f_idx]
                    
                    is_fri = (f_dow == 4 and f_hour >= 20) or f_dow in [5, 6]
                    is_tb = (f_idx - best_idx >= 12)
                    
                    if is_fri or is_tb:
                        exit_price = f_close
                        if pos_a_open:
                            pnl_a = (exit_price - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * lot_a * CONTRACT_SIZE
                            pnl_accum += (pnl_a - calc_friction(lot_a))
                        if pos_b_open:
                            pnl_b = (exit_price - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * lot_b * CONTRACT_SIZE
                            pnl_accum += (pnl_b - calc_friction(lot_b))
                        break
                        
                    if d == "BUY":
                        if not tp1_hit and f_low <= cur_sl_a:
                            pnl_g = (cur_sl_a - ep) * total_lot * CONTRACT_SIZE
                            pnl_accum = pnl_g - calc_friction(total_lot)
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
                                    break
                                elif f_low <= cur_sl_b:
                                    pnl_b = (cur_sl_b - ep) * lot_b * CONTRACT_SIZE
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    break
                                elif tp1_hit:
                                    r_gain = (f_high - ep) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        cur_sl_b = max(cur_sl_b, round(f_high - (TRAIL_DIST_R * sl_dist), 2))
                    else: # SELL
                        if not tp1_hit and f_high >= cur_sl_a:
                            pnl_g = (ep - cur_sl_a) * total_lot * CONTRACT_SIZE
                            pnl_accum = pnl_g - calc_friction(total_lot)
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
                                    break
                                elif f_high >= cur_sl_b:
                                    pnl_b = (ep - cur_sl_b) * lot_b * CONTRACT_SIZE
                                    pnl_accum += (pnl_b - calc_friction(lot_b))
                                    break
                                elif tp1_hit:
                                    r_gain = (ep - f_low) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        cur_sl_b = min(cur_sl_b, round(f_low + (TRAIL_DIST_R * sl_dist), 2))
                
                tot = round(pnl_accum, 2)
                equity += tot
                trades.append({'time': str(ts), 'win': 1 if tot > 0 else 0, 'pnl': tot})
                equity_curve.append(equity)
                
    tdf = pd.DataFrame(trades)
    n_trades = len(tdf)
    if n_trades == 0: return None
    wins = tdf[tdf['win'] == 1]
    losses = tdf[tdf['win'] == 0]
    wr = len(wins) / n_trades * 100
    gp = wins['pnl'].sum()
    gl = abs(losses['pnl'].sum())
    pf = gp / gl if gl > 0 else 999.0
    net_pnl = equity - INITIAL_EQUITY
    ret_pct = (equity / INITIAL_EQUITY - 1) * 100
    
    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    dds = (peaks - eq_arr) / peaks * 100
    max_dd = np.max(dds)
    
    return {
        'windows': n_windows,
        'tau': tau,
        'trades': n_trades,
        'tr_per_day': n_trades / 171.0,
        'win_rate': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'max_dd': max_dd
    }

print("\n--- SWEEPING NUMBER OF SESSIONS / TRADES PER DAY ---")
print(f"{'Target Windows':>14} | {'Trades':>6} | {'Tr/Day':>6} | {'WR%':>5} | {'PF':>5} | {'Net PnL':>10} | {'MaxDD%':>7}")
print("-" * 65)

for n_win in [3, 4, 5, 6, 8, 10, 12, 16, 20]:
    r = run_n_window_test(n_windows=n_win, tau=0.32)
    if r:
        print(f"{n_win:14d} | {r['trades']:6d} | {r['tr_per_day']:6.2f} | {r['win_rate']:5.1f} | {r['pf']:5.2f} | ${r['net_pnl']:9.2f} | {r['max_dd']:6.2f}%")
