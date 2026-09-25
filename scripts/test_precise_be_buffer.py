"""
Test proper Breakeven Buffer + Entropy Gate on 500 Modal
"""
import sys, os, pathlib
import numpy as np
import pandas as pd

from run_500_modal_backtest import df, preds, atr, opens, highs, lows, closes, ts_arr, dows, hours, minutes, day_ints, unique_days, day_to_indices, seq_offset, calc_friction, INITIAL_EQUITY, SPREAD_PRICE, SLIPPAGE_PRICE, COMMISSION_PER_LOT, CONTRACT_SIZE, SL_ATR_MULT, TP_MAX_R, BE_TRIGGER_R, TRAIL_TRIGGER_R, TRAIL_DIST_R, session_defs, FIXED_LOT

def test_be_buff(buff_val=0.25, tau_val=0.32, min_margin=0.0):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    friction_cost = calc_friction(FIXED_LOT)
    
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
                        p_hold = float(probs[0])
                        t_probs = probs[1:]
                        max_i = int(np.argmax(t_probs))
                        conf = float(t_probs[max_i])
                        margin = conf - p_hold
                        
                        if conf > best_conf and conf >= tau_val and margin >= min_margin:
                            best_conf = conf
                            best_idx = b_idx
                            best_act = max_i + 1
                            
            if best_idx is not None:
                d = "BUY" if best_act in [1, 2] else "SELL"
                atr_val = float(atr[best_idx]) if best_idx < len(atr) else 5.0
                sl_dist = round(atr_val * SL_ATR_MULT, 2)
                b_close = closes[best_idx]
                ts = ts_arr[best_idx]
                
                ep = round(b_close + SLIPPAGE_PRICE, 2) if d == "BUY" else round(b_close - SLIPPAGE_PRICE, 2)
                sl_p = round(ep - sl_dist, 2) if d == "BUY" else round(ep + sl_dist, 2)
                tp_max_p = round(ep + (TP_MAX_R * sl_dist), 2) if d == "BUY" else round(ep - (TP_MAX_R * sl_dist), 2)
                be_trigger_p = round(ep + (BE_TRIGGER_R * sl_dist), 2) if d == "BUY" else round(ep - (BE_TRIGGER_R * sl_dist), 2)
                
                cur_sl = sl_p
                be_activated = False
                pnl_net = 0.0
                exit_reason = None
                exit_price = None
                
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
                        exit_reason = "Friday Closeout" if is_fri else "Time Barrier (6h)"
                        pnl_raw = (exit_price - ep) * FIXED_LOT * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * FIXED_LOT * CONTRACT_SIZE
                        pnl_net = pnl_raw - friction_cost
                        break
                        
                    if d == "BUY":
                        if f_low <= cur_sl:
                            exit_price = cur_sl
                            exit_reason = "Breakeven Hit" if be_activated else "Stop Loss (-1.0R)"
                            pnl_raw = (cur_sl - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_high >= tp_max_p:
                            exit_price = tp_max_p
                            exit_reason = "Max TP (+2.5R)"
                            pnl_raw = (tp_max_p - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        else:
                            if not be_activated and f_high >= be_trigger_p:
                                be_activated = True
                                cur_sl = max(cur_sl, round(ep + buff_val, 2))
                            r_gain = (f_high - ep) / sl_dist
                            if r_gain >= TRAIL_TRIGGER_R:
                                trail_sl = round(f_high - (TRAIL_DIST_R * sl_dist), 2)
                                cur_sl = max(cur_sl, trail_sl)
                    else: # SELL
                        if f_high >= cur_sl:
                            exit_price = cur_sl
                            exit_reason = "Breakeven Hit" if be_activated else "Stop Loss (-1.0R)"
                            pnl_raw = (ep - cur_sl) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_low <= tp_max_p:
                            exit_price = tp_max_p
                            exit_reason = "Max TP (+2.5R)"
                            pnl_raw = (ep - tp_max_p) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        else:
                            if not be_activated and f_low <= be_trigger_p:
                                be_activated = True
                                cur_sl = min(cur_sl, round(ep - buff_val, 2))
                            r_gain = (ep - f_low) / sl_dist
                            if r_gain >= TRAIL_TRIGGER_R:
                                trail_sl = round(f_low + (TRAIL_DIST_R * sl_dist), 2)
                                cur_sl = min(cur_sl, trail_sl)
                
                tot = round(pnl_net, 2)
                equity += tot
                trades.append({'time': str(ts), 'month': str(ts)[:7], 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': exit_reason})
                equity_curve.append(equity)
                
    tdf = pd.DataFrame(trades)
    n_t = len(tdf)
    wins = tdf[tdf['win'] == 1]
    losses = tdf[tdf['win'] == 0]
    wr = len(wins) / n_t * 100
    gp = wins['pnl'].sum()
    gl = abs(losses['pnl'].sum())
    pf = gp / gl if gl > 0 else 999.0
    net_pnl = equity - INITIAL_EQUITY
    
    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    dds = (peaks - eq_arr) / peaks * 100
    max_dd = np.max(dds)
    min_eq = np.min(eq_arr)
    
    return {
        'buff': buff_val,
        'tau': tau_val,
        'margin': min_margin,
        'trades': n_t,
        'wins': len(wins),
        'losses': len(losses),
        'wr': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'max_dd': max_dd,
        'min_eq': min_eq,
        'final_eq': equity
    }

print(f"{'Buff':>5} | {'Tau':>5} | {'Margin':>6} | {'Trades':>6} | {'W / L':>11} | {'WR%':>6} | {'PF':>5} | {'Net PnL':>10} | {'MaxDD%':>7} | {'MinEq':>7}")
print("-" * 88)

for b in [0.20, 0.22, 0.25, 0.30]:
    for t in [0.32, 0.33, 0.35]:
        for m in [0.0, 0.01, 0.02]:
            r = test_be_buff(buff_val=b, tau_val=t, min_margin=m)
            print(f"{r['buff']:5.2f} | {r['tau']:5.2f} | {r['margin']:6.2f} | {r['trades']:6d} | {r['wins']:4d} / {r['losses']:4d} | {r['wr']:5.1f}% | {r['pf']:5.2f} | ${r['net_pnl']:9.2f} | {r['max_dd']:6.2f}% | ${r['min_eq']:6.1f}")
