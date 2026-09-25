import sys, pathlib
sys.path.insert(0, r'c:\Ngoding\bot_trading')
import pandas as pd, numpy as np, json

# Test the hypothesis on 2026 ($250 and $500)
# Let's inspect the effect of Daily Circuit Breaker:
# Rule: If cumulative losses on a single day reach 2 stop losses (or 2 consecutive losses on that day), stop trading for that day!

from scripts.run_250_backtest_2026_full import (
    df, preds, atr, opens, highs, lows, closes, ts_arr, dt_series, dows, hours, minutes, day_ints,
    unique_days, day_to_indices, session_defs, seq_offset, TAU_BASE, MARGIN_MIN,
    FIXED_LOT, CONTRACT_SIZE, friction_cost, SL_ATR_MULT, TP_MAX_R, BE_TRIGGER_R, BE_BUFFER_PRICE,
    RATCHET_12_R, TRAIL_TRIGGER_R, TRAIL_DIST_R, STALE_DECAY_BARS, STALE_DECAY_R, TIME_BARRIER_BARS,
    SLIPPAGE_PRICE
)

def run_with_circuit_breaker(initial_equity, max_daily_losses=2):
    equity = initial_equity
    equity_curve = [equity]
    trades = []
    
    for d_int in unique_days:
        day_idxs = day_to_indices[d_int]
        dow = dows[day_idxs[0]]
        if dow in [5, 6]:
            continue
            
        daily_losses = 0
        
        for w_st, w_en, s_name in session_defs:
            if max_daily_losses is not None and daily_losses >= max_daily_losses:
                # Circuit breaker triggered for remainder of day!
                break
                
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
                        
                        if conf > best_conf and conf >= TAU_BASE and margin >= MARGIN_MIN:
                            best_conf = conf
                            best_idx = b_idx
                            best_act = max_i + 1
                            
            if best_idx is not None:
                d = "BUY" if best_act in [1, 2] else "SELL"
                atr_val = float(atr[best_idx]) if best_idx < len(atr) else 10.0
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
                
                for f_idx in range(best_idx + 1, min(best_idx + 16, len(df))):
                    f_high = highs[f_idx]
                    f_low = lows[f_idx]
                    f_close = closes[f_idx]
                    f_dow = dows[f_idx]
                    f_hour = hours[f_idx]
                    f_ts = ts_arr[f_idx]
                    
                    is_fri = (f_dow == 4 and f_hour >= 20) or f_dow in [5, 6]
                    is_tb = (f_idx - best_idx >= TIME_BARRIER_BARS)
                    
                    if is_fri or is_tb:
                        exit_price = f_close
                        exit_reason = "Friday Closeout" if is_fri else f"Time Barrier ({TIME_BARRIER_BARS//2}h)"
                        pnl_raw = (exit_price - ep) * FIXED_LOT * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * FIXED_LOT * CONTRACT_SIZE
                        pnl_net = pnl_raw - friction_cost
                        break
                        
                    bars_held = f_idx - best_idx
                    if STALE_DECAY_BARS > 0 and bars_held >= STALE_DECAY_BARS and not be_activated:
                        decay_sl = round(ep - (STALE_DECAY_R * sl_dist), 2) if d == "BUY" else round(ep + (STALE_DECAY_R * sl_dist), 2)
                        if d == "BUY":
                            cur_sl = max(cur_sl, decay_sl)
                        else:
                            cur_sl = min(cur_sl, decay_sl)
                            
                    if d == "BUY":
                        if f_low <= cur_sl:
                            exit_price = cur_sl
                            exit_reason = "Protected Stop" if be_activated else "Stop Loss"
                            pnl_raw = (cur_sl - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_high >= tp_max_p:
                            exit_price = tp_max_p
                            exit_reason = f"Max TP (+{TP_MAX_R}R)"
                            pnl_raw = (tp_max_p - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        else:
                            if not be_activated and f_high >= be_trigger_p:
                                be_activated = True
                                cur_sl = max(cur_sl, round(ep + BE_BUFFER_PRICE, 2))
                            r_gain = (f_high - ep) / sl_dist
                            if RATCHET_12_R > 0 and r_gain >= 1.2:
                                cur_sl = max(cur_sl, round(ep + (RATCHET_12_R * sl_dist), 2))
                            if r_gain >= TRAIL_TRIGGER_R:
                                cur_sl = max(cur_sl, round(f_high - (TRAIL_DIST_R * sl_dist), 2))
                    else: # SELL
                        if f_high >= cur_sl:
                            exit_price = cur_sl
                            exit_reason = "Protected Stop" if be_activated else "Stop Loss"
                            pnl_raw = (ep - cur_sl) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_low <= tp_max_p:
                            exit_price = tp_max_p
                            exit_reason = f"Max TP (+{TP_MAX_R}R)"
                            pnl_raw = (ep - tp_max_p) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        else:
                            if not be_activated and f_low <= be_trigger_p:
                                be_activated = True
                                cur_sl = min(cur_sl, round(ep - BE_BUFFER_PRICE, 2))
                            r_gain = (ep - f_low) / sl_dist
                            if RATCHET_12_R > 0 and r_gain >= 1.2:
                                cur_sl = min(cur_sl, round(ep - (RATCHET_12_R * sl_dist), 2))
                            if r_gain >= TRAIL_TRIGGER_R:
                                cur_sl = min(cur_sl, round(f_low + (TRAIL_DIST_R * sl_dist), 2))
                                
                tot = round(pnl_net, 2)
                equity += tot
                if tot <= 0:
                    daily_losses += 1
                    
                trades.append({
                    'time': str(ts),
                    'month': str(ts)[:7],
                    'pnl': tot,
                    'win': 1 if tot > 0 else 0
                })
                equity_curve.append(equity)
                
    n_trades = len(trades)
    wins = [t for t in trades if t['win'] == 1]
    wr = len(wins)/n_trades*100 if n_trades > 0 else 0
    net = equity_curve[-1] - initial_equity
    peaks = np.maximum.accumulate(equity_curve)
    dds = (peaks - equity_curve)/peaks * 100
    max_dd = np.max(dds)
    min_eq = np.min(equity_curve)
    return {
        'trades': n_trades,
        'win_rate': wr,
        'net_pnl': net,
        'final_eq': equity_curve[-1],
        'max_dd': max_dd,
        'min_eq': min_eq
    }

print("=== TESTING CIRCUIT BREAKER ON 2026 ($250 MODAL) ===")
base = run_with_circuit_breaker(250.0, max_daily_losses=None)
print(f"BASELINE (No Breaker) : Net: ${base['net_pnl']:+8.2f} | WR: {base['win_rate']:.2f}% | Max DD: {base['max_dd']:.2f}% | Min Dip: ${base['min_eq']:.2f} | Trades: {base['trades']}")

for limit in [2, 1]:
    res = run_with_circuit_breaker(250.0, max_daily_losses=limit)
    print(f"LIMIT {limit} LOSS/DAY   : Net: ${res['net_pnl']:+8.2f} | WR: {res['win_rate']:.2f}% | Max DD: {res['max_dd']:.2f}% | Min Dip: ${res['min_eq']:.2f} | Trades: {res['trades']}")

print("\n=== TESTING CIRCUIT BREAKER ON 2026 ($500 MODAL) ===")
base500 = run_with_circuit_breaker(500.0, max_daily_losses=None)
print(f"BASELINE (No Breaker) : Net: ${base500['net_pnl']:+8.2f} | WR: {base500['win_rate']:.2f}% | Max DD: {base500['max_dd']:.2f}% | Min Dip: ${base500['min_eq']:.2f} | Trades: {base500['trades']}")

for limit in [2, 1]:
    res500 = run_with_circuit_breaker(500.0, max_daily_losses=limit)
    print(f"LIMIT {limit} LOSS/DAY   : Net: ${res500['net_pnl']:+8.2f} | WR: {res500['win_rate']:.2f}% | Max DD: {res500['max_dd']:.2f}% | Min Dip: ${res500['min_eq']:.2f} | Trades: {res500['trades']}")
