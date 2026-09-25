import sys, pathlib
sys.path.insert(0, r'c:\Ngoding\bot_trading')
import pandas as pd, numpy as np

# Let's test the "Momentum Half-Life" / "Surgical OMS Evolution":
# Hypothesis:
# 1. Early Soft BE / Lock at +0.7R (moves SL to Entry + friction or -0.1R)
# 2. Faster Stale Decay at Bar 6 (3 hours) instead of Bar 10 (5 hours) -> tightens SL to -0.4R or -0.5R

from scripts.run_500_backtest_2025_full import (
    df, preds, atr, opens, highs, lows, closes, ts_arr, dt_series, dows, hours, minutes, day_ints,
    session_defs, seq_offset, TAU_BASE, MARGIN_MIN,
    FIXED_LOT, CONTRACT_SIZE, friction_cost, SL_ATR_MULT, TP_MAX_R, BE_TRIGGER_R, BE_BUFFER_PRICE,
    RATCHET_12_R, TRAIL_TRIGGER_R, TRAIL_DIST_R, STALE_DECAY_BARS, STALE_DECAY_R, TIME_BARRIER_BARS,
    SLIPPAGE_PRICE
)

def run_surgical_oms(year, be_trig=0.7, stale_bar=6, stale_r=0.4):
    st_dt = np.datetime64(f'{year}-01-01T00:00:00')
    en_dt = np.datetime64(f'{year}-12-31T23:59:59') if year != 2026 else np.datetime64('2026-08-31T23:59:59')
    mask = (ts_arr >= st_dt) & (ts_arr <= en_dt)
    indices = np.where(mask)[0]
    unique_days = np.unique(day_ints[indices])
    day_to_indices = {}
    for idx in indices:
        d = day_ints[idx]
        if d not in day_to_indices:
            day_to_indices[d] = []
        day_to_indices[d].append(idx)
        
    equity = 500.0
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
                atr_val = float(atr[best_idx]) if best_idx < len(atr) else 5.0
                sl_dist = round(atr_val * SL_ATR_MULT, 2)
                b_close = closes[best_idx]
                ts = ts_arr[best_idx]
                
                ep = round(b_close + SLIPPAGE_PRICE, 2) if d == "BUY" else round(b_close - SLIPPAGE_PRICE, 2)
                sl_p = round(ep - sl_dist, 2) if d == "BUY" else round(ep + sl_dist, 2)
                tp_max_p = round(ep + (TP_MAX_R * sl_dist), 2) if d == "BUY" else round(ep - (TP_MAX_R * sl_dist), 2)
                be_trigger_p = round(ep + (be_trig * sl_dist), 2) if d == "BUY" else round(ep - (be_trig * sl_dist), 2)
                
                cur_sl = sl_p
                be_activated = False
                pnl_net = 0.0
                
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
                        pnl_raw = (exit_price - ep) * FIXED_LOT * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * FIXED_LOT * CONTRACT_SIZE
                        pnl_net = pnl_raw - friction_cost
                        break
                        
                    bars_held = f_idx - best_idx
                    # Surgical Stale Decay: after stale_bar, tighten SL
                    if stale_bar > 0 and bars_held >= stale_bar and not be_activated:
                        decay_sl = round(ep - (stale_r * sl_dist), 2) if d == "BUY" else round(ep + (stale_r * sl_dist), 2)
                        if d == "BUY":
                            cur_sl = max(cur_sl, decay_sl)
                        else:
                            cur_sl = min(cur_sl, decay_sl)
                            
                    if d == "BUY":
                        if f_low <= cur_sl:
                            exit_price = cur_sl
                            pnl_raw = (cur_sl - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_high >= tp_max_p:
                            exit_price = tp_max_p
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
                            pnl_raw = (ep - cur_sl) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_low <= tp_max_p:
                            exit_price = tp_max_p
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
                trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot})
                equity_curve.append(equity)
                
    n_trades = len(trades)
    wins = [t for t in trades if t['win'] == 1]
    wr = len(wins)/n_trades*100 if n_trades > 0 else 0
    net = equity_curve[-1] - 500.0
    peaks = np.maximum.accumulate(equity_curve)
    dds = (peaks - equity_curve)/peaks * 100
    max_dd = np.max(dds)
    min_eq = np.min(equity_curve)
    return {
        'trades': n_trades,
        'win_rate': wr,
        'net_pnl': net,
        'max_dd': max_dd,
        'min_eq': min_eq
    }

print("="*90)
print("TESTING SURGICAL OMS: EARLY BE (+0.75R) + STALE DECAY AT BAR 6 (SL -> -0.45R)")
print("="*90)
for yr in [2026, 2025, 2024, 2023]:
    # Baseline
    b = run_surgical_oms(yr, be_trig=1.0, stale_bar=10, stale_r=0.6)
    # Surgical OMS: BE at 0.75R, Stale Decay Bar 6 tightens to -0.45R
    s = run_surgical_oms(yr, be_trig=0.75, stale_bar=6, stale_r=0.45)
    print(f"[{yr}] BASELINE (BE 1.0R / Decay Bar 10): PnL: ${b['net_pnl']:+8.2f} | WR: {b['win_rate']:.1f}% | Max DD: {b['max_dd']:.2f}% | Min Dip: ${b['min_eq']:.2f}")
    print(f"[{yr}] SURGICAL (BE 0.75R / Decay Bar 6) : PnL: ${s['net_pnl']:+8.2f} | WR: {s['win_rate']:.1f}% | Max DD: {s['max_dd']:.2f}% | Min Dip: ${s['min_eq']:.2f}")
    diff_pnl = s['net_pnl'] - b['net_pnl']
    print(f"       DELTA -> PnL Diff: ${diff_pnl:+7.2f} | Max DD Drop: {b['max_dd'] - s['max_dd']:.2f}% | Dip Gain: +${s['min_eq'] - b['min_eq']:.2f}")
    print("-"*90)
