import sys, pathlib
sys.path.insert(0, r'c:\Ngoding\bot_trading')
import pandas as pd, numpy as np, json

from scripts.run_500_backtest_2025_full import (
    df, preds, atr, opens, highs, lows, closes, ts_arr, dt_series, dows, hours, minutes, day_ints,
    session_defs, seq_offset, TAU_BASE, MARGIN_MIN,
    FIXED_LOT, CONTRACT_SIZE, friction_cost, SL_ATR_MULT, TP_MAX_R, BE_BUFFER_PRICE,
    RATCHET_12_R, TRAIL_TRIGGER_R, TRAIL_DIST_R, TIME_BARRIER_BARS,
    SLIPPAGE_PRICE
)

def run_simulation(start_date, end_date, initial_equity, be_trig=0.75, stale_bar=6, stale_r=0.45):
    st_dt = np.datetime64(start_date)
    en_dt = np.datetime64(end_date)
    mask = (ts_arr >= st_dt) & (ts_arr <= en_dt)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    unique_days = np.unique(day_ints[indices])
    day_to_indices = {}
    for idx in indices:
        d = day_ints[idx]
        if d not in day_to_indices:
            day_to_indices[d] = []
        day_to_indices[d].append(idx)
        
    equity = initial_equity
    equity_curve = [equity]
    equity_ts = [ts_arr[indices[0]]]
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
                    if stale_bar > 0 and bars_held >= stale_bar and not be_activated:
                        decay_sl = round(ep - (stale_r * sl_dist), 2) if d == "BUY" else round(ep + (stale_r * sl_dist), 2)
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
                trades.append({
                    'time': str(ts),
                    'month': str(ts)[:7],
                    'pnl': tot,
                    'win': 1 if tot > 0 else 0,
                    'exit_reason': exit_reason
                })
                equity_curve.append(equity)
                equity_ts.append(ts)
                
    tdf = pd.DataFrame(trades)
    n_trades = len(trades)
    wins = [t for t in trades if t['win'] == 1]
    wr = len(wins)/n_trades*100 if n_trades > 0 else 0
    net = equity_curve[-1] - initial_equity
    return_pct = (net / initial_equity) * 100
    peaks = np.maximum.accumulate(equity_curve)
    dds = (peaks - equity_curve)/peaks * 100
    max_dd = np.max(dds)
    min_eq = np.min(equity_curve)
    
    # Monthly table
    monthly = []
    for m, grp in tdf.groupby('month'):
        m_w = len(grp[grp['win'] == 1])
        m_pnl = grp['pnl'].sum()
        m_wr = m_w / len(grp) * 100
        monthly.append({
            'month': m,
            'trades': len(grp),
            'wins': m_w,
            'win_rate': round(m_wr, 1),
            'pnl': round(m_pnl, 2)
        })
        
    return {
        'initial_equity': initial_equity,
        'trades': n_trades,
        'win_rate': wr,
        'net_pnl': net,
        'return_pct': return_pct,
        'final_equity': equity_curve[-1],
        'max_dd': max_dd,
        'min_eq': min_eq,
        'monthly': monthly
    }

periods = [
    ('2021 (Oct-Dec)', '2021-10-01T00:00:00', '2021-12-31T23:59:59'),
    ('2022 Full',      '2022-01-01T00:00:00', '2022-12-31T23:59:59'),
    ('2023 Full',      '2023-01-01T00:00:00', '2023-12-31T23:59:59'),
    ('2024 Full',      '2024-01-01T00:00:00', '2024-12-31T23:59:59'),
    ('2025 Full',      '2025-01-01T00:00:00', '2025-12-31T23:59:59'),
    ('2026 (Jan-Aug)', '2026-01-01T00:00:00', '2026-08-31T23:59:59')
]

print("="*105)
print("COMPREHENSIVE MULTI-YEAR AUDIT: KINETIC OMS (BE +0.75R / STALE DECAY BAR 6)")
print("="*105)

all_monthly_rows = []

for p_name, st, en in periods:
    res500 = run_simulation(st, en, 500.0, be_trig=0.75, stale_bar=6, stale_r=0.45)
    res250 = run_simulation(st, en, 250.0, be_trig=0.75, stale_bar=6, stale_r=0.45)
    res1000 = run_simulation(st, en, 1000.0, be_trig=0.75, stale_bar=6, stale_r=0.45)
    
    print(f"\n>>> PERIOD: {p_name} | Trades: {res500['trades']} | Win Rate: {res500['win_rate']:.2f}% | Net PnL: ${res500['net_pnl']:+8.2f}")
    print(f"    • Modal $250  : Return: {res250['return_pct']:+7.1f}% | Final: ${res250['final_equity']:8.2f} | Max DD: {res250['max_dd']:5.2f}% | Min Dip: ${res250['min_eq']:6.2f}")
    print(f"    • Modal $500  : Return: {res500['return_pct']:+7.1f}% | Final: ${res500['final_equity']:8.2f} | Max DD: {res500['max_dd']:5.2f}% | Min Dip: ${res500['min_eq']:6.2f}")
    print(f"    • Modal $1000 : Return: {res1000['return_pct']:+7.1f}% | Final: ${res1000['final_equity']:8.2f} | Max DD: {res1000['max_dd']:5.2f}% | Min Dip: ${res1000['min_eq']:6.2f}")
    
    for m in res500['monthly']:
        all_monthly_rows.append(m)

print("\n" + "="*85)
print("ALL MONTHS SUMMARY (2021 - 2026)")
print("="*85)
mdf = pd.DataFrame(all_monthly_rows)
print(mdf.to_string(index=False))
red_months = mdf[mdf['pnl'] < 0]
green_months = mdf[mdf['pnl'] >= 0]
print(f"\nTOTAL MONTHS AUDITED: {len(mdf)}")
print(f"GREEN MONTHS: {len(green_months)} ({len(green_months)/len(mdf)*100:.1f}%)")
print(f"RED MONTHS  : {len(red_months)} ({len(red_months)/len(mdf)*100:.1f}%)")
if len(red_months) > 0:
    print("\nRed months details:")
    print(red_months.to_string(index=False))
