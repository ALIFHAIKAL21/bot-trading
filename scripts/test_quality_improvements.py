"""
Experimentation: Quality Improvements for $500 Capital (Lot 0.01 Flat)
1. Confidence Margin Filter (p_best - p_hold >= margin)
2. Session-Adaptive Thresholds (Asia selective, London/NY momentum)
3. Positive Breakeven Buffer (Net +$0.25 on BE exit instead of -$0.06 loss)
4. Structural Alignment Gate
Target: Win Rate stably > 50%, Max Drawdown compressed to < 30%
"""

import sys, os, pathlib, json
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled_15ch.parquet"
preds_path = project_root / "checkpoints" / "predictions_15ch.npy"

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# Compute ATR(14)
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

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
ts_arr = df['timestamp_utc'].values.astype('datetime64[s]')
dt_series = df['timestamp_utc'].dt
dows = dt_series.dayofweek.values
hours = dt_series.hour.values
minutes = dt_series.minute.values
day_ints = dt_series.strftime('%Y%m%d').astype(int).values
ob_zones = df['order_block_zone'].values
ma_slopes = df['ma_ribbon_slope'].values

# $500 Account Parameters
INITIAL_EQUITY = 500.0
FIXED_LOT = 0.01
SPREAD_PRICE = 0.75 * 0.10     # $0.075 / oz
SLIPPAGE_PRICE = 0.3 * 0.10    # $0.03 / oz 2-way
COMMISSION_PER_LOT = 3.50      # $0.035 for 0.01 lot
CONTRACT_SIZE = 100.0          # 0.01 lot = 1 oz
friction_cost = round(SPREAD_PRICE * FIXED_LOT * CONTRACT_SIZE + SLIPPAGE_PRICE * FIXED_LOT * CONTRACT_SIZE * 2 + COMMISSION_PER_LOT * FIXED_LOT, 3) # ~$0.17

SL_ATR_MULT = 1.5
TP_MAX_R = 2.5
BE_TRIGGER_R = 1.0
TRAIL_TRIGGER_R = 1.5
TRAIL_DIST_R = 0.8
seq_offset = 63

st_dt = np.datetime64('2026-01-01T00:00:00')
en_dt = np.datetime64('2026-08-31T23:59:59')
mask_2026 = (ts_arr >= st_dt) & (ts_arr <= en_dt)
indices_2026 = np.where(mask_2026)[0]
unique_days = np.unique(day_ints[indices_2026])

day_to_indices = {}
for idx in indices_2026:
    d = day_ints[idx]
    if d not in day_to_indices:
        day_to_indices[d] = []
    day_to_indices[d].append(idx)

# Session definitions
session_defs = [
    (1*60, 4*60, "Asia Early", "asia"),
    (4*60 + 30, 7*60, "Asia Late", "asia"),
    (8*60 + 30, 12*60 + 30, "London Core", "london"),
    (13*60, 17*60, "NY Open", "ny"),
    (17*60 + 30, 21*60, "NY Core", "ny")
]

def simulate_improved(
    tau_base=0.32,
    tau_asia_boost=0.03,      # Asia threshold = tau_base + tau_asia_boost
    margin_min=0.03,          # Minimum p_best - p_hold
    be_profit_pips=3.0,       # Positive BE buffer (in pips: 3 pips = $0.30)
    use_structure_gate=True   # Avoid buying directly into heavy bearish order block
):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    equity_timestamps = [ts_arr[indices_2026[0]]]
    trades = []
    
    be_buffer_price = be_profit_pips * 0.10 # e.g. 3.0 pips = $0.30
    
    for d_int in unique_days:
        day_idxs = day_to_indices[d_int]
        dow = dows[day_idxs[0]]
        if dow in [5, 6]:
            continue
            
        for w_st, w_en, s_name, s_type in session_defs:
            best_idx = None
            best_conf = -1.0
            best_act = None
            
            # Adaptive threshold per session
            tau_target = (tau_base + tau_asia_boost) if s_type == "asia" else tau_base
            
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
                        
                        # 1. Check confidence and confidence margin
                        if conf >= tau_target and margin >= margin_min:
                            action_cand = max_i + 1
                            d_cand = "BUY" if action_cand in [1, 2] else "SELL"
                            
                            # 2. Structural Gate check
                            ok = True
                            if use_structure_gate:
                                ob = ob_zones[b_idx]
                                m_slp = ma_slopes[b_idx]
                                # Don't buy if market structure is deep bearish (-1) and sloping down hard
                                if d_cand == "BUY" and (ob < -0.5 or m_slp < -0.4): ok = False
                                if d_cand == "SELL" and (ob > 0.5 or m_slp > 0.4): ok = False
                                
                            if ok and conf > best_conf:
                                best_conf = conf
                                best_idx = b_idx
                                best_act = action_cand
                                
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
                exit_ts = None
                
                # Forward simulate bars
                for f_idx in range(best_idx + 1, min(best_idx + 16, len(df))):
                    f_high = highs[f_idx]
                    f_low = lows[f_idx]
                    f_close = closes[f_idx]
                    f_dow = dows[f_idx]
                    f_hour = hours[f_idx]
                    f_ts = ts_arr[f_idx]
                    
                    is_fri = (f_dow == 4 and f_hour >= 20) or f_dow in [5, 6]
                    is_tb = (f_idx - best_idx >= 12)
                    
                    if is_fri or is_tb:
                        exit_price = f_close
                        exit_reason = "Friday Closeout" if is_fri else "Time Barrier (6h)"
                        exit_ts = f_ts
                        pnl_raw = (exit_price - ep) * FIXED_LOT * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * FIXED_LOT * CONTRACT_SIZE
                        pnl_net = pnl_raw - friction_cost
                        break
                        
                    if d == "BUY":
                        if f_low <= cur_sl:
                            exit_price = cur_sl
                            exit_reason = "Breakeven Hit" if be_activated else "Stop Loss (-1.0R)"
                            exit_ts = f_ts
                            pnl_raw = (cur_sl - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_high >= tp_max_p:
                            exit_price = tp_max_p
                            exit_reason = "Max TP (+2.5R)"
                            exit_ts = f_ts
                            pnl_raw = (tp_max_p - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        else:
                            # Activate Breakeven with Positive Profit Buffer
                            if not be_activated and f_high >= be_trigger_p:
                                be_activated = True
                                # Lock in entry + positive buffer ($0.30)
                                cur_sl = max(cur_sl, round(ep + be_buffer_price, 2))
                            # Activate Trailing Stop if > 1.5R
                            r_gain = (f_high - ep) / sl_dist
                            if r_gain >= TRAIL_TRIGGER_R:
                                trail_sl = round(f_high - (TRAIL_DIST_R * sl_dist), 2)
                                cur_sl = max(cur_sl, trail_sl)
                    else: # SELL
                        if f_high >= cur_sl:
                            exit_price = cur_sl
                            exit_reason = "Breakeven Hit" if be_activated else "Stop Loss (-1.0R)"
                            exit_ts = f_ts
                            pnl_raw = (ep - cur_sl) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_low <= tp_max_p:
                            exit_price = tp_max_p
                            exit_reason = "Max TP (+2.5R)"
                            exit_ts = f_ts
                            pnl_raw = (ep - tp_max_p) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        else:
                            # Activate Breakeven with Positive Profit Buffer
                            if not be_activated and f_low <= be_trigger_p:
                                be_activated = True
                                cur_sl = min(cur_sl, round(ep - be_buffer_price, 2))
                            # Activate Trailing Stop if > 1.5R
                            r_gain = (ep - f_low) / sl_dist
                            if r_gain >= TRAIL_TRIGGER_R:
                                trail_sl = round(f_low + (TRAIL_DIST_R * sl_dist), 2)
                                cur_sl = min(cur_sl, trail_sl)
                
                tot = round(pnl_net, 2)
                equity += tot
                trades.append({
                    'time': str(ts),
                    'month': str(ts)[:7],
                    'dir': d,
                    'win': 1 if tot > 0 else 0,
                    'pnl': tot,
                    'lot': FIXED_LOT,
                    'conf': best_conf,
                    'session': s_name,
                    'exit_reason': exit_reason or 'Time Barrier'
                })
                equity_curve.append(equity)
                equity_timestamps.append(ts)
                
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
    min_eq = np.min(eq_arr)
    
    tdf['date'] = pd.to_datetime(tdf['time']).dt.date
    n_days = tdf['date'].nunique()
    tr_per_day = n_trades / 171.0
    
    cadence_dist = tdf.groupby('date').size()
    c45 = cadence_dist[cadence_dist.isin([4, 5])].count()
    pct_45 = c45 / n_days * 100
    
    monthly_pnl = tdf.groupby('month')['pnl'].sum().to_dict()
    
    return {
        'tau_base': tau_base,
        'tau_asia_boost': tau_asia_boost,
        'margin_min': margin_min,
        'be_pips': be_profit_pips,
        'gate': use_structure_gate,
        'trades': n_trades,
        'wins': len(wins),
        'losses': len(losses),
        'tr_per_day': tr_per_day,
        'pct_45': pct_45,
        'win_rate': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'ret_pct': ret_pct,
        'max_dd': max_dd,
        'min_equity': min_eq,
        'final_equity': equity,
        'monthly': monthly_pnl,
        'tdf': tdf,
        'eq_curve': equity_curve,
        'eq_ts': equity_timestamps
    }

print("Running Calibration Sweep for Quality Improvements on $500 Account...")
print(f"{'Tau':>5} | {'Asia+':>5} | {'Margin':>6} | {'BE_pip':>6} | {'Gate':>5} | {'Trades':>6} | {'W / L':>11} | {'WR%':>6} | {'PF':>5} | {'Net PnL':>10} | {'MaxDD%':>7} | {'MinEq':>7}")
print("-" * 105)

for tau in [0.32, 0.33, 0.34]:
    for asia_b in [0.02, 0.04]:
        for m_min in [0.02, 0.04]:
            for be_p in [2.5, 3.0, 3.5]:
                for gt in [True]:
                    res = simulate_improved(tau_base=tau, tau_asia_boost=asia_b, margin_min=m_min, be_profit_pips=be_p, use_structure_gate=gt)
                    if res:
                        print(f"{res['tau_base']:5.2f} | {res['tau_asia_boost']:5.2f} | {res['margin_min']:6.2f} | {res['be_pips']:6.1f} | {str(res['gate']):>5} | {res['trades']:6d} | {res['wins']:4d} / {res['losses']:4d} | {res['win_rate']:5.1f}% | {res['pf']:5.2f} | ${res['net_pnl']:9.2f} | {res['max_dd']:6.2f}% | ${res['min_equity']:6.1f}")
