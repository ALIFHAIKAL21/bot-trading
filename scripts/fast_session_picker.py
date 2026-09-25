"""
Fast Session Window Picker for Jan - Aug 2026 ($1,000 Capital)
4 Windows and 5 Windows per day
"""
import sys, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled_15ch.parquet"
preds_path = project_root / "checkpoints" / "predictions_15ch.npy"

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# ATR calculation
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

SPREAD_PRICE = 0.75 * 0.10
SLIPPAGE_PRICE = 0.3 * 0.10
COMMISSION = 3.50
CONTRACT_SIZE = 100.0
SL_ATR_MULT = 1.5
TP1_R = 1.0
TP2_R = 2.5
TRAIL_AFTER_R = 1.5
TRAIL_DIST_R = 0.8
INITIAL_EQUITY = 1000.0
seq_offset = 63

def calc_friction(lot):
    return round(SPREAD_PRICE * lot * CONTRACT_SIZE + SLIPPAGE_PRICE * lot * CONTRACT_SIZE * 2 + COMMISSION * lot, 2)

st_dt = np.datetime64('2026-01-01T00:00:00')
en_dt = np.datetime64('2026-08-31T23:59:59')

# Pre-filter 2026 indices
mask_2026 = (ts_arr >= st_dt) & (ts_arr <= en_dt)
indices_2026 = np.where(mask_2026)[0]
unique_days = np.unique(day_ints[indices_2026])

# Pre-group indices by day_int
day_to_indices = {}
for idx in indices_2026:
    d = day_ints[idx]
    if d not in day_to_indices:
        day_to_indices[d] = []
    day_to_indices[d].append(idx)

def run_session_picker(n_windows=4, tau_min=0.30, sizing="fixed_002"):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    
    if n_windows == 4:
        # 4 sessions across the day: Asia, London Open, NY Morning, NY Afternoon
        session_defs = [
            (1*60, 6*60 + 30, "Asia"),
            (8*60 + 30, 12*60 + 30, "London"),
            (13*60, 16*60 + 30, "NY Morning"),
            (17*60, 21*60, "NY Afternoon")
        ]
    elif n_windows == 5:
        # 5 sessions across the day: Asia Early, Asia Late/Pre-Lon, London Open, NY Open, NY Close
        session_defs = [
            (1*60, 4*60, "Asia Early"),
            (4*60 + 30, 7*60, "Asia Pre-Lon"),
            (8*60 + 30, 12*60 + 30, "London Open"),
            (13*60, 17*60, "NY Open"),
            (17*60 + 30, 21*60, "NY Close")
        ]
        
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
                # Skip quarantine
                if h == 7 or (h == 8 and m < 30):
                    continue
                # Skip Friday night close
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
                        if conf > best_conf and conf >= tau_min:
                            best_conf = conf
                            best_idx = b_idx
                            best_act = max_i + 1
                            
            if best_idx is not None:
                d = "BUY" if best_act in [1, 2] else "SELL"
                atr_val = float(atr[best_idx]) if best_idx < len(atr) else 5.0
                sl_dist = round(atr_val * SL_ATR_MULT, 2)
                b_close = closes[best_idx]
                ts = ts_arr[best_idx]
                
                if sizing == "fixed_002":
                    total_lot = 0.02
                    lot_a = 0.01
                    lot_b = 0.01
                elif sizing == "dynamic_1.5":
                    target_risk = equity * 0.015
                    raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                    total_lot = max(0.02, min(round(np.floor(raw_lot / 0.02) * 0.02, 2), 5.0))
                    lot_a = round(total_lot * 0.5, 2)
                    lot_b = round(total_lot - lot_a, 2)
                elif sizing == "dynamic_2.0":
                    target_risk = equity * 0.02
                    raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                    total_lot = max(0.02, min(round(np.floor(raw_lot / 0.02) * 0.02, 2), 5.0))
                    lot_a = round(total_lot * 0.5, 2)
                    lot_b = round(total_lot - lot_a, 2)
                    
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
                trades.append({
                    'time': str(ts),
                    'month': str(ts)[:7],
                    'dir': d,
                    'win': 1 if tot > 0 else 0,
                    'pnl': tot,
                    'lot': total_lot,
                    'conf': best_conf,
                    'session': s_name
                })
                equity_curve.append(equity)
                
    tdf = pd.DataFrame(trades)
    n_trades = len(tdf)
    if n_trades == 0:
        return None
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
    
    tdf['date'] = pd.to_datetime(tdf['time']).dt.date
    n_days = tdf['date'].nunique()
    tr_per_day = n_trades / 171.0
    
    return {
        'windows': n_windows,
        'tau': tau_min,
        'sizing': sizing,
        'trades': n_trades,
        'trades_per_day': tr_per_day,
        'active_days': n_days,
        'win_rate': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'return_pct': ret_pct,
        'max_dd': max_dd,
        'final_equity': equity,
        'tdf': tdf,
        'equity_curve': equity_curve
    }

print("\n--- Testing Fast Session Window Picker ---")
print(f"{'Win':>3} | {'Tau':>5} | {'Sizing':>11} | {'Trades':>6} | {'Tr/Day':>6} | {'WR%':>5} | {'PF':>5} | {'Net PnL':>10} | {'Ret%':>7} | {'MaxDD%':>7}")
print("-" * 88)

for w in [4, 5]:
    for t in [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]:
        for sz in ["fixed_002", "dynamic_1.5"]:
            res = run_session_picker(n_windows=w, tau_min=t, sizing=sz)
            if res:
                print(f"{res['windows']:3d} | {res['tau']:5.2f} | {res['sizing']:>11} | {res['trades']:6d} | {res['trades_per_day']:6.2f} | {res['win_rate']:5.1f} | {res['pf']:5.2f} | ${res['net_pnl']:9.2f} | {res['return_pct']:6.1f}% | {res['max_dd']:6.2f}%")
