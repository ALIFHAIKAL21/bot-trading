"""
Compare Cadence Selection Methods for Jan-Aug 2026:
Method 1: 4-5 Session Window Top-Sniper Picker (Asia, London Open, London Mid, NY Open, NY Mid)
Method 2: Bar-by-bar Sequential with Daily Cap 4-5 entries
Both with $1,000 capital, 15-channel model, realistic friction
"""
import numpy as np
import pandas as pd
import pathlib

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

st_dt = pd.to_datetime('2026-01-01', utc=True)
en_dt = pd.to_datetime('2026-08-31 23:59:59', utc=True)
sub_mask = (df['timestamp_utc'] >= st_dt) & (df['timestamp_utc'] <= en_dt)
sub_df = df[sub_mask].copy()
days = sorted(sub_df['timestamp_utc'].dt.strftime('%Y-%m-%d').unique())
print(f"Total Trading Days in 2026 Jan-Aug: {len(days)}")

# -------------------------------------------------------------
# METHOD 1: 4-5 Session Window Top-Sniper Picker
# Windows:
# 1. Asia: 01:00 - 06:30
# 2. London Open: 08:30 - 11:30
# 3. London Core: 11:30 - 14:00
# 4. NY Open: 14:00 - 17:30
# 5. NY Core: 17:30 - 21:00
# -------------------------------------------------------------
def run_session_window_sim(tau_min=0.30, n_windows=4, sizing="fixed_002"):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    
    if n_windows == 4:
        session_ranges = [
            (1, 6, 0, 30, "Asia"),
            (8, 12, 30, 0, "London Open"),
            (13, 16, 0, 30, "NY Open"),
            (17, 21, 0, 0, "NY Core")
        ]
    elif n_windows == 5:
        session_ranges = [
            (1, 6, 0, 30, "Asia Early"),
            (6, 7, 30, 0, "Asia Late"),
            (8, 12, 30, 0, "London Open"),
            (13, 16, 30, 30, "NY Open"),
            (17, 21, 0, 0, "NY Core")
        ]
    
    for day_str in days:
        day_mask = (df['timestamp_utc'].dt.strftime('%Y-%m-%d') == day_str) & sub_mask
        day_indices = df[day_mask].index
        day_ts = df.loc[day_indices[0], 'timestamp_utc']
        if day_ts.dayofweek in [5, 6]:
            continue
            
        for s_h_st, s_h_en, s_m_st, s_m_en, s_name in session_ranges:
            best_bar_idx = None
            best_conf = -1.0
            best_action = None
            
            for b_idx in day_indices:
                row = df.iloc[b_idx]
                h = row['timestamp_utc'].hour
                m = row['timestamp_utc'].minute
                
                # Check quarantine
                if h == 7 or (h == 8 and m < 30):
                    continue
                # Friday night cutoff
                if day_ts.dayofweek == 4 and h >= 18:
                    continue
                    
                # Time window match
                in_window = False
                t_val = h * 60 + m
                w_st = s_h_st * 60 + s_m_st
                w_en = s_h_en * 60 + s_m_en
                if w_st <= t_val <= w_en:
                    in_window = True
                    
                if in_window:
                    p_idx = b_idx - seq_offset
                    if 0 <= p_idx < len(preds):
                        probs = preds[p_idx]
                        t_probs = probs[1:]
                        max_idx = int(np.argmax(t_probs))
                        conf = float(t_probs[max_idx])
                        if conf > best_conf and conf >= tau_min:
                            best_conf = conf
                            best_bar_idx = b_idx
                            best_action = max_idx + 1
                            
            if best_bar_idx is not None:
                entry_row = df.iloc[best_bar_idx]
                entry_ts = entry_row["timestamp_utc"]
                bar_close = entry_row["close"]
                d = "BUY" if best_action in [1, 2] else "SELL"
                
                atr_val = float(atr[best_bar_idx]) if best_bar_idx < len(atr) else 5.0
                sl_dist = round(atr_val * SL_ATR_MULT, 2)
                
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
                    
                ep = round(bar_close + SLIPPAGE_PRICE, 2) if d == "BUY" else round(bar_close - SLIPPAGE_PRICE, 2)
                sl_p = round(ep - sl_dist, 2) if d == "BUY" else round(ep + sl_dist, 2)
                tp1_p = round(ep + (TP1_R * sl_dist), 2) if d == "BUY" else round(ep - (TP1_R * sl_dist), 2)
                tp2_p = round(ep + (TP2_R * sl_dist), 2) if d == "BUY" else round(ep - (TP2_R * sl_dist), 2)
                
                cur_sl_a = sl_p
                cur_sl_b = sl_p
                pos_a_open = True
                pos_b_open = True
                tp1_hit = False
                pnl_accum = 0.0
                
                for f_idx in range(best_bar_idx + 1, min(best_bar_idx + 16, len(df))):
                    f_row = df.iloc[f_idx]
                    f_ts = f_row["timestamp_utc"]
                    f_high = f_row["high"]
                    f_low = f_row["low"]
                    f_close = f_row["close"]
                    
                    is_fri_close = (f_ts.dayofweek == 4 and f_ts.hour >= 20) or f_ts.dayofweek in [5, 6]
                    is_tb = (f_idx - best_bar_idx >= 12)
                    
                    if is_fri_close or is_tb:
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
                    'time': entry_ts,
                    'month': str(entry_ts)[:7],
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

print("\n--- Testing Session Window Picker (4 vs 5 Windows/Day) ---")
print(f"{'Win':>3} | {'Tau':>5} | {'Sizing':>11} | {'Trades':>6} | {'Tr/Day':>6} | {'WR%':>5} | {'PF':>5} | {'Net PnL':>10} | {'Ret%':>7} | {'MaxDD%':>7}")
print("-" * 88)

for w in [4, 5]:
    for t in [0.25, 0.28, 0.30, 0.32, 0.34, 0.35]:
        for sz in ["fixed_002", "dynamic_1.5", "dynamic_2.0"]:
            res = run_session_window_sim(tau_min=t, n_windows=w, sizing=sz)
            if res:
                print(f"{res['windows']:3d} | {res['tau']:5.2f} | {res['sizing']:>11} | {res['trades']:6d} | {res['trades_per_day']:6.2f} | {res['win_rate']:5.1f} | {res['pf']:5.2f} | ${res['net_pnl']:9.2f} | {res['return_pct']:6.1f}% | {res['max_dd']:6.2f}%")
