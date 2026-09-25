"""
Empirical Comparison: 4-5 Trades/Day vs 8-10 Trades/Day vs 15-20 Trades/Day
Using the 15-Channel MOMENT Model on 2026 Jan - Aug (Modal $1,000)
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
mask_2026 = (ts_arr >= st_dt) & (ts_arr <= en_dt)
indices_2026 = np.where(mask_2026)[0]
unique_days = np.unique(day_ints[indices_2026])

# Simulation engine allowing high frequency entry
def simulate_frequency(max_daily=15, tau_val=0.30, max_concurrency=5, name="HighFreq"):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    open_positions = []
    current_day = -1
    daily_count = 0
    total_friction_paid = 0.0
    
    for bar_idx in range(len(df)):
        ts = ts_arr[bar_idx]
        if ts < st_dt:
            continue
        if ts > en_dt:
            break
            
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
        
        # 1. Update open positions
        remaining = []
        for pos in open_positions:
            bars_held = bar_idx - pos["entry_bar_idx"]
            d = pos["direction"]
            ep = pos["entry_price"]
            sl_dist = pos["sl_dist"]
            lot_a = pos["lot_a"]
            lot_b = pos["lot_b"]
            
            is_fri_close = (dow == 4 and hour >= 20) or dow in [5, 6]
            is_tb = (bars_held >= 14)
            
            if is_fri_close or is_tb:
                exit_p = b_close
                pnl_a = (exit_p - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_a * CONTRACT_SIZE
                pnl_b = (exit_p - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_b * CONTRACT_SIZE
                fric = calc_friction(lot_a) + calc_friction(lot_b)
                total_friction_paid += fric
                tot = round((pnl_a - calc_friction(lot_a)) + (pnl_b - calc_friction(lot_b)), 2)
                trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'month': str(pos['time'])[:7], 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'TB/Fri'})
                equity += tot
                continue
                
            if d == "BUY":
                if not pos["tp1_hit"] and b_low <= pos["cur_sl_a"]:
                    pnl_g = (pos["cur_sl_a"] - ep) * pos["total_lot"] * CONTRACT_SIZE
                    fric = calc_friction(pos["total_lot"])
                    total_friction_paid += fric
                    tot = round(pnl_g - fric, 2)
                    trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'month': str(pos['time'])[:7], 'dir': d, 'win': 0, 'pnl': tot, 'exit': 'SL'})
                    equity += tot
                    continue
                else:
                    if pos["pos_a_open"] and b_high >= pos["tp1_price"]:
                        pnl_a = (pos["tp1_price"] - ep) * lot_a * CONTRACT_SIZE
                        fric_a = calc_friction(lot_a)
                        total_friction_paid += fric_a
                        pos["pnl_net_accum"] += (pnl_a - fric_a)
                        pos["pos_a_open"] = False
                        pos["tp1_hit"] = True
                        f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                        pos["cur_sl_b"] = max(pos["cur_sl_b"], round(ep + f_p, 2))
                    if pos["pos_b_open"]:
                        if b_high >= pos["tp2_price"]:
                            pnl_b = (pos["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                            fric_b = calc_friction(lot_b)
                            total_friction_paid += fric_b
                            tot = round(pos["pnl_net_accum"] + pnl_b - fric_b, 2)
                            trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'month': str(pos['time'])[:7], 'dir': d, 'win': 1, 'pnl': tot, 'exit': 'TP2'})
                            equity += tot
                            continue
                        elif b_low <= pos["cur_sl_b"]:
                            pnl_b = (pos["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                            fric_b = calc_friction(lot_b)
                            total_friction_paid += fric_b
                            tot = round(pos["pnl_net_accum"] + pnl_b - fric_b, 2)
                            trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'month': str(pos['time'])[:7], 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'BE/Trail'})
                            equity += tot
                            continue
                        elif pos["tp1_hit"]:
                            r_gain = (b_high - ep) / sl_dist
                            if r_gain >= TRAIL_AFTER_R:
                                trail_sl = round(b_high - (TRAIL_DIST_R * sl_dist), 2)
                                pos["cur_sl_b"] = max(pos["cur_sl_b"], trail_sl)
            else: # SELL
                if not pos["tp1_hit"] and b_high >= pos["cur_sl_a"]:
                    pnl_g = (ep - pos["cur_sl_a"]) * pos["total_lot"] * CONTRACT_SIZE
                    fric = calc_friction(pos["total_lot"])
                    total_friction_paid += fric
                    tot = round(pnl_g - fric, 2)
                    trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'month': str(pos['time'])[:7], 'dir': d, 'win': 0, 'pnl': tot, 'exit': 'SL'})
                    equity += tot
                    continue
                else:
                    if pos["pos_a_open"] and b_low <= pos["tp1_price"]:
                        pnl_a = (ep - pos["tp1_price"]) * lot_a * CONTRACT_SIZE
                        fric_a = calc_friction(lot_a)
                        total_friction_paid += fric_a
                        pos["pnl_net_accum"] += (pnl_a - fric_a)
                        pos["pos_a_open"] = False
                        pos["tp1_hit"] = True
                        f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                        pos["cur_sl_b"] = min(pos["cur_sl_b"], round(ep - f_p, 2))
                    if pos["pos_b_open"]:
                        if b_low <= pos["tp2_price"]:
                            pnl_b = (ep - pos["tp2_price"]) * lot_b * CONTRACT_SIZE
                            fric_b = calc_friction(lot_b)
                            total_friction_paid += fric_b
                            tot = round(pos["pnl_net_accum"] + pnl_b - fric_b, 2)
                            trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'month': str(pos['time'])[:7], 'dir': d, 'win': 1, 'pnl': tot, 'exit': 'TP2'})
                            equity += tot
                            continue
                        elif b_high >= pos["cur_sl_b"]:
                            pnl_b = (ep - pos["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                            fric_b = calc_friction(lot_b)
                            total_friction_paid += fric_b
                            tot = round(pos["pnl_net_accum"] + pnl_b - fric_b, 2)
                            trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'month': str(pos['time'])[:7], 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'BE/Trail'})
                            equity += tot
                            continue
                        elif pos["tp1_hit"]:
                            r_gain = (ep - b_low) / sl_dist
                            if r_gain >= TRAIL_AFTER_R:
                                trail_sl = round(b_low + (TRAIL_DIST_R * sl_dist), 2)
                                pos["cur_sl_b"] = min(pos["cur_sl_b"], trail_sl)
            remaining.append(pos)
        open_positions = remaining
        
        # 2. Check Entry
        pred_idx = bar_idx - seq_offset
        if pred_idx >= 0 and pred_idx < len(preds) and len(open_positions) < max_concurrency and daily_count < max_daily:
            fri_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
            lon_quar = (hour == 7 or (hour == 8 and minute < 30))
            if not fri_freeze and not lon_quar:
                probs = preds[pred_idx]
                trade_probs = probs[1:]
                max_class_idx = int(np.argmax(trade_probs))
                action_class = max_class_idx + 1
                conf = float(trade_probs[max_class_idx])
                
                if conf >= tau_val:
                    d = "BUY" if action_class in [1, 2] else "SELL"
                    atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
                    sl_dist = round(atr_val * SL_ATR_MULT, 2)
                    
                    total_lot = 0.02
                    lot_a = 0.01
                    lot_b = 0.01
                    
                    ep = round(b_close + SLIPPAGE_PRICE, 2) if d == "BUY" else round(b_close - SLIPPAGE_PRICE, 2)
                    sl_p = round(ep - sl_dist, 2) if d == "BUY" else round(ep + sl_dist, 2)
                    tp1_p = round(ep + (TP1_R * sl_dist), 2) if d == "BUY" else round(ep - (TP1_R * sl_dist), 2)
                    tp2_p = round(ep + (TP2_R * sl_dist), 2) if d == "BUY" else round(ep - (TP2_R * sl_dist), 2)
                    
                    open_positions.append({
                        "entry_bar_idx": bar_idx,
                        "time": ts,
                        "direction": d,
                        "entry_price": ep,
                        "sl_dist": sl_dist,
                        "total_lot": total_lot,
                        "lot_a": lot_a,
                        "lot_b": lot_b,
                        "cur_sl_a": sl_p,
                        "cur_sl_b": sl_p,
                        "tp1_price": tp1_p,
                        "tp2_price": tp2_p,
                        "tp1_hit": False,
                        "pos_a_open": True,
                        "pos_b_open": True,
                        "pnl_net_accum": 0.0
                    })
                    daily_count += 1
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
    
    monthly_pnl = tdf.groupby('month')['pnl'].sum().to_dict()
    
    return {
        'name': name,
        'max_daily': max_daily,
        'tau': tau_val,
        'conc': max_concurrency,
        'trades': n_trades,
        'tr_per_day': tr_per_day,
        'active_days': n_days,
        'win_rate': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'ret_pct': ret_pct,
        'max_dd': max_dd,
        'final_equity': equity,
        'friction_paid': total_friction_paid,
        'monthly': monthly_pnl
    }

print("Running High-Frequency Test Suite (4-5 vs 8-10 vs 15-20 Trades/Day)...")
test_configs = [
    # Baseline
    {"name": "Baseline (4-5 tr/d)", "daily": 5, "tau": 0.35, "conc": 2},
    {"name": "Active (4-5 tr/d)", "daily": 5, "tau": 0.32, "conc": 3},
    # Medium-High (8-10 tr/d)
    {"name": "Med-High 8-10 tr/d", "daily": 10, "tau": 0.35, "conc": 4},
    {"name": "Med-High 8-10 tr/d", "daily": 10, "tau": 0.32, "conc": 4},
    # Ultra-High (15-20 tr/d)
    {"name": "Ultra-High 15 tr/d", "daily": 15, "tau": 0.32, "conc": 6},
    {"name": "Ultra-High 15 tr/d", "daily": 15, "tau": 0.30, "conc": 6},
    {"name": "Ultra-High 20 tr/d", "daily": 20, "tau": 0.30, "conc": 8},
    {"name": "Ultra-High 20 tr/d", "daily": 20, "tau": 0.28, "conc": 8},
]

print(f"{'Config Name':<22} | {'Daily':>5} | {'Tau':>5} | {'Conc':>4} | {'Trades':>6} | {'Tr/Day':>6} | {'WR%':>5} | {'PF':>5} | {'Net PnL':>10} | {'MaxDD%':>7} | {'Friction':>8}")
print("-" * 105)

for cfg in test_configs:
    r = simulate_frequency(max_daily=cfg['daily'], tau_val=cfg['tau'], max_concurrency=cfg['conc'], name=cfg['name'])
    if r:
        print(f"{r['name']:<22} | {r['max_daily']:5d} | {r['tau']:5.2f} | {r['conc']:4d} | {r['trades']:6d} | {r['tr_per_day']:6.2f} | {r['win_rate']:5.1f} | {r['pf']:5.2f} | ${r['net_pnl']:9.2f} | {r['max_dd']:6.2f}% | ${r['friction_paid']:7.1f}")
