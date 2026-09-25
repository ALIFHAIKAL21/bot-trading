"""
Sweep simulation parameter combinations for Jan - Aug 2026:
- Modal $1,000 USD
- Target frequency: 4 - 5 entries per day
- Date range: 2026-01-01 to 2026-08-31
- Model: predictions_15ch.npy
"""
import sys, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled_15ch.parquet"
preds_path = project_root / "checkpoints" / "predictions_15ch.npy"

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds_15ch = np.load(preds_path)

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
POS_A_PCT = 0.50
POS_B_PCT = 0.50
TP1_R = 1.0
TP2_R = 2.5
TRAIL_AFTER_R = 1.5
TRAIL_DIST_R = 0.8
INITIAL_EQUITY = 1000.0

def calc_friction(lot):
    return round(SPREAD_PRICE * lot * CONTRACT_SIZE + SLIPPAGE_PRICE * lot * CONTRACT_SIZE * 2 + COMMISSION * lot, 2)

st_dt = np.datetime64('2026-01-01T00:00:00')
en_dt = np.datetime64('2026-08-31T23:59:59')
seq_offset = 63

def run_sim(tau_val=0.36, max_daily=5, max_concurrency=2, sizing_mode="dynamic_1.5"):
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    open_positions = []
    current_day = -1
    daily_count = 0
    
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
        
        # 1. Update existing open positions
        remaining_positions = []
        for pos in open_positions:
            bars_held = bar_idx - pos["entry_bar_idx"]
            d = pos["direction"]
            ep = pos["entry_price"]
            sl_dist = pos["sl_dist"]
            lot_a = pos["lot_a"]
            lot_b = pos["lot_b"]
            
            is_fri_close = (dow == 4 and hour >= 20) or dow in [5, 6]
            is_tb = (bars_held >= 16)
            
            if is_fri_close or is_tb:
                exit_p = b_close
                pnl_a = (exit_p - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_a * CONTRACT_SIZE
                pnl_b = (exit_p - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_b * CONTRACT_SIZE
                tot = round((pnl_a - calc_friction(lot_a)) + (pnl_b - calc_friction(lot_b)), 2)
                trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'TB/Fri'})
                equity += tot
                continue
                
            if d == "BUY":
                if not pos["tp1_hit"] and b_low <= pos["cur_sl_a"]:
                    pnl_g = (pos["cur_sl_a"] - ep) * pos["total_lot"] * CONTRACT_SIZE
                    tot = round(pnl_g - calc_friction(pos["total_lot"]), 2)
                    trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'dir': d, 'win': 0, 'pnl': tot, 'exit': 'SL'})
                    equity += tot
                    continue
                else:
                    if pos["pos_a_open"] and b_high >= pos["tp1_price"]:
                        pnl_a = (pos["tp1_price"] - ep) * lot_a * CONTRACT_SIZE
                        pos["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                        pos["pos_a_open"] = False
                        pos["tp1_hit"] = True
                        f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                        pos["cur_sl_b"] = max(pos["cur_sl_b"], round(ep + f_p, 2))
                    if pos["pos_b_open"]:
                        if b_high >= pos["tp2_price"]:
                            pnl_b = (pos["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                            tot = round(pos["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                            trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'dir': d, 'win': 1, 'pnl': tot, 'exit': 'TP2'})
                            equity += tot
                            continue
                        elif b_low <= pos["cur_sl_b"]:
                            pnl_b = (pos["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                            tot = round(pos["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                            trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'BE/Trail'})
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
                    tot = round(pnl_g - calc_friction(pos["total_lot"]), 2)
                    trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'dir': d, 'win': 0, 'pnl': tot, 'exit': 'SL'})
                    equity += tot
                    continue
                else:
                    if pos["pos_a_open"] and b_low <= pos["tp1_price"]:
                        pnl_a = (ep - pos["tp1_price"]) * lot_a * CONTRACT_SIZE
                        pos["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                        pos["pos_a_open"] = False
                        pos["tp1_hit"] = True
                        f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                        pos["cur_sl_b"] = min(pos["cur_sl_b"], round(ep - f_p, 2))
                    if pos["pos_b_open"]:
                        if b_low <= pos["tp2_price"]:
                            pnl_b = (ep - pos["tp2_price"]) * lot_b * CONTRACT_SIZE
                            tot = round(pos["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                            trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'dir': d, 'win': 1, 'pnl': tot, 'exit': 'TP2'})
                            equity += tot
                            continue
                        elif b_high >= pos["cur_sl_b"]:
                            pnl_b = (ep - pos["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                            tot = round(pos["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                            trades.append({'bar': pos['entry_bar_idx'], 'time': str(pos['time']), 'dir': d, 'win': 1 if tot > 0 else 0, 'pnl': tot, 'exit': 'BE/Trail'})
                            equity += tot
                            continue
                        elif pos["tp1_hit"]:
                            r_gain = (ep - b_low) / sl_dist
                            if r_gain >= TRAIL_AFTER_R:
                                trail_sl = round(b_low + (TRAIL_DIST_R * sl_dist), 2)
                                pos["cur_sl_b"] = min(pos["cur_sl_b"], trail_sl)
            remaining_positions.append(pos)
        open_positions = remaining_positions
        
        # 2. Check Entry
        pred_idx = bar_idx - seq_offset
        if pred_idx >= 0 and pred_idx < len(preds_15ch) and len(open_positions) < max_concurrency and daily_count < max_daily:
            fri_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
            lon_quar = (hour == 7 or (hour == 8 and minute < 30))
            if not fri_freeze and not lon_quar:
                probs = preds_15ch[pred_idx]
                trade_probs = probs[1:]
                max_class_idx = int(np.argmax(trade_probs))
                action_class = max_class_idx + 1
                conf = float(trade_probs[max_class_idx])
                
                # Check confidence threshold
                if conf >= tau_val:
                    d = "BUY" if action_class in [1, 2] else "SELL"
                    atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
                    sl_dist = round(atr_val * SL_ATR_MULT, 2)
                    
                    if sizing_mode == "fixed_002":
                        total_lot = 0.02
                        lot_a = 0.01
                        lot_b = 0.01
                    else: # dynamic compounding 1.5%
                        target_risk = equity * 0.015
                        raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                        total_lot = max(0.02, min(round(np.floor(raw_lot / 0.02) * 0.02, 2), 5.0))
                        lot_a = round(total_lot * 0.5, 2)
                        lot_b = round(total_lot - lot_a, 2)
                    
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

    # Compute KPIs
    tdf = pd.DataFrame(trades)
    n_trades = len(tdf)
    if n_trades == 0:
        return None
        
    wins = tdf[tdf['win'] == 1]
    losses = tdf[tdf['win'] == 0]
    wr = len(wins) / n_trades * 100
    gross_p = wins['pnl'].sum()
    gross_l = abs(losses['pnl'].sum())
    pf = gross_p / gross_l if gross_l > 0 else 999.0
    net_pnl = equity - INITIAL_EQUITY
    ret_pct = (equity / INITIAL_EQUITY - 1) * 100
    
    # Drawdown
    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    dds = (peaks - eq_arr) / peaks * 100
    max_dd = np.max(dds)
    
    # Days active
    tdf['date'] = pd.to_datetime(tdf['time']).dt.date
    n_days = tdf['date'].nunique()
    trades_per_day = n_trades / 171.0 # 171 trading days
    trades_per_active_day = n_trades / n_days if n_days > 0 else 0
    
    return {
        'tau': tau_val,
        'max_daily': max_daily,
        'concurrency': max_concurrency,
        'sizing': sizing_mode,
        'trades': n_trades,
        'trades_per_day': trades_per_day,
        'trades_per_active_day': trades_per_active_day,
        'win_rate': wr,
        'pf': pf,
        'net_pnl': net_pnl,
        'return_pct': ret_pct,
        'max_dd': max_dd,
        'final_equity': equity
    }

print("Running parameter sweep for Jan-Aug 2026...")
print(f"{'Tau':>5} | {'Daily':>5} | {'Conc':>4} | {'Sizing':>10} | {'Trades':>6} | {'Tr/Day':>6} | {'WR%':>5} | {'PF':>5} | {'Net PnL':>10} | {'Ret%':>7} | {'MaxDD%':>7}")
print("-" * 88)

for tau in [0.35, 0.36, 0.38, 0.40]:
    for max_d in [4, 5]:
        for conc in [1, 2]:
            for sz in ["fixed_002", "dynamic_1.5"]:
                res = run_sim(tau_val=tau, max_daily=max_d, max_concurrency=conc, sizing_mode=sz)
                if res:
                    print(f"{res['tau']:5.2f} | {res['max_daily']:5d} | {res['concurrency']:4d} | {res['sizing']:>10} | {res['trades']:6d} | {res['trades_per_day']:6.2f} | {res['win_rate']:5.1f} | {res['pf']:5.2f} | ${res['net_pnl']:9.2f} | {res['return_pct']:6.1f}% | {res['max_dd']:6.2f}%")
