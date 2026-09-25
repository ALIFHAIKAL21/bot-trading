import sys, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

print("Loading data...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# H4 Data
df_h4 = df.set_index('timestamp_utc').resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

df_h4['h4_ema20'] = df_h4['close'].ewm(span=20, adjust=False).mean()
df_h4['h4_ema50'] = df_h4['close'].ewm(span=50, adjust=False).mean()
df_h4['h4_ema200'] = df_h4['close'].ewm(span=200, adjust=False).mean()
df_h4['h4_atr'] = (df_h4['high'] - df_h4['low']).rolling(14).mean()
df_h4['h4_slope20'] = (df_h4['h4_ema20'] - df_h4['h4_ema20'].shift(3)) / (df_h4['h4_atr'] + 1e-8)
df_h4['h4_slope50'] = (df_h4['h4_ema50'] - df_h4['h4_ema50'].shift(5)) / (df_h4['h4_atr'] + 1e-8)

# Distance to EMA50 and EMA200 in ATR units
df_h4['dist_e50_atr'] = (df_h4['close'] - df_h4['h4_ema50']) / (df_h4['h4_atr'] + 1e-8)
df_h4['dist_e200_atr'] = (df_h4['close'] - df_h4['h4_ema200']) / (df_h4['h4_atr'] + 1e-8)

# Distance to rolling 30-day high (180 H4 bars)
df_h4['h4_high_30d'] = df_h4['high'].rolling(180).max()
df_h4['dist_high_30d_atr'] = (df_h4['h4_high_30d'] - df_h4['close']) / (df_h4['h4_atr'] + 1e-8)

df_h4_lagged = df_h4[['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_slope50', 'dist_e50_atr', 'dist_e200_atr', 'dist_high_30d_atr']].shift(1)
df = df.merge(df_h4_lagged, on='timestamp_utc', how='left')
for col in ['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_slope50', 'dist_e50_atr', 'dist_e200_atr', 'dist_high_30d_atr']:
    df[col] = df[col].ffill().fillna(0)

# ATR(14)
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

# Numpy arrays
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
ts_arr = df['timestamp_utc'].values.astype('datetime64[s]')
ob_zones = df['order_block_zone'].values
ma_slopes = df['ma_ribbon_slope'].values

h4_slp20 = df['h4_slope20'].values
h4_slp50 = df['h4_slope50'].values
dist_e50 = df['dist_e50_atr'].values
dist_e200 = df['dist_e200_atr'].values
dist_high30 = df['dist_high_30d_atr'].values

dt_series = df['timestamp_utc'].dt
dows = dt_series.dayofweek.values
hours = dt_series.hour.values
minutes = dt_series.minute.values
day_ints = dt_series.strftime('%Y%m%d').astype(int).values

TAU = 0.355
INITIAL_EQUITY = 10_000.0
BASE_RISK = 0.01
SPREAD_PRICE = 0.75 * 0.10
SLIPPAGE_PRICE = 0.3 * 0.10
COMMISSION = 3.50
CONTRACT_SIZE = 100.0
SL_ATR_MULT = 1.5
POS_A_PCT = 0.40
POS_B_PCT = 0.60
TP1_R = 1.0
TP2_R = 2.5
TRAIL_AFTER_R = 1.5
TRAIL_DIST_R = 0.8
MAX_DAILY = 10

def calc_friction(lot):
    return round(SPREAD_PRICE * lot * CONTRACT_SIZE + SLIPPAGE_PRICE * lot * CONTRACT_SIZE * 2 + COMMISSION * lot, 2)

test_years = [
    {"year": 2021, "start_dt": np.datetime64('2021-09-23T12:30:00'), "end_dt": np.datetime64('2021-12-31T23:59:59')},
    {"year": 2022, "start_dt": np.datetime64('2022-01-01T00:00:00'), "end_dt": np.datetime64('2022-12-31T23:59:59')},
    {"year": 2023, "start_dt": np.datetime64('2023-01-01T00:00:00'), "end_dt": np.datetime64('2023-12-31T23:59:59')},
    {"year": 2024, "start_dt": np.datetime64('2024-01-01T00:00:00'), "end_dt": np.datetime64('2024-12-31T23:59:59')},
    {"year": 2025, "start_dt": np.datetime64('2025-01-01T00:00:00'), "end_dt": np.datetime64('2025-12-31T23:59:59')},
]

def eval_condition(gate_fn, convert_to_buy=False):
    seq_offset = 63
    n_bars = len(df)
    results = {}
    
    for y_cfg in test_years:
        year = y_cfg["year"]
        st = y_cfg["start_dt"]
        en = y_cfg["end_dt"]
        
        equity = INITIAL_EQUITY
        equity_curve = [equity]
        trades = []
        open_position = None
        current_day = -1
        daily_count = 0
        
        for bar_idx in range(n_bars):
            ts = ts_arr[bar_idx]
            in_range = (ts >= st) and (ts <= en)
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
            
            if open_position is not None:
                bars_held = bar_idx - open_position["entry_bar_idx"]
                d = open_position["direction"]
                ep = open_position["entry_price"]
                sl_dist = open_position["sl_dist"]
                lot_a = open_position["lot_a"]
                lot_b = open_position["lot_b"]
                
                is_fri_close = (dow == 4 and hour >= 20) or dow in [5, 6]
                is_tb = (bars_held >= 16)
                
                if is_fri_close or is_tb:
                    exit_p = b_close
                    pnl_a = (exit_p - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_a * CONTRACT_SIZE
                    pnl_b = (exit_p - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_b * CONTRACT_SIZE
                    tot = round((pnl_a - calc_friction(lot_a)) + (pnl_b - calc_friction(lot_b)), 2)
                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot})
                    equity += tot
                    open_position = None
                else:
                    if d == "BUY":
                        if not open_position["tp1_hit"] and b_low <= open_position["cur_sl_a"]:
                            pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                            tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                            trades.append({'win': 0, 'pnl': tot})
                            equity += tot
                            open_position = None
                        else:
                            if open_position["pos_a_open"] and b_high >= open_position["tp1_price"]:
                                pnl_a = (open_position["tp1_price"] - ep) * lot_a * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                                open_position["pos_a_open"] = False
                                open_position["tp1_hit"] = True
                                f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                                open_position["cur_sl_b"] = max(open_position["cur_sl_b"], round(ep + f_p, 2))
                                
                            if open_position["pos_b_open"]:
                                if b_high >= open_position["tp2_price"]:
                                    pnl_b = (open_position["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1, 'pnl': tot})
                                    equity += tot
                                    open_position = None
                                elif b_low <= open_position["cur_sl_b"]:
                                    pnl_b = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot})
                                    equity += tot
                                    open_position = None
                                elif open_position["tp1_hit"]:
                                    r_gain = (b_high - ep) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(b_high - (TRAIL_DIST_R * sl_dist), 2)
                                        open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                    else: # SELL
                        if not open_position["tp1_hit"] and b_high >= open_position["cur_sl_a"]:
                            pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * CONTRACT_SIZE
                            tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                            trades.append({'win': 0, 'pnl': tot})
                            equity += tot
                            open_position = None
                        else:
                            if open_position["pos_a_open"] and b_low <= open_position["tp1_price"]:
                                pnl_a = (ep - open_position["tp1_price"]) * lot_a * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                                open_position["pos_a_open"] = False
                                open_position["tp1_hit"] = True
                                f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                                open_position["cur_sl_b"] = min(open_position["cur_sl_b"], round(ep - f_p, 2))
                                
                            if open_position["pos_b_open"]:
                                if b_low <= open_position["tp2_price"]:
                                    pnl_b = (ep - open_position["tp2_price"]) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1, 'pnl': tot})
                                    equity += tot
                                    open_position = None
                                elif b_high >= open_position["cur_sl_b"]:
                                    pnl_b = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot})
                                    equity += tot
                                    open_position = None
                                elif open_position["tp1_hit"]:
                                    r_gain = (ep - b_low) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(b_low + (TRAIL_DIST_R * sl_dist), 2)
                                        open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)
            
            if in_range:
                equity_curve.append(equity)
                
            pred_idx = bar_idx - seq_offset
            if in_range and pred_idx >= 0 and pred_idx < len(preds) and open_position is None and daily_count < MAX_DAILY:
                fri_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
                lon_quar = (hour == 7 or (hour == 8 and minute < 30))
                if not fri_freeze and not lon_quar:
                    probs = preds[pred_idx]
                    trade_probs = probs[1:]
                    max_class_idx = int(np.argmax(trade_probs))
                    action_class = max_class_idx + 1
                    conf = float(trade_probs[max_class_idx])
                    p_hold = float(probs[0])
                    
                    if conf >= TAU and conf > p_hold:
                        d = "BUY" if action_class in [1, 2] else "SELL"
                        ob = ob_zones[bar_idx]
                        m_slp = ma_slopes[bar_idx]
                        
                        ok = True
                        if d == "BUY" and (ob < 0 or m_slp < -0.3): ok = False
                        if d == "SELL" and (ob > 0 or m_slp > 0.3): ok = False
                        
                        if ok and gate_fn is not None:
                            # Pass current market state to gate_fn
                            # gate_fn returns: (action, is_blocked)
                            # where action can be 'KEEP', 'CONVERT'
                            act, is_blocked = gate_fn(
                                bar_idx, d, h4_slp20[bar_idx], h4_slp50[bar_idx],
                                dist_e50[bar_idx], dist_e200[bar_idx], dist_high30[bar_idx]
                            )
                            if is_blocked:
                                ok = False
                            elif act == 'CONVERT':
                                d = "BUY" if d == "SELL" else "SELL"
                                
                        if ok:
                            atr_val = atr[bar_idx] if bar_idx < len(atr) else 5.0
                            sl_dist = round(atr_val * SL_ATR_MULT, 2)
                            if sl_dist > 0:
                                target_risk = equity * BASE_RISK
                                raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                                lot_size = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 50.0))
                                lot_a = max(0.01, round(lot_size * POS_A_PCT, 2))
                                lot_b = max(0.01, round(lot_size - lot_a, 2))
                                
                                if d == "BUY":
                                    ep = round(b_close + SLIPPAGE_PRICE, 2)
                                    sl_p = round(ep - sl_dist, 2)
                                    tp1_p = round(ep + (TP1_R * sl_dist), 2)
                                    tp2_p = round(ep + (TP2_R * sl_dist), 2)
                                else:
                                    ep = round(b_close - SLIPPAGE_PRICE, 2)
                                    sl_p = round(ep + sl_dist, 2)
                                    tp1_p = round(ep - (TP1_R * sl_dist), 2)
                                    tp2_p = round(ep - (TP2_R * sl_dist), 2)
                                    
                                open_position = {
                                    "direction": d, "entry_bar_idx": bar_idx,
                                    "entry_price": ep, "sl_dist": sl_dist,
                                    "cur_sl_a": sl_p, "cur_sl_b": sl_p, "tp1_price": tp1_p, "tp2_price": tp2_p,
                                    "total_lot": lot_size, "lot_a": lot_a, "lot_b": lot_b,
                                    "pos_a_open": True, "pos_b_open": True, "tp1_hit": False,
                                    "pnl_net_accum": 0.0
                                }
                                daily_count += 1
                                
        tdf = pd.DataFrame(trades)
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        max_dd = np.min((eq_arr - peak) / peak) * 100 if len(eq_arr) > 0 else 0
        n_tr = len(tdf)
        n_w = tdf['win'].sum() if n_tr > 0 else 0
        wr = (n_w / n_tr * 100) if n_tr > 0 else 0.0
        ret = ((equity - INITIAL_EQUITY) / INITIAL_EQUITY) * 100
        results[year] = {"trades": n_tr, "ret": round(ret, 2), "max_dd": round(max_dd, 2), "wr": round(wr, 1)}
        
    return results

print("=== BASELINE ===")
base = eval_condition(None)
for y, v in base.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")

# Gate A: Distance to EMA200 > 3.0 ATR + High Slope > 0.25 (Block Freight Train Shorting)
def gate_ema200_extension(bar_idx, d, s20, s50, de50, de200, dh30):
    # Only restrict SELL when price is severely overextended above EMA200 and slope is strong
    if d == "SELL" and (de200 > 3.5) and (s50 > 0.25) and (dh30 < 2.0):
        return ('KEEP', True) # Block suicide sell
    return ('KEEP', False)

print("\n=== GATE A: EMA200 Extension (Dist > 3.5 ATR, Slope50 > 0.25, DistHigh30 < 2 ATR) ===")
res_a = eval_condition(gate_ema200_extension)
for y, v in res_a.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")

# Gate B: Same as Gate A, but CONVERT suicide SELL to BUY continuation (No abstaining, pure trend riding!)
def gate_ema200_convert(bar_idx, d, s20, s50, de50, de200, dh30):
    if d == "SELL" and (de200 > 3.5) and (s50 > 0.25) and (dh30 < 2.0):
        return ('CONVERT', False) # Convert to BUY continuation!
    return ('KEEP', False)

print("\n=== GATE B: Smart Re-Orientation (Convert Parabolic Freight Train to BUY) ===")
res_b = eval_condition(gate_ema200_convert)
for y, v in res_b.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")

# Gate C: Macro Velocity Band (de200 > 2.5, s20 > 0.15, dh30 < 1.5)
def gate_c(bar_idx, d, s20, s50, de50, de200, dh30):
    if d == "SELL" and (de200 > 2.5) and (s20 > 0.15) and (dh30 < 1.5):
        return ('CONVERT', False)
    return ('KEEP', False)

print("\n=== GATE C: Velocity Band (Dist > 2.5 ATR, Slope20 > 0.15, DistHigh < 1.5 ATR) ===")
res_c = eval_condition(gate_c)
for y, v in res_c.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")
