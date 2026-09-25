import sys, pathlib, json
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

print("Loading data...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# H4 data
df_h4 = df.set_index('timestamp_utc').resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

df_h4['h4_ema20'] = df_h4['close'].ewm(span=20, adjust=False).mean()
df_h4['h4_ema50'] = df_h4['close'].ewm(span=50, adjust=False).mean()
df_h4['h4_ema200'] = df_h4['close'].ewm(span=200, adjust=False).mean()
df_h4['h4_atr'] = (df_h4['high'] - df_h4['low']).rolling(14).mean()
df_h4['h4_slope20'] = (df_h4['h4_ema20'] - df_h4['h4_ema20'].shift(3)) / (df_h4['h4_atr'] + 1e-8)

# 20-day high (120 H4 bars) and 50-day high (300 H4 bars)
df_h4['h4_high_20d'] = df_h4['high'].rolling(120).max()
df_h4['h4_high_50d'] = df_h4['high'].rolling(300).max()
df_h4['h4_ath_proximity'] = (df_h4['close'] >= df_h4['h4_high_20d'] * 0.998).astype(int)

df_h4_lagged = df_h4[['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_high_20d', 'h4_high_50d', 'h4_ath_proximity']].shift(1)
df = df.merge(df_h4_lagged, on='timestamp_utc', how='left')
for col in ['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_high_20d', 'h4_high_50d', 'h4_ath_proximity']:
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

# Convert all necessary columns into contiguous numpy arrays for maximum simulation speed
opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
ts_arr = df['timestamp_utc'].values.astype('datetime64[s]')
ob_zones = df['order_block_zone'].values
ma_slopes = df['ma_ribbon_slope'].values
h4_e20s = df['h4_ema20'].values
h4_e50s = df['h4_ema50'].values
h4_slopes = df['h4_slope20'].values
h4_aths = df['h4_ath_proximity'].values
is_blackouts = df.get('is_news_blackout', pd.Series(np.zeros(len(df), dtype=bool))).values
is_eligibles = df.get('entry_eligible', pd.Series(np.ones(len(df), dtype=bool))).values

# Precompute datetime parts
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

def simulate_engine(regime_mode="baseline", slope_cut=0.20, convert_parabolic=False, uncapped_runner=False):
    """
    regime_mode:
      - 'baseline': original logic
      - 'filter_parabolic': only block/convert when at ATH proximity AND slope > slope_cut
      - 'dynamic_trail': adjust risk/trailing on extended regimes
    """
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
            
            # Position management
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
                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot, 'direction': d})
                    equity += tot
                    open_position = None
                else:
                    if d == "BUY":
                        if not open_position["tp1_hit"] and b_low <= open_position["cur_sl_a"]:
                            pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                            tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                            trades.append({'win': 0, 'pnl': tot, 'direction': d})
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
                                is_uncapped = uncapped_runner and open_position.get("in_runner", False)
                                if not is_uncapped and b_high >= open_position["tp2_price"]:
                                    pnl_b = (open_position["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1, 'pnl': tot, 'direction': d})
                                    equity += tot
                                    open_position = None
                                elif b_low <= open_position["cur_sl_b"]:
                                    pnl_b = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot, 'direction': d})
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
                            trades.append({'win': 0, 'pnl': tot, 'direction': d})
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
                                is_uncapped = uncapped_runner and open_position.get("in_runner", False)
                                if not is_uncapped and b_low <= open_position["tp2_price"]:
                                    pnl_b = (ep - open_position["tp2_price"]) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1, 'pnl': tot, 'direction': d})
                                    equity += tot
                                    open_position = None
                                elif b_high >= open_position["cur_sl_b"]:
                                    pnl_b = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot, 'direction': d})
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
                if is_eligibles[bar_idx] and not is_blackouts[bar_idx]:
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
                            
                            in_runner = False
                            if regime_mode != "baseline" and ok:
                                h4_s = h4_slopes[bar_idx]
                                is_ath = bool(h4_aths[bar_idx])
                                
                                # Condition for Runaway Parabolic Bull:
                                # High slope AND price is at 20-day high!
                                is_parabolic_bull = (h4_s > slope_cut) and is_ath
                                is_parabolic_bear = (h4_s < -slope_cut)
                                
                                if is_parabolic_bull:
                                    if d == "SELL":
                                        if convert_parabolic:
                                            d = "BUY" # Convert to BUY trend rider!
                                            in_runner = True
                                        else:
                                            ok = False # Veto fatal counter-trend sell
                                    else:
                                        in_runner = True
                                elif is_parabolic_bear:
                                    if d == "BUY":
                                        if convert_parabolic:
                                            d = "SELL"
                                            in_runner = True
                                        else:
                                            ok = False
                                    else:
                                        in_runner = True
                                        
                            if ok:
                                atr_val = atr[bar_idx] if bar_idx < len(atr) else 5.0
                                sl_dist = round(atr_val * SL_ATR_MULT, 2)
                                if sl_dist > 0:
                                    target_risk = equity * BASE_RISK
                                    raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                                    lot_size = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 50.0))
                                    actual_risk = lot_size * sl_dist * CONTRACT_SIZE
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
                                        "pnl_net_accum": 0.0, "in_runner": in_runner
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

print("\n=== RUNNING 5-YEAR MULTI-REGIME TEST MATRIX ===")

print("\n1. BASELINE (ORIGINAL MODEL):")
b_res = simulate_engine(regime_mode="baseline")
for y, v in b_res.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")

# Test multiple surgical slope thresholds:
for thresh in [0.10, 0.15, 0.20, 0.25]:
    print(f"\n2. SURGICAL FILTER (ATH + Slope > {thresh}):")
    f_res = simulate_engine(regime_mode="filter_parabolic", slope_cut=thresh, convert_parabolic=False)
    for y, v in f_res.items():
        print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")

for thresh in [0.10, 0.15, 0.20]:
    print(f"\n3. SMART CONVERSION (Invert Parabolic SELL into BUY Rider, Slope > {thresh}):")
    c_res = simulate_engine(regime_mode="filter_parabolic", slope_cut=thresh, convert_parabolic=True)
    for y, v in c_res.items():
        print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")
