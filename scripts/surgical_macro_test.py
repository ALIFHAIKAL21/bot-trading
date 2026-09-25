"""
Deep Diagnostic: Why did 2022/2025 lose trades, and how to protect 2022/2025 while fixing 2023/2024?
Analyzing the winning vs losing counter-trend trades across all 5 years.
"""

import sys, os, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# Compute H4 indicators
df_h4 = df.set_index('timestamp_utc').resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

df_h4['h4_ema20'] = df_h4['close'].ewm(span=20, adjust=False).mean()
df_h4['h4_ema50'] = df_h4['close'].ewm(span=50, adjust=False).mean()
df_h4['h4_ema200'] = df_h4['close'].ewm(span=200, adjust=False).mean()
df_h4['h4_atr'] = (df_h4['high'] - df_h4['low']).rolling(14).mean()
df_h4['h4_slope20'] = (df_h4['h4_ema20'] - df_h4['h4_ema20'].shift(3)) / (df_h4['h4_atr'] + 1e-8)

# Highest high over last 20 days (120 H4 bars)
df_h4['h4_high_20d'] = df_h4['high'].rolling(120).max()
df_h4['h4_ath_proximity'] = (df_h4['close'] >= df_h4['h4_high_20d'] * 0.995).astype(int)

df_h4_lagged = df_h4[['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20', 'h4_ath_proximity']].shift(1)
df = df.merge(df_h4_lagged, on='timestamp_utc', how='left')
df['h4_ema20'] = df['h4_ema20'].ffill()
df['h4_ema50'] = df['h4_ema50'].ffill()
df['h4_ema200'] = df['h4_ema200'].ffill()
df['h4_slope20'] = df['h4_slope20'].ffill()
df['h4_ath_proximity'] = df['h4_ath_proximity'].ffill()

# ATR
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
    {"year": 2021, "start": "2021-09-23 12:30:00", "end": "2021-12-31 23:59:59"},
    {"year": 2022, "start": "2022-01-01 00:00:00", "end": "2022-12-31 23:59:59"},
    {"year": 2023, "start": "2023-01-01 00:00:00", "end": "2023-12-31 23:59:59"},
    {"year": 2024, "start": "2024-01-01 00:00:00", "end": "2024-12-31 23:59:59"},
    {"year": 2025, "start": "2025-01-01 00:00:00", "end": "2025-12-31 23:59:59"},
]

def run_experiment(slope_thresh=0.20, check_ath_only=True, convert_to_buy=False):
    ann_results = {}
    for y_cfg in test_years:
        year = y_cfg["year"]
        st_ts = pd.to_datetime(y_cfg["start"], utc=True)
        end_ts = pd.to_datetime(y_cfg["end"], utc=True)
        
        equity = INITIAL_EQUITY
        equity_curve = [equity]
        trades = []
        trade_counter = 0
        open_position = None
        current_day = None
        daily_count = 0
        seq_offset = 63
        
        for bar_idx in range(len(df)):
            row = df.iloc[bar_idx]
            ts = row["timestamp_utc"]
            bar_close = row["close"]
            bar_high = row["high"]
            bar_low = row["low"]
            in_date_range = (ts >= st_ts) and (ts <= end_ts)
            day_str = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]
            if day_str != current_day:
                current_day = day_str
                daily_count = 0
                
            if open_position is not None:
                bars_held = bar_idx - open_position["entry_bar_idx"]
                d = open_position["direction"]
                ep = open_position["entry_price"]
                sl_dist = open_position["sl_dist"]
                lot_a = open_position["lot_a"]
                lot_b = open_position["lot_b"]
                dow = ts.dayofweek
                hour = ts.hour
                is_fri_close = (dow == 4 and hour >= 20) or dow in [5, 6]
                is_tb = (bars_held >= 16)
                
                if is_fri_close or is_tb:
                    exit_p = bar_close
                    pnl_a = (exit_p - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_a * CONTRACT_SIZE
                    pnl_b = (exit_p - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_b * CONTRACT_SIZE
                    tot = round((pnl_a - calc_friction(lot_a)) + (pnl_b - calc_friction(lot_b)), 2)
                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot})
                    equity += tot
                    open_position = None
                else:
                    if d == "BUY":
                        if not open_position["tp1_hit"] and bar_low <= open_position["cur_sl_a"]:
                            pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                            tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                            trades.append({'win': 0, 'pnl': tot})
                            equity += tot
                            open_position = None
                        else:
                            if open_position["pos_a_open"] and bar_high >= open_position["tp1_price"]:
                                pnl_a = (open_position["tp1_price"] - ep) * lot_a * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                                open_position["pos_a_open"] = False
                                open_position["tp1_hit"] = True
                                f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                                open_position["cur_sl_b"] = max(open_position["cur_sl_b"], round(ep + f_p, 2))
                            if open_position["pos_b_open"]:
                                if bar_high >= open_position["tp2_price"]:
                                    pnl_b = (open_position["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1, 'pnl': tot})
                                    equity += tot
                                    open_position = None
                                elif bar_low <= open_position["cur_sl_b"]:
                                    pnl_b = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot})
                                    equity += tot
                                    open_position = None
                                elif open_position["tp1_hit"]:
                                    r_gain = (bar_high - ep) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(bar_high - (TRAIL_DIST_R * sl_dist), 2)
                                        open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                    else: # SELL
                        if not open_position["tp1_hit"] and bar_high >= open_position["cur_sl_a"]:
                            pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * CONTRACT_SIZE
                            tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                            trades.append({'win': 0, 'pnl': tot})
                            equity += tot
                            open_position = None
                        else:
                            if open_position["pos_a_open"] and bar_low <= open_position["tp1_price"]:
                                pnl_a = (ep - open_position["tp1_price"]) * lot_a * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                                open_position["pos_a_open"] = False
                                open_position["tp1_hit"] = True
                                f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                                open_position["cur_sl_b"] = min(open_position["cur_sl_b"], round(ep - f_p, 2))
                            if open_position["pos_b_open"]:
                                if bar_low <= open_position["tp2_price"]:
                                    pnl_b = (ep - open_position["tp2_price"]) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1, 'pnl': tot})
                                    equity += tot
                                    open_position = None
                                elif bar_high >= open_position["cur_sl_b"]:
                                    pnl_b = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                                    tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                                    trades.append({'win': 1 if tot > 0 else 0, 'pnl': tot})
                                    equity += tot
                                    open_position = None
                                elif open_position["tp1_hit"]:
                                    r_gain = (ep - bar_low) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(bar_low + (TRAIL_DIST_R * sl_dist), 2)
                                        open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)
            
            if in_date_range:
                equity_curve.append(equity)
                
            pred_idx = bar_idx - seq_offset
            if in_date_range and pred_idx >= 0 and pred_idx < len(preds) and open_position is None and daily_count < MAX_DAILY:
                dow = ts.dayofweek
                hour = ts.hour
                minute = ts.minute
                if not ((dow == 4 and hour >= 18) or dow in [5, 6]) and not (hour == 7 or (hour == 8 and minute < 30)):
                    probs = preds[pred_idx]
                    trade_probs = probs[1:]
                    max_class_idx = int(np.argmax(trade_probs))
                    action_class = max_class_idx + 1
                    conf = float(trade_probs[max_class_idx])
                    p_hold = float(probs[0])
                    
                    if conf >= TAU and conf > p_hold:
                        d = "BUY" if action_class in [1, 2] else "SELL"
                        ob = float(row.get("order_block_zone", 0.0))
                        m_slp = float(row.get("ma_ribbon_slope", 0.0))
                        
                        ok = True
                        if d == "BUY" and (ob < 0 or m_slp < -0.3): ok = False
                        if d == "SELL" and (ob > 0 or m_slp > 0.3): ok = False
                        
                        # SMART SURGICAL CONDITION:
                        # Only target the true monster breakout regime!
                        h4_slp = float(row.get("h4_slope20", 0.0))
                        is_ath = bool(row.get("h4_ath_proximity", 0)) if check_ath_only else True
                        
                        # Parabolic breakout: H4 Slope is sharply positive AND price is at 20-day high!
                        is_parabolic_bull = (h4_slp > slope_thresh) and is_ath
                        is_parabolic_bear = (h4_slp < -slope_thresh)
                        
                        if is_parabolic_bull and d == "SELL":
                            if convert_to_buy:
                                d = "BUY" # Convert momentum to BUY!
                            else:
                                ok = False # Only block this specific killer trap
                        elif is_parabolic_bear and d == "BUY":
                            ok = False
                            
                        if ok:
                            atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
                            sl_dist = round(atr_val * SL_ATR_MULT, 2)
                            if sl_dist > 0:
                                target_risk = equity * BASE_RISK
                                raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                                lot_size = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 50.0))
                                actual_risk = lot_size * sl_dist * CONTRACT_SIZE
                                lot_a = max(0.01, round(lot_size * POS_A_PCT, 2))
                                lot_b = max(0.01, round(lot_size - lot_a, 2))
                                
                                if d == "BUY":
                                    ep = round(bar_close + SLIPPAGE_PRICE, 2)
                                    sl_p = round(ep - sl_dist, 2)
                                    tp1_p = round(ep + (TP1_R * sl_dist), 2)
                                    tp2_p = round(ep + (TP2_R * sl_dist), 2)
                                else:
                                    ep = round(bar_close - SLIPPAGE_PRICE, 2)
                                    sl_p = round(ep + sl_dist, 2)
                                    tp1_p = round(ep - (TP1_R * sl_dist), 2)
                                    tp2_p = round(ep - (TP2_R * sl_dist), 2)
                                    
                                open_position = {
                                    "direction": d, "entry_bar_idx": bar_idx, "entry_timestamp": ts,
                                    "entry_price": ep, "sl_dist": sl_dist, "initial_sl": sl_p,
                                    "cur_sl_a": sl_p, "cur_sl_b": sl_p, "tp1_price": tp1_p, "tp2_price": tp2_p,
                                    "total_lot": lot_size, "lot_a": lot_a, "lot_b": lot_b,
                                    "risk_amount": actual_risk, "pos_a_open": True, "pos_b_open": True,
                                    "tp1_hit": False, "pnl_net_accum": 0.0
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
        ann_results[year] = {"trades": n_tr, "ret": round(ret, 2), "max_dd": round(max_dd, 2), "wr": round(wr, 1)}
        
    return ann_results

print("\n--- BASELINE ---")
print("2021: +6.19% | 2022: +38.01% | 2023: -9.23% | 2024: -11.55% | 2025: +23.29%")

print("\n--- UJI 1: Parabolic Breakout Gate (Slope > 0.15 + ATH Proximity) ---")
res1 = run_experiment(slope_thresh=0.15, check_ath_only=True, convert_to_buy=False)
for y, v in res1.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")

print("\n--- UJI 2: Smart Re-Orientation (Convert Parabolic Sells to BUY Continuation) ---")
res2 = run_experiment(slope_thresh=0.15, check_ath_only=True, convert_to_buy=True)
for y, v in res2.items():
    print(f"  {y}: Return {v['ret']:+6.2f}% | Max DD {v['max_dd']:6.2f}% | WR {v['wr']:4.1f}% | Trades {v['trades']}")
