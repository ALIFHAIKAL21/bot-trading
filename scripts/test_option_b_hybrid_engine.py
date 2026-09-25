"""
Option B: Zero-Retraining Hybrid Macro Decision Engine
Evaluates 2021 - 2025 like-for-like with Baseline.
Tests H4 Macro Trend Alignment & Pullback Continuation.
"""

import sys, os, pathlib, json
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

print("Memuat dataset dan prediksi...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# Hitung ATR(14)
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

# -------------------------------------------------------------
# 1. COMPUTE H4 MACRO TREND RIGOROUSLY (STRICTLY POINT-IN-TIME)
# -------------------------------------------------------------
print("Menghitung indikator H4 Macro tanpa lookahead bias...")
df_h4 = df.set_index('timestamp_utc').resample('4h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# H4 EMAs
df_h4['h4_ema20'] = df_h4['close'].ewm(span=20, adjust=False).mean()
df_h4['h4_ema50'] = df_h4['close'].ewm(span=50, adjust=False).mean()
df_h4['h4_ema200'] = df_h4['close'].ewm(span=200, adjust=False).mean()
df_h4['h4_atr'] = (df_h4['high'] - df_h4['low']).rolling(14).mean()

# H4 Slope (Normalized by ATR)
df_h4['h4_slope20'] = (df_h4['h4_ema20'] - df_h4['h4_ema20'].shift(3)) / (df_h4['h4_atr'] + 1e-8)

# Lag H4 by 1 bar so that M30 only knows the CLOSED H4 bar (zero lookahead)
df_h4_lagged = df_h4[['h4_ema20', 'h4_ema50', 'h4_ema200', 'h4_slope20']].shift(1)

# Merge back to M30 via forward fill
df = df.merge(df_h4_lagged, on='timestamp_utc', how='left')
df['h4_ema20'] = df['h4_ema20'].ffill()
df['h4_ema50'] = df['h4_ema50'].ffill()
df['h4_ema200'] = df['h4_ema200'].ffill()
df['h4_slope20'] = df['h4_slope20'].ffill()

# Parameter Eksekusi Standar (Baseline)
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
    {"year": 2021, "start": "2021-09-23 12:30:00", "end": "2021-12-31 23:59:59", "label": "2021 (Sep - Des)"},
    {"year": 2022, "start": "2022-01-01 00:00:00", "end": "2022-12-31 23:59:59", "label": "2022 (Jan - Des)"},
    {"year": 2023, "start": "2023-01-01 00:00:00", "end": "2023-12-31 23:59:59", "label": "2023 (Jan - Des)"},
    {"year": 2024, "start": "2024-01-01 00:00:00", "end": "2024-12-31 23:59:59", "label": "2024 (Jan - Des)"},
    {"year": 2025, "start": "2025-01-01 00:00:00", "end": "2025-12-31 23:59:59", "label": "2025 (Jan - Des)"},
]

def run_backtest(mode="baseline"):
    """
    mode = 'baseline' (Model M30 Asli tanpa macro filter)
    mode = 'option_b_align' (Macro Trend Alignment: Hanya ambil arah searah tren makro kuat)
    mode = 'option_b_smart' (Macro Alignment + Dynamic Uncapped Runner di arah tren makro)
    """
    results = {}
    
    for y_cfg in test_years:
        year = y_cfg["year"]
        st_ts = pd.to_datetime(y_cfg["start"], utc=True)
        end_ts = pd.to_datetime(y_cfg["end"], utc=True)
        
        n_bars = len(df)
        seq_offset = 63
        equity = INITIAL_EQUITY
        equity_curve = [equity]
        trades = []
        trade_counter = 0
        open_position = None
        
        current_day = None
        daily_count = 0
        
        for bar_idx in range(n_bars):
            row = df.iloc[bar_idx]
            ts = row["timestamp_utc"]
            bar_open, bar_high, bar_low, bar_close = row["open"], row["high"], row["low"], row["close"]
            is_blackout = bool(row.get("is_news_blackout", False))
            is_eligible = bool(row.get("entry_eligible", True))
            
            in_date_range = (ts >= st_ts) and (ts <= end_ts)
            day_str = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]
            if day_str != current_day:
                current_day = day_str
                daily_count = 0
                
            # Manage Open Position
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
                    if open_position["pos_a_open"]:
                        pnl_a = (exit_p - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_a * CONTRACT_SIZE
                        open_position["pnl_net_accum"] += (pnl_a - calc_friction(lot_a))
                        open_position["pos_a_open"] = False
                    if open_position["pos_b_open"]:
                        pnl_b = (exit_p - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_p) * lot_b * CONTRACT_SIZE
                        open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                        open_position["pos_b_open"] = False
                        
                    total_net = round(open_position["pnl_net_accum"], 2)
                    m_str = ts.strftime('%Y-%m')
                    trades.append({
                        'id': trade_counter, 'direction': d, 'pnl': total_net, 'win': 1 if total_net > 0 else 0,
                        'month': m_str, 'exit_reason': "Tutup Jumat" if is_fri_close else "Time Barrier"
                    })
                    trade_counter += 1
                    equity += total_net
                    open_position = None
                else:
                    if d == "BUY":
                        if not open_position["tp1_hit"] and bar_low <= open_position["cur_sl_a"]:
                            pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                            net_p = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                            m_str = ts.strftime('%Y-%m')
                            trades.append({'id': trade_counter, 'direction': d, 'pnl': net_p, 'win': 0, 'month': m_str, 'exit_reason': 'Stop Loss'})
                            trade_counter += 1
                            equity += net_p
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
                                # If mode option_b_smart, allow uncapped runner if in macro trend
                                is_uncapped = (mode == "option_b_smart") and open_position.get("in_macro_trend", False)
                                
                                if not is_uncapped and bar_high >= open_position["tp2_price"]:
                                    pnl_b = (open_position["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                                    open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                                    open_position["pos_b_open"] = False
                                    net_p = round(open_position["pnl_net_accum"], 2)
                                    m_str = ts.strftime('%Y-%m')
                                    trades.append({'id': trade_counter, 'direction': d, 'pnl': net_p, 'win': 1, 'month': m_str, 'exit_reason': 'Target TP2'})
                                    trade_counter += 1
                                    equity += net_p
                                    open_position = None
                                elif bar_low <= open_position["cur_sl_b"]:
                                    pnl_b = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                                    open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                                    open_position["pos_b_open"] = False
                                    net_p = round(open_position["pnl_net_accum"], 2)
                                    m_str = ts.strftime('%Y-%m')
                                    trades.append({'id': trade_counter, 'direction': d, 'pnl': net_p, 'win': 1 if net_p > 0 else 0, 'month': m_str, 'exit_reason': 'Breakeven / Trail'})
                                    trade_counter += 1
                                    equity += net_p
                                    open_position = None
                                elif open_position["tp1_hit"]:
                                    r_gain = (bar_high - ep) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(bar_high - (TRAIL_DIST_R * sl_dist), 2)
                                        open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                    else: # SELL
                        if not open_position["tp1_hit"] and bar_high >= open_position["cur_sl_a"]:
                            pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * CONTRACT_SIZE
                            net_p = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                            m_str = ts.strftime('%Y-%m')
                            trades.append({'id': trade_counter, 'direction': d, 'pnl': net_p, 'win': 0, 'month': m_str, 'exit_reason': 'Stop Loss'})
                            trade_counter += 1
                            equity += net_p
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
                                is_uncapped = (mode == "option_b_smart") and open_position.get("in_macro_trend", False)
                                
                                if not is_uncapped and bar_low <= open_position["tp2_price"]:
                                    pnl_b = (ep - open_position["tp2_price"]) * lot_b * CONTRACT_SIZE
                                    open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                                    open_position["pos_b_open"] = False
                                    net_p = round(open_position["pnl_net_accum"], 2)
                                    m_str = ts.strftime('%Y-%m')
                                    trades.append({'id': trade_counter, 'direction': d, 'pnl': net_p, 'win': 1, 'month': m_str, 'exit_reason': 'Target TP2'})
                                    trade_counter += 1
                                    equity += net_p
                                    open_position = None
                                elif bar_high >= open_position["cur_sl_b"]:
                                    pnl_b = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                                    open_position["pnl_net_accum"] += (pnl_b - calc_friction(lot_b))
                                    open_position["pos_b_open"] = False
                                    net_p = round(open_position["pnl_net_accum"], 2)
                                    m_str = ts.strftime('%Y-%m')
                                    trades.append({'id': trade_counter, 'direction': d, 'pnl': net_p, 'win': 1 if net_p > 0 else 0, 'month': m_str, 'exit_reason': 'Breakeven / Trail'})
                                    trade_counter += 1
                                    equity += net_p
                                    open_position = None
                                elif open_position["tp1_hit"]:
                                    r_gain = (ep - bar_low) / sl_dist
                                    if r_gain >= TRAIL_AFTER_R:
                                        trail_sl = round(bar_low + (TRAIL_DIST_R * sl_dist), 2)
                                        open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)
                                        
            if in_date_range:
                equity_curve.append(equity)
                
            # EVALUATE NEW ENTRY
            pred_idx = bar_idx - seq_offset
            if in_date_range and pred_idx >= 0 and pred_idx < len(preds) and open_position is None and daily_count < MAX_DAILY and is_eligible and not is_blackout:
                dow = ts.dayofweek
                hour = ts.hour
                minute = ts.minute
                
                friday_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
                london_quarantine = (hour == 7 or (hour == 8 and minute < 30))
                
                if not friday_freeze and not london_quarantine:
                    probs = preds[pred_idx]
                    trade_probs = probs[1:]
                    max_class_idx = int(np.argmax(trade_probs))
                    action_class = max_class_idx + 1
                    confidence = float(trade_probs[max_class_idx])
                    p_hold = float(probs[0])
                    
                    if confidence >= TAU and confidence > p_hold:
                        d = "BUY" if action_class in [1, 2] else "SELL"
                        ob_zone = float(row.get("order_block_zone", 0.0))
                        ma_slope = float(row.get("ma_ribbon_slope", 0.0))
                        
                        # BASELINE STRUCTURAL FILTER
                        ok = True
                        if d == "BUY" and (ob_zone < 0 or ma_slope < -0.3): ok = False
                        if d == "SELL" and (ob_zone > 0 or ma_slope > 0.3): ok = False
                        
                        # OPTION B: HYBRID MACRO ENGINE LOGIC
                        in_macro_trend = False
                        if mode in ["option_b_align", "option_b_smart"]:
                            h4_e20 = float(row.get("h4_ema20", 0.0))
                            h4_e50 = float(row.get("h4_ema50", 0.0))
                            h4_slp = float(row.get("h4_slope20", 0.0))
                            c_price = float(row.get("close", 0.0))
                            
                            # Deteksi Macro Bull Expansion
                            is_macro_bull = (c_price > h4_e50) and (h4_e20 > h4_e50) and (h4_slp > 0.05)
                            # Deteksi Macro Bear Expansion
                            is_macro_bear = (c_price < h4_e50) and (h4_e20 < h4_e50) and (h4_slp < -0.05)
                            
                            if is_macro_bull:
                                # DI TREN BULL BESAR (SEPERTI 2024):
                                # Dilarang keras ambil posisi SELL counter-trend!
                                if d == "SELL":
                                    ok = False
                                else:
                                    in_macro_trend = True
                            elif is_macro_bear:
                                # DI TREN BEAR BESAR:
                                # Dilarang keras ambil posisi BUY counter-trend!
                                if d == "BUY":
                                    ok = False
                                else:
                                    in_macro_trend = True
                            else:
                                # DI PASAR NORMAL / SIDEWAYS (2021, 2022, 2025):
                                # 100% BEBAS SEPERTI BASELINE!
                                pass
                                
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
                                    "risk_amount": actual_risk, "confidence": confidence,
                                    "pos_a_open": True, "pos_b_open": True, "tp1_hit": False, "tp2_hit": False,
                                    "pnl_net_accum": 0.0, "in_macro_trend": in_macro_trend
                                }
                                daily_count += 1
                                
        tdf = pd.DataFrame(trades)
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        max_dd = np.min((eq_arr - peak) / peak) * 100 if len(eq_arr) > 0 else 0
        n_tr = len(tdf)
        n_w = tdf['win'].sum() if n_tr > 0 else 0
        wr = (n_w / n_tr * 100) if n_tr > 0 else 0.0
        net_p = equity - INITIAL_EQUITY
        ret = (net_p / INITIAL_EQUITY) * 100
        
        gw = tdf[tdf['pnl'] > 0]['pnl'].sum() if n_tr > 0 else 0
        gl = abs(tdf[tdf['pnl'] < 0]['pnl'].sum()) if n_tr > 0 else 0
        pf = round(gw / gl, 2) if gl > 0 else 99.0
        
        results[year] = {
            "trades": n_tr,
            "wins": int(n_w),
            "losses": int(n_tr - n_w),
            "wr": round(wr, 1),
            "net_pnl": round(net_p, 2),
            "return_pct": round(ret, 2),
            "max_dd": round(max_dd, 2),
            "pf": pf
        }
        
    return results

print("\n" + "="*80)
print("1. RUNNING BASELINE (M30 Model Asli Sebelum Opsi B)...")
print("="*80)
res_baseline = run_backtest("baseline")

print("\n" + "="*80)
print("2. RUNNING OPTION B - MACRO ALIGNED ENGINE (Zero Retraining)...")
print("="*80)
res_opt_b = run_backtest("option_b_align")

print("\n" + "="*80)
print("3. RUNNING OPTION B - SMART RUNNER (Macro Aligned + Dynamic Uncapped)...")
print("="*80)
res_opt_b_smart = run_backtest("option_b_smart")

# Print Comparison Table
print("\n" + "#"*100)
print("HASIL PERBANDINGAN LENGKAP 5 TAHUN: BASELINE VS OPSI B")
print("#"*100)

comparison_rows = []
years = [2021, 2022, 2023, 2024, 2025]

print(f"{'Tahun':<6} | {'--- BASELINE SEBELUMNYA ---':<36} | {'--- OPSI B (HYBRID MACRO) ---':<36} | {'STATUS UPGRADE':<15}")
print(f"{'':<6} | {'Return':<10} {'Max DD':<10} {'WR':<6} {'PF':<6} | {'Return':<10} {'Max DD':<10} {'WR':<6} {'PF':<6} |")
print("-" * 105)

for yr in years:
    b = res_baseline[yr]
    o = res_opt_b[yr]
    
    b_ret_str = f"{b['return_pct']:+6.2f}%"
    b_dd_str = f"{b['max_dd']:6.2f}%"
    b_wr_str = f"{b['wr']:4.1f}%"
    b_pf_str = f"{b['pf']:4.2f}"
    
    o_ret_str = f"{o['return_pct']:+6.2f}%"
    o_dd_str = f"{o['max_dd']:6.2f}%"
    o_wr_str = f"{o['wr']:4.1f}%"
    o_pf_str = f"{o['pf']:4.2f}"
    
    # Check impact
    diff = o['return_pct'] - b['return_pct']
    if diff > 5.0:
        status = "MENINGKAT DRASTIS (PASS)"
    elif diff >= -0.5:
        status = "TERJAGA STABIL (PASS)"
    else:
        status = "PERLU ADJUST"
        
    print(f"{yr:<6} | {b_ret_str:<10} {b_dd_str:<10} {b_wr_str:<6} {b_pf_str:<6} | {o_ret_str:<10} {o_dd_str:<10} {o_wr_str:<6} {o_pf_str:<6} | {status}")

# Save JSON
out_json = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\option_b_audit_comparison.json')
with open(out_json, 'w') as f:
    json.dump({
        "baseline": res_baseline,
        "option_b_align": res_opt_b,
        "option_b_smart": res_opt_b_smart
    }, f, indent=2)

print(f"\nHasil audit lengkap tersimpan di: {out_json}")
