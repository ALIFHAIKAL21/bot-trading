import sys, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

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

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
ts_arr = df['timestamp_utc'].values.astype('datetime64[s]')
ob_zones = df['order_block_zone'].values
ma_slopes = df['ma_ribbon_slope'].values
dt_series = df['timestamp_utc'].dt
dows = dt_series.dayofweek.values
hours = dt_series.hour.values
minutes = dt_series.minute.values
day_ints = dt_series.strftime('%Y%m%d').astype(int).values
years = dt_series.year.values

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

seq_offset = 63
equity = INITIAL_EQUITY
all_trades = []
open_position = None
current_day = -1
daily_count = 0

for bar_idx in range(len(df)):
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
    yr = years[bar_idx]
    
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
            all_trades.append({'year': open_position['year'], 'direction': d, 'pnl': tot, 'win': 1 if tot > 0 else 0, 'exit': 'TB/Fri', 'entry_bar': open_position["entry_bar_idx"]})
            equity += tot
            open_position = None
        else:
            if d == "BUY":
                if not open_position["tp1_hit"] and b_low <= open_position["cur_sl_a"]:
                    pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                    tot = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                    all_trades.append({'year': open_position['year'], 'direction': d, 'pnl': tot, 'win': 0, 'exit': 'SL', 'entry_bar': open_position["entry_bar_idx"]})
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
                            all_trades.append({'year': open_position['year'], 'direction': d, 'pnl': tot, 'win': 1, 'exit': 'TP2', 'entry_bar': open_position["entry_bar_idx"]})
                            equity += tot
                            open_position = None
                        elif b_low <= open_position["cur_sl_b"]:
                            pnl_b = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                            tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                            all_trades.append({'year': open_position['year'], 'direction': d, 'pnl': tot, 'win': 1 if tot > 0 else 0, 'exit': 'BE/Trail', 'entry_bar': open_position["entry_bar_idx"]})
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
                    all_trades.append({'year': open_position['year'], 'direction': d, 'pnl': tot, 'win': 0, 'exit': 'SL', 'entry_bar': open_position["entry_bar_idx"]})
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
                            all_trades.append({'year': open_position['year'], 'direction': d, 'pnl': tot, 'win': 1, 'exit': 'TP2', 'entry_bar': open_position["entry_bar_idx"]})
                            equity += tot
                            open_position = None
                        elif b_high >= open_position["cur_sl_b"]:
                            pnl_b = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                            tot = round(open_position["pnl_net_accum"] + pnl_b - calc_friction(lot_b), 2)
                            all_trades.append({'year': open_position['year'], 'direction': d, 'pnl': tot, 'win': 1 if tot > 0 else 0, 'exit': 'BE/Trail', 'entry_bar': open_position["entry_bar_idx"]})
                            equity += tot
                            open_position = None
                        elif open_position["tp1_hit"]:
                            r_gain = (ep - b_low) / sl_dist
                            if r_gain >= TRAIL_AFTER_R:
                                trail_sl = round(b_low + (TRAIL_DIST_R * sl_dist), 2)
                                open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)
                                
    pred_idx = bar_idx - seq_offset
    if pred_idx >= 0 and pred_idx < len(preds) and open_position is None and daily_count < MAX_DAILY:
        probs = preds[pred_idx]
        trade_probs = probs[1:]
        max_idx = int(np.argmax(trade_probs))
        action_class = max_idx + 1
        conf = float(trade_probs[max_idx])
        p_hold = float(probs[0])
        
        if conf >= TAU and conf > p_hold:
            d = "BUY" if action_class in [1, 2] else "SELL"
            ob = ob_zones[bar_idx]
            m_slp = ma_slopes[bar_idx]
            ok = True
            if d == "BUY" and (ob < 0 or m_slp < -0.3): ok = False
            if d == "SELL" and (ob > 0 or m_slp > 0.3): ok = False
            
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
                        "direction": d, "entry_bar_idx": bar_idx, "year": yr,
                        "entry_price": ep, "sl_dist": sl_dist,
                        "cur_sl_a": sl_p, "cur_sl_b": sl_p, "tp1_price": tp1_p, "tp2_price": tp2_p,
                        "total_lot": lot_size, "lot_a": lot_a, "lot_b": lot_b,
                        "pos_a_open": True, "pos_b_open": True, "tp1_hit": False,
                        "pnl_net_accum": 0.0
                    }
                    daily_count += 1

tdf = pd.DataFrame(all_trades)
print("\n--- PERFORMANCE BY YEAR AND DIRECTION (BASELINE) ---")
for yr in [2021, 2022, 2023, 2024, 2025]:
    sub = tdf[tdf['year'] == yr]
    print(f"\nYEAR {yr}: Total Trades = {len(sub)}, Net PnL = ${sub['pnl'].sum():.2f}")
    for d in ['BUY', 'SELL']:
        d_sub = sub[sub['direction'] == d]
        if len(d_sub) > 0:
            wr = (d_sub['win'].sum() / len(d_sub)) * 100
            print(f"  {d}: N={len(d_sub)}, WinRate={wr:.1f}%, PnL=${d_sub['pnl'].sum():.2f}")
