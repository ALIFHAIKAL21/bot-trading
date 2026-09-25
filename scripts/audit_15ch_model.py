"""
Deep Diagnostic & 5-Year Institutional Audit of the Newly Trained 15-Channel MOMENT Model
========================================================================================
Checkpoint: checkpoints/best_moment_15ch_lora.pt
Dataset:    data/processed/xauusd_m30_labeled_15ch.parquet
"""
import sys, pathlib, json
import torch
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.moment_model import MOMENTConfig, MOMENTClassifier
from src.models.dataset import XAUTimeSeriesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Audit Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

data_path = project_root / "data" / "processed" / "xauusd_m30_labeled_15ch.parquet"
ckpt_path = project_root / "checkpoints" / "best_moment_15ch_lora.pt"

print(f"Loading 15-channel dataset: {data_path}...")
df = pd.read_parquet(data_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

# 1. Load Checkpoint and Rebuild Model
print(f"Loading checkpoint: {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model_cfg = ckpt.get("config", None)
if model_cfg is None:
    model_cfg = MOMENTConfig(
        n_channels=15, seq_len=64, patch_len=8, patch_stride=8,
        d_model=1024, num_layers=6, num_heads=16, d_ff=2816,
        dropout=0.2, num_classes=5, use_lora=True, lora_r=32, lora_alpha=64
    )

model = MOMENTClassifier(model_cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model successfully restored from Epoch {ckpt.get('epoch', '?')} (Composite Score: {ckpt.get('composite_score', 0):.4f})")

# 2. Generate Predictions Across Entire 58,000 Bars
print("\nGenerating full-history inference predictions in batches of 256...")
dataset = XAUTimeSeriesDataset(df, sequence_length=64)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

all_probs = []
with torch.no_grad():
    for x_b, _ in loader:
        x_b = x_b.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x_b)
            probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

preds_15ch = np.concatenate(all_probs, axis=0)
print(f"Generated predictions shape: {preds_15ch.shape}")

# Save predictions for record
np.save(project_root / "checkpoints" / "predictions_15ch.npy", preds_15ch)

# 3. Simulate Institutional Execution Engine (2021 - 2026)
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

test_periods = [
    {"year": "2021", "start_dt": np.datetime64('2021-09-23T12:30:00'), "end_dt": np.datetime64('2021-12-31T23:59:59')},
    {"year": "2022", "start_dt": np.datetime64('2022-01-01T00:00:00'), "end_dt": np.datetime64('2022-12-31T23:59:59')},
    {"year": "2023", "start_dt": np.datetime64('2023-01-01T00:00:00'), "end_dt": np.datetime64('2023-12-31T23:59:59')},
    {"year": "2024", "start_dt": np.datetime64('2024-01-01T00:00:00'), "end_dt": np.datetime64('2024-12-31T23:59:59')},
    {"year": "2025", "start_dt": np.datetime64('2025-01-01T00:00:00'), "end_dt": np.datetime64('2025-12-31T23:59:59')},
    {"year": "2026 (Q1)", "start_dt": np.datetime64('2026-01-01T00:00:00'), "end_dt": np.datetime64('2026-03-31T23:59:59')},
]

def simulate_backtest(preds_matrix, tau_val=0.355):
    seq_offset = 63
    n_bars = len(df)
    results = {}
    
    for p_cfg in test_periods:
        period_lbl = p_cfg["year"]
        st = p_cfg["start_dt"]
        en = p_cfg["end_dt"]
        
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
                                if b_high >= open_position["tp2_price"]:
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
                                if b_low <= open_position["tp2_price"]:
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
            if in_range and pred_idx >= 0 and pred_idx < len(preds_matrix) and open_position is None and daily_count < MAX_DAILY:
                fri_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
                lon_quar = (hour == 7 or (hour == 8 and minute < 30))
                if not fri_freeze and not lon_quar:
                    probs = preds_matrix[pred_idx]
                    trade_probs = probs[1:]
                    max_class_idx = int(np.argmax(trade_probs))
                    action_class = max_class_idx + 1
                    conf = float(trade_probs[max_class_idx])
                    p_hold = float(probs[0])
                    
                    if conf >= tau_val and conf > p_hold:
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
        
        buy_sub = tdf[tdf['direction'] == 'BUY'] if n_tr > 0 else pd.DataFrame()
        sell_sub = tdf[tdf['direction'] == 'SELL'] if n_tr > 0 else pd.DataFrame()
        
        results[period_lbl] = {
            "trades": n_tr,
            "ret": round(ret, 2),
            "max_dd": round(max_dd, 2),
            "wr": round(wr, 1),
            "buy_trades": len(buy_sub),
            "buy_wr": round((buy_sub['win'].sum() / len(buy_sub) * 100) if len(buy_sub) > 0 else 0, 1),
            "buy_pnl": round(buy_sub['pnl'].sum() if len(buy_sub) > 0 else 0, 2),
            "sell_trades": len(sell_sub),
            "sell_wr": round((sell_sub['win'].sum() / len(sell_sub) * 100) if len(sell_sub) > 0 else 0, 1),
            "sell_pnl": round(sell_sub['pnl'].sum() if len(sell_sub) > 0 else 0, 2),
        }
        
    return results

print("\n=======================================================")
print("EVALUASI HASIL TRAINING MODEL 15-CHANNEL (2021 - 2026)")
print("=======================================================")

# Test across tau thresholds
for t_val in [0.355, 0.380, 0.400]:
    print(f"\n--- HASIL BACKTEST DENGAN TAU = {t_val:.3f} ---")
    b_res = simulate_backtest(preds_15ch, tau_val=t_val)
    for p, v in b_res.items():
        print(f"  {p:10s} | Return: {v['ret']:+7.2f}% | Max DD: {v['max_dd']:6.2f}% | WR: {v['wr']:4.1f}% | Trades: {v['trades']:3d} (BUY: {v['buy_trades']} / SELL: {v['sell_trades']})")
