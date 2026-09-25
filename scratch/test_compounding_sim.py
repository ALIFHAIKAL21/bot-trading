import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np, pandas as pd

df = pd.read_parquet(r'c:\Ngoding\xau_deep_sniper\data\processed\xauusd_m30_labeled_15ch.parquet')
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(r'c:\Ngoding\xau_deep_sniper\checkpoints\predictions_15ch.npy')

from src.frame.constants import (
    SPREAD_PIPS, SLIPPAGE_PIPS, COMMISSION_PER_LOT, CONTRACT_SIZE,
    SL_ATR_MULT, TP_MAX_R, BE_TRIGGER_R, BE_BUFFER_PRICE, RATCHET_12_R,
    TRAIL_TRIGGER_R, TRAIL_DIST_R, STALE_DECAY_BARS, STALE_DECAY_R, TIME_BARRIER_BARS,
    TAU_BASE, UNCERTAINTY_MARGIN
)

# Test 2026
st_dt = np.datetime64('2026-01-01T00:00:00')
en_dt = np.datetime64('2026-08-31T23:59:59')

ts_arr = df['timestamp_utc'].values.astype('datetime64[s]')
mask = (ts_arr >= st_dt) & (ts_arr <= en_dt)
indices = np.where(mask)[0]

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

opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
dt_series = df['timestamp_utc'].dt
dows, hours, minutes = dt_series.dayofweek.values, dt_series.hour.values, dt_series.minute.values
day_ints = dt_series.strftime('%Y%m%d').astype(int).values

unique_days = np.unique(day_ints[indices])
day_to_indices = {}
for idx in indices:
    d = day_ints[idx]
    if d not in day_to_indices:
        day_to_indices[d] = []
    day_to_indices[d].append(idx)

session_defs = [
    (1*60, 4*60, "Asia Early"),
    (4*60 + 30, 7*60, "Asia Late"),
    (8*60 + 30, 12*60 + 30, "London Core"),
    (13*60, 17*60, "NY Open"),
    (17*60 + 30, 21*60, "NY Core")
]

capital = 500.0
eq_flat = capital
eq_dyn = capital

curve_flat = [eq_flat]
curve_dyn = [eq_dyn]
lots_dyn = []

for d_int in unique_days:
    day_idxs = day_to_indices[d_int]
    dow = dows[day_idxs[0]]
    if dow in [5, 6]: continue

    for w_st, w_en, s_name in session_defs:
        best_idx, best_conf, best_act = None, -1.0, None
        for b_idx in day_idxs:
            h, m = hours[b_idx], minutes[b_idx]
            if h == 7 or (h == 8 and m < 30): continue
            if dow == 4 and h >= 18: continue
            t_val = h * 60 + m
            if w_st <= t_val <= w_en:
                p_idx = b_idx - 63
                if 0 <= p_idx < len(preds):
                    probs = preds[p_idx]
                    p_hold = float(probs[0])
                    t_probs = probs[1:]
                    max_i = int(np.argmax(t_probs))
                    conf = float(t_probs[max_i])
                    if conf > best_conf and conf >= TAU_BASE and (conf - p_hold) >= UNCERTAINTY_MARGIN:
                        best_conf = conf
                        best_idx = b_idx
                        best_act = max_i + 1

        if best_idx is not None:
            d = "BUY" if best_act in [1, 2] else "SELL"
            atr_val = float(atr[best_idx]) if best_idx < len(atr) else 5.0
            sl_dist = round(atr_val * SL_ATR_MULT, 2)
            b_close = closes[best_idx]
            ep = round(b_close + 0.10, 2) if d == "BUY" else round(b_close - 0.10, 2)
            sl_p = round(ep - sl_dist, 2) if d == "BUY" else round(ep + sl_dist, 2)
            tp_max_p = round(ep + (TP_MAX_R * sl_dist), 2) if d == "BUY" else round(ep - (TP_MAX_R * sl_dist), 2)
            be_trigger_p = round(ep + (BE_TRIGGER_R * sl_dist), 2) if d == "BUY" else round(ep - (BE_TRIGGER_R * sl_dist), 2)

            cur_sl = sl_p
            be_activated = False
            exit_price = None

            for f_idx in range(best_idx + 1, min(best_idx + 16, len(df))):
                f_high, f_low, f_close = highs[f_idx], lows[f_idx], closes[f_idx]
                f_dow, f_hour = dows[f_idx], hours[f_idx]
                is_fri = (f_dow == 4 and f_hour >= 20) or f_dow in [5, 6]
                is_tb = (f_idx - best_idx >= TIME_BARRIER_BARS)

                if is_fri or is_tb:
                    exit_price = f_close
                    break

                bars_held = f_idx - best_idx
                if STALE_DECAY_BARS > 0 and bars_held >= STALE_DECAY_BARS and not be_activated:
                    decay_sl = round(ep - (STALE_DECAY_R * sl_dist), 2) if d == "BUY" else round(ep + (STALE_DECAY_R * sl_dist), 2)
                    cur_sl = max(cur_sl, decay_sl) if d == "BUY" else min(cur_sl, decay_sl)

                if d == "BUY":
                    if f_low <= cur_sl: exit_price = cur_sl; break
                    elif f_high >= tp_max_p: exit_price = tp_max_p; break
                    else:
                        if not be_activated and f_high >= be_trigger_p:
                            be_activated = True; cur_sl = max(cur_sl, round(ep + BE_BUFFER_PRICE, 2))
                        r_gain = (f_high - ep) / sl_dist
                        if RATCHET_12_R > 0 and r_gain >= 1.2: cur_sl = max(cur_sl, round(ep + (RATCHET_12_R * sl_dist), 2))
                        if r_gain >= TRAIL_TRIGGER_R: cur_sl = max(cur_sl, round(f_high - (TRAIL_DIST_R * sl_dist), 2))
                else:
                    if f_high >= cur_sl: exit_price = cur_sl; break
                    elif f_low <= tp_max_p: exit_price = tp_max_p; break
                    else:
                        if not be_activated and f_low <= be_trigger_p:
                            be_activated = True; cur_sl = min(cur_sl, round(ep - BE_BUFFER_PRICE, 2))
                        r_gain = (ep - f_low) / sl_dist
                        if RATCHET_12_R > 0 and r_gain >= 1.2: cur_sl = min(cur_sl, round(ep - (RATCHET_12_R * sl_dist), 2))
                        if r_gain >= TRAIL_TRIGGER_R: cur_sl = min(cur_sl, round(f_low + (TRAIL_DIST_R * sl_dist), 2))

            if exit_price is None: exit_price = closes[min(best_idx + 15, len(df)-1)]

            # Flat Mode (0.01 lot)
            price_diff = (exit_price - ep) if d == "BUY" else (ep - exit_price)
            fric_flat = round((SPREAD_PIPS * 0.10 * 0.01 * CONTRACT_SIZE) + (SLIPPAGE_PIPS * 0.10 * 0.01 * CONTRACT_SIZE * 2) + (COMMISSION_PER_LOT * 0.01), 3)
            pnl_flat = round(price_diff * 0.01 * CONTRACT_SIZE - fric_flat, 2)
            eq_flat += pnl_flat
            curve_flat.append(eq_flat)

            # Dynamic Compounding Mode
            # Lot scales with equity: 0.01 per $500 balance, clamped
            dyn_lot = round(max(0.01, min(5.0, (eq_dyn / capital) * 0.01)), 2)
            lots_dyn.append(dyn_lot)
            fric_dyn = round((SPREAD_PIPS * 0.10 * dyn_lot * CONTRACT_SIZE) + (SLIPPAGE_PIPS * 0.10 * dyn_lot * CONTRACT_SIZE * 2) + (COMMISSION_PER_LOT * dyn_lot), 3)
            pnl_dyn = round(price_diff * dyn_lot * CONTRACT_SIZE - fric_dyn, 2)
            eq_dyn += pnl_dyn
            curve_dyn.append(eq_dyn)

print(f"--- 2026 AUDIT (Initial: ${capital}) ---")
print(f"Flat 0.01 Lot      : Final Eq: ${eq_flat:,.2f} | Net PnL: ${eq_flat - capital:+,.2f} (+{((eq_flat-capital)/capital)*100:.1f}%)")
print(f"Dynamic Compounding: Final Eq: ${eq_dyn:,.2f} | Net PnL: ${eq_dyn - capital:+,.2f} (+{((eq_dyn-capital)/capital)*100:.1f}%)")
print(f"Peak Lot Reached   : {max(lots_dyn):.2f} Lot (Min: {min(lots_dyn):.2f}, Final: {lots_dyn[-1]:.2f})")
