"""
Fine-tuning and Optimization for $200 Capital (Jan - Mar 2026)
Exploring:
1. SL ATR Multipliers: 0.8, 0.9, 1.0, 1.1, 1.2, 1.3
2. TP Ratios: TP 1.0R, 1.2R, 1.5R, 1.8R, 2.0R
3. Compounding thresholds: 0.01 lot base, scaling to 0.02 at $400, 0.03 at $800, etc.
4. Trailing Stop & Breakeven
"""

import sys, os, pathlib, json
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

START_DATE = "2026-01-01 00:00:00"
END_DATE = "2026-03-31 23:59:59"
INITIAL_EQUITY = 200.0

sub_mask = (df['timestamp_utc'] >= START_DATE) & (df['timestamp_utc'] <= END_DATE)
sub_indices = df[sub_mask].index
days = sorted(df.loc[sub_indices, 'timestamp_utc'].dt.strftime('%Y-%m-%d').unique())
seq_offset = 63

def evaluate_candidate(sl_atr_mult=1.0, tp_r=1.5, scale_up=False, tau_gate=0.0):
    spread_price = 0.75 * 0.10
    slippage_price = 0.3 * 0.10
    commission = 3.50
    contract_size = 100.0

    def calc_friction(lot):
        return round(spread_price * lot * contract_size + slippage_price * lot * contract_size * 2 + commission * lot, 2)

    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    trade_counter = 0
    margin_called = False
    min_equity = equity

    session_ranges = [
        (1, 7, "Sesi Pagi (Asia)"),
        (8, 13, "Sesi Siang (London)"),
        (14, 20, "Sesi Sore (New York)")
    ]

    for day_str in days:
        if margin_called:
            break

        day_mask = (df['timestamp_utc'].dt.strftime('%Y-%m-%d') == day_str) & sub_mask
        day_indices = df[day_mask].index
        day_ts = df.loc[day_indices[0], 'timestamp_utc']
        if day_ts.dayofweek in [5, 6]:
            continue

        for s_start, s_end, s_name in session_ranges:
            best_bar_idx = None
            best_conf = -1.0
            best_action = None

            for b_idx in day_indices:
                row = df.iloc[b_idx]
                h = row['timestamp_utc'].hour
                m = row['timestamp_utc'].minute
                if h == 7 or (h == 8 and m < 30):
                    continue
                if s_start <= h <= s_end:
                    p_idx = b_idx - seq_offset
                    if 0 <= p_idx < len(preds):
                        probs = preds[p_idx]
                        t_probs = probs[1:]
                        max_idx = int(np.argmax(t_probs))
                        conf = float(t_probs[max_idx])
                        if conf > best_conf:
                            best_conf = conf
                            best_bar_idx = b_idx
                            best_action = max_idx + 1

            if best_bar_idx is not None and best_conf >= tau_gate:
                if equity <= 25.0:
                    margin_called = True
                    break

                entry_row = df.iloc[best_bar_idx]
                entry_ts = entry_row["timestamp_utc"]
                bar_close = entry_row["close"]
                d = "BUY" if best_action in [1, 2] else "SELL"

                atr_val = float(atr[best_bar_idx]) if best_bar_idx < len(atr) else 5.0
                sl_dist = round(atr_val * sl_atr_mult, 2)

                # Sizing
                if not scale_up:
                    lot_size = 0.01
                else:
                    if equity < 400.0:
                        lot_size = 0.01
                    elif equity < 800.0:
                        lot_size = 0.02
                    elif equity < 1400.0:
                        lot_size = 0.03
                    elif equity < 2000.0:
                        lot_size = 0.04
                    else:
                        lot_size = 0.05

                if d == "BUY":
                    ep = round(bar_close + slippage_price, 2)
                    sl_p = round(ep - sl_dist, 2)
                    tp_p = round(ep + (tp_r * sl_dist), 2)
                else:
                    ep = round(bar_close - slippage_price, 2)
                    sl_p = round(ep + sl_dist, 2)
                    tp_p = round(ep - (tp_r * sl_dist), 2)

                pnl_accum = 0.0
                exit_reason = None
                exit_ts = None

                for f_idx in range(best_bar_idx + 1, min(best_bar_idx + 16, len(df))):
                    f_row = df.iloc[f_idx]
                    f_ts = f_row["timestamp_utc"]
                    f_high = f_row["high"]
                    f_low = f_row["low"]
                    f_close = f_row["close"]
                    f_dow = f_ts.dayofweek
                    f_hour = f_ts.hour

                    is_fri_close = (f_dow == 4 and f_hour >= 20)
                    is_tb = (f_idx - best_bar_idx >= 12)

                    if is_fri_close or is_tb:
                        exit_reason = "Tutup Jumat" if is_fri_close else "Time Barrier"
                        exit_ts = f_ts
                        pnl_g = (f_close - ep) * lot_size * contract_size if d == "BUY" else (ep - f_close) * lot_size * contract_size
                        pnl_accum = pnl_g - calc_friction(lot_size)
                        break
                    else:
                        if d == "BUY":
                            if f_low <= sl_p:
                                pnl_g = (sl_p - ep) * lot_size * contract_size
                                pnl_accum = pnl_g - calc_friction(lot_size)
                                exit_reason = "Stop Loss"
                                exit_ts = f_ts
                                break
                            elif f_high >= tp_p:
                                pnl_g = (tp_p - ep) * lot_size * contract_size
                                pnl_accum = pnl_g - calc_friction(lot_size)
                                exit_reason = "Take Profit"
                                exit_ts = f_ts
                                break
                        else:
                            if f_high >= sl_p:
                                pnl_g = (ep - sl_p) * lot_size * contract_size
                                pnl_accum = pnl_g - calc_friction(lot_size)
                                exit_reason = "Stop Loss"
                                exit_ts = f_ts
                                break
                            elif f_low <= tp_p:
                                pnl_g = (ep - tp_p) * lot_size * contract_size
                                pnl_accum = pnl_g - calc_friction(lot_size)
                                exit_reason = "Take Profit"
                                exit_ts = f_ts
                                break

                pnl_final = round(pnl_accum, 2)
                trades.append({
                    'id': trade_counter + 1,
                    'month': entry_ts.strftime('%Y-%m'),
                    'day': day_str,
                    'pnl': pnl_final,
                    'win': 1 if pnl_final > 0 else 0,
                    'lot': lot_size,
                    'exit_reason': exit_reason
                })
                trade_counter += 1
                equity += pnl_final
                equity_curve.append(equity)
                if equity < min_equity:
                    min_equity = equity

    tdf = pd.DataFrame(trades)
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = np.min((eq_arr - peak) / peak) * 100 if len(eq_arr) > 0 else 0.0

    return {
        "sl_mult": sl_atr_mult,
        "tp_r": tp_r,
        "scale_up": scale_up,
        "tau_gate": tau_gate,
        "final_equity": round(equity, 2),
        "min_equity": round(min_equity, 2),
        "net_pnl": round(equity - INITIAL_EQUITY, 2),
        "return_pct": round((equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100, 2),
        "max_dd_pct": round(max_dd, 2),
        "total_trades": len(tdf),
        "win_rate": round(tdf['win'].sum() / len(tdf) * 100, 2) if len(tdf) > 0 else 0.0,
        "margin_called": margin_called,
        "tdf": tdf,
        "eq_curve": eq_arr
    }

# RUN SWEEP
print("Sweeping SL ATR Multiplier & TP Ratios...")
sweep_results = []
for sl_m in [0.8, 0.9, 1.0, 1.1, 1.2]:
    for tp_r in [1.0, 1.2, 1.5, 1.8, 2.0]:
        for scale in [False, True]:
            res = evaluate_candidate(sl_atr_mult=sl_m, tp_r=tp_r, scale_up=scale, tau_gate=0.0)
            sweep_results.append(res)

# Sort by final equity
sweep_results.sort(key=lambda x: x["final_equity"], reverse=True)

print("\nTOP 10 KONFIGURASI PROFIT TERTINGGI DENGAN MODAL $200 (Q1 2026):")
print("-" * 100)
for idx, r in enumerate(sweep_results[:10]):
    status = "MARGIN CALL" if r["margin_called"] else "SURVIVED"
    sc_txt = "Scale-Up" if r["scale_up"] else "Fixed 0.01"
    print(f"#{idx+1:2d} | SL: {r['sl_mult']:.1f} ATR | TP: {r['tp_r']:.1f}R | {sc_txt:10s} | Saldo: ${r['final_equity']:8.2f} (+{r['return_pct']:6.1f}%) | Min: ${r['min_equity']:6.2f} | MaxDD: {r['max_dd_pct']:5.1f}% | WR: {r['win_rate']:4.1f}% | {status}")

# Monthly breakdown of #1
best = sweep_results[0]
print(f"\nDETAIL BULAN KE BULAN UNTUK JUARA #1 (SL {best['sl_mult']} ATR, TP {best['tp_r']}R, {'Scale-Up' if best['scale_up'] else 'Fixed 0.01'}):")
tdf_b = best["tdf"]
cur_e = 200.0
for m in ['2026-01', '2026-02', '2026-03']:
    sub = tdf_b[tdf_b['month'] == m]
    sw = sub['win'].sum()
    spnl = sub['pnl'].sum()
    pct = (spnl / cur_e) * 100
    cur_e += spnl
    print(f"  {m} | Trades: {len(sub):2d} | Menang: {sw:2d} ({sw/len(sub)*100:4.1f}%) | PnL: ${spnl:+7.2f} ({pct:+6.1f}%) | Saldo: ${cur_e:8.2f}")

# Also check Best Fixed 0.01
best_fixed = next(r for r in sweep_results if not r["scale_up"])
print(f"\nDETAIL BULAN KE BULAN UNTUK JUARA FIXED 0.01 LOT (SL {best_fixed['sl_mult']} ATR, TP {best_fixed['tp_r']}R):")
tdf_f = best_fixed["tdf"]
cur_e = 200.0
for m in ['2026-01', '2026-02', '2026-03']:
    sub = tdf_f[tdf_f['month'] == m]
    sw = sub['win'].sum()
    spnl = sub['pnl'].sum()
    pct = (spnl / cur_e) * 100
    cur_e += spnl
    print(f"  {m} | Trades: {len(sub):2d} | Menang: {sw:2d} ({sw/len(sub)*100:4.1f}%) | PnL: ${spnl:+7.2f} ({pct:+6.1f}%) | Saldo: ${cur_e:8.2f}")

# Save best config
best_summary = {
    "top_config": {
        "sl_mult": best["sl_mult"],
        "tp_r": best["tp_r"],
        "scale_up": best["scale_up"],
        "final_equity": best["final_equity"],
        "net_pnl": best["net_pnl"],
        "return_pct": best["return_pct"],
        "max_dd_pct": best["max_dd_pct"],
        "min_equity": best["min_equity"],
        "win_rate": best["win_rate"],
        "total_trades": best["total_trades"],
    },
    "best_fixed_config": {
        "sl_mult": best_fixed["sl_mult"],
        "tp_r": best_fixed["tp_r"],
        "final_equity": best_fixed["final_equity"],
        "net_pnl": best_fixed["net_pnl"],
        "return_pct": best_fixed["return_pct"],
        "max_dd_pct": best_fixed["max_dd_pct"],
        "min_equity": best_fixed["min_equity"],
        "win_rate": best_fixed["win_rate"],
        "total_trades": best_fixed["total_trades"],
    }
}

with open(r'c:\Ngoding\bot_trading\scratch\best_200_dollar_config.json', 'w') as f:
    json.dump(best_summary, f, indent=2)

print("\nDone fine-tuning!")
