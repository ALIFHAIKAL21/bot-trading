"""
Comprehensive Stress Test & Profit Optimization for Jan - Mar 2026 ($200k Modal)
Tests multiple execution regimes, risk fractions, compounding, and stress scenarios.
"""

import sys, os, pathlib, json
import numpy as np
import pandas as pd
from typing import Dict, List, Any

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
sys.path.insert(0, str(project_root))
sys.path.insert(0, r'c:\Ngoding\bot_trading')

from scripts.backtest_engine_adaptive import AdaptiveBacktestEngine

df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

print("Loading data & predictions...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# Compute ATR(14)
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
INITIAL_EQUITY = 200_000.0

# -------------------------------------------------------------
# 1. EVALUATE REGIME 1: 3 TRADES PER DAY (Asia, London, NY)
# -------------------------------------------------------------
def run_3trades_per_day(risk_pct=0.01, compounding=True, fixed_lot=None, stress_friction=False):
    spread_price = (1.5 if stress_friction else 0.75) * 0.10
    slippage_price = (0.8 if stress_friction else 0.3) * 0.10
    commission = 7.0 if stress_friction else 3.50
    contract_size = 100.0
    sl_atr_mult = 1.5
    tp1_r = 1.0
    tp2_r = 2.5
    trail_after_r = 1.5
    trail_dist_r = 0.8
    seq_offset = 63

    def calc_friction(lot):
        return round(spread_price * lot * contract_size + slippage_price * lot * contract_size * 2 + commission * lot, 2)

    sub_mask = (df['timestamp_utc'] >= START_DATE) & (df['timestamp_utc'] <= END_DATE)
    sub_indices = df[sub_mask].index
    days = sorted(df.loc[sub_indices, 'timestamp_utc'].dt.strftime('%Y-%m-%d').unique())

    equity = INITIAL_EQUITY
    equity_curve = [equity]
    trades = []
    trade_counter = 0

    session_ranges = [
        (1, 7, "Sesi Pagi (Asia)"),
        (8, 13, "Sesi Siang (London)"),
        (14, 20, "Sesi Sore (New York)")
    ]

    daily_pnl = {}

    for day_str in days:
        day_mask = (df['timestamp_utc'].dt.strftime('%Y-%m-%d') == day_str) & sub_mask
        day_indices = df[day_mask].index
        day_ts = df.loc[day_indices[0], 'timestamp_utc']
        if day_ts.dayofweek in [5, 6]:
            continue

        day_start_equity = equity

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

            if best_bar_idx is not None:
                entry_row = df.iloc[best_bar_idx]
                entry_ts = entry_row["timestamp_utc"]
                bar_close = entry_row["close"]
                d = "BUY" if best_action in [1, 2] else "SELL"

                atr_val = float(atr[best_bar_idx]) if best_bar_idx < len(atr) else 5.0
                sl_dist = round(atr_val * sl_atr_mult, 2)

                if fixed_lot is not None:
                    total_lot = fixed_lot
                else:
                    base_eq = equity if compounding else INITIAL_EQUITY
                    target_risk = base_eq * risk_pct
                    raw_lot = target_risk / (sl_dist * contract_size)
                    total_lot = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 50.0))

                lot_a = max(0.01, round(total_lot * 0.5, 2))
                lot_b = max(0.01, round(total_lot - lot_a, 2))
                actual_risk = total_lot * sl_dist * contract_size

                if d == "BUY":
                    ep = round(bar_close + slippage_price, 2)
                    sl_p = round(ep - sl_dist, 2)
                    tp1_p = round(ep + (tp1_r * sl_dist), 2)
                    tp2_p = round(ep + (tp2_r * sl_dist), 2)
                else:
                    ep = round(bar_close - slippage_price, 2)
                    sl_p = round(ep + sl_dist, 2)
                    tp1_p = round(ep - (tp1_r * sl_dist), 2)
                    tp2_p = round(ep - (tp2_r * sl_dist), 2)

                cur_sl_a = sl_p
                cur_sl_b = sl_p
                pos_a_open = True
                pos_b_open = True if lot_b > 0 else False
                tp1_hit = False
                pnl_accum = 0.0
                exit_ts = None
                exit_price = None
                exit_reason = None

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
                        exit_price = f_close
                        if pos_a_open:
                            pnl_a = (exit_price - ep) * lot_a * contract_size if d == "BUY" else (ep - exit_price) * lot_a * contract_size
                            pnl_accum += (pnl_a - calc_friction(lot_a))
                            pos_a_open = False
                        if pos_b_open:
                            pnl_b = (exit_price - ep) * lot_b * contract_size if d == "BUY" else (ep - exit_price) * lot_b * contract_size
                            pnl_accum += (pnl_b - calc_friction(lot_b))
                            pos_b_open = False
                        break
                    else:
                        if d == "BUY":
                            if not tp1_hit and f_low <= cur_sl_a:
                                pnl_g = (cur_sl_a - ep) * total_lot * contract_size
                                pnl_accum = round(pnl_g - calc_friction(total_lot), 2)
                                exit_ts = f_ts
                                exit_price = cur_sl_a
                                exit_reason = "Stop Loss (-1.0R)"
                                pos_a_open = False
                                pos_b_open = False
                                break
                            else:
                                if pos_a_open and f_high >= tp1_p:
                                    pnl_a = (tp1_p - ep) * lot_a * contract_size
                                    pnl_accum += (pnl_a - calc_friction(lot_a))
                                    pos_a_open = False
                                    tp1_hit = True
                                    f_p = spread_price + (commission / contract_size)
                                    cur_sl_b = max(cur_sl_b, round(ep + f_p, 2))

                                if pos_b_open:
                                    if f_high >= tp2_p:
                                        pnl_b = (tp2_p - ep) * lot_b * contract_size
                                        pnl_accum += (pnl_b - calc_friction(lot_b))
                                        pos_b_open = False
                                        exit_ts = f_ts
                                        exit_price = tp2_p
                                        exit_reason = "Take Profit (+2.5R)"
                                        break
                                    elif f_low <= cur_sl_b:
                                        pnl_b = (cur_sl_b - ep) * lot_b * contract_size
                                        pnl_accum += (pnl_b - calc_friction(lot_b))
                                        pos_b_open = False
                                        exit_ts = f_ts
                                        exit_price = cur_sl_b
                                        exit_reason = "Breakeven" if tp1_hit else "Stop Loss"
                                        break
                                    elif tp1_hit:
                                        r_gain = (f_high - ep) / sl_dist
                                        if r_gain >= trail_after_r:
                                            trail_sl = round(f_high - (trail_dist_r * sl_dist), 2)
                                            cur_sl_b = max(cur_sl_b, trail_sl)
                        else:  # SELL
                            if not tp1_hit and f_high >= cur_sl_a:
                                pnl_g = (ep - cur_sl_a) * total_lot * contract_size
                                pnl_accum = round(pnl_g - calc_friction(total_lot), 2)
                                exit_ts = f_ts
                                exit_price = cur_sl_a
                                exit_reason = "Stop Loss (-1.0R)"
                                pos_a_open = False
                                pos_b_open = False
                                break
                            else:
                                if pos_a_open and f_low <= tp1_p:
                                    pnl_a = (ep - tp1_p) * lot_a * contract_size
                                    pnl_accum += (pnl_a - calc_friction(lot_a))
                                    pos_a_open = False
                                    tp1_hit = True
                                    f_p = spread_price + (commission / contract_size)
                                    cur_sl_b = min(cur_sl_b, round(ep - f_p, 2))

                                if pos_b_open:
                                    if f_low <= tp2_p:
                                        pnl_b = (ep - tp2_p) * lot_b * contract_size
                                        pnl_accum += (pnl_b - calc_friction(lot_b))
                                        pos_b_open = False
                                        exit_ts = f_ts
                                        exit_price = tp2_p
                                        exit_reason = "Take Profit (+2.5R)"
                                        break
                                    elif f_high >= cur_sl_b:
                                        pnl_b = (ep - cur_sl_b) * lot_b * contract_size
                                        pnl_accum += (pnl_b - calc_friction(lot_b))
                                        pos_b_open = False
                                        exit_ts = f_ts
                                        exit_price = cur_sl_b
                                        exit_reason = "Breakeven" if tp1_hit else "Stop Loss"
                                        break
                                    elif tp1_hit:
                                        r_gain = (ep - f_low) / sl_dist
                                        if r_gain >= trail_after_r:
                                            trail_sl = round(f_low + (trail_dist_r * sl_dist), 2)
                                            cur_sl_b = min(cur_sl_b, trail_sl)

                pnl_final = round(pnl_accum, 2)
                r_mult = round(pnl_final / actual_risk, 2) if actual_risk > 0 else 0.0
                m_str = entry_ts.strftime('%Y-%m')
                trades.append({
                    'id': trade_counter + 1,
                    'month': m_str,
                    'day': day_str,
                    'session': s_name,
                    'direction': d,
                    'entry_ts': entry_ts,
                    'lot': total_lot,
                    'pnl': pnl_final,
                    'r_mult': r_mult,
                    'win': 1 if pnl_final > 0 else 0,
                    'exit_reason': exit_reason if exit_reason else "Selesai Sesi"
                })
                trade_counter += 1
                equity += pnl_final
                equity_curve.append(equity)

        daily_pnl[day_str] = equity - day_start_equity

    tdf = pd.DataFrame(trades)
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = np.min((eq_arr - peak) / peak) * 100

    # Max daily DD
    max_daily_loss_pct = 0.0
    for day_str, pnl_d in daily_pnl.items():
        # find equity at start of day
        loss_pct = (pnl_d / INITIAL_EQUITY) * 100
        if loss_pct < max_daily_loss_pct:
            max_daily_loss_pct = loss_pct

    return tdf, equity, eq_arr, max_dd, max_daily_loss_pct

# -------------------------------------------------------------
# 2. RUN SYSTEMATIC SWEEP & STRESS TESTS
# -------------------------------------------------------------
print("\n" + "="*80)
print("EXPERIMENT 1: 3-Trades/Day Execution on $200k Modal (Jan - Mar 2026)")
print("="*80)

results_regime1 = []
configs_regime1 = [
    {"name": "3T/Day Fixed 0.50 Lot (Very Safe)", "risk_pct": 0.005, "compounding": False, "fixed_lot": 0.50},
    {"name": "3T/Day Fixed 1.00 Lot (Standard)", "risk_pct": 0.01, "compounding": False, "fixed_lot": 1.00},
    {"name": "3T/Day Fixed 2.00 Lot (Aggressive)", "risk_pct": 0.02, "compounding": False, "fixed_lot": 2.00},
    {"name": "3T/Day Dynamic 0.5% Risk (Compounding)", "risk_pct": 0.005, "compounding": True, "fixed_lot": None},
    {"name": "3T/Day Dynamic 1.0% Risk (Compounding)", "risk_pct": 0.010, "compounding": True, "fixed_lot": None},
    {"name": "3T/Day Dynamic 1.5% Risk (Compounding)", "risk_pct": 0.015, "compounding": True, "fixed_lot": None},
    {"name": "3T/Day Dynamic 2.0% Risk (Compounding)", "risk_pct": 0.020, "compounding": True, "fixed_lot": None},
    {"name": "3T/Day Dynamic 1.0% Risk + 2x Friction STRESS", "risk_pct": 0.010, "compounding": True, "fixed_lot": None, "stress_friction": True},
]

for cfg in configs_regime1:
    tdf, eq, eq_arr, max_dd, max_daily_dd = run_3trades_per_day(
        risk_pct=cfg["risk_pct"],
        compounding=cfg["compounding"],
        fixed_lot=cfg.get("fixed_lot"),
        stress_friction=cfg.get("stress_friction", False)
    )
    net_pnl = eq - INITIAL_EQUITY
    ret_pct = (net_pnl / INITIAL_EQUITY) * 100
    wins = tdf['win'].sum()
    wr = (wins / len(tdf)) * 100
    gw = tdf[tdf['pnl'] > 0]['pnl'].sum()
    gl = abs(tdf[tdf['pnl'] < 0]['pnl'].sum())
    pf = (gw / gl) if gl > 0 else 99.0

    # Monthly breakdown
    m_pnl = {}
    for m in ['2026-01', '2026-02', '2026-03']:
        sub = tdf[tdf['month'] == m]
        m_pnl[m] = sub['pnl'].sum()

    results_regime1.append({
        "Config": cfg["name"],
        "FinalEquity": eq,
        "NetProfit": net_pnl,
        "ReturnPct": ret_pct,
        "MaxDD": max_dd,
        "MaxDailyDD": max_daily_dd,
        "WinRate": wr,
        "PF": pf,
        "Jan_PnL": m_pnl.get('2026-01', 0),
        "Feb_PnL": m_pnl.get('2026-02', 0),
        "Mar_PnL": m_pnl.get('2026-03', 0),
    })

df_res1 = pd.DataFrame(results_regime1)
print(df_res1[["Config", "FinalEquity", "NetProfit", "ReturnPct", "MaxDD", "MaxDailyDD", "WinRate", "PF"]].to_string(index=False))

# -------------------------------------------------------------
# 3. EVALUATE REGIME 2: ADAPTIVE INSTITUTIONAL SNIPER ($200k Modal)
# -------------------------------------------------------------
print("\n" + "="*80)
print("EXPERIMENT 2: Adaptive Institutional Sniper Engine ($200k Modal, Jan - Mar 2026)")
print("="*80)

results_regime2 = []
configs_regime2 = [
    {"name": "Adaptive Sniper (0.5% Risk, Tau=0.34)", "risk": 0.005, "tau": 0.34, "defensive": True},
    {"name": "Adaptive Sniper (1.0% Risk, Tau=0.34)", "risk": 0.010, "tau": 0.34, "defensive": True},
    {"name": "Adaptive Sniper (1.5% Risk, Tau=0.34)", "risk": 0.015, "tau": 0.34, "defensive": True},
    {"name": "Adaptive Sniper (2.0% Risk, Tau=0.34)", "risk": 0.020, "tau": 0.34, "defensive": True},
    {"name": "Adaptive Sniper (1.0% Risk, High Tau=0.40)", "risk": 0.010, "tau": 0.40, "defensive": True},
    {"name": "Adaptive Sniper (1.5% Risk, High Tau=0.40)", "risk": 0.015, "tau": 0.40, "defensive": True},
    {"name": "Adaptive Sniper (1.0% Risk, 2x Friction STRESS)", "risk": 0.010, "tau": 0.34, "defensive": True, "spread": 1.5, "slip": 0.8},
]

for cfg in configs_regime2:
    engine = AdaptiveBacktestEngine(
        initial_equity=INITIAL_EQUITY,
        base_risk_fraction=cfg["risk"],
        confidence_tau=cfg["tau"],
        sl_atr_multiplier=1.5,
        spread_pip=cfg.get("spread", 0.75),
        commission_per_lot=3.50 if not cfg.get("spread") else 7.0,
        slippage_pip=cfg.get("slip", 0.3),
        time_barrier_bars=16,
        contract_size=100.0,
        enable_dual_playbook=True,
        enable_scratch_exit=True,
        enable_defensive_scaling=cfg["defensive"],
        defensive_risk_fraction=cfg["risk"] * 0.5,
        enable_uncapped_runner=True,
        enable_regime_gating=True,
        max_daily_entries=10,
        quarantine_london_open=True,
        use_structural_filter=True,
        pos_a_pct=0.40,
        pos_b_pct=0.60,
        tp1_r_ratio=1.0,
        tp2_r_ratio=2.5,
        trail_after_r=1.5,
        trail_distance_r=0.75
    )

    res = engine.run(
        df=df,
        predictions=preds,
        atr_values=atr,
        start_date="2026-01-01",
        end_date="2026-03-31 23:59:59"
    )

    net_pnl = res.final_equity - INITIAL_EQUITY
    ret_pct = (net_pnl / INITIAL_EQUITY) * 100

    results_regime2.append({
        "Config": cfg["name"],
        "Trades": res.total_trades,
        "FinalEquity": res.final_equity,
        "NetProfit": net_pnl,
        "ReturnPct": ret_pct,
        "MaxDD": res.max_drawdown_pct,
        "WinRate": res.win_rate,
        "PF": res.net_profit_factor,
        "Sharpe": res.sharpe_ratio,
        "Breakeven": res.breakeven_trades,
    })

df_res2 = pd.DataFrame(results_regime2)
print(df_res2[["Config", "Trades", "FinalEquity", "NetProfit", "ReturnPct", "MaxDD", "WinRate", "PF", "Sharpe"]].to_string(index=False))

# Save summary json
output_summary = {
    "regime1_3trades_daily": results_regime1,
    "regime2_adaptive_sniper": results_regime2
}

with open(r'c:\Ngoding\bot_trading\scratch\stress_test_q1_2026_200k.json', 'w') as f:
    json.dump(output_summary, f, indent=2)

print("\nDone! Results saved to scratch/stress_test_q1_2026_200k.json")
