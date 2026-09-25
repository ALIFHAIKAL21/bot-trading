"""
Test tau thresholds across 5 years
"""

import sys, os, pathlib, math
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
sys.path.insert(0, str(project_root / "src"))
from validation.backtest_engine import RealisticBacktestEngine

df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

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

for tau in [0.34, 0.35, 0.355, 0.36, 0.365]:
    annual_pnl = []
    annual_wr = []
    annual_trades = []
    annual_dds = []
    years = [2022, 2023, 2024, 2025, 2026]

    for y in years:
        engine = RealisticBacktestEngine(
            initial_equity=10_000.0,
            risk_fraction=0.01,
            confidence_tau=tau,
            sl_atr_multiplier=1.5,
            spread_pip=0.75,
            commission_per_lot=3.50,
            slippage_pip=0.3,
            time_barrier_bars=16,
            contract_size=100.0,
            use_partial_tp=True,
            pos_a_pct=0.40,
            pos_b_pct=0.60,
            tp1_r_ratio=1.0,
            tp2_r_ratio=2.5,
            be_trigger_r=1.0,
            trail_after_r=1.5,
            trail_distance_r=0.8,
            max_daily_entries=10,
            quarantine_london_open=True,
            use_structural_filter=True
        )
        res = engine.run(df=df, predictions=preds, atr_values=atr, start_date=f"{y}-01-01", end_date=f"{y}-08-31 23:59:59")
        peak = np.maximum.accumulate(res.equity_curve)
        mdd = np.min((res.equity_curve - peak) / peak) * 100 if len(res.equity_curve) > 0 else 0
        annual_pnl.append(res.net_pnl)
        annual_wr.append(res.win_rate * 100)
        annual_trades.append(res.total_trades)
        annual_dds.append(mdd)

    tot_pnl = sum(annual_pnl)
    tot_trades = sum(annual_trades)
    avg_dd = np.mean(annual_dds)
    green_years = sum(1 for p in annual_pnl if p > 0)
    print(f"Tau: {tau:0.3f} | Tot Trades: {tot_trades:3d} | Tot PnL: ${tot_pnl:+7.2f} ({tot_pnl/10000*100:+5.2f}%) | Green Years: {green_years}/5 | Avg MaxDD: {avg_dd:5.2f}%")
    for y, t, p, w, d in zip(years, annual_trades, annual_pnl, annual_wr, annual_dds):
        print(f"   {y}: Trades={t:2d}, WR={w:4.1f}%, PnL=${p:+7.2f}, DD={d:5.2f}%")
