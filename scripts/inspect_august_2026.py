"""
Inspect August 2026 trades in Baseline Tier 3 Sniper
"""

import sys, os, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, r'c:\Ngoding\bot_trading')
sys.path.insert(0, r'c:\Ngoding\bot_trading\scripts')

from scripts.backtest_engine_adaptive import AdaptiveBacktestEngine

df_path = r'c:\Ngoding\xau_deep_sniper\data\processed\xauusd_m30_labeled.parquet'
preds_path = r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy'

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

engine = AdaptiveBacktestEngine(
    initial_equity=10_000.0,
    base_risk_fraction=0.01,
    confidence_tau=0.34,
    sl_atr_multiplier=1.5,
    spread_pip=0.75,
    commission_per_lot=3.50,
    slippage_pip=0.3,
    time_barrier_bars=16,
    enable_dual_playbook=False,
    enable_scratch_exit=False,
    enable_defensive_scaling=False,
    enable_uncapped_runner=False,
    enable_regime_gating=False,
    max_daily_entries=10,
    quarantine_london_open=True,
    use_structural_filter=True,
)

res = engine.run(df=df, predictions=preds, atr_values=atr, start_date="2026-08-01", end_date="2026-08-31 23:59:59")
print(f"August 2026 Trades: {len(res.trades)}")
for t in res.trades:
    print(f"ID: {t.trade_id} | {t.direction} | Entry: {t.entry_timestamp} | Exit: {t.exit_timestamp} | ExitReason: {t.exit_reason:<12} | PnL: ${t.pnl_net:+7.2f} | R: {t.r_multiple:+5.2f}")
