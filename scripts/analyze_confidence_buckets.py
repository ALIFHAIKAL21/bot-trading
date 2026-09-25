"""
Analyze win rate and expectancy across confidence buckets and parameters
"""

import sys, os, pathlib
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

engine = RealisticBacktestEngine(
    initial_equity=10_000.0,
    risk_fraction=0.01,
    confidence_tau=0.30,  # lower tau to capture all signals
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

res = engine.run(df=df, predictions=preds, atr_values=atr, start_date="2022-01-01", end_date="2026-08-31 23:59:59")
trades = [{
    'id': t.trade_id,
    'direction': t.direction,
    'ts': t.entry_timestamp,
    'pnl': t.pnl_net,
    'r': t.r_multiple,
    'conf': t.confidence,
    'win': 1 if t.pnl_net > 0 else 0
} for t in res.trades]
tdf = pd.DataFrame(trades)
tdf['conf_bucket'] = pd.cut(tdf['conf'], bins=[0.30, 0.33, 0.36, 0.40, 0.50, 1.0])
print("--- TRADES BY CONFIDENCE BUCKET (2022 - 2026) ---")
gb = tdf.groupby('conf_bucket', observed=False).agg(
    Trades=('pnl', 'count'),
    Wins=('win', 'sum'),
    WR=('win', 'mean'),
    TotalPnL=('pnl', 'sum'),
    AvgPnL=('pnl', 'mean')
)
gb['WR'] = gb['WR'] * 100
print(gb)
