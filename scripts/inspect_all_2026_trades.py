"""
Inspect all 2026 trades from RealisticBacktestEngine
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

engine = RealisticBacktestEngine(
    initial_equity=10_000.0,
    risk_fraction=0.01,
    confidence_tau=0.34,
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

res = engine.run(df=df, predictions=preds, atr_values=atr, start_date="2026-01-01", end_date="2026-08-31 23:59:59")
trades = [{
    'id': t.trade_id,
    'direction': t.direction,
    'entry_ts': t.entry_timestamp,
    'exit_ts': t.exit_timestamp,
    'entry_price': t.entry_price,
    'exit_price': t.exit_price,
    'pnl_net': t.pnl_net,
    'r_mult': t.r_multiple,
    'exit_reason': t.exit_reason,
    'confidence': t.confidence,
    'tp1_hit': t.tp1_hit,
    'tp2_hit': t.tp2_hit
} for t in res.trades]
tdf = pd.DataFrame(trades)
tdf['entry_ts'] = pd.to_datetime(tdf['entry_ts'])
tdf['month'] = tdf['entry_ts'].dt.strftime('%Y-%m')

for m in sorted(tdf['month'].unique()):
    sub = tdf[tdf['month'] == m]
    wins = (sub['pnl_net'] > 0).sum()
    print(f"\n=== Month {m} (Total: {len(sub)}, Wins: {wins}, WR: {wins/len(sub)*100:.1f}%, Net PnL: ${sub['pnl_net'].sum():+.2f}) ===")
    for _, r in sub.iterrows():
        print(f"  {r['direction']:4s} | {r['entry_ts']} | Conf: {r['confidence']:.3f} | Exit: {r['exit_reason']:<12s} | PnL: ${r['pnl_net']:+7.2f} (R: {r['r_mult']:+5.2f})")
