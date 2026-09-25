"""
Ablation of Adaptive Components on 2026 (Jan - Aug)
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

configs = [
    ("1. Baseline Tier 3 Sniper", dict(enable_dual_playbook=False, enable_scratch_exit=False, enable_defensive_scaling=False, enable_uncapped_runner=False, enable_regime_gating=False)),
    ("2. + Uncapped Runner Only", dict(enable_dual_playbook=False, enable_scratch_exit=False, enable_defensive_scaling=False, enable_uncapped_runner=True, enable_regime_gating=False)),
    ("3. + Defensive Scaling Only", dict(enable_dual_playbook=False, enable_scratch_exit=False, enable_defensive_scaling=True, enable_uncapped_runner=False, enable_regime_gating=False)),
    ("4. + Scratch Exit Only", dict(enable_dual_playbook=False, enable_scratch_exit=True, enable_defensive_scaling=False, enable_uncapped_runner=False, enable_regime_gating=False)),
    ("5. + Regime Gating Only", dict(enable_dual_playbook=False, enable_scratch_exit=False, enable_defensive_scaling=False, enable_uncapped_runner=False, enable_regime_gating=True)),
    ("6. + Dual Playbook Only", dict(enable_dual_playbook=True, enable_scratch_exit=False, enable_defensive_scaling=False, enable_uncapped_runner=False, enable_regime_gating=False)),
]

print(f"{'Config Name':<30} | {'Trades':<6} | {'WR (%)':<6} | {'Net PnL ($)':<12} | {'Return':<7} | {'Max DD':<7} | {'PF':<5}")
print("-" * 85)

for name, cfg in configs:
    engine = AdaptiveBacktestEngine(
        initial_equity=10_000.0,
        base_risk_fraction=0.01,
        confidence_tau=0.34,
        sl_atr_multiplier=1.5,
        spread_pip=0.75,
        commission_per_lot=3.50,
        slippage_pip=0.3,
        time_barrier_bars=16,
        max_daily_entries=10,
        quarantine_london_open=True,
        use_structural_filter=True,
        **cfg
    )
    res = engine.run(df=df, predictions=preds, atr_values=atr, start_date="2026-01-01", end_date="2026-08-31 23:59:59")
    n_t = res.total_trades
    wr = res.win_rate * 100
    pnl = res.net_pnl
    ret = (pnl / 10000.0) * 100
    peak = np.maximum.accumulate(res.equity_curve)
    max_dd = np.min((res.equity_curve - peak) / peak) * 100 if len(res.equity_curve) > 0 else 0
    pf = res.net_profit_factor
    print(f"{name:<30} | {n_t:<6} | {wr:<6.1f} | ${pnl:<11.2f} | {ret:<+6.2f}% | {max_dd:<6.2f}% | {pf:<5.2f}")
