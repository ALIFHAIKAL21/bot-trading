import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.momentum_backtest import MomentumBacktestEngine


df_bars = pd.read_parquet('data/xau/xauusd_1h_clean.parquet')
df_setups = pd.read_parquet('data/xau/momentum_setups_1h.parquet')

df_bars['timestamp_utc'] = pd.to_datetime(df_bars['timestamp_utc'], utc=True)
df_setups['timestamp_utc'] = pd.to_datetime(df_setups['timestamp_utc'], utc=True)

# Regime definition
median_spread = df_setups['sma_spread'].abs().median()
df_setups['is_sideways'] = (df_setups['sma_spread'].abs() < median_spread)

# Boundary setup
is_buy_edge = (df_setups['direction'] == 'BUY') & ((df_setups['inside_demand_zone'] == 1) | (df_setups['smi_os_reversal_bull'] == 1) | (df_setups['premium_discount_ratio'] < 0.4))
is_sell_edge = (df_setups['direction'] == 'SELL') & ((df_setups['inside_supply_zone'] == 1) | (df_setups['smi_ob_reversal_bear'] == 1) | (df_setups['premium_discount_ratio'] > 0.6))
df_setups['is_edge'] = is_buy_edge | is_sell_edge

# Select eligible setups:
# 1. In sideways: ONLY take edge setups (mean reversion at range boundaries)
# 2. In trending: take all momentum setups
eligible_mask = (~df_setups['is_sideways']) | (df_setups['is_sideways'] & df_setups['is_edge'])
df_setups_adaptive = df_setups[eligible_mask].copy()

# For setups in sideways, evaluate RR 1:1; for trending, evaluate RR 1:2
# In our backtest engine, let's see how both target_rr=1 and target_rr=2 perform across 2021-2026

print(f"Total original setups: {len(df_setups)}")
print(f"Total adaptive filtered setups: {len(df_setups_adaptive)}")

# Let's inspect 2023 specifically with RR 1:1 vs RR 1:2
for rr in [1, 2]:
    engine = MomentumBacktestEngine(initial_equity=10000.0, risk_pct=0.01)
    # Assign uniform prob 1.0 to test the underlying edge
    df_setups_adaptive['prob_win'] = 1.0
    summary = engine.run(df_bars, df_setups_adaptive, target_rr=rr, confidence_tau=0.0)
    
    trades = pd.DataFrame([{
        'year': t.exit_time.year,
        'pnl': t.pnl_net,
        'win': t.pnl_net > 0
    } for t in summary.trades])
    
    print(f"\n=== BACKTEST RESULTS (Adaptive Setups, RR 1:{rr}) ===")
    for year, grp in trades.groupby('year'):
        print(f"Year {year} | Trades: {len(grp):<4} | WinRate: {(grp['win'].mean()*100):.1f}% | Net PnL: ${grp['pnl'].sum():,.2f}")
    print(f"TOTAL: Trades: {len(trades)} | Net PnL: ${trades['pnl'].sum():,.2f}")
