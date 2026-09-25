"""
Test Adaptive Backtest Engine on Jan - Aug 2026
"""

import sys, os, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, r'c:\Ngoding\bot_trading')
sys.path.insert(0, r'c:\Ngoding\bot_trading\scripts')

from scripts.backtest_engine_adaptive import AdaptiveBacktestEngine

df_path = r'c:\Ngoding\xau_deep_sniper\data\processed\xauusd_m30_labeled.parquet'
preds_path = r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy'

print("Loading dataset & predictions...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# Compute ATR(14)
high = df["high"].values
low = df["low"].values
close = df["close"].values
tr = np.zeros(len(df))
tr[0] = high[0] - low[0]
for i in range(1, len(df)):
    tr[i] = max(
        high[i] - low[i],
        abs(high[i] - close[i - 1]),
        abs(low[i] - close[i - 1]),
    )
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
    enable_dual_playbook=True,
    enable_scratch_exit=True,
    scratch_bars=5,
    scratch_loss_threshold_r=0.25,
    enable_defensive_scaling=True,
    defensive_loss_streak=2,
    defensive_risk_fraction=0.005,
    enable_uncapped_runner=True,
    enable_regime_gating=True,
    max_daily_entries=10,
    quarantine_london_open=True,
    use_structural_filter=True,
)

res = engine.run(
    df=df,
    predictions=preds,
    atr_values=atr,
    start_date="2026-01-01",
    end_date="2026-08-31 23:59:59"
)

trades = []
for t in res.trades:
    trades.append({
        'id': t.trade_id,
        'direction': t.direction,
        'playbook': t.playbook,
        'entry_ts': t.entry_timestamp,
        'exit_ts': t.exit_timestamp,
        'pnl_net': t.pnl_net,
        'r_mult': t.r_multiple,
        'exit_reason': t.exit_reason,
    })
tdf = pd.DataFrame(trades)
print(f"\nTotal Trades: {len(tdf)}")
if len(tdf) > 0:
    tdf['entry_ts'] = pd.to_datetime(tdf['entry_ts'])
    tdf['month'] = tdf['entry_ts'].dt.strftime('%Y-%m')
    wins = (tdf['pnl_net'] > 0).sum()
    print(f"Wins: {wins}, Losses: {len(tdf) - wins}, Win Rate: {wins/len(tdf)*100:.1f}%")
    print(f"Net PnL: ${tdf['pnl_net'].sum():,.2f} ({tdf['pnl_net'].sum()/10000*100:.2f}%)")
    peak = np.maximum.accumulate(res.equity_curve)
    max_dd = np.min((res.equity_curve - peak) / peak) * 100
    print(f"Max DD: {max_dd:.2f}%, Profit Factor: {res.net_profit_factor:.2f}")

    print("\n--- MONTHLY BREAKDOWN (2026) ---")
    cur_eq = 10000.0
    for m in [f"2026-{i:02d}" for i in range(1, 9)]:
        sub = tdf[tdf['month'] == m]
        if len(sub) > 0:
            sw = (sub['pnl_net'] > 0).sum()
            spnl = sub['pnl_net'].sum()
            spct = spnl / cur_eq * 100
            cur_eq += spnl
            print(f"{m} | Trades: {len(sub):2d} | Win: {sw:2d} ({sw/len(sub)*100:4.1f}%) | PnL: ${spnl:+7.2f} ({spct:+5.2f}%)")
        else:
            print(f"{m} | Trades:  0 | Win:  0 ( 0.0%) | PnL:   $0.00 (+0.00%)")
