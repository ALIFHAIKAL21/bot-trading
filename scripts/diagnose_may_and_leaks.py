"""
Deep Diagnostic of May 2026 & Edge Leak Analysis on $500 Capital (Lot 0.01)
"""
import sys, pathlib
import numpy as np
import pandas as pd

from run_500_improved_production import df, preds, atr, opens, highs, lows, closes, ts_arr, dows, hours, minutes, day_ints, unique_days, day_to_indices, seq_offset, calc_friction, INITIAL_EQUITY, SPREAD_PRICE, SLIPPAGE_PRICE, COMMISSION_PER_LOT, CONTRACT_SIZE, SL_ATR_MULT, TP_MAX_R, BE_TRIGGER_R, BE_BUFFER_PRICE, TRAIL_TRIGGER_R, TRAIL_DIST_R, TAU_BASE, MARGIN_MIN, session_defs, FIXED_LOT, run_improved_backtest

tdf, eq_curve, eq_ts = run_improved_backtest()

print("\n--- MONTHLY BREAKDOWN METRICS ---")
for m, grp in tdf.groupby('month'):
    wins = grp[grp['win'] == 1]
    losses = grp[grp['win'] == 0]
    pnl = grp['pnl'].sum()
    avg_w = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_l = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
    print(f"Month {m}: Trades={len(grp):2d} | Wins={len(wins):2d} | Losses={len(losses):2d} | WR={len(wins)/len(grp)*100:5.1f}% | AvgWin=${avg_w:5.2f} | AvgLoss=${avg_l:5.2f} | Net PnL=${pnl:+7.2f}")

# Deep dive into May 2026
may_trades = tdf[tdf['month'] == '2026-05']
print(f"\n--- MAY 2026 EXIT REASONS BREAKDOWN ({len(may_trades)} Trades) ---")
for reason, count in may_trades['exit_reason'].value_counts().items():
    sub = may_trades[may_trades['exit_reason'] == reason]
    pnl = sub['pnl'].sum()
    print(f"  {reason:<22}: Count={count:2d} ({count/len(may_trades)*100:4.1f}%) | Net PnL=${pnl:+7.2f}")

# Look at Directional Breakdown in May
print(f"\n--- MAY 2026 DIRECTIONAL PERFORMANCE ---")
for d, grp in may_trades.groupby('dir'):
    wins = grp[grp['win'] == 1]
    losses = grp[grp['win'] == 0]
    pnl = grp['pnl'].sum()
    print(f"  Direction {d}: Trades={len(grp):2d} | Wins={len(wins):2d} | Losses={len(losses):2d} | WR={len(wins)/len(grp)*100:5.1f}% | Net PnL=${pnl:+7.2f}")

# Look at Sessions in May
print(f"\n--- MAY 2026 SESSION PERFORMANCE ---")
for s, grp in may_trades.groupby('session'):
    wins = grp[grp['win'] == 1]
    losses = grp[grp['win'] == 0]
    pnl = grp['pnl'].sum()
    print(f"  {s:<25}: Trades={len(grp):2d} | WR={len(wins)/len(grp)*100:5.1f}% | Net PnL=${pnl:+7.2f}")
