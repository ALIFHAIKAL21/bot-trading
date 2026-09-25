"""
Check 2026 performance under higher confidence thresholds (tau = 0.40, 0.42, 0.45)
"""
import sys, pathlib
import pandas as pd
from deep_monthly_audit import run_detailed_simulation

print("\n--- 2026 MONTHLY BREAKDOWN FOR TAU = 0.400 ---")
tdf_40 = run_detailed_simulation(tau_val=0.400)
sub_26 = tdf_40[tdf_40['year'] == 2026]
for m, grp in sub_26.groupby('month'):
    pnl = grp['pnl'].sum()
    wr = grp['win'].mean() * 100
    n_b = len(grp[grp['dir'] == 'BUY'])
    n_s = len(grp[grp['dir'] == 'SELL'])
    print(f"  Month {m}: Trades={len(grp):2d} (B:{n_b:2d}, S:{n_s:2d}) | WR={wr:4.1f}% | Net PnL=${pnl:+8.2f}")

win_money = sub_26[sub_26['pnl'] > 0]['pnl'].sum()
loss_money = abs(sub_26[sub_26['pnl'] < 0]['pnl'].sum())
pf = (win_money / loss_money) if loss_money > 0 else 999.0
print(f"  -> 2026 FULL YEAR SUMMARY (TAU=0.40): Total PnL=${sub_26['pnl'].sum():+8.2f} | PF: {pf:.2f} | WR: {sub_26['win'].mean()*100:.1f}%")

print("\n--- 2026 MONTHLY BREAKDOWN FOR TAU = 0.420 ---")
tdf_42 = run_detailed_simulation(tau_val=0.420)
sub_26_42 = tdf_42[tdf_42['year'] == 2026]
for m, grp in sub_26_42.groupby('month'):
    pnl = grp['pnl'].sum()
    wr = grp['win'].mean() * 100
    n_b = len(grp[grp['dir'] == 'BUY'])
    n_s = len(grp[grp['dir'] == 'SELL'])
    print(f"  Month {m}: Trades={len(grp):2d} (B:{n_b:2d}, S:{n_s:2d}) | WR={wr:4.1f}% | Net PnL=${pnl:+8.2f}")

win_money = sub_26_42[sub_26_42['pnl'] > 0]['pnl'].sum()
loss_money = abs(sub_26_42[sub_26_42['pnl'] < 0]['pnl'].sum())
pf = (win_money / loss_money) if loss_money > 0 else 999.0
print(f"  -> 2026 FULL YEAR SUMMARY (TAU=0.42): Total PnL=${sub_26_42['pnl'].sum():+8.2f} | PF: {pf:.2f} | WR: {sub_26_42['win'].mean()*100:.1f}%")
