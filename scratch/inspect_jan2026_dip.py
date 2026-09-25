import sys, pathlib
sys.path.insert(0, r'c:\Ngoding\bot_trading')
import pandas as pd, numpy as np, json
from scripts.run_250_backtest_2026_full import run_2026_backtest_250

trades, eq_curve, eq_ts = run_2026_backtest_250()
tdf = pd.DataFrame(trades)
jan = tdf[tdf['month'] == '2026-01']
print(f"Jan trades: {len(jan)}, Wins: {len(jan[jan['pnl'] > 0])}, Losses: {len(jan[jan['pnl'] <= 0])}")

running_eq = [eq_curve[i+1] for i in range(len(trades))]
min_idx = int(np.argmin(running_eq))
print(f"Lowest dip at trade #{min_idx}: ${running_eq[min_idx]:.2f} on {trades[min_idx]['time']}")

print("\n--- Sequence of 12 trades leading to the lowest dip ---")
for i in range(max(0, min_idx - 11), min_idx + 1):
    t = trades[i]
    print(f"#{i:3d} | {t['time']} | {t['dir']} | PnL: ${t['pnl']:+6.2f} | Bal: ${running_eq[i]:7.2f} | Exit: {t['exit_reason']:20s} | SL_dist: ${t['sl_dist']:.2f}")
