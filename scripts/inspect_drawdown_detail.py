"""
Inspect Drawdown Period on $500 Account
Where does the drawdown peak occur, and what trades caused it?
"""
import sys, pathlib
import numpy as np
import pandas as pd

from test_precise_be_buffer import test_be_buff

res = test_be_buff(buff_val=0.25, tau_val=0.33, min_margin=0.01)

tdf = res['tdf']
tdf['cum_pnl'] = tdf['pnl'].cumsum()
tdf['equity'] = 500.0 + tdf['cum_pnl']
tdf['peak'] = tdf['equity'].cummax()
tdf['dd_pct'] = (tdf['peak'] - tdf['equity']) / tdf['peak'] * 100

max_dd_idx = tdf['dd_pct'].idxmax()
worst_row = tdf.loc[max_dd_idx]
print(f"Max Drawdown: {worst_row['dd_pct']:.2f}% on {worst_row['time']}")
print(f"Equity at Peak: ${worst_row['peak']:.2f} | Equity at Trough: ${worst_row['equity']:.2f}")

# Look at the period around max drawdown
start_search = max(0, max_dd_idx - 15)
end_search = min(len(tdf), max_dd_idx + 10)
print("\n--- Trades leading to Max Drawdown ---")
sub = tdf.iloc[start_search:end_search]
for idx, r in sub.iterrows():
    print(f"{r['time']} | Dir: {r['dir']:<4} | Win: {r['win']} | PnL: ${r['pnl']:+6.2f} | Eq: ${r['equity']:7.2f} | DD: {r['dd_pct']:5.1f}% | Exit: {r['exit']}")
