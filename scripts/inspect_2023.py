import sys, pathlib
import pandas as pd
from sell_diagnostics import tdf

tdf_23 = tdf[tdf['year'] == 2023].copy()
tdf_23['month'] = pd.to_datetime(tdf_23['entry_price']).apply(lambda x: '') # placeholder
print("2023 Trade Exit Reasons and PnL by Direction:")
print(tdf_23.groupby(['direction', 'exit'])['pnl_net_accum'].agg(['count', 'sum']))

print("\nWin rate by direction in 2023:")
for d in ['BUY', 'SELL']:
    sub = tdf_23[tdf_23['direction'] == d]
    print(f"  {d}: N={len(sub)}, Win={sub['win'].sum()} ({sub['win'].mean()*100:.1f}%), PnL=${sub['pnl_net_accum'].sum():.2f}")
