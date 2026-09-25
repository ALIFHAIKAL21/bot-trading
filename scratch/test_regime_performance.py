import pandas as pd
import numpy as np

# Load setups and bars
df_setups = pd.read_parquet('data/xau/momentum_setups_1h.parquet')
df_setups['timestamp_utc'] = pd.to_datetime(df_setups['timestamp_utc'], utc=True)
df_setups['year'] = df_setups['timestamp_utc'].dt.year

# Classify regime based on market structure and volatility
# Sideways regime indicators:
# 1. sma_spread is tight (abs(sma_spread) < median)
# 2. atr_14 is lower or in consolidation
# 3. premium_discount_ratio is defined between swings
median_spread = df_setups['sma_spread'].abs().median()
df_setups['is_sideways'] = (df_setups['sma_spread'].abs() < median_spread)

print(f"Total setups: {len(df_setups)}")
print(f"Sideways setups: {df_setups['is_sideways'].sum()} ({df_setups['is_sideways'].mean()*100:.1f}%)")

for year, grp in df_setups.groupby('year'):
    sw = grp[grp['is_sideways']]
    tr = grp[~grp['is_sideways']]
    print(f"Year {year}: Total={len(grp)} | Sideways={len(sw)} ({len(sw)/len(grp)*100:.1f}%)")
    
    # Sideways edge trades with RR 1:1
    sw_edge = sw[
        ((sw['direction'] == 'BUY') & ((sw['inside_demand_zone'] == 1) | (sw['smi_os_reversal_bull'] == 1) | (sw['premium_discount_ratio'] < 0.4))) |
        ((sw['direction'] == 'SELL') & ((sw['inside_supply_zone'] == 1) | (sw['smi_ob_reversal_bear'] == 1) | (sw['premium_discount_ratio'] > 0.6)))
    ]
    sw_wr_1r = (sw_edge['win_1r'] == 1).mean() * 100 if len(sw_edge) > 0 else 0
    sw_wr_2r = (sw_edge['win_2r'] == 1).mean() * 100 if len(sw_edge) > 0 else 0
    
    # Trending trades with RR 1:2
    tr_wr_2r = (tr['win_2r'] == 1).mean() * 100 if len(tr) > 0 else 0
    
    print(f"   -> Sideways Boundary Setups: N={len(sw_edge):<4} | Win 1R: {sw_wr_1r:.1f}% | Win 2R: {sw_wr_2r:.1f}%")
    print(f"   -> Trending Setups:          N={len(tr):<4} | Win 2R: {tr_wr_2r:.1f}%")
