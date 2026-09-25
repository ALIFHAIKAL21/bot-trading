import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backtest.momentum_backtest import MomentumBacktestEngine

# Load setups and bars
df_bars = pd.read_parquet('data/xau/xauusd_1h_clean.parquet')
df_setups = pd.read_parquet('data/xau/momentum_setups_1h.parquet')

df_bars['timestamp_utc'] = pd.to_datetime(df_bars['timestamp_utc'], utc=True)
df_setups['timestamp_utc'] = pd.to_datetime(df_setups['timestamp_utc'], utc=True)

# 1. Feature Engineering for Sideways vs Trending Regime
# Sideways characteristics:
# - Low ATR relative to price
# - Narrow MA spread (SMA 9 and 21 are entangled)
# - Price bouncing between swings
df_setups['ma_entangled'] = (df_setups['sma_spread'].abs() < df_setups['sma_spread'].abs().quantile(0.4)).astype(int)
df_setups['is_mid_range'] = df_setups['premium_discount_ratio'].between(0.40, 0.60).astype(int)

# Sideways Reversal Edge:
# BUY near Demand / Discount (< 0.4) with SMI OS
df_setups['is_sideways_buy_edge'] = (
    (df_setups['direction'] == 'BUY') & 
    ((df_setups['inside_demand_zone'] == 1) | (df_setups['smi_os_reversal_bull'] == 1) | (df_setups['premium_discount_ratio'] < 0.38))
).astype(int)

# SELL near Supply / Premium (> 0.6) with SMI OB
df_setups['is_sideways_sell_edge'] = (
    (df_setups['direction'] == 'SELL') & 
    ((df_setups['inside_supply_zone'] == 1) | (df_setups['smi_ob_reversal_bear'] == 1) | (df_setups['premium_discount_ratio'] > 0.62))
).astype(int)

df_setups['is_range_edge'] = (df_setups['is_sideways_buy_edge'] | df_setups['is_sideways_sell_edge']).astype(int)

# Target adaptation:
# When in sideways (ma_entangled == 1 or low spread), target is win_1r!
# When trending, target is win_2r!
df_setups['adaptive_target'] = np.where(df_setups['ma_entangled'] == 1, df_setups['win_1r'], df_setups['win_2r'])

# Let's inspect the correlation and stats
print("=== SETUP REGIME DISTRIBUTION ===")
print(f"Total setups: {len(df_setups)}")
print(f"MA Entangled (Sideways): {df_setups['ma_entangled'].sum()} ({df_setups['ma_entangled'].mean()*100:.1f}%)")
print(f"Range Edges: {df_setups['is_range_edge'].sum()} ({df_setups['is_range_edge'].mean()*100:.1f}%)")
print(f"Mid-range traps: {df_setups['is_mid_range'].sum()} ({df_setups['is_mid_range'].mean()*100:.1f}%)")

# Filter out false mid-range MA crosses during sideways
# In sideways, taking an MA cross in the middle of the range (40%-60%) is a known trap
trap_mask = (df_setups['ma_entangled'] == 1) & (df_setups['is_mid_range'] == 1) & (df_setups['primary_trigger'] == 'sma_cross')
print(f"Trapped MA Crosses in Sideways mid-range: {trap_mask.sum()}")
print(f"Win Rate of Trapped Setups (2R): {(df_setups.loc[trap_mask, 'win_2r']==1).mean()*100:.1f}%")
print(f"Win Rate of Trapped Setups (1R): {(df_setups.loc[trap_mask, 'win_1r']==1).mean()*100:.1f}%")
