import pandas as pd
import numpy as np

df_setups = pd.read_parquet('data/xau/momentum_setups_1h.parquet')
df_setups['timestamp_utc'] = pd.to_datetime(df_setups['timestamp_utc'], utc=True)
df_setups['year'] = df_setups['timestamp_utc'].dt.year

df_2023 = df_setups[df_setups['year'] == 2023].copy()

# Look at 2023 trades that were actually taken in our previous backtest
# In the previous backtest, the model used confidence_tau = 0.33 on win_2r
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.momentum_classifier import MomentumClassifier

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False)
oof_probs = np.zeros(len(df_setups))
for fold, (train_idx, val_idx) in enumerate(kf.split(df_setups)):
    clf = MomentumClassifier(target_col='win_2r')
    clf.fit(df_setups.iloc[train_idx])
    oof_probs[val_idx] = clf.predict_proba(df_setups.iloc[val_idx])

df_setups['prob_win'] = oof_probs
taken_2023 = df_setups[(df_setups['year'] == 2023) & (df_setups['prob_win'] >= 0.33)].copy()

print(f"Total trades taken in 2023: {len(taken_2023)}")
print(f"2023 Taken Win Rate (2R): {(taken_2023['win_2r']==1).mean()*100:.2f}%")
print(f"2023 Taken Win Rate (1R): {(taken_2023['win_1r']==1).mean()*100:.2f}%")

print("\n--- 2023 Taken Trades breakdown by trigger ---")
for trig, grp in taken_2023.groupby('primary_trigger'):
    print(f"Trigger {trig:<20}: N={len(grp):<4} | Win 2R: {(grp['win_2r']==1).mean()*100:.1f}% | Win 1R: {(grp['win_1r']==1).mean()*100:.1f}%")

print("\n--- 2023 Taken Trades: Range Edge vs Non-Edge ---")
is_buy_edge = (taken_2023['direction'] == 'BUY') & ((taken_2023['inside_demand_zone'] == 1) | (taken_2023['smi_os_reversal_bull'] == 1) | (taken_2023['premium_discount_ratio'] < 0.4))
is_sell_edge = (taken_2023['direction'] == 'SELL') & ((taken_2023['inside_supply_zone'] == 1) | (taken_2023['smi_ob_reversal_bear'] == 1) | (taken_2023['premium_discount_ratio'] > 0.6))
taken_2023['is_edge'] = is_buy_edge | is_sell_edge

edges = taken_2023[taken_2023['is_edge']]
non_edges = taken_2023[~taken_2023['is_edge']]

print(f"Edge Trades (Demand/SMI OS, Supply/SMI OB): N={len(edges)} | Win 2R: {(edges['win_2r']==1).mean()*100:.1f}% | Win 1R: {(edges['win_1r']==1).mean()*100:.1f}%")
print(f"Non-Edge Trades (Chasing Middle/Breaks):    N={len(non_edges)} | Win 2R: {(non_edges['win_2r']==1).mean()*100:.1f}% | Win 1R: {(non_edges['win_1r']==1).mean()*100:.1f}%")
