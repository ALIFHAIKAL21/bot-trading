import pandas as pd
import numpy as np

df = pd.read_parquet('data/xau/momentum_setups_1h.parquet')
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
df['year'] = df['timestamp_utc'].dt.year

df_2023 = df[df['year'] == 2023]
print("=== 2023 PERFORMANCE PER PRIMARY TRIGGER ===")
for trig, grp in df_2023.groupby('primary_trigger'):
    print(f"Trigger: {trig:<25} | N={len(grp):<4} | Win 1R: {(grp['win_1r']==1).mean()*100:.1f}% | Win 2R: {(grp['win_2r']==1).mean()*100:.1f}%")

print("\n=== ALL YEARS (2021-2026) PERFORMANCE PER PRIMARY TRIGGER ===")
for trig, grp in df.groupby('primary_trigger'):
    print(f"Trigger: {trig:<25} | N={len(grp):<4} | Win 1R: {(grp['win_1r']==1).mean()*100:.1f}% | Win 2R: {(grp['win_2r']==1).mean()*100:.1f}%")
