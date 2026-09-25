import sys, pathlib
import pandas as pd

sys.path.append(r'c:\Ngoding\xau_deep_sniper')
from src.pipeline.feature_pipeline import temporal_split

data_dir = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\data\processed')
df_15ch = pd.read_parquet(data_dir / 'xauusd_m30_labeled_15ch.parquet')

print(f"Splitting 15-channel dataset ({len(df_15ch)} rows)...")
train_df, test_df = temporal_split(df_15ch, train_ratio=0.80, purge_bars=16, embargo_bars=48)

train_path = data_dir / 'xauusd_m30_train_labeled_15ch.parquet'
test_path = data_dir / 'xauusd_m30_test_labeled_15ch.parquet'

train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)

print(f"Saved Train: {train_path} ({len(train_df)} rows)")
print(f"Saved Test:  {test_path} ({len(test_df)} rows)")
