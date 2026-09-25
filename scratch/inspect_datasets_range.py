import pandas as pd
import pathlib

data_dir = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\data\processed')
for p in data_dir.glob('*.parquet'):
    try:
        df = pd.read_parquet(p, columns=['timestamp_utc'])
        t_min = df['timestamp_utc'].min()
        t_max = df['timestamp_utc'].max()
        print(f"{p.name:35s} | Rows: {len(df):6,d} | Min: {t_min} | Max: {t_max}")
    except Exception as e:
        print(f"{p.name:35s} | Error: {e}")
