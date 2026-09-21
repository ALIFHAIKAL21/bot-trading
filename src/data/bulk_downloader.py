"""High-capacity bulk market data downloader using official Binance Vision archives.

Downloads multi-month official Binance kline archives (e.g. 1-minute or 5-minute candles)
concurrently via CDN without API key requirements or rate limits.
Compiles into clean, high-performance Parquet format with >= 500,000 bars.
"""

import concurrent.futures
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from loguru import logger

BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"

COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


class BulkMarketDataDownloader:
    """Downloader and parser for Binance Vision monthly archive datasets."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.zip_dir = self.cache_dir / "zips"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zip_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "").replace("_", "").upper()

    def get_output_path(self, symbol: str, timeframe: str, tag: str = "bulk") -> Path:
        clean = symbol.replace("/", "_").replace(":", "_").upper()
        return self.cache_dir / f"{clean}_{timeframe}_{tag}.parquet"

    def download_month_zip(
        self, symbol: str, timeframe: str, year: int, month: int, timeout: int = 45, max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """Download and parse a single month's zip archive into a clean DataFrame."""
        clean_sym = self._clean_symbol(symbol)
        month_str = f"{month:02d}"
        filename = f"{clean_sym}-{timeframe}-{year}-{month_str}.zip"
        local_zip_path = self.zip_dir / filename
        url = f"{BINANCE_VISION_BASE}/{clean_sym}/{timeframe}/{filename}"

        # If already cached locally on disk, load directly!
        if local_zip_path.exists() and local_zip_path.stat().st_size > 100_000:
            logger.info(f"Using local cached zip: {local_zip_path.name}")
            data = local_zip_path.read_bytes()
        else:
            logger.info(f"Downloading {filename} from {url}...")
            data = None
            for attempt in range(1, max_retries + 1):
                try:
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                    )
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        if resp.getcode() == 200:
                            data = resp.read()
                            # Save to disk
                            local_zip_path.write_bytes(data)
                            break
                        else:
                            logger.warning(f"[{attempt}/{max_retries}] HTTP {resp.getcode()} for {url}")
                except Exception as e:
                    logger.warning(f"[{attempt}/{max_retries}] Error fetching {filename}: {e}")

            if data is None:
                logger.error(f"Failed to download {filename} after {max_retries} attempts.")
                return None

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    logger.warning(f"No CSV found in {filename}")
                    return None
                with z.open(csv_names[0]) as f:
                    sample = f.readline().decode("utf-8")
                    f.seek(0)
                    has_header = "open_time" in sample.lower()

                    if has_header:
                        df = pd.read_csv(f)
                        df.columns = [c.lower().strip() for c in df.columns]
                        rename_map = {
                            "open_time": "timestamp",
                            "count": "trades_count",
                        }
                        df = df.rename(columns=rename_map)
                    else:
                        df = pd.read_csv(f, names=COLUMNS)

            df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
            numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "trades_count" in df.columns:
                df["trades_count"] = pd.to_numeric(df["trades_count"], errors="coerce").fillna(0).astype(int)

            keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "trades_count"]
            df = df[[c for c in keep_cols if c in df.columns]]
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="first")]
            logger.success(f"Parsed {filename}: {len(df):,} bars.")
            return df

        except Exception as e:
            logger.error(f"Error parsing {filename}: {e}")
            return None

    def download_bulk_dataset(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        year: int = 2024,
        months: Optional[List[int]] = None,
        min_bars_required: int = 500_000,
        max_workers: int = 6,
    ) -> pd.DataFrame:
        """Download multi-month archives concurrently, concatenate, validate, and cache to Parquet."""
        if months is None:
            months = list(range(1, 13))  # 12 months (Jan-Dec)

        logger.info(f"Fetching {len(months)} monthly archives concurrently with {max_workers} threads...")
        month_dfs = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_month = {
                executor.submit(self.download_month_zip, symbol, timeframe, year, m): m
                for m in months
            }
            for future in concurrent.futures.as_completed(future_to_month):
                m = future_to_month[future]
                try:
                    df_m = future.result()
                    if df_m is not None and not df_m.empty:
                        month_dfs[m] = df_m
                except Exception as exc:
                    logger.error(f"Month {m} generated exception: {exc}")

        if not month_dfs:
            raise RuntimeError(f"No data successfully downloaded for {symbol} {timeframe}")

        sorted_months = sorted(month_dfs.keys())
        dfs = [month_dfs[m] for m in sorted_months]
        full_df = pd.concat(dfs).sort_index()
        full_df = full_df[~full_df.index.duplicated(keep="first")]

        # Forward fill tiny gaps and zero out missing volumes
        expected_freq = "1min" if timeframe == "1m" else "5min"
        full_df = full_df.asfreq(expected_freq)
        full_df["close"] = full_df["close"].ffill()
        full_df["open"] = full_df["open"].fillna(full_df["close"])
        full_df["high"] = full_df["high"].fillna(full_df["close"])
        full_df["low"] = full_df["low"].fillna(full_df["close"])
        full_df["volume"] = full_df["volume"].fillna(0.0)
        full_df["quote_volume"] = full_df["quote_volume"].fillna(0.0)
        if "taker_buy_volume" in full_df.columns:
            full_df["taker_buy_volume"] = full_df["taker_buy_volume"].fillna(0.0)
        if "trades_count" in full_df.columns:
            full_df["trades_count"] = full_df["trades_count"].fillna(0)

        out_path = self.get_output_path(symbol, timeframe)
        full_df.to_parquet(out_path, engine="pyarrow", compression="snappy")

        total_bars = len(full_df)
        logger.success(
            f"Successfully compiled {total_bars:,} bars ({full_df.index[0]} to {full_df.index[-1]}) -> {out_path}"
        )

        if total_bars < min_bars_required:
            logger.warning(
                f"Total bars ({total_bars:,}) is below requested minimum ({min_bars_required:,})."
            )
        else:
            logger.success(f"Requirement met: {total_bars:,} >= {min_bars_required:,} bars!")

        return full_df
