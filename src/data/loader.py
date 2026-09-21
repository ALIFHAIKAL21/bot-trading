"""Market data loader with CCXT Binance fetcher, gap detection, float64 precision, and manifests."""

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

try:
    import ccxt
except ImportError:
    ccxt = None

try:
    import yfinance as yf
except ImportError:
    yf = None


class MarketDataLoader:
    """Unified OHLCV loader supporting CCXT and yfinance fallback with Parquet caching."""

    def __init__(
        self,
        cache_dir: str = "data/cache",
        primary_source: str = "binance",
        gap_fill_policy: str = "ffill_zero_vol",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.primary_source = primary_source
        self.gap_fill_policy = gap_fill_policy
        self._exchange = None
        self.data_quality_stats: Dict[str, Dict] = {}
        self.data_provenance: Dict[str, Dict] = {}

    def _get_ccxt_exchange(self):
        """Initialize CCXT Binance instance configured with Binance Vision public API."""
        if self._exchange is None and ccxt is not None:
            self._exchange = ccxt.binance(
                {
                    "enableRateLimit": True,
                    "timeout": 30000,
                    "options": {"defaultType": "spot"},
                    "urls": {
                        "api": {
                            "public": "https://data-api.binance.vision/api/v3",
                            "fapiPublic": "https://data-api.binance.vision/api/v3",
                            "dapiPublic": "https://data-api.binance.vision/api/v3",
                            "eapiPublic": "https://data-api.binance.vision/api/v3",
                        }
                    },
                }
            )
        return self._exchange

    @staticmethod
    def _clean_symbol_for_filename(symbol: str) -> str:
        return symbol.replace("/", "_").replace(":", "_").replace("^", "").upper()

    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        clean_sym = self._clean_symbol_for_filename(symbol)
        return self.cache_dir / f"{clean_sym}_{timeframe}.parquet"

    def fetch_ccxt_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since_ms: Optional[int] = None,
        limit: int = 1000,
        max_retries: int = 5,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV from CCXT with pagination and rate limit handling."""
        exchange = self._get_ccxt_exchange()
        if exchange is None:
            raise RuntimeError("ccxt is not installed or unavailable.")

        target_symbol = "PAXG/USDT" if symbol.upper() in ("XAU/USD", "XAUUSD", "GOLD") else symbol

        all_candles = []
        current_since = since_ms
        logger.info(f"Fetching CCXT data for {symbol} (as {target_symbol}, {timeframe}) starting from {since_ms}...")

        batch_count = 0
        while True:
            retries = 0
            batch = None
            while retries < max_retries:
                try:
                    batch = exchange.fetch_ohlcv(
                        target_symbol, timeframe=timeframe, since=current_since, limit=limit
                    )
                    break
                except Exception as e:
                    retries += 1
                    backoff = min(2**retries, 15)
                    logger.warning(
                        f"Retry {retries}/{max_retries} fetching {symbol} via CCXT: {e}. Backing off {backoff}s..."
                    )
                    time.sleep(backoff)

            if not batch:
                if len(all_candles) == 0:
                    logger.warning(f"CCXT returned 0 candles for {symbol} after {max_retries} retries.")
                break

            all_candles.extend(batch)
            batch_count += 1
            last_ts = batch[-1][0]

            # If fewer candles returned than limit or reached present time, stop pagination
            if len(batch) < limit or (current_since is not None and last_ts <= current_since):
                break

            # Advance since to next millisecond
            current_since = last_ts + 1
            time.sleep(exchange.rateLimit / 1000.0)

            # Safeguard: break if last_ts is within 1 timeframe of now
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            tf_ms = 3600 * 1000 if timeframe == "1h" else 86400 * 1000
            if last_ts >= now_ms - tf_ms:
                break

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # Enforce float64 precision
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float64)

        df["is_zero_volume"] = (df["volume"] == 0.0) | df["volume"].isna()
        df["is_gap_filled"] = False

        logger.success(
            f"CCXT pagination complete for {symbol}: fetched {len(df):,} bars across {batch_count} batches."
        )
        return df

    def fetch_yfinance_ohlcv(
        self, symbol: str, timeframe: str = "1h", history_days: int = 1095
    ) -> pd.DataFrame:
        """Fallback loader via yfinance (handles .JK stocks and crypto when CCXT is blocked)."""
        if yf is None:
            raise RuntimeError("yfinance is not installed or unavailable.")

        yf_symbol = symbol
        if symbol.upper() in ("XAU/USD", "XAUUSD", "GOLD"):
            yf_symbol = "GC=F"
        elif "/" in symbol:
            base, quote = symbol.split("/")
            if quote == "USDT":
                yf_symbol = f"{base}-USD"
            else:
                yf_symbol = f"{base}-{quote}"

        yf_interval = "1h" if timeframe == "1h" else "1d"
        # yfinance 1h data is restricted to max 730 days
        period_str = "730d" if timeframe == "1h" else f"{history_days}d"

        logger.warning("=" * 70)
        logger.warning(f"FALLBACK WARNING: Fetching yfinance data for {yf_symbol} ({yf_interval})")
        logger.warning("yfinance 1h data is strictly capped at ~730 days and uses float32 prices.")
        logger.warning("=" * 70)

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(interval=yf_interval, period=period_str)

        if df.empty:
            logger.warning(f"No data returned by yfinance for {yf_symbol}")
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        req_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in req_cols if c in df.columns]
        df = df[available_cols].copy()

        # Enforce float64 precision
        for col in available_cols:
            df[col] = df[col].astype(np.float64)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp"
        df.sort_index(inplace=True)
        df["is_zero_volume"] = (df["volume"] == 0.0) | df["volume"].isna()
        df["is_gap_filled"] = False
        return df

    def clean_and_fill_gaps(self, df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
        """Detect gaps in datetime index and apply filling policy, tagging gap bars."""
        if df.empty:
            return df

        df = df[~df.index.duplicated(keep="first")].sort_index()

        # Reindex to uniform regular frequency
        full_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz
        )
        missing_count = len(full_idx) - len(df)

        if missing_count > 0:
            missing_bars_idx = full_idx.difference(df.index)
            logger.info(
                f"Detected {missing_count} missing bars in {freq} series. Applying '{self.gap_fill_policy}' policy."
            )
            df = df.reindex(full_idx)
            df.loc[missing_bars_idx, "is_gap_filled"] = True
            df.loc[missing_bars_idx, "is_zero_volume"] = True

            if self.gap_fill_policy == "ffill_zero_vol":
                df["close"] = df["close"].ffill()
                df["open"] = df["open"].fillna(df["close"])
                df["high"] = df["high"].fillna(df["close"])
                df["low"] = df["low"].fillna(df["close"])
                df["volume"] = df["volume"].fillna(0.0)
            else:
                df = df.ffill()
        else:
            if "is_gap_filled" not in df.columns:
                df["is_gap_filled"] = False

        df.index.name = "timestamp"
        return df

    def load_or_fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        history_days: int = 1095,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Load from local Parquet cache if valid, otherwise fetch from primary source."""
        cache_path = self.get_cache_path(symbol, timeframe)

        if not force_refresh and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty and len(df) >= 20000:
                    logger.info(
                        f"Loaded {len(df):,} bars for {symbol} ({timeframe}) from cache: {cache_path}"
                    )
                    self._record_provenance(symbol, timeframe, "parquet_cache", df, cache_path)
                    return df
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_path}: {e}. Refetching...")

        # Calculate start timestamp for CCXT
        start_dt = datetime.now(timezone.utc) - timedelta(days=history_days)
        since_ms = int(start_dt.timestamp() * 1000)

        df = pd.DataFrame()
        used_source = "none"

        # If symbol is IDX stock (e.g. BBCA.JK), use yfinance directly
        if symbol.endswith(".JK"):
            df = self.fetch_yfinance_ohlcv(symbol, timeframe, history_days)
            used_source = "yfinance_stock"
        else:
            # Try CCXT Binance first
            try:
                df = self.fetch_ccxt_ohlcv(symbol, timeframe, since_ms)
                if not df.empty:
                    used_source = "ccxt_binance_vision"
            except Exception as e:
                logger.warning(f"CCXT fetch error for {symbol}: {e}. Trying fallback.")

            # Fallback to yfinance if CCXT failed or returned empty
            if df.empty:
                df = self.fetch_yfinance_ohlcv(symbol, timeframe, history_days)
                if not df.empty:
                    used_source = "yfinance_fallback"

        if df.empty:
            raise ValueError(f"Could not load data for {symbol} from either CCXT or yfinance.")

        # Clean and gap-fill
        df = self.clean_and_fill_gaps(df, freq="1h" if timeframe == "1h" else "1d")

        # Save to Parquet cache
        df.to_parquet(cache_path)
        logger.success(f"Cached {len(df):,} bars for {symbol} to {cache_path}")

        self._record_provenance(symbol, timeframe, used_source, df, cache_path)
        return df

    def _record_provenance(
        self, symbol: str, timeframe: str, source: str, df: pd.DataFrame, cache_path: Path
    ) -> None:
        """Record data provenance and quality statistics."""
        n_total = len(df)
        n_zero_vol = int(df["is_zero_volume"].sum()) if "is_zero_volume" in df.columns else 0
        n_gap_filled = int(df["is_gap_filled"].sum()) if "is_gap_filled" in df.columns else 0

        # Compute SHA256 of cache file if it exists
        sha = "unknown"
        if cache_path.exists():
            h = hashlib.sha256()
            with open(cache_path, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            sha = h.hexdigest()

        key = f"{symbol}_{timeframe}"
        self.data_quality_stats[key] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_bars": n_total,
            "zero_volume_bars": n_zero_vol,
            "zero_volume_pct": round(n_zero_vol / max(n_total, 1) * 100.0, 3),
            "gap_filled_bars": n_gap_filled,
            "gap_filled_pct": round(n_gap_filled / max(n_total, 1) * 100.0, 3),
            "start_timestamp": str(df.index[0]),
            "end_timestamp": str(df.index[-1]),
            "dtype_prices": str(df["close"].dtype),
            "dtype_volume": str(df["volume"].dtype),
        }

        self.data_provenance[key] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": source,
            "cache_file": str(cache_path),
            "cache_sha256": sha,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def save_reports(self, output_dir: str = "reports") -> None:
        """Persist data_quality.json and data_manifest.json."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "data_quality.json", "w") as f:
            json.dump(self.data_quality_stats, f, indent=2)

        with open(out / "data_manifest.json", "w") as f:
            json.dump(self.data_provenance, f, indent=2)

        logger.info(f"Saved data quality and provenance reports to {output_dir}")
