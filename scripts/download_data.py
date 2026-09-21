"""CLI script to download historical OHLCV data and cache to Parquet."""

import argparse
import sys
from pathlib import Path

# Add repository root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.data.loader import MarketDataLoader
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Download and cache historical market data.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to download (e.g. BTC/USDT ETH/USDT)")
    parser.add_argument("--timeframe", type=str, default=None, help="Candle timeframe (e.g. 1h)")
    parser.add_argument("--days", type=int, default=None, help="History days to download")
    parser.add_argument("--force", action="store_true", help="Force refresh cache from exchange")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = args.symbols or cfg.market.symbols
    timeframe = args.timeframe or cfg.market.timeframe
    history_days = args.days or cfg.market.history_days

    logger.info(f"Starting data download for symbols: {symbols} | timeframe: {timeframe} | history: {history_days} days")
    loader = MarketDataLoader(
        cache_dir=cfg.data.cache_dir,
        primary_source=cfg.data.primary_source,
        gap_fill_policy=cfg.data.gap_fill_policy,
    )

    results = {}
    for sym in symbols:
        try:
            logger.info(f"Fetching {sym}...")
            df = loader.load_or_fetch(
                symbol=sym,
                timeframe=timeframe,
                history_days=history_days,
                force_refresh=args.force,
            )
            results[sym] = {
                "bars": len(df),
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "cache_file": str(loader.get_cache_path(sym, timeframe)),
            }
            logger.success(f"Successfully processed {sym}: {len(df)} bars from {df.index.min()} to {df.index.max()}")
        except Exception as e:
            logger.error(f"Failed to fetch {sym}: {e}")
            results[sym] = {"error": str(e)}

    loader.save_reports()
    logger.info("Download summary:")
    for sym, res in results.items():
        logger.info(f"  {sym}: {res}")


if __name__ == "__main__":
    main()
