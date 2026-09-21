"""CLI script to download and cache 500,000+ bars dataset from Binance Vision."""

import argparse
import sys
from pathlib import Path

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.data.bulk_downloader import BulkMarketDataDownloader


def main():
    parser = argparse.ArgumentParser(description="Download bulk 500k+ bars dataset from Binance Vision.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol (e.g. BTC/USDT, ETH/USDT)")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe (e.g. 1m or 5m)")
    parser.add_argument("--year", type=int, default=2024, help="Year of data to download")
    parser.add_argument("--min-bars", type=int, default=500_000, help="Minimum bars requirement")
    args = parser.parse_args()

    logger.info(f"Initiating bulk download for {args.symbol} ({args.timeframe}) for year {args.year}...")
    downloader = BulkMarketDataDownloader()
    df = downloader.download_bulk_dataset(
        symbol=args.symbol,
        timeframe=args.timeframe,
        year=args.year,
        min_bars_required=args.min_bars,
    )
    logger.success(f"Ingested {len(df):,} total bars successfully!")


if __name__ == "__main__":
    main()
