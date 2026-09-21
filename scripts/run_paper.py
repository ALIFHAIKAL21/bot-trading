"""CLI script to run paper trading service or single cycle."""

import argparse
import sys
from pathlib import Path

# Add repository root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import uvicorn
from src.service.scheduler import PaperTradingScheduler
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Paper trading runner.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--once", action="store_true", help="Run single evaluation cycle and exit")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI REST server")
    parser.add_argument("--loop", action="store_true", help="Run background scheduler loop")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.serve:
        logger.info(f"Launching FastAPI service on {cfg.service.host}:{cfg.service.port}...")
        uvicorn.run(
            "src.service.app:app",
            host=cfg.service.host,
            port=cfg.service.port,
            reload=False,
        )
    elif args.once:
        logger.info("Executing single paper trading step...")
        scheduler = PaperTradingScheduler(args.config)
        for sym in cfg.market.symbols:
            scheduler.run_step(sym)
    else:
        # Default: run loop
        scheduler = PaperTradingScheduler(args.config)
        scheduler.run_loop()


if __name__ == "__main__":
    main()
