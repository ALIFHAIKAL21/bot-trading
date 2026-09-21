"""Unified cross-platform CLI runner for the multi-model quant trading system."""

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON_EXE = str(Path(".venv/Scripts/python.exe") if sys.platform == "win32" else Path(".venv/bin/python"))


def run_cmd(cmd_list):
    print(f"\n[RUN] {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Quant Trading System Runner")
    parser.add_argument("--setup", action="store_true", help="Setup environment and install dependencies")
    parser.add_argument("--data", action="store_true", help="Download and cache historical data")
    parser.add_argument("--train", action="store_true", help="Train all models with purged CV and stacking")
    parser.add_argument("--backtest", action="store_true", help="Run backtest, baselines, and ablation")
    parser.add_argument("--test", action="store_true", help="Run pytest unit and leakage test suite")
    parser.add_argument("--bulk-data", action="store_true", help="Download 500,000+ bulk dataset from Binance Vision")
    parser.add_argument("--train-deep", action="store_true", help="Fine-tune deep neural network on GPU with 500k dataset")
    parser.add_argument("--paper", action="store_true", help="Run single cycle of paper trading execution")
    parser.add_argument("--paper-loop", action="store_true", help="Run continuous 24/7 paper trading scheduler daemon")
    parser.add_argument("--scalp-5m", action="store_true", help="Run 5-minute scalping demonstration replay and open positions")
    parser.add_argument("--scalp-live", action="store_true", help="Run live real-time 5-minute scalper daemon")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI REST microservice")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--all", action="store_true", help="Run full pipeline end-to-end: data -> train -> backtest -> test")
    args = parser.parse_args()

    if args.setup:
        run_cmd(["uv", "pip", "install", "-r", "requirements.txt", "--python", PYTHON_EXE])

    elif args.bulk_data:
        run_cmd([PYTHON_EXE, "scripts/download_bulk_data.py"])

    elif args.train_deep:
        run_cmd([PYTHON_EXE, "scripts/train_deep.py"])

    elif args.data:
        run_cmd([PYTHON_EXE, "scripts/download_data.py"])

    elif args.train:
        run_cmd([PYTHON_EXE, "scripts/train_all.py"])

    elif args.backtest:
        run_cmd([PYTHON_EXE, "scripts/backtest.py"])

    elif args.test:
        run_cmd([PYTHON_EXE, "-m", "pytest", "tests/", "-v"])

    elif args.scalp_5m:
        run_cmd([PYTHON_EXE, "scripts/test_scalp_5m.py", "--replay"])

    elif args.scalp_live:
        run_cmd([PYTHON_EXE, "scripts/test_scalp_5m.py", "--live"])

    elif args.paper:
        run_cmd([PYTHON_EXE, "scripts/run_paper.py", "--once"])

    elif args.paper_loop:
        run_cmd([PYTHON_EXE, "scripts/run_paper.py", "--loop"])


    elif args.serve:
        run_cmd([PYTHON_EXE, "scripts/run_paper.py", "--serve"])

    elif args.dashboard:
        run_cmd([PYTHON_EXE, "-m", "streamlit", "run", "src/dashboard/app.py", "--server.port", "8502"])

    elif args.all:
        print("=== Step 1/4: Data Ingestion ===")
        run_cmd([PYTHON_EXE, "scripts/download_data.py"])

        print("\n=== Step 2/4: Model Training & CV Stacking ===")
        run_cmd([PYTHON_EXE, "scripts/train_all.py"])

        print("\n=== Step 3/4: Backtesting & Ablation ===")
        run_cmd([PYTHON_EXE, "scripts/backtest.py"])

        print("\n=== Step 4/4: Validation Tests ===")
        run_cmd([PYTHON_EXE, "-m", "pytest", "tests/", "-v"])

        print("\n=== ALL PIPELINE STEPS SUCCEEDED! ===")
        print("Launch the interactive dashboard with: python run.py --dashboard")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
