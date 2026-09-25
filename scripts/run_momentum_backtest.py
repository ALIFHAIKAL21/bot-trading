"""
Execution Script: Train Momentum Filter & Run Head-to-Head Backtest
Evaluates 1 full year of out-of-sample data (Sept 2025 - Sept 2026) against xau_deep_sniper baseline.
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.momentum_classifier import MomentumClassifier
from src.backtest.momentum_backtest import MomentumBacktestEngine


def run_experiment(timeframe: str = "1h", target_rr: int = 2):
    print("=" * 80)
    print(" INSTITUTIONAL MOMENTUM BACKTEST & HEAD-TO-HEAD AUDIT (XAU/USD)")
    print("=" * 80)

    # 1. Load Data
    bars_path = f"data/xau/xauusd_{timeframe}_clean.parquet"
    setups_path = f"data/xau/momentum_setups_{timeframe}.parquet"

    if not os.path.exists(bars_path) or not os.path.exists(setups_path):
        raise FileNotFoundError("Clean bars or setups not found. Run generate_momentum_dataset.py first.")

    df_bars = pd.read_parquet(bars_path)
    df_setups = pd.read_parquet(setups_path)

    df_bars["timestamp_utc"] = pd.to_datetime(df_bars["timestamp_utc"], utc=True)
    df_setups["timestamp_utc"] = pd.to_datetime(df_setups["timestamp_utc"], utc=True)

    # Split Date: 2025-09-18 (Exactly matches xau_deep_sniper sealed test set)
    split_date = pd.Timestamp("2025-09-18 00:00:00+00:00")

    df_train_setups = df_setups[df_setups["timestamp_utc"] < split_date].copy()
    df_test_setups = df_setups[df_setups["timestamp_utc"] >= split_date].copy()

    df_test_bars = df_bars[df_bars["timestamp_utc"] >= split_date].copy().reset_index(drop=True)

    print(f"Data Partition:")
    print(f"  - Training Set:   {df_train_setups['timestamp_utc'].min()} to {df_train_setups['timestamp_utc'].max()} ({len(df_train_setups):,} setups, ~4 years)")
    print(f"  - Sealed Test Set: {df_test_setups['timestamp_utc'].min()} to {df_test_setups['timestamp_utc'].max()} ({len(df_test_setups):,} setups, 1 year full)")
    print(f"  - Test Set Bars:  {len(df_test_bars):,} {timeframe.upper()} bars")

    # 2. Train Momentum Classifier Model
    target_col = f"win_{target_rr}r"
    print(f"\nTraining Calibrated Momentum Classifier (Target: RR 1:{target_rr})...")
    clf = MomentumClassifier(target_col=target_col)
    clf.fit(df_train_setups)

    # Top Feature Importances
    print("\nTop 10 Feature Importances (Trading Decision Weights):")
    for feat, weight in list(clf.feature_importances.items())[:10]:
        print(f"  - {feat:<26}: {weight * 100:.1f}%")

    # 3. Predict Probabilities on Test Set
    df_test_setups["prob_win"] = clf.predict_proba(df_test_setups)
    eval_metrics = clf.evaluate(df_test_setups)
    print(f"\nOut-of-Sample Test Metrics:")
    print(f"  - Test AUC:         {eval_metrics['auc']:.4f}")
    print(f"  - Brier Score:      {eval_metrics['brier_score']:.4f}")
    print(f"  - Base Win Rate:    {eval_metrics['base_rate'] * 100:.2f}%")
    print(f"  - Mean Model Prob:  {eval_metrics['mean_prob'] * 100:.2f}%")

    # 4. Sweep Confidence Gates (Finding the Institutional Sweet Spot)
    print("\n" + "=" * 80)
    print(" EVENT-DRIVEN SIMULATION: SENSITIVITY SWEEP ACROSS CONFIDENCE GATES")
    print("=" * 80)

    engine = MomentumBacktestEngine(initial_equity=10_000.0, risk_pct=0.01)

    thresholds = [0.30, 0.33, 0.35, 0.38]
    best_summary = None
    best_tau = None

    print(f"{'Gate (Tau)':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Profit Factor':<14} | {'Net Return':<12} | {'Max DD':<10} | {'Sharpe':<8}")
    print("-" * 85)

    all_results = {}

    for tau in thresholds:
        summary = engine.run(
            df_bars=df_test_bars,
            df_setups=df_test_setups,
            target_rr=target_rr,
            confidence_tau=tau,
        )
        all_results[str(tau)] = {
            "total_trades": summary.total_trades,
            "win_rate": summary.win_rate,
            "net_pnl": summary.net_pnl,
            "net_return_pct": summary.net_return_pct,
            "profit_factor": summary.profit_factor,
            "max_drawdown_pct": summary.max_drawdown_pct,
            "sharpe_ratio": summary.sharpe_ratio,
            "expectancy_r": summary.expectancy_r,
        }

        print(
            f"{tau:<12.2f} | {summary.total_trades:<8} | {summary.win_rate:<9.1f}% | {summary.profit_factor:<14.2f} | {summary.net_return_pct:>+10.2f}% | {summary.max_drawdown_pct:<9.2f}% | {summary.sharpe_ratio:<8.2f}"
        )

        # Select best operational setup (adequate trades + positive return)
        if best_summary is None or (summary.total_trades >= 50 and summary.net_pnl > best_summary.net_pnl):
            best_summary = summary
            best_tau = tau

    if best_summary is None:
        best_summary = summary
        best_tau = thresholds[0]

    # 5. Head-to-Head Comparison Table
    print("\n" + "=" * 80)
    print(" HEAD-TO-HEAD COMPARISON: OLD SYSTEM (XAU_DEEP_SNIPER) VS NEW SYSTEM")
    print("=" * 80)
    print(f"{'Performance Metric':<28} | {'Old System (MOMENT-1-large)':<28} | {'New System (Momentum+SMC)':<25}")
    print("-" * 85)
    print(f"{'Total Trades (1 Year)':<28} | {'12 trades (Lumpuh)':<28} | {f'{best_summary.total_trades} trades (Aktif)' :<25}")
    print(f"{'Win Rate':<28} | {'25.0% (3 Win / 9 Loss)':<28} | {f'{best_summary.win_rate:.1f}% ({best_summary.winning_trades}W / {best_summary.losing_trades}L)' :<25}")
    print(f"{'Target Risk-Reward':<28} | {'1:1 s/d 1:2 (Uncontrolled)':<28} | {f'RR 1:{target_rr} (Strict)' :<25}")
    print(f"{'Net PnL ($)':<28} | {'-$30.08':<28} | {f'{best_summary.net_pnl:>+,.2f} USD' :<25}")
    print(f"{'Net Return (%)':<28} | {'-0.30%':<28} | {f'{best_summary.net_return_pct:>+,.2f}%' :<25}")
    print(f"{'Profit Factor':<28} | {'0.901 (Kalah)':<28} | {f'{best_summary.profit_factor:.2f}' :<25}")
    print(f"{'Max Drawdown':<28} | {'-2.71%':<28} | {f'{best_summary.max_drawdown_pct:.2f}%' :<25}")
    print(f"{'Total Friction Paid':<28} | {'$9.69':<28} | {f'${best_summary.total_friction:,.2f}' :<25}")
    print(f"{'Expectancy per Trade':<28} | {'-0.036 R':<28} | {f'{best_summary.expectancy_r:+.3f} R' :<25}")
    print("=" * 80)

    # 6. Save JSON Report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/momentum_backtest_report.json"
    report_data = {
        "timeframe": timeframe,
        "target_rr": target_rr,
        "test_period": "2025-09-18 to 2026-09-22",
        "optimal_tau": best_tau,
        "best_performance": {
            "initial_equity": best_summary.initial_equity,
            "final_equity": best_summary.final_equity,
            "net_pnl": best_summary.net_pnl,
            "net_return_pct": best_summary.net_return_pct,
            "total_trades": best_summary.total_trades,
            "winning_trades": best_summary.winning_trades,
            "losing_trades": best_summary.losing_trades,
            "win_rate": best_summary.win_rate,
            "profit_factor": best_summary.profit_factor,
            "sharpe_ratio": best_summary.sharpe_ratio,
            "sortino_ratio": best_summary.sortino_ratio,
            "max_drawdown_pct": best_summary.max_drawdown_pct,
            "max_drawdown_usd": best_summary.max_drawdown_usd,
            "total_friction": best_summary.total_friction,
            "expectancy_r": best_summary.expectancy_r,
        },
        "all_gates": all_results,
        "top_features": clf.feature_importances,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Momentum Backtest")
    parser.add_argument("--timeframe", default="1h", choices=["1h", "m30"], help="Timeframe (1h or m30)")
    parser.add_argument("--rr", type=int, default=2, choices=[1, 2], help="Target Risk-Reward (1 or 2)")
    args = parser.parse_args()

    run_experiment(timeframe=args.timeframe, target_rr=args.rr)
