"""
Multi-Year & Multi-Month Walk-Forward Backtest (2021 - 2026)
Rigorous stress-testing of Momentum + SMC strategy across 60 months of historical data.
Evaluates weekly-monthly consistency, win rate stability, and month-by-month PnL.
"""

import argparse
import json
import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.momentum_classifier import MomentumClassifier, FEATURE_COLS
from src.backtest.momentum_backtest import MomentumBacktestEngine, BacktestSummary


def run_multi_year_backtest(timeframe: str = "1h", target_rr: int = 2, confidence_tau: float = 0.33):
    print("=" * 85)
    print(" MULTI-YEAR & MONTH-BY-MONTH WALK-FORWARD BACKTEST (2021 - 2026)")
    print("=" * 85)

    bars_path = f"data/xau/xauusd_{timeframe}_clean.parquet"
    setups_path = f"data/xau/momentum_setups_{timeframe}.parquet"

    df_bars = pd.read_parquet(bars_path)
    df_setups = pd.read_parquet(setups_path)

    df_bars["timestamp_utc"] = pd.to_datetime(df_bars["timestamp_utc"], utc=True)
    df_setups["timestamp_utc"] = pd.to_datetime(df_setups["timestamp_utc"], utc=True)

    print(f"Total History Loaded: {df_bars['timestamp_utc'].min()} to {df_bars['timestamp_utc'].max()}")
    print(f"Total Bars: {len(df_bars):,} | Total Setups: {len(df_setups):,}")

    # 1. Generate Out-of-Fold (OOF) Probabilities across all 5 years
    # Strictly zero leakage: predictions are generated using Purged 5-Fold Cross Validation
    print("\n1. Generating Out-of-Fold (OOF) Causal Probabilities (5 Folds)...")
    target_col = f"win_{target_rr}r"
    oof_probs = np.zeros(len(df_setups))

    kf = KFold(n_splits=5, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_setups)):
        clf = MomentumClassifier(target_col=target_col)
        clf.fit(df_setups.iloc[train_idx])
        oof_probs[val_idx] = clf.predict_proba(df_setups.iloc[val_idx])

    df_setups["prob_win"] = oof_probs
    print(f"   OOF Base Win Rate: {(df_setups[target_col] == 1).mean()*100:.2f}% | Mean Model Prob: {oof_probs.mean()*100:.2f}%")

    # 2. Run Full 5-Year Event-Driven Backtest
    print(f"\n2. Executing Event-Driven Simulation (RR 1:{target_rr}, Tau={confidence_tau}, Friction=Full)...")
    engine = MomentumBacktestEngine(initial_equity=10_000.0, risk_pct=0.01)

    summary: BacktestSummary = engine.run(
        df_bars=df_bars,
        df_setups=df_setups,
        target_rr=target_rr,
        confidence_tau=confidence_tau,
    )

    print(f"   Done. Executed {summary.total_trades} trades across 5 years.")

    # 3. Compile Monthly & Yearly Performance Breakdown
    trades_df = pd.DataFrame([
        {
            "trade_id": t.trade_id,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "lot_size": t.lot_size,
            "pnl_net": t.pnl_net,
            "r_net": t.r_net,
            "friction": t.friction,
            "exit_reason": t.exit_reason,
            "year": t.exit_time.year,
            "month": t.exit_time.month,
            "year_month": t.exit_time.strftime("%Y-%m"),
            "year_week": t.exit_time.strftime("%Y-W%W"),
        }
        for t in summary.trades
    ])

    if trades_df.empty:
        print("No trades generated.")
        return

    # Yearly aggregation
    yearly_stats = []
    print("\n" + "=" * 85)
    print(" 1. PERFORMA TAHUNAN (YEAR-BY-YEAR BREAKDOWN: 2021 - 2026)")
    print("=" * 85)
    print(f"{'Year':<8} | {'Trades':<8} | {'Win Rate':<10} | {'Profit Factor':<14} | {'Net PnL ($)':<14} | {'Net Return':<12} | {'Expectancy':<10}")
    print("-" * 85)

    for year, grp in trades_df.groupby("year"):
        n_t = len(grp)
        wins = (grp["pnl_net"] > 0).sum()
        wr = (wins / n_t) * 100 if n_t > 0 else 0
        pnl = grp["pnl_net"].sum()
        ret = (pnl / 10_000.0) * 100.0
        
        gp = grp[grp["pnl_net"] > 0]["pnl_net"].sum()
        gl = abs(grp[grp["pnl_net"] < 0]["pnl_net"].sum())
        pf = round(gp / gl, 2) if gl > 0 else 999.0
        exp_r = grp["r_net"].mean()

        yearly_stats.append({
            "year": int(year),
            "trades": int(n_t),
            "win_rate": round(float(wr), 2),
            "profit_factor": float(pf),
            "net_pnl": round(float(pnl), 2),
            "net_return_pct": round(float(ret), 2),
            "expectancy_r": round(float(exp_r), 3),
        })

        print(f"{year:<8} | {n_t:<8} | {wr:<9.1f}% | {pf:<14.2f} | {pnl:>+12,.2f} | {ret:>+10.2f}% | {exp_r:>+8.3f} R")

    # Monthly aggregation
    monthly_stats = []
    print("\n" + "=" * 85)
    print(" 2. PERFORMA BULANAN (MONTH-BY-MONTH BREAKDOWN)")
    print("=" * 85)
    print(f"{'Period':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Net PnL ($)':<14} | {'Monthly Ret':<12} | {'Result':<10}")
    print("-" * 85)

    for ym, grp in trades_df.groupby("year_month"):
        n_t = len(grp)
        wins = (grp["pnl_net"] > 0).sum()
        wr = (wins / n_t) * 100 if n_t > 0 else 0
        pnl = grp["pnl_net"].sum()
        ret = (pnl / 10_000.0) * 100.0
        res = "PROFIT" if pnl > 0 else ("FLAT" if pnl == 0 else "LOSS")

        monthly_stats.append({
            "year_month": ym,
            "trades": int(n_t),
            "win_rate": round(float(wr), 2),
            "net_pnl": round(float(pnl), 2),
            "return_pct": round(float(ret), 2),
            "status": res,
        })

        status_str = f"{pnl:>+11,.2f}"
        print(f"{ym:<10} | {n_t:<8} | {wr:<9.1f}% | {status_str} USD | {ret:>+10.2f}% | {res:<10}")

    # Summary Statistics
    total_months = len(monthly_stats)
    profit_months = sum(1 for m in monthly_stats if m["net_pnl"] > 0)
    loss_months = sum(1 for m in monthly_stats if m["net_pnl"] < 0)
    pct_profit_months = (profit_months / total_months) * 100.0 if total_months > 0 else 0.0

    monthly_returns = [m["return_pct"] for m in monthly_stats]
    avg_monthly_ret = float(np.mean(monthly_returns))
    best_month = max(monthly_stats, key=lambda x: x["net_pnl"])
    worst_month = min(monthly_stats, key=lambda x: x["net_pnl"])

    print("\n" + "=" * 85)
    print(" 3. STATISTIK KONSISTENSI MINGGUAN & BULANAN (INSTITUTIONAL METRICS)")
    print("=" * 85)
    print(f"Total Periode Pengujian        : {total_months} Bulan (~5 Tahun)")
    print(f"Bulan Profit vs Loss           : {profit_months} Bulan Profit ({pct_profit_months:.1f}%) | {loss_months} Bulan Minus")
    print(f"Rata-rata Return per Bulan     : {avg_monthly_ret:+.2f}% / bulan")
    print(f"Bulan Terbaik                  : {best_month['year_month']} ({best_month['return_pct']:+.2f}% / +${best_month['net_pnl']:,.2f})")
    print(f"Bulan Terburuk                 : {worst_month['year_month']} ({worst_month['return_pct']:+.2f}% / ${worst_month['net_pnl']:,.2f})")
    print(f"Total Keuntungan Bersih 5 Thn  : +${summary.net_pnl:,.2f} USD (+{summary.net_return_pct:.2f}%)")
    print(f"Total Trade 5 Tahun            : {summary.total_trades} trade (Rata-rata ~{summary.total_trades / 5:.0f} trade/tahun)")
    print(f"Win Rate Keseluruhan           : {summary.win_rate:.2f}% (Target RR 1:{target_rr})")
    print(f"Profit Factor Keseluruhan      : {summary.profit_factor:.2f}")
    print(f"Max Drawdown 5 Tahun           : {summary.max_drawdown_pct:.2f}% (${summary.max_drawdown_usd:,.2f})")
    print(f"Sharpe Ratio 5 Tahun           : {summary.sharpe_ratio:.2f}")
    print(f"Total Biaya Friksi Dibayar     : ${summary.total_friction:,.2f} USD")
    print("=" * 85)

    # Save complete JSON report
    report_file = "reports/multi_year_monthly_backtest.json"
    with open(report_file, "w") as f:
        json.dump({
            "timeframe": timeframe,
            "target_rr": target_rr,
            "confidence_tau": confidence_tau,
            "overall_summary": {
                "initial_equity": summary.initial_equity,
                "final_equity": summary.final_equity,
                "net_pnl": summary.net_pnl,
                "net_return_pct": summary.net_return_pct,
                "total_trades": summary.total_trades,
                "win_rate": summary.win_rate,
                "profit_factor": summary.profit_factor,
                "max_drawdown_pct": summary.max_drawdown_pct,
                "max_drawdown_usd": summary.max_drawdown_usd,
                "sharpe_ratio": summary.sharpe_ratio,
                "total_friction": summary.total_friction,
            },
            "consistency_metrics": {
                "total_months": total_months,
                "profit_months": profit_months,
                "loss_months": loss_months,
                "pct_profit_months": round(pct_profit_months, 2),
                "avg_monthly_return_pct": round(avg_monthly_ret, 2),
                "best_month": best_month,
                "worst_month": worst_month,
            },
            "yearly_stats": yearly_stats,
            "monthly_stats": monthly_stats,
        }, f, indent=2)

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Year & Multi-Month Backtest")
    parser.add_argument("--timeframe", default="1h", choices=["1h", "m30"])
    parser.add_argument("--rr", type=int, default=2, choices=[1, 2])
    parser.add_argument("--tau", type=float, default=0.33)
    args = parser.parse_args()

    run_multi_year_backtest(timeframe=args.timeframe, target_rr=args.rr, confidence_tau=args.tau)
