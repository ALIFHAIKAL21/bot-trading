"""
CLI Script: Generate Discrete Momentum Dataset (Point 3 & Point 4)
Processes 5 years of XAU/USD historical data with TradingView Default Indicators
and Causal Market Structure, creating labeled transaction setups for Deep Learning.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.indicator_signals import compute_all_indicators
from src.features.market_structure import compute_market_structure
from src.labels.momentum_labeler import extract_momentum_setups


def generate_dataset(timeframe: str = "1h", max_holding_bars: int = 48) -> pd.DataFrame:
    data_path = f"data/xau/xauusd_{timeframe}_clean.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Clean data file not found: {data_path}")

    print("=" * 70)
    print(f" GENERATING MOMENTUM DATASET: XAU/USD {timeframe.upper()} (5-YEAR HISTORICAL)")
    print("=" * 70)
    print(f"1. Loading clean data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"   Total raw bars: {len(df):,}")
    print(f"   Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")

    print("\n2. Computing TradingView Default Indicators...")
    print("   - MA Cross: Fast SMA 9 x Slow SMA 21 (with EMA 9/21 enrichment)")
    print("   - SMI: %K=10, %K Smooth=3, %K Double Smooth=3, %D Signal=10, Bands=[+40, -40]")
    df_ind = compute_all_indicators(df)

    print("\n3. Computing Causal Market Structure (Point 4)...")
    print("   - Support & Resistance Zones from confirmed Swings (lookback=3)")
    print("   - Supply & Demand Order Blocks (1.5x ATR impulsive expansion)")
    print("   - Liquidity Sweeps (SSL & BSL fakeouts)")
    print("   - Premium / Discount Valuation Zones")
    df_ms = compute_market_structure(df_ind)
    df_full = pd.concat([df_ind, df_ms], axis=1)

    print("\n4. Extracting Momentum Setups & Simulating Multi-RR Outcomes (Point 3)...")
    print(f"   - Target 1: RR 1:1 (Take Profit = 1R)")
    print(f"   - Target 2: RR 1:2 (Take Profit = 2R)")
    print(f"   - Stop Loss: Local Swing Structure or 1.5x ATR")
    print(f"   - Max holding window: {max_holding_bars} bars")
    print(f"   - News filter: Total trade pause during is_news_blackout == True")
    
    setups = extract_momentum_setups(df_full, max_holding_bars=max_holding_bars)
    
    # Save dataset
    out_dir = "data/xau"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"momentum_setups_{timeframe}.parquet")
    setups.to_parquet(out_path)
    print(f"\nSaved labeled dataset to: {out_path} ({os.path.getsize(out_path):,} bytes)")

    # Print Detailed Quantitative Analysis
    print("\n" + "=" * 70)
    print(" QUANTITATIVE AUDIT & STATISTICAL METRICS")
    print("=" * 70)
    total_trades = len(setups)
    print(f"Total Transactions Generated: {total_trades:,}")
    
    # Direction breakdown
    buy_count = (setups["direction"] == "BUY").sum()
    sell_count = (setups["direction"] == "SELL").sum()
    print(f"\n[Trade Direction]")
    print(f"  BUY Setups:  {buy_count:,} ({buy_count / total_trades * 100:.1f}%)")
    print(f"  SELL Setups: {sell_count:,} ({sell_count / total_trades * 100:.1f}%)")
    
    # Trigger breakdown
    print(f"\n[Trigger Breakdown]")
    for trig, count in setups["primary_trigger"].value_counts().items():
        print(f"  - {trig:<20}: {count:,} ({count / total_trades * 100:.1f}%)")
        
    # Outcome 1R
    win_1r = (setups["result_1r"] == 1).sum()
    loss_1r = (setups["result_1r"] == -1).sum()
    timeout_1r = (setups["result_1r"] == 0).sum()
    wr_1r = win_1r / (win_1r + loss_1r) if (win_1r + loss_1r) > 0 else 0
    exp_1r = (wr_1r * 1.0) - ((1 - wr_1r) * 1.0)
    
    print(f"\n[Performance: Target RR 1:1]")
    print(f"  Wins:     {win_1r:,} ({win_1r / total_trades * 100:.1f}%)")
    print(f"  Losses:   {loss_1r:,} ({loss_1r / total_trades * 100:.1f}%)")
    print(f"  Timeouts: {timeout_1r:,} ({timeout_1r / total_trades * 100:.1f}%)")
    print(f"  Win Rate (resolved): {wr_1r * 100:.2f}%")
    print(f"  Expectancy (1R):     {exp_1r:+.3f} R / trade")

    # Outcome 2R
    win_2r = (setups["result_2r"] == 1).sum()
    loss_2r = (setups["result_2r"] == -1).sum()
    timeout_2r = (setups["result_2r"] == 0).sum()
    wr_2r = win_2r / (win_2r + loss_2r) if (win_2r + loss_2r) > 0 else 0
    exp_2r = (wr_2r * 2.0) - ((1 - wr_2r) * 1.0)

    print(f"\n[Performance: Target RR 1:2]")
    print(f"  Wins:     {win_2r:,} ({win_2r / total_trades * 100:.1f}%)")
    print(f"  Losses:   {loss_2r:,} ({loss_2r / total_trades * 100:.1f}%)")
    print(f"  Timeouts: {timeout_2r:,} ({timeout_2r / total_trades * 100:.1f}%)")
    print(f"  Win Rate (resolved): {wr_2r * 100:.2f}% (Break-even threshold = 33.33%)")
    print(f"  Expectancy (2R):     {exp_2r:+.3f} R / trade")

    # Market Structure Confluence Analysis
    print(f"\n[Market Structure Confluence Edges]")
    
    # 1. Demand Zone Buys
    dem_buys = setups[(setups["direction"] == "BUY") & (setups["inside_demand_zone"] == 1)]
    if len(dem_buys) > 0:
        w1 = (dem_buys["result_1r"] == 1).mean()
        w2 = (dem_buys["result_2r"] == 1).mean()
        e2 = (w2 * 2.0) - ((1 - w2) * 1.0)
        print(f"  - BUY in Demand Zone (N={len(dem_buys)}): WR 1R={w1*100:.1f}%, WR 2R={w2*100:.1f}%, Expectancy 2R={e2:+.3f}R")
        
    # 2. Supply Zone Sells
    sup_sells = setups[(setups["direction"] == "SELL") & (setups["inside_supply_zone"] == 1)]
    if len(sup_sells) > 0:
        w1 = (sup_sells["result_1r"] == 1).mean()
        w2 = (sup_sells["result_2r"] == 1).mean()
        e2 = (w2 * 2.0) - ((1 - w2) * 1.0)
        print(f"  - SELL in Supply Zone (N={len(sup_sells)}): WR 1R={w1*100:.1f}%, WR 2R={w2*100:.1f}%, Expectancy 2R={e2:+.3f}R")

    # 3. Discount Zone Buys (< 0.3)
    disc_buys = setups[(setups["direction"] == "BUY") & (setups["premium_discount_ratio"] < 0.35)]
    if len(disc_buys) > 0:
        w1 = (disc_buys["result_1r"] == 1).mean()
        w2 = (disc_buys["result_2r"] == 1).mean()
        e2 = (w2 * 2.0) - ((1 - w2) * 1.0)
        print(f"  - BUY in Deep Discount (N={len(disc_buys)}): WR 1R={w1*100:.1f}%, WR 2R={w2*100:.1f}%, Expectancy 2R={e2:+.3f}R")

    # 4. Premium Zone Sells (> 0.65)
    prem_sells = setups[(setups["direction"] == "SELL") & (setups["premium_discount_ratio"] > 0.65)]
    if len(prem_sells) > 0:
        w1 = (prem_sells["result_1r"] == 1).mean()
        w2 = (prem_sells["result_2r"] == 1).mean()
        e2 = (w2 * 2.0) - ((1 - w2) * 1.0)
        print(f"  - SELL in Deep Premium (N={len(prem_sells)}): WR 1R={w1*100:.1f}%, WR 2R={w2*100:.1f}%, Expectancy 2R={e2:+.3f}R")

    print("=" * 70)
    return setups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Momentum Dataset")
    parser.add_argument("--timeframe", choices=["1h", "m30"], default="1h", help="Timeframe (1h or m30)")
    parser.add_argument("--max-bars", type=int, default=48, help="Max holding bars for trade resolution")
    args = parser.parse_args()

    generate_dataset(timeframe=args.timeframe, max_holding_bars=args.max_bars)
