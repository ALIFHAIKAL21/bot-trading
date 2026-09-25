"""
Unit tests for Momentum Setup Extractor & Multi-RR Labeler (Point 3).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.indicator_signals import compute_all_indicators
from src.features.market_structure import compute_market_structure
from src.labels.momentum_labeler import extract_momentum_setups


def create_trend_scenario():
    """Creates a synthetic trend with a clear MA cross and outcome."""
    n = 100
    # Create an initial downtrend then strong uptrend
    closes = np.zeros(n)
    for i in range(40):
        closes[i] = 2000.0 - i * 2.0
    for i in range(40, n):
        closes[i] = closes[39] + (i - 39) * 5.0
        
    highs = closes + 2.0
    lows = closes - 2.0
    opens = closes - 1.0
    
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [500] * n,
        "is_news_blackout": [False] * n,
    })
    return df


def test_momentum_setup_extraction():
    df = create_trend_scenario()
    df = compute_all_indicators(df)
    ms = compute_market_structure(df)
    df_full = pd.concat([df, ms], axis=1)
    
    setups = extract_momentum_setups(df_full, max_holding_bars=20)
    assert len(setups) > 0, "Should extract at least one setup"
    
    # Check that each setup has valid RR labels
    assert "result_1r" in setups.columns
    assert "result_2r" in setups.columns
    assert "direction" in setups.columns
    assert "entry_price" in setups.columns
    assert "sl_price" in setups.columns
    assert "tp1_price" in setups.columns
    assert "tp2_price" in setups.columns
    
    # Check that 1R results are valid {-1, 0, 1}
    assert set(setups["result_1r"].unique()).issubset({-1, 0, 1})
    assert set(setups["result_2r"].unique()).issubset({-1, 0, 1})
    
    # In a strong uptrend, buy setups should hit 1R or 2R
    buy_setups = setups[setups["direction"] == "BUY"]
    if len(buy_setups) > 0:
        assert (buy_setups["result_1r"] == 1).any()


def test_news_blackout_filter():
    df = create_trend_scenario()
    # Set news blackout on all bars
    df["is_news_blackout"] = True
    df = compute_all_indicators(df)
    ms = compute_market_structure(df)
    df_full = pd.concat([df, ms], axis=1)
    
    setups = extract_momentum_setups(df_full, max_holding_bars=20)
    assert len(setups) == 0, "No setups should be extracted during total news blackout"
