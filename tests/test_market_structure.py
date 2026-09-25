"""
Unit tests for Market Structure module (Point 4: SMC, Support/Resistance, Liquidity Sweeps).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.market_structure import compute_atr, compute_market_structure


def create_synthetic_ohlcv(n: int = 100) -> pd.DataFrame:
    """Creates synthetic OHLCV data for testing."""
    np.random.seed(42)
    closes = 2000.0 + np.cumsum(np.random.randn(n) * 5.0)
    highs = closes + np.random.uniform(1.0, 5.0, size=n)
    lows = closes - np.random.uniform(1.0, 5.0, size=n)
    opens = (highs + lows) / 2.0
    
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(100, 1000, size=n),
    })


def test_compute_atr():
    df = create_synthetic_ohlcv(50)
    atr = compute_atr(df, period=14)
    assert len(atr) == 50
    assert not atr.iloc[-1] <= 0
    assert not np.isnan(atr.iloc[-1])


def test_market_structure_causality():
    df = create_synthetic_ohlcv(100)
    ms = compute_market_structure(df, swing_lookback=3)
    
    # Check output columns
    expected_cols = [
        "atr_14", "dist_to_support_atr", "dist_to_resistance_atr",
        "recent_swing_high", "recent_swing_low", "liq_sweep_bull",
        "liq_sweep_bear", "dist_to_demand_atr", "dist_to_supply_atr",
        "inside_demand_zone", "inside_supply_zone", "premium_discount_ratio"
    ]
    for col in expected_cols:
        assert col in ms.columns, f"Missing column {col}"
        
    # Check causality: First few bars before swing lookback should not have swings
    assert np.isnan(ms["recent_swing_high"].iloc[0])
    assert np.isnan(ms["recent_swing_low"].iloc[0])
    
    # Premium discount ratio is strictly in [0, 1]
    valid_pdr = ms["premium_discount_ratio"].dropna()
    assert (valid_pdr >= 0.0).all()
    assert (valid_pdr <= 1.0).all()


def test_liquidity_sweep_logic():
    # Construct a specific sweep scenario:
    # Bar 0-6: form a clear swing high at bar 3 (lookback=3)
    # Then later bar spikes above it but closes below
    df = pd.DataFrame({
        "open":  [100, 105, 110, 120, 110, 105, 100, 100, 115],
        "high":  [102, 107, 112, 125, 112, 107, 102, 102, 128],  # Bar 8 spikes to 128 > 125
        "low":   [ 98, 103, 108, 115, 108, 103,  98,  98, 110],
        "close": [101, 106, 111, 120, 110, 105, 101, 101, 122],  # Bar 8 closes 122 < 125 (Sweep!)
        "volume": [100] * 9
    })
    ms = compute_market_structure(df, swing_lookback=3)
    # Bar 8 should register bear sweep (BSL)
    assert ms["liq_sweep_bear"].iloc[8] == 1
