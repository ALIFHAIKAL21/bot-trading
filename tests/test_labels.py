"""Unit tests for triple-barrier labeling and sample uniqueness."""

import numpy as np
import pandas as pd
import pytest

from src.labels.triple_barrier import TripleBarrierLabeler
from src.utils.config import LabelConfig


def test_triple_barrier_take_profit():
    """Verify that an immediate upward surge triggers TP label (+1)."""
    n = 30
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Base price 100, then immediate surge to 150
    closes = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    # Spike at bar 2
    highs[2] = 150.0
    closes[2] = 145.0

    df = pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1000.0,
            "atr_14": 2.0,  # TP = 100 + 2*2 = 104, so 150 hits TP easily
        },
        index=timestamps,
    )

    labeler = TripleBarrierLabeler(LabelConfig(tp_multiplier=2.0, sl_multiplier=1.5, vertical_barrier=10))
    res = labeler.label_barriers(df)

    assert res.loc[timestamps[0], "target_label"] == 1
    assert res.loc[timestamps[0], "target_binary_long"] == 1
    assert res.loc[timestamps[0], "target_touch_idx"] == 2


def test_triple_barrier_stop_loss():
    """Verify that an immediate downward drop triggers SL label (-1)."""
    n = 30
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    # Drop at bar 3
    lows[3] = 70.0
    closes[3] = 75.0

    df = pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1000.0,
            "atr_14": 2.0,  # SL = 100 - 1.5*2 = 97, so 70 hits SL easily
        },
        index=timestamps,
    )

    labeler = TripleBarrierLabeler(LabelConfig(tp_multiplier=2.0, sl_multiplier=1.5, vertical_barrier=10))
    res = labeler.label_barriers(df)

    assert res.loc[timestamps[0], "target_label"] == -1
    assert res.loc[timestamps[0], "target_binary_long"] == 0
    assert res.loc[timestamps[0], "target_touch_idx"] == 3


def test_triple_barrier_vertical_timeout():
    """Verify that a flat series expires at vertical barrier with label 0."""
    n = 30
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)

    df = pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1000.0,
            "atr_14": 2.0,  # TP=104, SL=97 -> neither is touched
        },
        index=timestamps,
    )

    labeler = TripleBarrierLabeler(LabelConfig(tp_multiplier=2.0, sl_multiplier=1.5, vertical_barrier=10))
    res = labeler.label_barriers(df)

    assert res.loc[timestamps[0], "target_label"] == 0
    assert res.loc[timestamps[0], "target_touch_idx"] == 10
