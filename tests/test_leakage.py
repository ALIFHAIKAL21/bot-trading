"""Rigorous causality, leakage prevention, and hold-out discipline test suite.

Includes:
1. Truncate-future test: features computed on data[:t] must equal features computed on the full data at t.
2. Label-shuffle test: models trained on randomly permuted labels must collapse to AUC ~0.50.
3. Higher-timeframe causality test: 4h/1d features strictly reflect completed HTF bars.
4. Whitelist assertion test: passing any label/weight/forward column raises LeakageViolationError.
5. Hold-out guard test: unauthorized access to LOCKED_TEST raises HoldoutViolationError.
"""

import os
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.data.holdout_guard import (
    HoldoutViolationError,
    guard_locked_test,
    partition_dataset,
)
from src.features.feature_pipeline import (
    FeaturePipeline,
    LeakageViolationError,
    validate_feature_columns,
)
from src.labels.triple_barrier import TripleBarrierLabeler
from src.models.tabular_model import TabularDirectionModel


def generate_synthetic_ohlcv(n_bars: int = 500, start_price: float = 50000.0) -> pd.DataFrame:
    """Generate realistic synthetic 1h OHLCV data with random walk."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    returns = np.random.normal(0.0002, 0.01, size=n_bars)
    prices = start_price * np.exp(np.cumsum(returns))

    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.005, size=n_bars)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.005, size=n_bars)))
    opens = (highs + lows) / 2.0 + np.random.normal(0, 0.001, size=n_bars) * prices
    closes = prices
    volumes = np.random.lognormal(mean=5.0, sigma=1.0, size=n_bars)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=timestamps,
    )


def test_truncate_future_exact_match():
    """Truncate-future test: features computed on data[:t] equal features computed on full data at t."""
    df_full = generate_synthetic_ohlcv(n_bars=300)
    pipeline = FeaturePipeline()
    df_feat_full = pipeline.build_features(df_full)
    feature_cols = pipeline.get_feature_columns(df_feat_full)

    # Test at multiple cutoff points past rolling warmup (e.g. t=150, t=200, t=250)
    for t_cutoff in [150, 200, 250]:
        df_trunc = df_full.iloc[: t_cutoff + 1].copy()
        df_feat_trunc = pipeline.build_features(df_trunc)

        for col in feature_cols:
            full_val = df_feat_full.loc[df_full.index[t_cutoff], col]
            trunc_val = df_feat_trunc.loc[df_trunc.index[t_cutoff], col]

            if np.isnan(full_val):
                assert np.isnan(trunc_val), f"NaN mismatch for {col} at t={t_cutoff}"
            else:
                assert np.isclose(
                    full_val, trunc_val, rtol=1e-6, atol=1e-6
                ), f"Leakage detected in feature '{col}' at t={t_cutoff}: full={full_val}, trunc={trunc_val}"


def test_feature_whitelist_and_leakage_assertion():
    """Strict whitelist audit: passing label/weight/forward column raises LeakageViolationError."""
    # 1. Reject label / weight columns
    forbidden_cols = ["sample_weight", "target_label", "target_binary_long", "fwd_ret_12", "forward_vol"]
    for col in forbidden_cols:
        with pytest.raises(LeakageViolationError) as exc_info:
            validate_feature_columns(["log_ret_1", col])
        assert "CRITICAL LEAKAGE DETECTED" in str(exc_info.value)

    # 2. Reject non-whitelisted columns
    whitelist = ["log_ret_1", "rsi_14"]
    with pytest.raises(LeakageViolationError) as exc_info:
        validate_feature_columns(["log_ret_1", "random_unapproved_column"], whitelist=whitelist)
    assert "FEATURE WHITELIST VIOLATION" in str(exc_info.value)

    # 3. Model B must reject sample_weight in X_train
    lgbm = TabularDirectionModel()
    X_bad = pd.DataFrame({"log_ret_1": [0.01, -0.02], "sample_weight": [1.0, 1.0]})
    y = np.array([1, 0])
    with pytest.raises(LeakageViolationError):
        lgbm.fit(X_bad, y)


def test_label_shuffle_auc_collapse():
    """Label-shuffle test: Model B trained on shuffled labels must collapse to AUC ~0.50."""
    np.random.seed(42)
    df = generate_synthetic_ohlcv(n_bars=600)
    pipeline = FeaturePipeline()
    df_feat = pipeline.build_features(df).dropna()
    feature_cols = pipeline.get_feature_columns(df_feat)

    # Create synthetic binary target and randomly shuffle it
    n = len(df_feat)
    y_shuffled = np.random.binomial(1, 0.5, size=n)

    split = int(n * 0.7)
    X_train, X_val = df_feat[feature_cols].iloc[:split], df_feat[feature_cols].iloc[split:]
    y_train, y_val = y_shuffled[:split], y_shuffled[split:]

    lgbm = TabularDirectionModel()
    lgbm.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    metrics = lgbm.evaluate(X_val, y_val)

    # Shuffled label AUC must collapse to near 0.50 (well within [0.42, 0.58])
    assert 0.42 <= metrics["auc"] <= 0.58, (
        f"Label-shuffled model failed to collapse to ~0.50! AUC was {metrics['auc']:.4f}"
    )


def test_higher_timeframe_causality_shift():
    """Verify HTF 4h and 1d trends strictly reflect completed HTF bars and never partial candles."""
    df = generate_synthetic_ohlcv(n_bars=200)
    pipeline = FeaturePipeline()
    df_feat = pipeline.build_features(df)

    assert "htf_4h_trend" in df_feat.columns
    assert "htf_1d_trend" in df_feat.columns

    # Check 4h trend: within each 4-hour bin, the feature value must remain constant
    # and only change at the start of the next 4-hour cycle
    h4 = df_feat["htf_4h_trend"].dropna()
    # At least 100 valid bars
    assert len(h4) >= 100

    # Ensure 4 consecutive bars within a 4h candle have the exact same value
    for i in range(20, len(h4) - 4, 4):
        vals = h4.iloc[i : i + 4].values
        assert np.allclose(vals, vals[0]), f"HTF 4h trend changed mid-candle at index {i}!"


def test_holdout_guard_unauthorized_access():
    """Verify hold-out guard blocks unauthorized access to LOCKED_TEST slice."""
    df = generate_synthetic_ohlcv(n_bars=500)
    df_val, df_locked, partition_info = partition_dataset(df)

    assert len(df_val) == 450
    assert len(df_locked) == 50
    cutoff_ts = partition_info["locked_cutoff_timestamp"]

    # Unauthorized access to data with timestamps >= cutoff_ts must raise HoldoutViolationError
    with pytest.raises(HoldoutViolationError) as exc_info:
        guard_locked_test(df_locked, cutoff_ts)
    assert "CRITICAL DISCIPLINE VIOLATION" in str(exc_info.value)

    # VALIDATION data passes freely
    df_cleared = guard_locked_test(df_val, cutoff_ts)
    assert len(df_cleared) == len(df_val)

    # Authorized access (ALLOW_LOCKED_TEST_ACCESS=1 with final_report script name)
    with patch.dict(os.environ, {"ALLOW_LOCKED_TEST_ACCESS": "1"}):
        with patch("sys.argv", ["scripts/final_report.py"]):
            df_auth = guard_locked_test(df_locked, cutoff_ts)
            assert len(df_auth) == 50


def test_triple_barrier_labeler():
    """Verify triple-barrier labels adhere to constraints and positive uniqueness weights."""
    df = generate_synthetic_ohlcv(n_bars=200)
    pipeline = FeaturePipeline()
    df_feat = pipeline.build_features(df)

    labeler = TripleBarrierLabeler()
    df_labeled = labeler.label_barriers(df_feat)

    assert "target_label" in df_labeled.columns
    assert "target_binary_long" in df_labeled.columns
    assert "sample_weight" in df_labeled.columns

    unique_labels = set(df_labeled["target_label"].unique())
    assert unique_labels.issubset({-1, 0, 1})

    weights = df_labeled["sample_weight"].values
    assert (weights > 0).all()
    assert np.isclose(np.mean(weights), 1.0, atol=0.05)
