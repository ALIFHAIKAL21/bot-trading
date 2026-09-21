"""Unit tests for Institutional Adaptivity Layer (A1–A7)."""

import numpy as np
import pandas as pd
import pytest

from src.adaptive.drift_monitor import DriftMonitor, calculate_psi
from src.adaptive.model_registry import ModelRegistry, ModelStatus
from src.adaptive.online_weights import OnlineWeightManager
from src.adaptive.retrainer import RollingRetrainer


def test_online_weight_manager_hedge():
    """Verify Hedge/EWA online weight updates increase weight of superior model."""
    manager = OnlineWeightManager(models=["model_b", "model_c"], eta=2.0, min_weight=0.05)
    init_weights = manager.get_weights()
    assert init_weights["model_b"] == 0.50
    assert init_weights["model_c"] == 0.50

    # Model B consistently predicts accurate probabilities (0.8 vs 1.0)
    # Model C predicts inaccurate probabilities (0.2 vs 1.0)
    for _ in range(10):
        manager.update(y_true=1.0, predictions={"model_b": 0.85, "model_c": 0.20})

    updated = manager.get_weights()
    assert updated["model_b"] > updated["model_c"]
    assert np.isclose(sum(updated.values()), 1.0)
    assert updated["model_c"] >= 0.05  # Min weight floor respected


def test_drift_monitor_psi():
    """Verify PSI accurately detects distribution stability vs critical drift."""
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    identical = np.random.normal(0, 1, 1000)
    shifted = np.random.normal(2.5, 1, 1000)  # Strong mean shift

    # Identical should yield near-zero PSI
    psi_stable = calculate_psi(ref, identical)
    assert psi_stable < 0.10

    # Shifted should exceed critical threshold (0.25)
    psi_drift = calculate_psi(ref, shifted)
    assert psi_drift > 0.25

    # Test full DriftMonitor
    df_ref = pd.DataFrame({"feat1": ref, "feat2": ref})
    df_cur = pd.DataFrame({"feat1": shifted, "feat2": identical})

    monitor = DriftMonitor(df_ref, feature_cols=["feat1", "feat2"])
    report = monitor.evaluate_drift(df_cur)

    assert report.status == "CRITICAL"
    assert "feat1" in report.drifting_features
    assert report.retrain_recommended is True


def test_circuit_breaker_auto_degrade():
    """Verify A4: Circuit breaker trips and zeroes weight when model performance degrades."""
    registry = ModelRegistry(brier_circuit_breaker_threshold=0.28, shadow_evaluation_window=50)
    registry.register_model(name="model_b", version="v1.0", status=ModelStatus.CHAMPION, initial_weight=0.50)

    # Feed 30 terrible predictions (predicting 0.90 when target is 0.0) -> Brier = 0.81
    for _ in range(30):
        registry.record_step_outcome(model_name="model_b", pred_prob=0.90, y_true=0.0)

    record = registry.models["model_b"]
    assert record.circuit_breaker_tripped is True
    assert record.status == ModelStatus.DEGRADED
    assert record.active_weight == 0.0

    active_weights = registry.get_active_weights()
    assert active_weights["model_b"] == 0.0


def test_champion_challenger_promotion_and_rollback():
    """Verify A5: Challenger is promoted when outperforming champion, with rollback capability."""
    registry = ModelRegistry(shadow_evaluation_window=30)
    registry.register_model(name="model_b", version="v1.0", status=ModelStatus.CHAMPION, initial_weight=0.60)
    registry.register_model(name="model_b_challenger", version="v1.1", status=ModelStatus.CHALLENGER)

    # Feed 35 steps: Challenger is accurate (pred 0.8 vs y=1), Champion is inaccurate (pred 0.4 vs y=1)
    for _ in range(35):
        registry.record_step_outcome("model_b", pred_prob=0.40, y_true=1.0)
        registry.record_step_outcome("model_b_challenger", pred_prob=0.85, y_true=1.0)

    promoted = registry.evaluate_challenger("model_b_challenger", "model_b")
    assert promoted is True
    assert registry.models["model_b_challenger"].status == ModelStatus.CHAMPION
    assert registry.models["model_b"].status == ModelStatus.SHADOW

    # Verify rollback
    rolled_back = registry.rollback_model("model_b")
    assert rolled_back is True
    assert registry.models["model_b"].status == ModelStatus.CHAMPION


def test_rolling_retrainer_causal_slices():
    """Verify A1: Retrainer enforces causal embargo between training and validation slices."""
    dates = pd.date_range("2024-01-01", periods=2000, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": np.random.randn(2000)}, index=dates)

    retrainer = RollingRetrainer(
        retrain_frequency_bars=720,
        training_window_bars=1000,
        embargo_bars=12,
        validation_slice_bars=200,
    )

    train_slice, val_slice = retrainer.prepare_causal_slices(df)

    assert len(val_slice) == 200
    assert len(train_slice) == 1000

    # Ensure gap between train_slice end and val_slice start is EXACTLY embargo_bars
    train_end_idx = df.index.get_loc(train_slice.index[-1])
    val_start_idx = df.index.get_loc(val_slice.index[0])
    gap_bars = val_start_idx - train_end_idx - 1
    assert gap_bars == 12
