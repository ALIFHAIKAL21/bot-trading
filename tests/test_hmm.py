"""Unit tests for Causal Gaussian HMM regime detector."""

import numpy as np
import pandas as pd
import pytest

from src.models.regime_model import CausalRegimeHMM
from src.utils.config import ModelEConfig


def test_causal_regime_hmm():
    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    df = pd.DataFrame(
        {
            "log_ret_1": np.random.normal(0, 0.01, size=n),
            "realized_vol_12": np.abs(np.random.normal(0.01, 0.005, size=n)),
            "vol_zscore_24": np.random.normal(0, 1.0, size=n),
        },
        index=timestamps,
    )

    hmm = CausalRegimeHMM(ModelEConfig(n_components=3, n_iter=20))
    hmm.fit(df)

    probs = hmm.predict_filtered_proba(df)
    assert len(probs) == n
    assert "regime_p0" in probs.columns
    assert "regime_p1" in probs.columns
    assert "regime_p2" in probs.columns

    # Check probabilities sum to 1.0
    row_sums = probs.sum(axis=1).values
    assert np.allclose(row_sums, 1.0, atol=1e-4)
