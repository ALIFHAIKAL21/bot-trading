"""Unit tests for Model C (Deep Sequence Multi-Task Network)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models.deep_sequence_model import DeepSequenceModel, MultiTaskSequenceNetwork, RevIN


def test_revin_normalization():
    import torch

    revin = RevIN(num_features=4)
    x = torch.randn(8, 30, 4)
    x_norm = revin(x)
    assert x_norm.shape == x.shape
    # Mean across sequence dimension should be close to zero
    assert torch.allclose(x_norm.mean(dim=1), torch.zeros_like(x_norm.mean(dim=1)), atol=1e-4)


def test_deep_network_multi_task_forward():
    import torch

    net = MultiTaskSequenceNetwork(in_features=6, d_model=32, n_layers=1)
    x = torch.randn(4, 20, 6)
    p_dir, p_ret, p_vol = net(x)

    assert p_dir.shape == (4,)
    assert p_ret.shape == (4,)
    assert p_vol.shape == (4,)

    # Direction probability in [0, 1]
    assert (p_dir >= 0.0).all() and (p_dir <= 1.0).all()
    # Volatility strictly positive
    assert (p_vol > 0.0).all()


def test_deep_sequence_model_predict_proba():
    # Construct synthetic DataFrame
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "ret_1": np.random.randn(100) * 0.01,
            "ret_5": np.random.randn(100) * 0.02,
            "rsi_14": np.random.uniform(0.2, 0.8, 100),
            "taker_buy_ratio": np.random.uniform(0.3, 0.7, 100),
        },
        index=dates,
    )

    model = DeepSequenceModel()
    preds = model.predict_proba(df)

    assert len(preds) == len(df)
    assert "model_c_dir_prob" in preds.columns
    assert "model_c_fwd_ret" in preds.columns
    assert "model_c_fwd_vol" in preds.columns
    assert not preds.isnull().any().any()
