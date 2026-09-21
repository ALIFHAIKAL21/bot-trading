"""Unit tests for configuration loading and validation."""

import pytest
from src.utils.config import load_config, AppConfig, resolve_device


def test_load_default_config():
    cfg = load_config("config/config.yaml")
    assert isinstance(cfg, AppConfig)
    assert "BTC/USDT" in cfg.market.symbols
    assert cfg.market.timeframe == "1h"
    assert cfg.market.horizon == 12
    assert cfg.risk.max_leverage <= 1.0
    assert cfg.backtest.taker_fee == 0.0010
    assert cfg.models.model_a.enabled is True
    assert cfg.models.model_b.enabled is True
    assert cfg.models.model_c.enabled is True


def test_resolve_device():
    dev = resolve_device("auto")
    assert dev in ["cuda", "cpu"]
    dev_cpu = resolve_device("cpu")
    assert dev_cpu == "cpu"
