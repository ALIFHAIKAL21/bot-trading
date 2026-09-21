"""Unit test suite for Phase 0 security, authentication, and execution modes."""

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.service.app import app
from src.utils.config import AppConfig
from src.utils.security import (
    determine_execution_mode,
    generate_run_manifest,
    get_active_models,
    validate_api_key_safety,
    verify_api_token,
)


@pytest.fixture
def mock_cfg():
    cfg = AppConfig()
    cfg.service.live_execution_enabled = False
    return cfg


def test_determine_execution_mode_paper(mock_cfg):
    """Verify default config always produces PAPER mode."""
    mock_cfg.service.live_execution_enabled = False
    with patch.dict(os.environ, {"CONFIRM_LIVE": "NO"}):
        mode = determine_execution_mode(mock_cfg)
        assert mode == "PAPER"


def test_determine_execution_mode_interlock_blocked(mock_cfg):
    """Verify live config without CONFIRM_LIVE=YES is blocked and falls back to PAPER."""
    mock_cfg.service.live_execution_enabled = True
    with patch.dict(os.environ, {"CONFIRM_LIVE": "NO"}):
        mode = determine_execution_mode(mock_cfg)
        assert mode == "PAPER"

    with patch.dict(os.environ, {}, clear=True):
        mode = determine_execution_mode(mock_cfg)
        assert mode == "PAPER"


def test_determine_execution_mode_testnet(mock_cfg):
    """Verify live config with CONFIRM_LIVE=YES and TESTNET defaults to TESTNET."""
    mock_cfg.service.live_execution_enabled = True
    with patch.dict(os.environ, {"CONFIRM_LIVE": "YES", "BINANCE_USE_TESTNET": "true"}):
        mode = determine_execution_mode(mock_cfg)
        assert mode == "TESTNET"


def test_determine_execution_mode_live(mock_cfg):
    """Verify live mode requires explicit opt-out of testnet and CONFIRM_LIVE=YES."""
    mock_cfg.service.live_execution_enabled = True
    with patch.dict(os.environ, {"CONFIRM_LIVE": "YES", "BINANCE_USE_TESTNET": "false"}):
        mode = determine_execution_mode(mock_cfg)
        assert mode == "LIVE"


def test_api_token_verification():
    """Verify API token verification with valid and invalid tokens."""
    with patch.dict(os.environ, {"API_TOKEN": "my_secret_token_123"}):
        # Direct function check
        token = verify_api_token(x_api_token="my_secret_token_123")
        assert token == "my_secret_token_123"

        token_bearer = verify_api_token(authorization="Bearer my_secret_token_123")
        assert token_bearer == "my_secret_token_123"

        with pytest.raises(HTTPException) as exc_info:
            verify_api_token(x_api_token="wrong_token")
        assert exc_info.value.status_code == 401

        with pytest.raises(HTTPException) as exc_info:
            verify_api_token(x_api_token=None, authorization=None)
        assert exc_info.value.status_code == 401


def test_fastapi_endpoints_auth():
    """Verify FastAPI client rejects unauthorized POST requests to /trade/step."""
    client = TestClient(app)

    # GET /health should be public
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "execution_mode" in data
    assert "active_models" in data
    assert data["auth_required_for_post"] is True

    # POST /trade/step without auth must fail with 401
    payload = {
        "symbol": "BTC/USDT",
        "close_price": 50000.0,
        "target_weight": 0.2,
        "prob_long": 0.55,
        "confidence": 0.1,
        "regime_id": 0,
    }
    res_unauth = client.post("/trade/step", json=payload)
    assert res_unauth.status_code == 401

    # POST /trade/step with invalid auth must fail with 401
    res_bad = client.post("/trade/step", json=payload, headers={"X-API-Token": "invalid_token"})
    assert res_bad.status_code == 401


def test_validate_api_key_safety_rejection():
    """Verify system raises PermissionError if exchange key has withdrawal permissions."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_permissions.return_value = {"trade": True, "withdraw": True}

    with pytest.raises(PermissionError) as exc_info:
        validate_api_key_safety(mock_exchange)
    assert "WITHDRAWAL permissions" in str(exc_info.value)

    # Safe permissions
    mock_exchange_safe = MagicMock()
    mock_exchange_safe.fetch_permissions.return_value = {"trade": True, "withdraw": False}
    # Should not raise
    validate_api_key_safety(mock_exchange_safe)


def test_active_models_audit(mock_cfg):
    """Verify active models audit includes all 5 models and meta-learner."""
    models = get_active_models(mock_cfg)
    assert "model_a_chronos" in models
    assert "model_b_lightgbm" in models
    assert "model_c_deep" in models
    assert "model_d_sentiment" in models
    assert "model_e_hmm" in models
    assert "meta_learner" in models
    assert models["model_b_lightgbm"] == "ENABLED"


def test_generate_run_manifest(mock_cfg, tmp_path):
    """Verify run manifest creates verifiable JSON artifact."""
    manifest = generate_run_manifest(mock_cfg, output_dir=str(tmp_path))
    assert "timestamp_utc" in manifest
    assert "git_commit" in manifest
    assert "config_sha256" in manifest
    assert "active_models" in manifest
    assert manifest["execution_mode"] == "PAPER"

    # Verify latest pointer was written
    latest_file = tmp_path / "latest_run_manifest.json"
    assert latest_file.exists()
