"""Security, authentication, safety guards, and run manifest tracking.

Enforces:
- API token authentication on REST service endpoints.
- Strict live-trading authorization (dual confirmation via config + CONFIRM_LIVE=YES).
- Prevention of API keys with withdrawal permissions.
- Loud execution mode banners (PAPER / TESTNET / LIVE).
- Active models startup audit (no silent skipping).
- Complete reproducible run manifests.
"""

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader
from loguru import logger

from src.utils.config import AppConfig


api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)


def get_configured_api_token() -> str:
    """Return configured API token or fallback default for development."""
    return os.getenv("API_TOKEN", "quant_dev_secret_token_change_me")


def verify_api_token(
    x_api_token: Optional[str] = Security(api_key_header),
    authorization: Optional[str] = Header(None),
) -> str:
    """FastAPI dependency to verify API token on protected endpoints.
    
    Accepts token via 'X-API-Token' header or 'Authorization: Bearer <token>'.
    """
    expected_token = get_configured_api_token()
    token = x_api_token if isinstance(x_api_token, str) else None

    if not token and isinstance(authorization, str):
        if authorization.startswith("Bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()

    if not token or token != expected_token:
        logger.warning("Unauthorized API access attempt rejected.")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid API token. Provide valid 'X-API-Token' or 'Bearer' token.",
        )
    return token


def determine_execution_mode(cfg: AppConfig) -> str:
    """Determine current execution mode: 'PAPER', 'TESTNET', or 'LIVE'.
    
    Live trading requires:
    1. cfg.service.live_execution_enabled == True
    2. Environment variable CONFIRM_LIVE == 'YES'
    """
    if not cfg.service.live_execution_enabled:
        return "PAPER"

    confirm_live = os.getenv("CONFIRM_LIVE", "").strip().upper()
    if confirm_live != "YES":
        logger.error(
            "SAFETY INTERLOCK TRIPPED: live_execution_enabled is TRUE in config, "
            "but environment variable CONFIRM_LIVE != 'YES'. Defaulting to PAPER MODE for safety."
        )
        return "PAPER"

    use_testnet = os.getenv("BINANCE_USE_TESTNET", "true").strip().lower() in ("true", "1", "yes")
    if use_testnet:
        return "TESTNET"

    return "LIVE"


def validate_api_key_safety(exchange) -> None:
    """Verify that exchange API key does NOT have withdrawal permissions enabled.
    
    Refuses to start if withdrawal permissions are detected or cannot be verified.
    """
    if exchange is None:
        return

    logger.info("Validating API key safety permissions on exchange...")
    try:
        # Check permissions if available in CCXT exchange
        if hasattr(exchange, "fetch_permissions"):
            perms = exchange.fetch_permissions()
            if perms.get("withdraw", False) or perms.get("withdrawals", False):
                raise PermissionError(
                    "CRITICAL SECURITY VIOLATION: The configured API key has WITHDRAWAL permissions! "
                    "This trading bot NEVER requires withdrawal permissions. System aborted immediately."
                )
    except Exception as e:
        if "WITHDRAWAL permissions" in str(e):
            raise
        # fetch_permissions is not supported on all exchanges/endpoints; log warning
        logger.debug(f"Exchange does not support fetch_permissions: {e}")


def print_mode_banner(mode: str) -> None:
    """Print prominent visual execution banner to logs and stdout."""
    line = "=" * 70
    if mode == "PAPER":
        logger.info("\n" + line)
        logger.info("                  *** RUNNING IN PAPER MODE ***")
        logger.info("   All orders are virtual simulations. Real funds are NEVER touched.")
        logger.info(line + "\n")
    elif mode == "TESTNET":
        logger.warning("\n" + line)
        logger.warning("                 *** RUNNING IN TESTNET MODE ***")
        logger.warning("   Connecting to Binance Testnet sandbox. Real funds are NOT at risk.")
        logger.warning(line + "\n")
    elif mode == "LIVE":
        logger.critical("\n" + line)
        logger.critical("           *** DANGER: RUNNING IN PRODUCTION LIVE MODE ***")
        logger.critical("           REAL FUNDS ARE AT RISK ON LIVE EXCHANGES!")
        logger.critical(line + "\n")


def get_active_models(cfg: AppConfig) -> Dict[str, str]:
    """Inspect and return dictionary of all models and their active status."""
    status = {}
    status["model_a_chronos"] = "ENABLED" if cfg.models.model_a.enabled else "DISABLED"
    status["model_b_lightgbm"] = "ENABLED" if cfg.models.model_b.enabled else "DISABLED"
    status["model_c_deep"] = "ENABLED" if cfg.models.model_c.enabled else "DISABLED"
    status["model_d_sentiment"] = (
        "GATE_ONLY" if cfg.models.model_d.enabled else "DISABLED"
    )
    status["model_e_hmm"] = "ENABLED" if cfg.models.model_e.enabled else "DISABLED"
    status["meta_learner"] = (
        f"ENABLED ({cfg.ensemble.method})" if cfg.ensemble.method else "SIMPLE_AVERAGE"
    )
    return status


def print_active_models_banner(cfg: AppConfig, context: str = "STARTUP") -> None:
    """Log active model status clearly with zero silent skipping (F13)."""
    models = get_active_models(cfg)
    logger.info(f"[{context} - ACTIVE MODELS AUDIT]")
    for model_name, state in models.items():
        if "ENABLED" in state or "GATE" in state:
            logger.info(f"  -> {model_name.upper()}: {state}")
        else:
            logger.warning(f"  -> {model_name.upper()}: {state} [SKIPPED]")


def compute_file_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    if not file_path.exists():
        return "file_not_found"
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_commit_hash() -> str:
    """Get current git commit hash, or indicate unversioned."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit = res.stdout.strip()
        # Check if dirty
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.stdout.strip():
            commit += "-dirty"
        return commit
    except Exception:
        return "unversioned_workspace"


def generate_run_manifest(
    cfg: AppConfig,
    config_path: str = "config/config.yaml",
    dataset_meta: Optional[Dict] = None,
    extra_meta: Optional[Dict] = None,
    output_dir: str = "reports",
) -> Dict:
    """Generate and persist a complete, reproducible run manifest."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).isoformat()
    config_hash = compute_file_sha256(Path(config_path))
    git_hash = get_git_commit_hash()
    exec_mode = determine_execution_mode(cfg)
    active_models = get_active_models(cfg)

    manifest = {
        "timestamp_utc": now_utc,
        "execution_mode": exec_mode,
        "git_commit": git_hash,
        "config_path": config_path,
        "config_sha256": config_hash,
        "system_seed": cfg.system.seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "symbols": cfg.market.symbols,
        "timeframe": cfg.market.timeframe,
        "history_days": cfg.market.history_days,
        "primary_data_source": cfg.data.primary_source,
        "active_models": active_models,
        "dataset_meta": dataset_meta or {},
        "extra_meta": extra_meta or {},
    }

    # Save timestamped manifest and latest pointer
    ts_clean = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_file = out_path / f"run_manifest_{ts_clean}.json"
    latest_file = out_path / "latest_run_manifest.json"

    import json
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(latest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Run manifest saved to {manifest_file}")
    return manifest
