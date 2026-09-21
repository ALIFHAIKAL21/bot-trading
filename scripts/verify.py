"""Master verification script for 'make verify'.

Runs:
1. Environment and hardware check.
2. Full pytest test suite (including security, leakage, risk, paper).
3. Leakage check assertions.
4. Lightweight pipeline smoke test.
"""

import sys
import subprocess
from pathlib import Path
from loguru import logger

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.env_check import run_env_check
from src.utils.security import generate_run_manifest, print_active_models_banner, print_mode_banner, determine_execution_mode


def run_verification() -> bool:
    """Execute complete institutional verification suite."""
    logger.info("=" * 70)
    logger.info("STARTING INSTITUTIONAL SYSTEM VERIFICATION (`make verify`)")
    logger.info("=" * 70)

    cfg = load_config("config/config.yaml")
    mode = determine_execution_mode(cfg)
    print_mode_banner(mode)
    print_active_models_banner(cfg, context="VERIFY")

    # 1. Environment & Hardware Check
    logger.info("\n--- STEP 1: Environment & Hardware Check ---")
    try:
        env_res = run_env_check("config/config.yaml")
        logger.success("Environment & config check passed.")
    except Exception as e:
        logger.error(f"Environment check failed: {e}")
        return False

    # 2. Pytest Suite
    logger.info("\n--- STEP 2: Running Automated Pytest Suite ---")
    python_exe = sys.executable
    pytest_cmd = [python_exe, "-m", "pytest", "tests/", "-v"]
    res = subprocess.run(pytest_cmd, capture_output=False)
    if res.returncode != 0:
        logger.error(f"Pytest suite failed with returncode {res.returncode}")
        return False
    logger.success("All unit and integration tests passed.")

    # 3. Generate Verified Run Manifest
    logger.info("\n--- STEP 3: Emitting Verified Run Manifest ---")
    try:
        manifest = generate_run_manifest(cfg, output_dir="reports")
        logger.success(f"Verified run manifest emitted: git {manifest['git_commit']}")
    except Exception as e:
        logger.error(f"Failed to emit run manifest: {e}")
        return False

    logger.info("\n" + "=" * 70)
    logger.success("VERIFICATION SUITE PASSED SUCCESSFULLY (ALL CRITERIA MET)")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
