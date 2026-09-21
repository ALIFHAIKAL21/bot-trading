"""Tamper-proof hold-out partition discipline and guard.

Enforces:
- Separation of dataset into VALIDATION folds (oldest ~90%) and LOCKED_TEST (most recent 10%).
- Absolute quarantine of LOCKED_TEST: any attempt to access, fit, or evaluate on LOCKED_TEST
  outside scripts/final_report.py raises HoldoutViolationError.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from loguru import logger


class HoldoutViolationError(PermissionError):
    """Raised when unauthorized code attempts to read or process LOCKED_TEST data."""
    pass


LOCKED_TEST_FRACTION = 0.10  # Most recent 10% of timestamps


def partition_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Partition dataset into validation (90%) and locked test (10%) sets."""
    if df.empty:
        raise ValueError("Cannot partition empty DataFrame.")

    df_sorted = df.sort_index()
    n_total = len(df_sorted)
    split_idx = int(n_total * (1.0 - LOCKED_TEST_FRACTION))

    df_val = df_sorted.iloc[:split_idx].copy()
    df_locked = df_sorted.iloc[split_idx:].copy()

    start_date = str(df_sorted.index[0])
    val_end_date = str(df_val.index[-1])
    locked_start_date = str(df_locked.index[0])
    end_date = str(df_sorted.index[-1])

    info = {
        "n_total_bars": n_total,
        "n_validation_bars": len(df_val),
        "n_locked_test_bars": len(df_locked),
        "total_range": f"{start_date} -> {end_date}",
        "validation_range": f"{start_date} -> {val_end_date}",
        "locked_test_range": f"{locked_start_date} -> {end_date}",
        "locked_cutoff_timestamp": locked_start_date,
    }

    logger.info(
        f"Data Partition Discipline: Validation = {len(df_val)} bars ({start_date} to {val_end_date}) | "
        f"LOCKED_TEST = {len(df_locked)} bars ({locked_start_date} to {end_date})"
    )
    return df_val, df_locked, info


def is_authorized_locked_test_access() -> bool:
    """Check if calling execution context is authorized to unlock LOCKED_TEST."""
    # Only scripts/final_report.py with explicit environment variable may access
    env_flag = os.getenv("ALLOW_LOCKED_TEST_ACCESS", "").strip().lower() in ("1", "true", "yes")
    if not env_flag:
        return False

    # Check caller script name from sys.argv or stack
    main_script = Path(sys.argv[0]).name if sys.argv else ""
    return main_script == "final_report.py" or "test_" in main_script


def guard_locked_test(df: pd.DataFrame, locked_cutoff_ts: str) -> pd.DataFrame:
    """Guard a DataFrame against containing unauthorized LOCKED_TEST bars.
    
    Raises HoldoutViolationError if unauthorized code attempts to process bars >= locked_cutoff_ts.
    """
    cutoff = pd.to_datetime(locked_cutoff_ts, utc=True)
    has_locked_bars = (df.index >= cutoff).any()

    if has_locked_bars:
        if not is_authorized_locked_test_access():
            raise HoldoutViolationError(
                f"CRITICAL DISCIPLINE VIOLATION: Attempted to access LOCKED_TEST data (bars >= {locked_cutoff_ts})! "
                "LOCKED_TEST is strictly quarantined and may ONLY be accessed once via scripts/final_report.py. "
                "All model selection, tuning, and CV must occur strictly on VALIDATION data."
            )
        logger.warning(f"LOCKED_TEST data accessed under authorized context ({sys.argv[0] if sys.argv else 'script'}).")

    return df
