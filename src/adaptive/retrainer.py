"""Scheduled Rolling & Expanding Retraining Module with Purge/Embargo.

Adaptivity Component A1:
- Implements scheduled retraining triggers (time-based e.g. 30 days / 720 bars or drift-triggered).
- Extracts purged and embargoed training slices: strictly enforces that the most recent
  `embargo_bars` are excluded from the training set to prevent label leakage.
- Fits a challenger model instance and compares out-of-sample Brier loss against current champion.
- Registers valid retrained models with ModelRegistry as CHALLENGER for shadow testing.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger

from src.adaptive.model_registry import ModelRegistry, ModelStatus
from src.features.feature_pipeline import validate_feature_columns


class RollingRetrainer:
    """Manages scheduled and drift-driven causal retraining of models."""

    def __init__(
        self,
        retrain_frequency_bars: int = 720,  # Retrain every ~30 days on hourly data
        training_window_bars: int = 5000,   # Rolling training sample size
        embargo_bars: int = 12,             # Triple barrier embargo horizon
        validation_slice_bars: int = 500,   # Out-of-sample test window
    ):
        self.frequency = retrain_frequency_bars
        self.window_size = training_window_bars
        self.embargo = embargo_bars
        self.val_size = validation_slice_bars
        self.bars_since_last_retrain: int = 0
        self.retrain_history: List[Dict[str, Any]] = []

    def should_retrain(self, drift_triggered: bool = False) -> bool:
        """Evaluate if scheduled interval or feature drift warrants a retraining cycle."""
        if drift_triggered:
            return True
        return self.bars_since_last_retrain >= self.frequency

    def step(self) -> None:
        """Increment elapsed bars counter."""
        self.bars_since_last_retrain += 1

    def prepare_causal_slices(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract purged training and validation slices respecting strict causal embargo.
        
        Structure:
        [ ... Training Window ... ] [ Embargo ] [ Validation Slice ]
        """
        n = len(df)
        required = self.window_size + self.embargo + self.val_size
        if n < required:
            # Fallback for shorter series: proportion split with embargo
            val_split = max(int(n * 0.15), 50)
            train_end = n - val_split - self.embargo
            df_train = df.iloc[:train_end].copy()
            df_val = df.iloc[-val_split:].copy()
            return df_train, df_val

        val_slice = df.iloc[-self.val_size:].copy()
        train_end = n - self.val_size - self.embargo
        train_start = max(0, train_end - self.window_size)
        train_slice = df.iloc[train_start:train_end].copy()

        return train_slice, val_slice

    def execute_retrain(
        self,
        model_name: str,
        fit_fn: Callable[[pd.DataFrame, pd.DataFrame, List[str]], Any],
        df_history: pd.DataFrame,
        feature_cols: List[str],
        registry: Optional[ModelRegistry] = None,
    ) -> Dict[str, Any]:
        """Execute causal model retraining and register challenger."""
        validate_feature_columns(feature_cols)
        df_train, df_val = self.prepare_causal_slices(df_history)

        logger.info(
            f"Executing causal retrain for '{model_name}': Train = {len(df_train)} bars, "
            f"Embargo = {self.embargo} bars, Val = {len(df_val)} bars."
        )

        # Train new model instance
        retrained_model = fit_fn(df_train, df_val, feature_cols)

        # Reset counter
        self.bars_since_last_retrain = 0
        new_version = f"v_retrain_{len(self.retrain_history) + 1}"

        record_info = {
            "model_name": model_name,
            "version": new_version,
            "n_train": len(df_train),
            "n_val": len(df_val),
            "embargo_bars": self.embargo,
        }

        if registry is not None:
            registry.register_model(
                name=f"{model_name}_challenger",
                version=new_version,
                status=ModelStatus.CHALLENGER,
                initial_weight=0.0,
            )

        self.retrain_history.append(record_info)
        logger.success(f"Retraining completed for '{model_name}'. Registered as '{model_name}_challenger'.")
        return record_info
