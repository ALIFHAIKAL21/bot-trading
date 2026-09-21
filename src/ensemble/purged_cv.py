"""Purged Walk-Forward Cross-Validation with Embargo.

Implements López de Prado's Purged K-Fold / Walk-Forward CV:
- Purges training samples whose label horizons overlap into the test/validation period.
- Applies an embargo zone (>= horizon bars) to prevent serial correlation leakage.
- Generates clean Out-Of-Fold (OOF) prediction masks.
"""

from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class PurgedWalkForwardCV:
    """Purged Walk-Forward cross-validator with embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 12,
        embargo_bars: int = 12,
        holdout_ratio: float = 0.15,
    ):
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_bars = embargo_bars
        self.holdout_ratio = holdout_ratio

    def split(
        self,
        df: pd.DataFrame,
        touch_idx_col: str = "target_touch_idx",
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate (train_indices, val_indices) splits with purging and embargo."""
        n_total = len(df)
        n_holdout = int(n_total * self.holdout_ratio)
        n_cv = n_total - n_holdout

        touch_indices = (
            df[touch_idx_col].values
            if touch_idx_col in df.columns
            else np.arange(n_total) + self.horizon
        )

        fold_size = n_cv // (self.n_splits + 1)

        for fold in range(self.n_splits):
            val_start = fold_size * (fold + 1)
            val_end = min(val_start + fold_size, n_cv)

            val_indices = np.arange(val_start, val_end)

            # Training set: expanding window from 0 to val_start
            raw_train_indices = np.arange(0, val_start)

            # Purging: remove any training sample whose label window overlaps into [val_start, val_end]
            # i.e., touch_idx >= val_start
            purged_train_mask = touch_indices[raw_train_indices] < val_start
            train_indices = raw_train_indices[purged_train_mask]

            # Ensure minimum viable training set size
            if len(train_indices) < 50:
                continue

            yield train_indices, val_indices

    def get_holdout_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Split into CV dataset and final untouched holdout dataset."""
        n_total = len(df)
        n_holdout = int(n_total * self.holdout_ratio)
        n_cv = n_total - n_holdout

        cv_indices = np.arange(0, n_cv)
        holdout_indices = np.arange(n_cv + self.embargo_bars, n_total)

        return cv_indices, holdout_indices
