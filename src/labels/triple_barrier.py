"""Triple-barrier labeling and sample uniqueness weighting.

Implements Marcos López de Prado's Triple-Barrier Method:
- Take-profit and stop-loss scaled by ATR.
- Vertical barrier at fixed horizon (e.g. 12 bars).
- Continuous forward returns for regression heads.
- Concurrency-based sample uniqueness weights.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import LabelConfig


class TripleBarrierLabeler:
    """Computes triple-barrier discrete labels, forward returns, and sample weights."""

    def __init__(self, config: Optional[LabelConfig] = None):
        self.config = config or LabelConfig()

    def label_barriers(
        self,
        df: pd.DataFrame,
        atr_col: str = "atr_14",
    ) -> pd.DataFrame:
        """Apply triple-barrier labeling to OHLCV DataFrame with ATR.

        Returns DataFrame with columns:
        - `target_label`: {-1 (SL hit), 0 (Vertical barrier / flat), +1 (TP hit)}
        - `target_binary_long`: {1 if target_label == +1, else 0}
        - `target_fwd_ret`: log return over the vertical horizon
        - `target_touch_idx`: index of the bar where barrier was touched
        - `sample_weight`: López de Prado average uniqueness weight
        """
        if df.empty or atr_col not in df.columns:
            raise ValueError(f"DataFrame must contain ATR column '{atr_col}'")

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        atrs = df["atr_14"].values
        n = len(df)
        horizon = self.config.vertical_barrier
        tp_mult = self.config.tp_multiplier
        sl_mult = self.config.sl_multiplier

        labels = np.zeros(n, dtype=np.int32)
        touch_indices = np.zeros(n, dtype=np.int64)
        fwd_returns = np.zeros(n, dtype=np.float64)

        for i in range(n):
            p0 = closes[i]
            atr = atrs[i]
            if np.isnan(atr) or atr <= 0:
                labels[i] = 0
                touch_indices[i] = i
                fwd_returns[i] = 0.0
                continue

            tp_price = p0 + tp_mult * atr
            sl_price = p0 - sl_mult * atr

            max_j = min(i + horizon, n - 1)
            fwd_returns[i] = np.log(closes[max_j] / (p0 + 1e-9))

            touched = False
            for j in range(i + 1, max_j + 1):
                h_j = highs[j]
                l_j = lows[j]

                # Check if both barriers hit in the same bar (conservative: trigger SL)
                if h_j >= tp_price and l_j <= sl_price:
                    labels[i] = -1
                    touch_indices[i] = j
                    touched = True
                    break
                elif h_j >= tp_price:
                    labels[i] = 1
                    touch_indices[i] = j
                    touched = True
                    break
                elif l_j <= sl_price:
                    labels[i] = -1
                    touch_indices[i] = j
                    touched = True
                    break

            if not touched:
                labels[i] = 0
                touch_indices[i] = max_j

        # Compute sample uniqueness weights
        weights = self._compute_sample_weights(n, touch_indices)

        result_df = df.copy()
        result_df["target_label"] = labels
        result_df["target_binary_long"] = (labels == 1).astype(int)
        result_df["target_fwd_ret"] = fwd_returns
        result_df["target_touch_idx"] = touch_indices
        result_df["sample_weight"] = weights

        return result_df

    def _compute_sample_weights(self, n: int, touch_indices: np.ndarray) -> np.ndarray:
        """Calculate average uniqueness of overlapping event windows."""
        # Concurrency array c[t]: number of active events at bar t
        concurrency = np.zeros(n, dtype=np.int32)
        for i in range(n):
            t_end = touch_indices[i]
            concurrency[i : t_end + 1] += 1

        concurrency = np.maximum(concurrency, 1)

        # Uniqueness of event i: average(1 / concurrency) across [i, touch_indices[i]]
        uniqueness = np.zeros(n, dtype=np.float64)
        for i in range(n):
            t_end = touch_indices[i]
            window_c = concurrency[i : t_end + 1]
            uniqueness[i] = np.mean(1.0 / window_c)

        # Normalize weights so mean is 1.0
        mean_u = np.mean(uniqueness)
        if mean_u > 0:
            weights = uniqueness / mean_u
        else:
            weights = np.ones(n, dtype=np.float64)

        return weights
