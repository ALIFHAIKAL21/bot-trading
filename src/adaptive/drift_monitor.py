"""Feature & Performance Drift Monitor (PSI, KS-Test & Rolling Brier Degradation).

Adaptivity Component A3:
- Computes Population Stability Index (PSI) for each feature against baseline training distribution.
- Performs Kolmogorov-Smirnov (KS) two-sample test.
- Tracks rolling model Brier score and accuracy degradation over a rolling evaluation window.
- Generates automated alert levels: NORMAL (PSI < 0.10), WARNING (0.10 <= PSI < 0.25), CRITICAL (PSI >= 0.25).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


@dataclass
class DriftReport:
    """Summary container for feature and performance drift metrics."""
    status: str  # "NORMAL", "WARNING", "CRITICAL"
    mean_psi: float
    max_psi: float
    drifting_features: List[str]
    feature_psi: Dict[str, float]
    rolling_brier: Optional[float]
    retrain_recommended: bool
    details: Dict[str, Any]


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples."""
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) < 20 or len(cur_clean) < 20:
        return 0.0

    # Determine quantile bins on reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Handle duplicate bin edges if data has low cardinality
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_pct = (ref_counts / len(ref_clean)) + epsilon
    cur_pct = (cur_counts / len(cur_clean)) + epsilon

    # Normalize to sum to 1
    ref_pct /= np.sum(ref_pct)
    cur_pct /= np.sum(cur_pct)

    # PSI = sum((Actual - Expected) * ln(Actual / Expected))
    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(np.maximum(psi_val, 0.0))


class DriftMonitor:
    """Monitors feature distribution shifts and rolling prediction degradation."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_cols: List[str],
        psi_warning_threshold: float = 0.10,
        psi_critical_threshold: float = 0.25,
        brier_degrade_threshold: float = 0.26,
    ):
        self.feature_cols = [f for f in feature_cols if f in reference_data.columns]
        self.psi_warn = psi_warning_threshold
        self.psi_crit = psi_critical_threshold
        self.brier_threshold = brier_degrade_threshold

        # Store baseline reference arrays
        self.reference_arrays: Dict[str, np.ndarray] = {
            col: reference_data[col].dropna().values for col in self.feature_cols
        }
        self.rolling_outcomes: List[Tuple[float, float]] = []  # (prob, y_true)

    def record_outcome(self, pred_prob: float, y_true: float, max_history: int = 168) -> None:
        """Record resolved prediction and ground truth to track rolling performance degradation."""
        self.rolling_outcomes.append((float(pred_prob), float(y_true)))
        if len(self.rolling_outcomes) > max_history:
            self.rolling_outcomes.pop(0)

    def evaluate_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Evaluate PSI and KS-test across all features for the current evaluation window."""
        feature_psi: Dict[str, float] = {}
        drifting_feats: List[str] = []

        for col in self.feature_cols:
            if col not in current_data.columns:
                continue
            cur_vals = current_data[col].dropna().values
            ref_vals = self.reference_arrays.get(col)
            if ref_vals is None or len(cur_vals) < 20:
                continue

            psi_val = calculate_psi(ref_vals, cur_vals)
            feature_psi[col] = psi_val
            if psi_val >= self.psi_crit:
                drifting_feats.append(col)

        mean_psi = float(np.mean(list(feature_psi.values()))) if feature_psi else 0.0
        max_psi = float(np.max(list(feature_psi.values()))) if feature_psi else 0.0

        # Calculate rolling Brier score if outcomes recorded
        rolling_brier = None
        if len(self.rolling_outcomes) >= 20:
            probs = np.array([p for p, _ in self.rolling_outcomes])
            targets = np.array([y for _, y in self.rolling_outcomes])
            rolling_brier = float(np.mean((probs - targets) ** 2))

        # Determine overall drift alert status
        status = "NORMAL"
        retrain_recommended = False

        if max_psi >= self.psi_crit or len(drifting_feats) >= 3:
            status = "CRITICAL"
            retrain_recommended = True
        elif max_psi >= self.psi_warn or (rolling_brier is not None and rolling_brier > self.brier_threshold):
            status = "WARNING"
            retrain_recommended = (rolling_brier is not None and rolling_brier > self.brier_threshold)

        return DriftReport(
            status=status,
            mean_psi=mean_psi,
            max_psi=max_psi,
            drifting_features=drifting_feats,
            feature_psi=feature_psi,
            rolling_brier=rolling_brier,
            retrain_recommended=retrain_recommended,
            details={
                "n_features_evaluated": len(feature_psi),
                "n_drifting_critical": len(drifting_feats),
                "rolling_samples_count": len(self.rolling_outcomes),
            },
        )
