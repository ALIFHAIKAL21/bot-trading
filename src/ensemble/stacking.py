"""Stacking meta-learner with constrained non-negative weights and honest fallback protocol.

Enforces:
- Non-negative convex weighting (w_i >= 0, sum(w_i) = 1) to prevent unstable negative weights.
- Strict training on out-of-fold base model predictions.
- Automated protocol: Stacker is kept ONLY if it beats simple average on out-of-fold Brier score/log-loss.
  Otherwise, it defaults to the simple average baseline.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.utils.config import EnsembleConfig


class StackingMetaModel:
    """Meta-learner combining out-of-fold predictions with non-negative constrained optimization."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.feature_names: List[str] = []
        self.weights: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.active_method: str = "simple_average"
        self.comparison_metrics: Dict[str, Dict] = {}
        self._fitted = False

    def fit(
        self,
        oof_features: pd.DataFrame,
        y_true: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "StackingMetaModel":
        """Fit meta-learner with constrained non-negative weights and compare against simple average."""
        # Clean out-of-fold matrix
        valid_mask = ~oof_features.isna().any(axis=1)
        if valid_mask.sum() < 50:
            raise ValueError(f"Insufficient valid OOF samples ({valid_mask.sum()}) to train meta-model.")

        clean_oof = oof_features.loc[valid_mask]
        y_clean = y_true[valid_mask]
        self.feature_names = list(clean_oof.columns)
        n_features = len(self.feature_names)

        # Baseline: Simple Average across probability columns
        p_cols = [c for c in self.feature_names if "p_long" in c or "prob" in c]
        if not p_cols:
            p_cols = self.feature_names

        simple_avg_probs = clean_oof[p_cols].mean(axis=1).values
        brier_simple = float(brier_score_loss(y_clean, simple_avg_probs))
        loss_simple = float(log_loss(y_clean, np.clip(simple_avg_probs, 1e-5, 1.0 - 1e-5)))
        auc_simple = float(roc_auc_score(y_clean, simple_avg_probs)) if len(np.unique(y_clean)) > 1 else 0.5

        logger.info(
            f"Simple Average Baseline (OOF) -> AUC: {auc_simple:.4f} | Brier: {brier_simple:.4f} | LogLoss: {loss_simple:.4f}"
        )

        # Constrained non-negative weighting optimization on probability inputs
        X_prob = clean_oof[p_cols].values
        n_p = len(p_cols)

        def objective(w: np.ndarray) -> float:
            p_ens = np.dot(X_prob, w)
            return float(np.mean((p_ens - y_clean) ** 2))

        # Initial weights: equal weights
        w0 = np.full(n_p, 1.0 / n_p)
        # Bounds: [0.0, 1.0] for each base model weight
        bounds = [(0.0, 1.0) for _ in range(n_p)]
        # Constraint: sum(weights) == 1.0
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(objective, w0, bounds=bounds, constraints=constraints, method="SLSQP")
        if res.success:
            optimal_weights = res.x
            optimal_weights = np.maximum(optimal_weights, 0.0)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        else:
            logger.warning("Constrained optimization did not converge; falling back to equal weights.")
            optimal_weights = w0

        opt_probs = np.dot(X_prob, optimal_weights)
        brier_opt = float(brier_score_loss(y_clean, opt_probs))
        loss_opt = float(log_loss(y_clean, np.clip(opt_probs, 1e-5, 1.0 - 1e-5)))
        auc_opt = float(roc_auc_score(y_clean, opt_probs)) if len(np.unique(y_clean)) > 1 else 0.5

        # Decision Protocol: Only keep stacker if it beats simple average on Brier score
        weight_dict = {col: round(float(w), 4) for col, w in zip(p_cols, optimal_weights)}
        logger.info(f"Constrained Stacker Weights: {weight_dict}")
        logger.info(
            f"Constrained Stacker (OOF) -> AUC: {auc_opt:.4f} | Brier: {brier_opt:.4f} | LogLoss: {loss_opt:.4f}"
        )

        self.comparison_metrics = {
            "simple_average": {"auc": auc_simple, "brier": brier_simple, "log_loss": loss_simple},
            "constrained_stacker": {"auc": auc_opt, "brier": brier_opt, "log_loss": loss_opt, "weights": weight_dict},
        }

        # Stacker beat criterion: lower Brier score and lower log-loss
        if brier_opt < brier_simple and loss_opt <= loss_simple:
            self.weights = optimal_weights
            self.active_method = "constrained_stacker"
            logger.success("Stacker protocol: Constrained Stacker OUTPERFORMS Simple Average. Keeping stacker.")
        else:
            self.weights = w0
            self.active_method = "simple_average"
            logger.warning(
                "Stacker protocol: Constrained Stacker did NOT outperform Simple Average on OOF Brier/LogLoss. "
                "Defaulting to Simple Average as required by F9."
            )

        self._fitted = True
        return self

    def predict_proba(self, base_predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate final ensemble P(long) using active method (stacker or simple average)."""
        p_cols = [c for c in base_predictions_df.columns if "p_long" in c or "prob" in c or "dir_prob" in c]
        if not p_cols:
            p_cols = list(base_predictions_df.columns)

        # Baseline simple average
        simple_avg = base_predictions_df[p_cols].mean(axis=1).values

        if not self._fitted or self.weights is None:
            return pd.DataFrame(
                {
                    "meta_p_long": simple_avg,
                    "meta_confidence": np.abs(simple_avg - 0.5) * 2.0,
                    "simple_avg_p_long": simple_avg,
                    "ensemble_method": "simple_average",
                },
                index=base_predictions_df.index,
            )

        if self.active_method == "constrained_stacker":
            # Filter aligned probability columns
            matched_cols = [c for c in p_cols if c in self.feature_names]
            if len(matched_cols) == len(self.weights):
                stacked_probs = np.dot(base_predictions_df[matched_cols].values, self.weights)
            else:
                stacked_probs = simple_avg
        else:
            stacked_probs = simple_avg

        stacked_probs = np.clip(stacked_probs, 1e-4, 1.0 - 1e-4)
        confidence = np.abs(stacked_probs - 0.5) * 2.0

        return pd.DataFrame(
            {
                "meta_p_long": stacked_probs,
                "meta_confidence": confidence,
                "simple_avg_p_long": simple_avg,
                "ensemble_method": self.active_method,
            },
            index=base_predictions_df.index,
        )

    def evaluate(self, oof_df: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble predictions against ground truth."""
        preds = self.predict_proba(oof_df)
        p_meta = preds["meta_p_long"].values
        p_avg = preds["simple_avg_p_long"].values

        auc_meta = roc_auc_score(y_true, p_meta) if len(np.unique(y_true)) > 1 else 0.5
        auc_avg = roc_auc_score(y_true, p_avg) if len(np.unique(y_true)) > 1 else 0.5
        brier_meta = brier_score_loss(y_true, p_meta)
        brier_avg = brier_score_loss(y_true, p_avg)

        return {
            "meta_auc": float(auc_meta),
            "simple_avg_auc": float(auc_avg),
            "meta_brier": float(brier_meta),
            "simple_avg_brier": float(brier_avg),
            "active_method": self.active_method,
        }
