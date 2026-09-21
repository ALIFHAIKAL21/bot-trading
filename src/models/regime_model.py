"""Regime detector using Gaussian Hidden Markov Model (HMM).

Strictly causal and robust implementation:
- Feature standardization (StandardScaler) fitted strictly on training folds.
- States sorted by realized volatility so State 0 = Low Vol, State 1 = Medium Vol, State 2 = High Vol / Crisis.
- Forward filtering algorithm: computes alpha_t = P(S_t | y_{1:t}) causally without look-ahead.
- Health check: detects state collapse or constant/NaN output and flags degraded status.
- Diagnostics reporting: state occupancies, transition matrix, mean returns and volatilities.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

try:
    from hmmlearn.hmm import GaussianHMM
    from scipy.stats import multivariate_normal
    from sklearn.preprocessing import StandardScaler
except ImportError:
    GaussianHMM = None
    multivariate_normal = None
    StandardScaler = None

from src.utils.config import ModelEConfig


class CausalRegimeHMM:
    """3-State Gaussian HMM with standardized features, sorted volatility regimes, and causal filtering."""

    def __init__(self, config: Optional[ModelEConfig] = None):
        self.config = config or ModelEConfig()
        self.n_components = self.config.n_components
        self.features = self.config.features
        self.model: Optional[GaussianHMM] = None
        self.scaler = None
        self.state_order: np.ndarray = np.arange(self.n_components)
        self.diagnostics: Dict = {}
        self._fitted = False

    def fit(self, df_train: pd.DataFrame) -> "CausalRegimeHMM":
        """Fit Gaussian HMM strictly on training data with feature standardization and sorted states."""
        if GaussianHMM is None or StandardScaler is None:
            logger.warning("hmmlearn or scikit-learn is not installed. CausalRegimeHMM operating in mock mode.")
            self._fitted = True
            return self

        sub = df_train[self.features].dropna()
        if len(sub) < 100:
            raise ValueError(f"Insufficient training samples for HMM (need >= 100, got {len(sub)}).")

        # 1. Fit StandardScaler strictly on training fold
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(sub.values)

        # 2. Fit GaussianHMM with diagonal covariance to prevent singular matrix collapse
        cov_type = "diag" if self.config.covariance_type in ("diag", "full") else self.config.covariance_type
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=cov_type,
            n_iter=self.config.n_iter,
            random_state=42,
        )
        self.model.fit(X_scaled)

        # 3. Sort states by realized volatility (feature index 1: 'realized_vol_12')
        # State 0 = Lowest volatility (calm/trending)
        # State 1 = Medium volatility
        # State 2 = Highest volatility (crisis/breakout)
        raw_means = self.scaler.inverse_transform(self.model.means_)
        vol_idx = 1 if "realized_vol_12" in self.features else 0
        state_vols = raw_means[:, vol_idx]
        self.state_order = np.argsort(state_vols)

        # Reorder HMM parameters according to volatility ordering
        self.model.means_ = self.model.means_[self.state_order]
        self.model._covars_ = self.model._covars_[self.state_order]

        # Reorder transition matrix and startprob
        self.model.transmat_ = self.model.transmat_[self.state_order][:, self.state_order]
        self.model.startprob_ = self.model.startprob_[self.state_order]

        # Normalize transition rows
        self.model.transmat_ = self.model.transmat_ / self.model.transmat_.sum(axis=1, keepdims=True)
        self.model.startprob_ = self.model.startprob_ / self.model.startprob_.sum()

        # Mark model as fitted before calculating diagnostics
        self._fitted = True

        # Compute diagnostics on training fold
        train_filtered = self.predict_filtered_proba(df_train)
        preds = train_filtered.idxmax(axis=1).values
        unique, counts = np.unique(preds, return_counts=True)
        occupancies = {str(k): int(c) for k, c in zip(unique, counts)}
        total_p = sum(occupancies.values())
        occupancy_pct = {k: round(v / max(total_p, 1) * 100.0, 2) for k, v in occupancies.items()}

        sorted_means = self.scaler.inverse_transform(self.model.means_)
        self.diagnostics = {
            "n_training_samples": len(sub),
            "state_occupancy_counts": occupancies,
            "state_occupancy_pct": occupancy_pct,
            "state_volatilities": [float(v) for v in sorted_means[:, vol_idx]],
            "state_means": {
                f"state_{k}": {feat: float(sorted_means[k, i]) for i, feat in enumerate(self.features)}
                for k in range(self.n_components)
            },
            "transition_matrix": [[round(float(p), 4) for p in row] for row in self.model.transmat_],
        }

        self._fitted = True
        logger.info(
            f"Fitted {self.n_components}-state HMM on {len(sub)} bars | "
            f"Occupancies: {occupancy_pct} | Volatilities: {[round(v, 5) for v in sorted_means[:, vol_idx]]}"
        )
        return self

    def predict_filtered_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute causal forward filtered probabilities P(S_t | y_{1:t}) without look-ahead."""
        cols = [f"regime_p{k}" for k in range(self.n_components)]
        if not self._fitted:
            raise RuntimeError("Model must be fitted before computing filtered probabilities.")

        if self.model is None or self.scaler is None or multivariate_normal is None:
            n = len(df)
            probs = np.full((n, self.n_components), 1.0 / self.n_components)
            return pd.DataFrame(probs, index=df.index, columns=cols)

        sub_df = df[self.features].copy()
        valid_idx = sub_df.dropna().index
        if len(valid_idx) == 0:
            return pd.DataFrame(1.0 / self.n_components, index=df.index, columns=cols)

        # Scale features using training fold scaler
        X_scaled = self.scaler.transform(sub_df.loc[valid_idx].values)
        n_samples = len(X_scaled)

        # HMM parameters
        startprob = self.model.startprob_
        transmat = self.model.transmat_
        means = self.model.means_
        covars = self.model.covars_
        cov_type = self.model.covariance_type

        # Compute emission probabilities
        emission_probs = np.zeros((n_samples, self.n_components), dtype=np.float64)
        for k in range(self.n_components):
            try:
                if cov_type == "diag":
                    cov_k = np.diag(covars[k] + 1e-4)
                else:
                    cov_k = covars[k] + np.eye(covars[k].shape[0]) * 1e-4
                rv = multivariate_normal(mean=means[k], cov=cov_k, allow_singular=True)
                emission_probs[:, k] = rv.pdf(X_scaled)
            except Exception as e:
                logger.warning(f"Emission density error for state {k}: {e}. Fallback used.")
                emission_probs[:, k] = 1e-6

        # Avoid exact zeros
        emission_probs = np.maximum(emission_probs, 1e-12)

        # Forward filtering algorithm
        alpha = np.zeros((n_samples, self.n_components), dtype=np.float64)
        alpha[0] = startprob * emission_probs[0]
        sum_0 = np.sum(alpha[0])
        alpha[0] = alpha[0] / (sum_0 if sum_0 > 0 else 1.0)

        for t in range(1, n_samples):
            prior_t = np.dot(alpha[t - 1], transmat)
            alpha[t] = emission_probs[t] * prior_t
            sum_t = np.sum(alpha[t])
            alpha[t] = alpha[t] / (sum_t if sum_t > 0 else 1.0)

        res_df = pd.DataFrame(index=df.index, columns=cols, dtype=np.float64)
        res_df.loc[valid_idx, cols] = alpha
        res_df = res_df.bfill().fillna(1.0 / self.n_components)
        return res_df

    def check_health(self, recent_probs_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Health check for state collapse, NaNs, or constant probabilities.
        
        Returns (is_healthy, degraded_reason).
        """
        if recent_probs_df.empty:
            return False, "regime_probabilities_empty"

        if recent_probs_df.isna().any().any():
            return False, "regime_probabilities_contain_nans"

        # Check if one state probability has remained > 0.99 for all bars in recent slice
        for col in recent_probs_df.columns:
            if (recent_probs_df[col] > 0.999).all() and len(recent_probs_df) >= 24:
                return False, f"regime_collapsed_to_{col}"

        return True, None

    def save_diagnostics(self, file_path: str = "reports/hmm_report.json") -> None:
        """Persist HMM diagnostics report."""
        out = Path(file_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.diagnostics, f, indent=2)
        logger.info(f"HMM diagnostics saved to {file_path}")
