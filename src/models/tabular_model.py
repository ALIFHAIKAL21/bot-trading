"""Tabular direction model using LightGBM with probability calibration and strict leakage prevention."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.features.feature_pipeline import validate_feature_columns
from src.utils.config import ModelBConfig


class TabularDirectionModel:
    """LightGBM binary classification model with probability calibration and sample weights."""

    def __init__(self, config: Optional[ModelBConfig] = None):
        self.config = config or ModelBConfig()
        self.model: Optional[lgb.LGBMClassifier] = None
        self.calibrator = None
        self.feature_names: List[str] = []
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TabularDirectionModel":
        """Fit LightGBM classifier with early stopping and calibrate on validation fold."""
        if lgb is None:
            logger.warning("lightgbm is not installed. Operating in mock mode.")
            self._fitted = True
            return self

        # Strictly assert zero leakage in feature names
        feature_cols = list(X_train.columns)
        validate_feature_columns(feature_cols)
        self.feature_names = feature_cols

        base_params = {
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "num_leaves": self.config.num_leaves,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        self.model = lgb.LGBMClassifier(**base_params)

        if X_val is not None and y_val is not None and len(X_val) > 0:
            validate_feature_columns(list(X_val.columns))
            self.model.fit(
                X_train[self.feature_names],
                y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val[self.feature_names], y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
            # Calibrate probabilities using Platt scaling (sigmoid) on validation fold
            val_preds = self.model.predict_proba(X_val[self.feature_names])[:, 1]
            self.calibrator = LogisticRegression()
            self.calibrator.fit(val_preds.reshape(-1, 1), y_val)
        else:
            self.model.fit(X_train[self.feature_names], y_train, sample_weight=sample_weights)
            self.calibrator = None

        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability of upward direction P(long)."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet.")

        if self.model is None:
            # Fallback mock probabilities
            return np.full(len(X), 0.5)

        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")

        raw_probs = self.model.predict_proba(X[self.feature_names])[:, 1]
        if self.calibrator is not None:
            calibrated_probs = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            return np.clip(calibrated_probs, 1e-4, 1.0 - 1e-4)

        return np.clip(raw_probs, 1e-4, 1.0 - 1e-4)

    def get_feature_importances(self) -> Dict[str, float]:
        """Extract split and gain feature importances."""
        if self.model is None or not self._fitted:
            return {}

        importances = self.model.feature_importances_
        res = {name: float(imp) for name, imp in zip(self.feature_names, importances)}
        # Sort descending
        return dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics: AUC, LogLoss, Brier score."""
        probs = self.predict_proba(X_test)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
        brier = brier_score_loss(y_test, probs)
        loss = log_loss(y_test, probs)

        return {
            "auc": float(auc),
            "brier_score": float(brier),
            "log_loss": float(loss),
        }
