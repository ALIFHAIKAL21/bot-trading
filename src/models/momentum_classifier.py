"""
Momentum Setup Quality Filter Model (Point 3 & Point 4)
Classifies whether a candidate momentum setup will reach RR 1:2 (or RR 1:1) before hitting SL.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import lightgbm as lgb
from loguru import logger

# Strictly whitelisted features (Market Structure + Indicators + Setup characteristics)
FEATURE_COLS: List[str] = [
    # Market Structure (Point 4)
    "dist_to_support_atr",
    "dist_to_resistance_atr",
    "dist_to_demand_atr",
    "dist_to_supply_atr",
    "inside_demand_zone",
    "inside_supply_zone",
    "liq_sweep_bull",
    "liq_sweep_bear",
    "premium_discount_ratio",
    "risk_atr_ratio",
    "atr_14",
    # Indicators (TradingView default: MA Cross & SMI)
    "sma_spread",
    "sma_trend",
    "sma_bars_since_cross",
    "ema_spread",
    "ema_trend",
    "ema_bars_since_cross",
    "smi_val",
    "smi_signal",
    "smi_hist",
    "smi_zone",
    "smi_cross_bull",
    "smi_cross_bear",
    "smi_os_reversal_bull",
    "smi_ob_reversal_bear",
    "smi_zero_cross_bull",
    "smi_zero_cross_bear",
    # Setup Context
    "direction_num",
    "trigger_count",
]


class MomentumClassifier:
    """
    Calibrated Decision Filter Model to evaluate momentum trade setups.
    Predicts P(Win) at target Risk-to-Reward ratio (e.g. 1:2).
    """

    def __init__(
        self,
        target_col: str = "win_2r",
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 25,
    ):
        self.target_col = target_col
        self.feature_cols = FEATURE_COLS
        self.base_model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42,
            verbose=-1,
        )
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.feature_importances: Dict[str, float] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values and selects whitelisted features."""
        X = df[self.feature_cols].copy()
        for col in self.feature_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
            else:
                X[col] = 0.0
        return X

    def fit(self, df_train: pd.DataFrame) -> "MomentumClassifier":
        """Fit model with 5-fold cross-validation probability calibration."""
        X_train = self._prepare_features(df_train)
        y_train = df_train[self.target_col].values

        logger.info(
            f"Training MomentumClassifier on {len(df_train):,} setups (Target: {self.target_col}, Base Win Rate: {y_train.mean()*100:.2f}%)..."
        )

        # 1. Fit base model to get feature importances
        self.base_model.fit(X_train, y_train)
        raw_imp = self.base_model.feature_importances_
        total_imp = max(1, sum(raw_imp))
        self.feature_importances = {
            col: round(float(imp / total_imp), 4)
            for col, imp in sorted(zip(self.feature_cols, raw_imp), key=lambda x: x[1], reverse=True)
        }

        # 2. Calibrate model probabilities using 5-fold CV
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method="sigmoid",
            cv=5,
        )
        self.calibrated_model.fit(X_train, y_train)

        # Training diagnostics
        train_probs = self.calibrated_model.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, train_probs)
        brier = brier_score_loss(y_train, train_probs)
        logger.info(f"Model calibrated. Train AUC: {auc:.4f}, Brier Score: {brier:.4f}")

        return self

    def predict_proba(self, df_eval: pd.DataFrame) -> np.ndarray:
        """Returns array of P(Win) for each setup in df_eval."""
        if self.calibrated_model is None:
            raise RuntimeError("Model is not fitted yet.")
        X_eval = self._prepare_features(df_eval)
        probs = self.calibrated_model.predict_proba(X_eval)[:, 1]
        return probs

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluates model performance on out-of-sample data."""
        y_true = df_test[self.target_col].values
        y_prob = self.predict_proba(df_test)

        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        brier = brier_score_loss(y_true, y_prob)
        loss = log_loss(y_true, y_prob)

        return {
            "auc": round(float(auc), 4),
            "brier_score": round(float(brier), 4),
            "log_loss": round(float(loss), 4),
            "mean_prob": round(float(y_prob.mean()), 4),
            "base_rate": round(float(y_true.mean()), 4),
        }
