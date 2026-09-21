"""Empirical CV evaluation comparing leaked vs clean features, and HTF feature ablation (F1, F2)."""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from loguru import logger

# Add repo root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import MarketDataLoader
from src.data.holdout_guard import partition_dataset, guard_locked_test
from src.ensemble.purged_cv import PurgedWalkForwardCV
from src.features.feature_pipeline import FeaturePipeline
from src.labels.triple_barrier import TripleBarrierLabeler
from src.models.tabular_model import TabularDirectionModel
from src.utils.config import load_config


def run_leakage_cv_study(config_path: str = "config/config.yaml") -> Dict:
    """Run rigorous cross-validation comparison:
    1. Leaked features (with sample_weight & unconstrained inputs)
    2. Clean features (strict whitelist, sample_weight purged)
    3. Clean features without htf_1d_trend (F2 ablation)
    """
    logger.info("=" * 70)
    logger.info("STARTING PHASE 1 LEAKAGE & HTF EMPIRICAL CV COMPARISON")
    logger.info("=" * 70)

    cfg = load_config(config_path)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load CCXT 3-year data
    loader = MarketDataLoader(
        cache_dir=cfg.data.cache_dir,
        primary_source=cfg.data.primary_source,
        gap_fill_policy=cfg.data.gap_fill_policy,
    )
    primary_sym = cfg.market.symbols[0]
    df_raw = loader.load_or_fetch(primary_sym, timeframe=cfg.market.timeframe, history_days=cfg.market.history_days)
    logger.info(f"Loaded {len(df_raw)} bars for {primary_sym}")

    # 2. Partition into VALIDATION (90%) and LOCKED_TEST (10%)
    df_val_pool, df_locked, partition_info = partition_dataset(df_raw)
    cutoff_ts = partition_info["locked_cutoff_timestamp"]
    guard_locked_test(df_val_pool, cutoff_ts)

    # 3. Feature extraction and labeling on VALIDATION pool
    pipeline = FeaturePipeline(cfg.features)
    labeler = TripleBarrierLabeler(cfg.labels)

    df_feat = pipeline.build_features(df_val_pool)
    df_labeled = labeler.label_barriers(df_feat)
    df_clean = df_labeled.iloc[72:].copy()  # Warmup strip
    logger.info(f"Usable validation bars after 72h warmup: {len(df_clean)}")

    # Standard clean whitelist features
    clean_features = pipeline.get_feature_columns(df_clean)
    logger.info(f"Clean whitelist features ({len(clean_features)}): {clean_features}")

    # Feature sets to test
    feature_sets = {
        "leaked_with_sample_weight": clean_features + ["sample_weight"],
        "clean_with_htf_1d": clean_features,
        "clean_without_htf_1d": [f for f in clean_features if f != "htf_1d_trend"],
    }

    cv = PurgedWalkForwardCV(
        n_splits=cfg.ensemble.cv_splits,
        horizon=cfg.market.horizon,
        embargo_bars=cfg.ensemble.embargo_bars,
        holdout_ratio=0.0,  # Already partitioned externally
    )

    y_all = df_clean["target_binary_long"].values
    w_all = df_clean["sample_weight"].values

    cv_results = {}

    for set_name, f_cols in feature_sets.items():
        logger.info(f"\nEvaluating Feature Set: '{set_name}' ({len(f_cols)} features)...")
        fold_evals = []

        for fold, (tr_idx, val_idx) in enumerate(cv.split(df_clean)):
            X_tr = df_clean[f_cols].iloc[tr_idx]
            y_tr = y_all[tr_idx]
            w_tr = w_all[tr_idx]

            X_val = df_clean[f_cols].iloc[val_idx]
            y_val = y_all[val_idx]

            # Fit Model B
            # Note: TabularDirectionModel.fit() has a strict validate_feature_columns check.
            # For testing the 'leaked' baseline, we instantiate LGBM directly to measure before vs after.
            if "sample_weight" in f_cols:
                import lightgbm as lgb
                from sklearn.metrics import roc_auc_score, brier_score_loss

                model = lgb.LGBMClassifier(
                    n_estimators=cfg.models.model_b.n_estimators,
                    learning_rate=cfg.models.model_b.learning_rate,
                    max_depth=cfg.models.model_b.max_depth,
                    num_leaves=cfg.models.model_b.num_leaves,
                    random_state=42,
                    verbose=-1,
                )
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                probs = model.predict_proba(X_val)[:, 1]
                auc = float(roc_auc_score(y_val, probs))
                brier = float(brier_score_loss(y_val, probs))
                val_eval = {"fold": fold, "auc": auc, "brier_score": brier}
            else:
                model = TabularDirectionModel(cfg.models.model_b)
                model.fit(X_tr, y_tr, sample_weights=w_tr, X_val=X_val, y_val=y_val)
                val_eval = {"fold": fold, **model.evaluate(X_val, y_val)}

            fold_evals.append(val_eval)
            logger.info(f"  Fold {fold}: AUC = {val_eval['auc']:.4f} | Brier = {val_eval['brier_score']:.4f}")

        aucs = [f["auc"] for f in fold_evals]
        briers = [f["brier_score"] for f in fold_evals]
        cv_results[set_name] = {
            "features_count": len(f_cols),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_brier": float(np.mean(briers)),
            "std_brier": float(np.std(briers)),
            "folds": fold_evals,
        }
        logger.success(
            f"Set '{set_name}' Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} | Brier: {np.mean(briers):.4f}"
        )

    # Save summary report
    out_file = reports_dir / "leakage_cv_comparison.json"
    with open(out_file, "w") as f:
        json.dump(cv_results, f, indent=2)

    logger.success(f"Empirical CV results saved to {out_file}")
    return cv_results


if __name__ == "__main__":
    run_leakage_cv_study()
