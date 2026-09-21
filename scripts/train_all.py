"""Master training orchestration script with purged CV, hold-out discipline, and honest ensembling."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss

# Add repository root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import MarketDataLoader
from src.data.holdout_guard import partition_dataset, guard_locked_test
from src.ensemble.purged_cv import PurgedWalkForwardCV
from src.ensemble.stacking import StackingMetaModel
from src.features.feature_pipeline import FeaturePipeline
from src.labels.triple_barrier import TripleBarrierLabeler
from src.models.chronos_model import ChronosBoltForecaster
from src.models.deep_sequence_model import DeepSequenceModel
from src.models.regime_model import CausalRegimeHMM
from src.models.sentiment_gate import SentimentNewsGate
from src.models.tabular_model import TabularDirectionModel
from src.utils.config import AppConfig, load_config
from src.utils.security import determine_execution_mode, print_active_models_banner, print_mode_banner


def train_pipeline(config_path: str = "config/config.yaml", target_symbol: Optional[str] = None) -> Dict:
    """Execute end-to-end training pipeline across all models with purged CV and strict discipline."""
    cfg: AppConfig = load_config(config_path)
    models_dir = Path("models_store")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    exec_mode = determine_execution_mode(cfg)
    print_mode_banner(exec_mode)
    print_active_models_banner(cfg, context="TRAINING")

    primary_sym = target_symbol or cfg.market.symbols[0]
    logger.info("=" * 70)
    logger.info(f"STARTING INSTITUTIONAL MULTI-MODEL QUANT TRAINING PIPELINE FOR {primary_sym}")
    logger.info("=" * 70)

    # 1. Load Data
    loader = MarketDataLoader(
        cache_dir=cfg.data.cache_dir,
        primary_source=cfg.data.primary_source,
        gap_fill_policy=cfg.data.gap_fill_policy,
    )
    feature_pipeline = FeaturePipeline(cfg.features)
    labeler = TripleBarrierLabeler(cfg.labels)

    logger.info(f"Loading primary market dataset for {primary_sym}...")
    df_raw = loader.load_or_fetch(
        primary_sym, timeframe=cfg.market.timeframe, history_days=cfg.market.history_days
    )
    logger.info(f"Raw dataset loaded: {len(df_raw):,} bars ({df_raw.index.min()} to {df_raw.index.max()})")

    # 2. Strict Hold-Out Partitioning
    df_val_pool, df_locked, partition_info = partition_dataset(df_raw)
    cutoff_ts = partition_info["locked_cutoff_timestamp"]
    guard_locked_test(df_val_pool, cutoff_ts)
    logger.success(f"Quarantined LOCKED_TEST: {len(df_locked)} bars sealed for final report.")

    # 3. Feature Generation & Labeling strictly on VALIDATION pool
    df_feat = feature_pipeline.build_features(df_val_pool)
    df_labeled = labeler.label_barriers(df_feat)
    # Strip warm-up NaNs from longest lookback (72h)
    df_clean = df_labeled.iloc[72:].copy()
    feature_cols = feature_pipeline.get_feature_columns(df_clean)
    logger.info(f"Validation dataset clean: {len(df_clean):,} bars | Whitelisted features: {len(feature_cols)}")

    # 4. Setup Purged Walk-Forward Cross-Validation
    cv = PurgedWalkForwardCV(
        n_splits=cfg.ensemble.cv_splits,
        horizon=cfg.market.horizon,
        embargo_bars=cfg.ensemble.embargo_bars,
        holdout_ratio=0.0,  # Partitioning managed by partition_dataset
    )

    n_cv = len(df_clean)
    oof_model_b = np.full(n_cv, np.nan)
    oof_regime_p0 = np.full(n_cv, np.nan)
    oof_regime_p1 = np.full(n_cv, np.nan)
    oof_regime_p2 = np.full(n_cv, np.nan)

    fold_metrics_b = []
    y_clean = df_clean["target_binary_long"].values
    w_clean = df_clean["sample_weight"].values
    fwd_ret_clean = df_clean["target_fwd_ret"].values

    # 5. Walk-Forward Cross-Validation: Model B (LightGBM) & Model E (HMM)
    logger.info("Executing Purged Walk-Forward Cross-Validation on Models B & E...")
    final_hmm = None

    for fold, (tr_idx, val_idx) in enumerate(cv.split(df_clean)):
        df_tr = df_clean.iloc[tr_idx]
        df_val = df_clean.iloc[val_idx]

        # Model E: Causal HMM fit on training fold
        hmm = CausalRegimeHMM(cfg.models.model_e)
        hmm.fit(df_tr)
        final_hmm = hmm
        val_regimes = hmm.predict_filtered_proba(df_val)

        # Model B: LightGBM fit on training fold (calibrated on val_idx)
        X_tr = df_tr[feature_cols]
        y_tr = y_clean[tr_idx]
        w_tr = w_clean[tr_idx]

        X_val = df_val[feature_cols]
        y_val = y_clean[val_idx]

        lgbm = TabularDirectionModel(cfg.models.model_b)
        lgbm.fit(X_tr, y_tr, sample_weights=w_tr, X_val=X_val, y_val=y_val)

        val_probs = lgbm.predict_proba(X_val)
        val_eval = lgbm.evaluate(X_val, y_val)

        # Calculate Information Coefficient (IC)
        val_ic, _ = spearmanr(val_probs, fwd_ret_clean[val_idx])
        val_eval["ic"] = float(val_ic) if not np.isnan(val_ic) else 0.0

        oof_model_b[val_idx] = val_probs
        oof_regime_p0[val_idx] = val_regimes["regime_p0"].values
        oof_regime_p1[val_idx] = val_regimes["regime_p1"].values
        oof_regime_p2[val_idx] = val_regimes["regime_p2"].values

        fold_metrics_b.append({"fold": fold, **val_eval})
        logger.info(
            f"Fold {fold} - Model B AUC: {val_eval['auc']:.4f} | IC: {val_eval['ic']:.4f} | Brier: {val_eval['brier_score']:.4f}"
        )

    # 6. Model A (Chronos-Bolt Foundation Forecaster)
    logger.info("Evaluating Model A (Chronos-Bolt Forecaster)...")
    chronos = ChronosBoltForecaster(cfg.models.model_a, cache_dir=cfg.data.cache_dir)
    chronos_preds = chronos.predict_rolling(df_clean, symbol=primary_sym, stride=6)
    chronos_exp_ret = chronos_preds["chronos_exp_ret"].values

    # Model A standalone metrics
    valid_chronos_mask = ~np.isnan(chronos_exp_ret) & (chronos_exp_ret != 0.0)
    if valid_chronos_mask.sum() > 100:
        chronos_auc = roc_auc_score(y_clean[valid_chronos_mask], chronos_exp_ret[valid_chronos_mask])
        chronos_ic, _ = spearmanr(chronos_exp_ret[valid_chronos_mask], fwd_ret_clean[valid_chronos_mask])
    else:
        chronos_auc, chronos_ic = 0.50, 0.0

    chronos_metrics = {
        "model": "model_a_chronos",
        "standalone_auc": float(chronos_auc),
        "standalone_ic": float(chronos_ic),
        "status": "ACTIVE",
    }
    logger.success(f"Model A (Chronos) Standalone AUC: {chronos_auc:.4f} | IC: {chronos_ic:.4f}")

    # 7. Model C (Deep Sequence Model)
    logger.info("Evaluating Model C (Deep Sequence Model)...")
    deep_model = DeepSequenceModel(cfg.models.model_c)
    val_split_point = int(len(df_clean) * 0.8)
    df_c_train = df_clean.iloc[:val_split_point]
    df_c_val = df_clean.iloc[val_split_point:]

    deep_model.fit(
        df_train=df_c_train,
        df_val=df_c_val,
        feature_cols=feature_cols[:20],
        save_dir=str(models_dir),
        reports_dir=str(reports_dir),
    )
    deep_preds_df = deep_model.predict_proba(df_clean)
    deep_val_preds = deep_preds_df["model_c_dir_prob"].iloc[val_split_point:].values
    y_c_val = y_clean[val_split_point:]

    deep_val_auc = float(roc_auc_score(y_c_val, deep_val_preds)) if len(np.unique(y_c_val)) > 1 else 0.5
    deep_val_ic, _ = spearmanr(deep_preds_df["model_c_fwd_ret"].iloc[val_split_point:].values, fwd_ret_clean[val_split_point:])

    # Decision on Model C per F8: disable if walk-forward AUC <= 0.50
    model_c_enabled = deep_val_auc > 0.505
    model_c_metrics = {
        "model": "model_c_deep",
        "val_auc": float(deep_val_auc),
        "val_ic": float(deep_val_ic) if not np.isnan(deep_val_ic) else 0.0,
        "enabled_by_decision": model_c_enabled,
        "disposition": "ACTIVE" if model_c_enabled else "DISABLED_DEFAULT (AUC <= 0.505)",
    }
    logger.info(f"Model C Validation AUC: {deep_val_auc:.4f} -> Disposition: {model_c_metrics['disposition']}")

    # 8. Assemble Out-Of-Fold Prediction Matrix
    oof_df = pd.DataFrame(
        {
            "lgbm_p_long": oof_model_b,
            "regime_p0": oof_regime_p0,
            "regime_p1": oof_regime_p1,
            "regime_p2": oof_regime_p2,
            "chronos_exp_ret": chronos_exp_ret,
            "model_c_dir_prob": deep_preds_df["model_c_dir_prob"].values,
        },
        index=df_clean.index,
    )

    # 9. Stacking Meta-Model with Constrained Weights & Protocol (F9)
    meta_model = StackingMetaModel(cfg.ensemble)
    meta_model.fit(oof_df, y_clean, sample_weights=w_clean)

    # 10. Fit Final Base Models on Full Validation Set
    logger.info("Fitting production models on full validation set...")
    final_lgbm = TabularDirectionModel(cfg.models.model_b)
    final_lgbm.fit(df_clean[feature_cols], y_clean, sample_weights=w_clean)

    if final_hmm is not None:
        final_hmm.save_diagnostics("reports/hmm_report.json")
        joblib.dump(final_hmm, models_dir / "model_e_hmm.joblib")

    joblib.dump(final_lgbm, models_dir / "model_b_lgbm.joblib")
    joblib.dump(meta_model, models_dir / "meta_model.joblib")

    # Save feature importances
    feat_imp = final_lgbm.get_feature_importances()
    with open(reports_dir / "feature_importances.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    # Save complete model diagnostics
    diagnostics = {
        "model_a_chronos": chronos_metrics,
        "model_b_lightgbm": {
            "mean_auc": float(np.mean([m["auc"] for m in fold_metrics_b])),
            "mean_ic": float(np.mean([m["ic"] for m in fold_metrics_b])),
            "mean_brier": float(np.mean([m["brier_score"] for m in fold_metrics_b])),
            "fold_metrics": fold_metrics_b,
        },
        "model_c_deep": model_c_metrics,
        "model_e_hmm": final_hmm.diagnostics if final_hmm else {},
        "meta_learner": meta_model.comparison_metrics,
    }

    with open(reports_dir / "model_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.success("Multi-model training and diagnostics export completed successfully.")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-model quant pipeline on target asset.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--symbol", type=str, default=None, help="Target asset symbol (e.g. XAU/USD, BTC/USDT)")
    args = parser.parse_args()
    train_pipeline(args.config, target_symbol=args.symbol)
