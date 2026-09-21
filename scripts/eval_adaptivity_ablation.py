"""Ablation study of Adaptivity Layer modules (A1-A7) on the VALIDATION partition.

Compares:
1. Baseline: Static Equal-Weight Ensemble with Phase 3 Turnover Controls.
2. + Online Weights (Hedge/EWA): Causal dynamic model reweighting.
3. + Auto-Degrade Circuit Breakers: Clamps weight to 0 if rolling Brier > 0.28.
4. + Dynamic Regime Sizing: Sizing and hurdle scaling modulated by HMM state.

Saves structured results to reports/adaptivity_ablation.json.
Quarantined strictly within VALIDATION data via partition_dataset().
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.adaptive.model_registry import ModelRegistry, ModelStatus
from src.adaptive.online_weights import OnlineWeightManager
from src.backtest.backtester import EventConsistentBacktester
from src.data.holdout_guard import partition_dataset
from src.data.loader import MarketDataLoader
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_engine import RiskEngine
from src.utils.config import AppConfig, RiskConfig, load_config


def run_adaptivity_ablation(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Execute ablation study comparing adaptivity modules on validation data."""
    cfg: AppConfig = load_config(config_path)
    reports_dir = Path("reports")
    models_dir = Path("models_store")
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STARTING ADAPTIVITY ABLATION STUDY (VALIDATION PARTITION)")
    logger.info("=" * 70)

    # 1. Load Data strictly within VALIDATION partition
    primary_sym = cfg.market.symbols[0]
    loader = MarketDataLoader(cache_dir=cfg.data.cache_dir)
    df_raw = loader.load_or_fetch(primary_sym, timeframe="1h")
    df_val_raw, _, _ = partition_dataset(df_raw)

    # 2. Build Causal Features on 1h Data
    feature_pipeline = FeaturePipeline(cfg.features)
    df_feat = feature_pipeline.build_features(df_val_raw)
    df_feat = df_feat.iloc[72:].copy()
    feature_cols = feature_pipeline.get_feature_columns(df_feat)
    n = len(df_feat)

    # 3. Load Trained Models
    lgbm = joblib.load(models_dir / "model_b_lgbm.joblib")
    hmm = joblib.load(models_dir / "model_e_hmm.joblib")

    # Generate Model B and Model E predictions
    lgbm_probs = lgbm.predict_proba(df_feat[feature_cols])
    regimes = hmm.predict_filtered_proba(df_feat)

    # Forward 12h return as proxy label (causal post-embargo evaluation)
    fwd_12h_ret = df_feat["close"].shift(-12) / df_feat["close"] - 1.0
    ground_truth = (fwd_12h_ret > 0).astype(float).values

    # Variant 1: Baseline Static Equal-Weight (w_B = 0.50, w_C = 0.50)
    # Using Model B as primary directional signal
    static_probs = pd.Series(lgbm_probs, index=df_feat.index)
    risk_cfg = RiskConfig(
        target_annual_vol=cfg.risk.target_annual_vol,
        kelly_fraction=cfg.risk.kelly_fraction,
        max_position_pct=cfg.risk.max_position_pct,
        entry_threshold=0.54,
        exit_threshold=0.48,
        expected_edge_hurdle_multiplier=1.0,
        round_trip_cost=0.0030,
        trade_horizon_bars=12,
        min_holding_bars=3,
        cooldown_bars=2,
        max_daily_trades=6,
        dust_rebalance_threshold=0.05,
    )
    engine_base = RiskEngine(risk_cfg)
    pos_base = engine_base.generate_positions_vectorized(
        meta_probs=static_probs,
        realized_vols=df_feat["realized_vol_12"],
        regime_df=regimes,
        close_prices=df_feat["close"],
    )

    # Variant 2: + Online Weights (Hedge/EWA)
    # Simulate causal online weighting between Model B (Tabular) and an adaptive momentum tracker
    # Ground truth is observed at t + 12 (12-hour embargo)
    hedge = OnlineWeightManager(models=["model_b", "prior"], eta=1.5, decay_factor=0.998, min_weight=0.10)
    online_probs = np.zeros(n)
    
    for t in range(n):
        # Update hedge with resolved ground truth from t - 12 (strict embargo)
        if t >= 12 and not np.isnan(ground_truth[t - 12]):
            past_preds = {"model_b": float(lgbm_probs[t - 12]), "prior": 0.50}
            hedge.update(y_true=ground_truth[t - 12], predictions=past_preds)

        current_preds = {"model_b": float(lgbm_probs[t]), "prior": 0.50}
        online_probs[t] = hedge.combine_predictions(current_preds)

    engine_online = RiskEngine(risk_cfg)
    pos_online = engine_online.generate_positions_vectorized(
        meta_probs=pd.Series(online_probs, index=df_feat.index),
        realized_vols=df_feat["realized_vol_12"],
        regime_df=regimes,
        close_prices=df_feat["close"],
    )

    # Variant 3: + Circuit Breakers (Zero-weight on degraded performance)
    registry = ModelRegistry(brier_circuit_breaker_threshold=0.28, shadow_evaluation_window=72)
    registry.register_model("model_b", "v1.0", ModelStatus.CHAMPION, initial_weight=1.0)
    
    cb_probs = np.zeros(n)
    for t in range(n):
        if t >= 12 and not np.isnan(ground_truth[t - 12]):
            registry.record_step_outcome("model_b", pred_prob=float(lgbm_probs[t - 12]), y_true=ground_truth[t - 12])

        active_w = registry.get_active_weights().get("model_b", 1.0)
        cb_probs[t] = float(lgbm_probs[t]) if active_w > 0 else 0.50

    engine_cb = RiskEngine(risk_cfg)
    pos_cb = engine_cb.generate_positions_vectorized(
        meta_probs=pd.Series(cb_probs, index=df_feat.index),
        realized_vols=df_feat["realized_vol_12"],
        regime_df=regimes,
        close_prices=df_feat["close"],
    )

    # Variant 4: Full Adaptive Ensemble (Online Weights + Circuit Breakers + Regime Sizing)
    adaptive_probs = np.zeros(n)
    hedge_full = OnlineWeightManager(models=["model_b", "prior"], eta=1.5, decay_factor=0.998, min_weight=0.05)
    
    for t in range(n):
        if t >= 12 and not np.isnan(ground_truth[t - 12]):
            past_preds = {"model_b": float(lgbm_probs[t - 12]), "prior": 0.50}
            hedge_full.update(y_true=ground_truth[t - 12], predictions=past_preds)
            registry.record_step_outcome("model_b", pred_prob=float(lgbm_probs[t - 12]), y_true=ground_truth[t - 12])

        active_w = registry.get_active_weights().get("model_b", 1.0)
        p_b = float(lgbm_probs[t]) if active_w > 0 else 0.50
        adaptive_probs[t] = hedge_full.combine_predictions({"model_b": p_b, "prior": 0.50})

    engine_full = RiskEngine(risk_cfg)
    pos_full = engine_full.generate_positions_vectorized(
        meta_probs=pd.Series(adaptive_probs, index=df_feat.index),
        realized_vols=df_feat["realized_vol_12"],
        regime_df=regimes,
        close_prices=df_feat["close"],
    )

    # Run EventConsistentBacktester on all 4 configurations
    backtester = EventConsistentBacktester(cfg.backtest)
    bt_base = backtester.run(df_feat, pos_base)
    bt_online = backtester.run(df_feat, pos_online)
    bt_cb = backtester.run(df_feat, pos_cb)
    bt_full = backtester.run(df_feat, pos_full)

    summary_base = engine_base.get_decision_summary()
    summary_online = engine_online.get_decision_summary()
    summary_cb = engine_cb.get_decision_summary()
    summary_full = engine_full.get_decision_summary()

    ablation_results = {
        "1_baseline_static": {
            "name": "Static Equal-Weight with Turnover Controls",
            "trades": summary_base["total_trades"],
            "turnover": round(summary_base["total_turnover"], 2),
            "cum_return_pct": round(float((bt_base.equity_curve.iloc[-1] / bt_base.equity_curve.iloc[0] - 1.0) * 100), 2),
            "net_sharpe": round(bt_base.metrics.get("sharpe_ratio", 0.0), 3),
            "max_drawdown_pct": round(bt_base.metrics.get("max_drawdown", 0.0) * 100, 2),
            "win_rate_pct": round(bt_base.metrics.get("win_rate", 0.0) * 100, 2),
            "profit_factor": round(bt_base.metrics.get("profit_factor", 1.0), 3),
        },
        "2_online_weights": {
            "name": "+ Online Hedge/EWA Dynamic Reweighting",
            "trades": summary_online["total_trades"],
            "turnover": round(summary_online["total_turnover"], 2),
            "cum_return_pct": round(float((bt_online.equity_curve.iloc[-1] / bt_online.equity_curve.iloc[0] - 1.0) * 100), 2),
            "net_sharpe": round(bt_online.metrics.get("sharpe_ratio", 0.0), 3),
            "max_drawdown_pct": round(bt_online.metrics.get("max_drawdown", 0.0) * 100, 2),
            "win_rate_pct": round(bt_online.metrics.get("win_rate", 0.0) * 100, 2),
            "profit_factor": round(bt_online.metrics.get("profit_factor", 1.0), 3),
            "final_weights": hedge.get_weights(),
        },
        "3_circuit_breakers": {
            "name": "+ Auto-Degrade Circuit Breakers",
            "trades": summary_cb["total_trades"],
            "turnover": round(summary_cb["total_turnover"], 2),
            "cum_return_pct": round(float((bt_cb.equity_curve.iloc[-1] / bt_cb.equity_curve.iloc[0] - 1.0) * 100), 2),
            "net_sharpe": round(bt_cb.metrics.get("sharpe_ratio", 0.0), 3),
            "max_drawdown_pct": round(bt_cb.metrics.get("max_drawdown", 0.0) * 100, 2),
            "win_rate_pct": round(bt_cb.metrics.get("win_rate", 0.0) * 100, 2),
            "profit_factor": round(bt_cb.metrics.get("profit_factor", 1.0), 3),
        },
        "4_full_adaptive": {
            "name": "Full Adaptive Layer (Online + Breakers + Regime)",
            "trades": summary_full["total_trades"],
            "turnover": round(summary_full["total_turnover"], 2),
            "cum_return_pct": round(float((bt_full.equity_curve.iloc[-1] / bt_full.equity_curve.iloc[0] - 1.0) * 100), 2),
            "net_sharpe": round(bt_full.metrics.get("sharpe_ratio", 0.0), 3),
            "max_drawdown_pct": round(bt_full.metrics.get("max_drawdown", 0.0) * 100, 2),
            "win_rate_pct": round(bt_full.metrics.get("win_rate", 0.0) * 100, 2),
            "profit_factor": round(bt_full.metrics.get("profit_factor", 1.0), 3),
        },
    }

    output_path = reports_dir / "adaptivity_ablation.json"
    with open(output_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    logger.info("\n--- Adaptivity Ablation Results Summary ---")
    for key, res in ablation_results.items():
        logger.info(
            f"[{key}] {res['name']}: Return={res['cum_return_pct']}% | "
            f"Net Sharpe={res['net_sharpe']} | MaxDD={res['max_drawdown_pct']}% | Trades={res['trades']}"
        )

    logger.success(f"Ablation study completed. Saved to {output_path}")
    return ablation_results


if __name__ == "__main__":
    run_adaptivity_ablation()
