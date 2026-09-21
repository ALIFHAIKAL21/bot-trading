"""Backtest runner comparing Stacking Meta-Model against baselines, single models, and ablation."""

import argparse
import json
import sys
from pathlib import Path

# Add repository root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.backtester import EventConsistentBacktester
from src.backtest.metrics import (
    bootstrap_sharpe_ci,
    calculate_deflated_sharpe_ratio,
    run_monte_carlo_random_baseline,
)
from src.data.loader import MarketDataLoader
from src.ensemble.purged_cv import PurgedWalkForwardCV
from src.features.feature_pipeline import FeaturePipeline
from src.labels.triple_barrier import TripleBarrierLabeler
from src.models.chronos_model import ChronosBoltForecaster
from src.models.deep_sequence_model import DeepSequenceModel
from src.risk.risk_engine import RiskEngine
from src.utils.config import AppConfig, load_config


def run_full_backtest(config_path: str = "config/config.yaml") -> Dict:
    """Execute complete institutional-grade backtest with realistic costs and baselines."""
    cfg: AppConfig = load_config(config_path)
    reports_dir = Path("reports")
    models_dir = Path("models_store")
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RUNNING EVENT-CONSISTENT BACKTEST & STATISTICAL BENCHMARKS")
    logger.info("=" * 60)

    # 1. Load Data
    # 1. Load Data strictly within VALIDATION partition
    primary_sym = cfg.market.symbols[0]
    loader = MarketDataLoader(cache_dir=cfg.data.cache_dir)
    feature_pipeline = FeaturePipeline(cfg.features)
    labeler = TripleBarrierLabeler(cfg.labels)

    from src.data.holdout_guard import partition_dataset
    df_raw = loader.load_or_fetch(
        primary_sym, timeframe=cfg.market.timeframe, history_days=cfg.market.history_days
    )
    df_val_raw, _, _ = partition_dataset(df_raw)

    df_feat = feature_pipeline.build_features(df_val_raw)
    df_labeled = labeler.label_barriers(df_feat)
    df_clean = df_labeled.iloc[72:].copy()
    feature_cols = feature_pipeline.get_feature_columns(df_clean)

    # Get untouched validation holdout test set (last 15% of validation bars)
    cv = PurgedWalkForwardCV(n_splits=cfg.ensemble.cv_splits, horizon=cfg.market.horizon)
    _, holdout_indices = cv.get_holdout_split(df_clean)
    df_test = df_clean.iloc[holdout_indices].copy()
    logger.info(f"Validation Holdout Period: {df_test.index.min()} to {df_test.index.max()} ({len(df_test)} bars)")

    # 2. Load Models
    lgbm = joblib.load(models_dir / "model_b_lgbm.joblib")
    hmm = joblib.load(models_dir / "model_e_hmm.joblib")
    meta_model = joblib.load(models_dir / "meta_model.joblib")

    chronos = ChronosBoltForecaster(cfg.models.model_a, cache_dir=cfg.data.cache_dir)
    chronos_preds = chronos.predict_rolling(df_clean, symbol=primary_sym, stride=6).iloc[holdout_indices]

    deep_model = DeepSequenceModel(cfg.models.model_c)
    checkpoint_path = models_dir / "model_c_best.pt"
    if checkpoint_path.exists():
        deep_model.load(checkpoint_path, feature_cols)
    deep_preds = deep_model.predict_proba(df_test)

    # Model E Regimes
    regimes = hmm.predict_filtered_proba(df_test)

    # Model B Probabilities
    lgbm_probs = lgbm.predict_proba(df_test[feature_cols])

    # 3. Assemble Meta-Model Features
    meta_input = pd.DataFrame(
        {
            "lgbm_p_long": lgbm_probs,
            "regime_p0": regimes["regime_p0"].values,
            "regime_p1": regimes["regime_p1"].values,
            "regime_p2": regimes["regime_p2"].values,
            "chronos_exp_ret": chronos_preds["chronos_exp_ret"].values,
            "chronos_skew": chronos_preds["chronos_skew"].values,
            "chronos_uncertainty": chronos_preds["chronos_uncertainty"].values,
            "model_c_dir_prob": deep_preds["model_c_dir_prob"].values,
            "model_c_fwd_ret": deep_preds["model_c_fwd_ret"].values,
        },
        index=df_test.index,
    )

    meta_preds = meta_model.predict_proba(meta_input)
    meta_p_long = meta_preds["meta_p_long"]
    simple_avg_p_long = meta_preds["simple_avg_p_long"]

    # 4. Generate Positions via Risk Engine
    risk_engine = RiskEngine(cfg.risk)
    meta_positions = risk_engine.generate_positions_vectorized(
        meta_probs=meta_p_long,
        realized_vols=df_test["realized_vol_12"],
        regime_df=regimes,
        close_prices=df_test["close"],
    )

    # Baseline positions
    bnh_positions = pd.Series(1.0, index=df_test.index)
    simple_avg_positions = risk_engine.generate_positions_vectorized(
        meta_probs=pd.Series(simple_avg_p_long, index=df_test.index),
        realized_vols=df_test["realized_vol_12"],
        regime_df=regimes,
        close_prices=df_test["close"],
    )
    lgbm_positions = risk_engine.generate_positions_vectorized(
        meta_probs=pd.Series(lgbm_probs, index=df_test.index),
        realized_vols=df_test["realized_vol_12"],
        regime_df=regimes,
        close_prices=df_test["close"],
    )

    # Time-Shuffled Signal Baseline (F12)
    np.random.seed(42)
    shuffled_arr = meta_positions.values.copy()
    np.random.shuffle(shuffled_arr)
    shuffled_positions = pd.Series(shuffled_arr, index=df_test.index)

    # 5. Run Backtester on all strategies
    backtester = EventConsistentBacktester(cfg.backtest)

    res_meta = backtester.run(df_test, meta_positions)
    res_bnh = backtester.run(df_test, bnh_positions)
    res_avg = backtester.run(df_test, simple_avg_positions)
    res_lgbm = backtester.run(df_test, lgbm_positions)
    res_shuffled = backtester.run(df_test, shuffled_positions)

    # 6. Statistical Significance & Baselines
    # Deflated Sharpe Ratio
    dsr = calculate_deflated_sharpe_ratio(
        sharpe_hat=res_meta.metrics.get("sharpe_ratio", 0.0),
        n_trials=15,
        returns=res_meta.returns.values,
    )

    # Bootstrap 95% CI on Sharpe
    ci_low, ci_high = bootstrap_sharpe_ci(res_meta.returns.values, n_bootstraps=1000)

    # Monte Carlo Random Baseline
    bar_returns = np.diff(df_test["close"].values, prepend=df_test["close"].values[0]) / (
        df_test["close"].values + 1e-9
    )
    mc_bench = run_monte_carlo_random_baseline(bar_returns, n_sims=cfg.backtest.monte_carlo_runs)

    # 7. Ablation Analysis: Systematically drop each model
    ablation_results = {}
    models_to_ablate = [
        ("minus_model_a_chronos", ["chronos_exp_ret", "chronos_skew", "chronos_uncertainty"]),
        ("minus_model_b_lgbm", ["lgbm_p_long"]),
        ("minus_model_c_deep", ["model_c_dir_prob", "model_c_fwd_ret"]),
        ("minus_model_e_regime", ["regime_p0", "regime_p1", "regime_p2"]),
    ]

    for abl_name, drop_cols in models_to_ablate:
        abl_input = meta_input.copy()
        abl_input[drop_cols] = 0.0  # Zero out model signals
        abl_probs = meta_model.predict_proba(abl_input)["meta_p_long"]
        abl_pos = risk_engine.generate_positions_vectorized(
            meta_probs=abl_probs,
            realized_vols=df_test["realized_vol_12"],
            regime_df=regimes if "minus_model_e_regime" not in abl_name else None,
        )
        abl_res = backtester.run(df_test, abl_pos)
        ablation_results[abl_name] = {
            "sharpe_ratio": round(abl_res.metrics.get("sharpe_ratio", 0.0), 3),
            "cagr": round(abl_res.metrics.get("cagr", 0.0), 4),
            "max_drawdown": round(abl_res.metrics.get("max_drawdown", 0.0), 4),
        }

    # 8. Compile Comprehensive Results Manifest
    results = {
        "market": {"symbol": primary_sym, "bars": len(df_test)},
        "strategies": {
            "meta_model_stacked": {
                **{k: round(v, 4) for k, v in res_meta.metrics.items()},
                "deflated_sharpe_ratio": round(dsr, 4),
                "sharpe_95ci_low": round(ci_low, 3),
                "sharpe_95ci_high": round(ci_high, 3),
            },
            "simple_average_ensemble": {k: round(v, 4) for k, v in res_avg.metrics.items()},
            "best_single_lgbm": {k: round(v, 4) for k, v in res_lgbm.metrics.items()},
            "time_shuffled_signal_baseline": {k: round(v, 4) for k, v in res_shuffled.metrics.items()},
            "buy_and_hold_benchmark": {k: round(v, 4) for k, v in res_bnh.metrics.items()},
            "monte_carlo_random_baseline": {k: round(v, 4) for k, v in mc_bench.items()},
        },
        "ablation_table": ablation_results,
    }

    # Save to reports
    with open(reports_dir / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save equity curve CSV
    equity_df = pd.DataFrame(
        {
            "Meta_Model": res_meta.equity_curve,
            "Simple_Avg": res_avg.equity_curve,
            "LGBM_Single": res_lgbm.equity_curve,
            "Buy_and_Hold": res_bnh.equity_curve,
        }
    )
    equity_df.to_csv(reports_dir / "equity_curves.csv")

    logger.success(f"Backtest completed successfully. Results saved to {reports_dir}/backtest_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    run_full_backtest(args.config)
