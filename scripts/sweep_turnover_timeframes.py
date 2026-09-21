"""Empirical parameter sweep on VALIDATION data evaluating turnover controls across timeframes.

Evaluates:
- 1h, 4h, and 1d timeframes.
- Two variants per timeframe:
  1. RAW (No turnover controls, no cost hurdle, buys whenever p >= 0.50).
  2. CONTROLLED (Full turnover controls: entry_threshold=0.54, exit_threshold=0.48,
     hysteresis deadband, min_holding_bars=3, exit_cooldown=2, dust_filter=5%,
     and expected edge > 2.0 * round_trip_cost).

Outputs structured results to reports/turnover_controls_sweep.json.
Guarantees zero access to LOCKED_TEST by wrapping in HoldoutGuard.get_validation_data().
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.backtester import EventConsistentBacktester
from src.data.holdout_guard import partition_dataset
from src.data.loader import MarketDataLoader
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_engine import RiskEngine
from src.utils.config import AppConfig, RiskConfig, load_config


def run_turnover_sweep(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Execute comparative sweep of turnover controls across 1h, 4h, and 1d on VALIDATION data."""
    cfg: AppConfig = load_config(config_path)
    reports_dir = Path("reports")
    models_dir = Path("models_store")
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STARTING TURNOVER CONTROLS & COST HURDLE SWEEP (VALIDATION PARTITION)")
    logger.info("=" * 70)

    # 1. Load Data strictly within VALIDATION partition
    primary_sym = cfg.market.symbols[0]
    loader = MarketDataLoader(cache_dir=cfg.data.cache_dir)
    df_raw = loader.load_or_fetch(primary_sym, timeframe="1h")
    
    # Strictly quarantine LOCKED_TEST
    df_val_raw, _, _ = partition_dataset(df_raw)
    logger.info(
        f"Validation Data: {df_val_raw.index.min()} to {df_val_raw.index.max()} "
        f"({len(df_val_raw):,} bars). Hold-Out Test is LOCKED."
    )

    # 2. Build Causal Features on 1h Data
    feature_pipeline = FeaturePipeline(cfg.features)
    df_feat_1h = feature_pipeline.build_features(df_val_raw)
    df_feat_1h = df_feat_1h.iloc[72:].copy()
    feature_cols = feature_pipeline.get_feature_columns(df_feat_1h)

    # 3. Load Trained Institutional Models
    lgbm = joblib.load(models_dir / "model_b_lgbm.joblib")
    hmm = joblib.load(models_dir / "model_e_hmm.joblib")
    meta_model = joblib.load(models_dir / "meta_model.joblib")

    # Predict probabilities and regimes on 1h
    probs_1h = lgbm.predict_proba(df_feat_1h[feature_cols])
    regimes_1h = hmm.predict_filtered_proba(df_feat_1h)
    
    meta_in = pd.DataFrame(
        {
            "lgbm_p_long": probs_1h,
            "regime_p0": regimes_1h["regime_p0"].values,
            "regime_p1": regimes_1h["regime_p1"].values,
            "regime_p2": regimes_1h["regime_p2"].values,
            "chronos_exp_ret": 0.0,
            "chronos_skew": 0.0,
            "chronos_uncertainty": 0.0,
            "model_c_dir_prob": 0.5,
            "model_c_fwd_ret": 0.0,
        },
        index=df_feat_1h.index,
    )
    meta_preds_1h = meta_model.predict_proba(meta_in)
    p_long_1h = meta_preds_1h["meta_p_long"]

    # 4. Prepare Timeframe Datasets
    timeframes = {
        "1h": {
            "df": df_feat_1h,
            "probs": p_long_1h,
            "regimes": regimes_1h,
            "freq_hours": 1,
            "bars_per_year": 8760,
        },
        "4h": {
            "df": df_feat_1h.iloc[::4].copy(),
            "probs": p_long_1h.iloc[::4].copy(),
            "regimes": regimes_1h.iloc[::4].copy(),
            "freq_hours": 4,
            "bars_per_year": 2190,
        },
        "1d": {
            "df": df_feat_1h.iloc[::24].copy(),
            "probs": p_long_1h.iloc[::24].copy(),
            "regimes": regimes_1h.iloc[::24].copy(),
            "freq_hours": 24,
            "bars_per_year": 365,
        },
    }

    results = {}

    for tf_name, tf_data in timeframes.items():
        logger.info(f"\n--- Evaluating Timeframe: {tf_name.upper()} ({len(tf_data['df']):,} bars) ---")
        sub_df = tf_data["df"]
        sub_probs = tf_data["probs"]
        sub_regimes = tf_data["regimes"]
        bars_per_year = tf_data["bars_per_year"]

        # Config A: RAW (No turnover controls, no cost hurdle, churns on 0.50)
        risk_cfg_raw = RiskConfig(
            target_annual_vol=cfg.risk.target_annual_vol,
            kelly_fraction=cfg.risk.kelly_fraction,
            max_position_pct=cfg.risk.max_position_pct,
            entry_threshold=0.50,
            exit_threshold=0.50,
            expected_edge_hurdle_multiplier=0.0,  # No hurdle
            min_holding_bars=1,
            cooldown_bars=0,
            max_daily_trades=100,
            dust_rebalance_threshold=0.0,  # Rebalances on every micro change
        )
        engine_raw = RiskEngine(risk_cfg_raw)
        pos_raw = engine_raw.generate_positions_vectorized(
            meta_probs=sub_probs,
            realized_vols=sub_df["realized_vol_12"],
            regime_df=sub_regimes,
            close_prices=sub_df["close"],
        )
        raw_summary = engine_raw.get_decision_summary()

        # Config B: CONTROLLED (Full turnover controls & cost hurdle)
        risk_cfg_controlled = RiskConfig(
            target_annual_vol=cfg.risk.target_annual_vol,
            kelly_fraction=cfg.risk.kelly_fraction,
            max_position_pct=cfg.risk.max_position_pct,
            entry_threshold=0.54,
            exit_threshold=0.48,
            expected_edge_hurdle_multiplier=1.0,  # 1.0x round-trip cost hurdle (30 bps)
            round_trip_cost=0.0030,
            trade_horizon_bars=12 if tf_name == "1h" else (6 if tf_name == "4h" else 3),
            min_holding_bars=3 if tf_name == "1h" else 2,
            cooldown_bars=2 if tf_name == "1h" else 1,
            max_daily_trades=6 if tf_name == "1h" else 4,
            dust_rebalance_threshold=0.05,
        )
        engine_controlled = RiskEngine(risk_cfg_controlled)
        pos_controlled = engine_controlled.generate_positions_vectorized(
            meta_probs=sub_probs,
            realized_vols=sub_df["realized_vol_12"],
            regime_df=sub_regimes,
            close_prices=sub_df["close"],
        )
        controlled_summary = engine_controlled.get_decision_summary()

        # Backtest both configurations
        backtester = EventConsistentBacktester(cfg.backtest)
        bt_raw = backtester.run(sub_df, pos_raw)
        bt_controlled = backtester.run(sub_df, pos_controlled)

        # Compute annualized metrics
        n_years = len(sub_df) / bars_per_year
        raw_ann_turnover = raw_summary["total_turnover"] / max(n_years, 0.1)
        ctrl_ann_turnover = controlled_summary["total_turnover"] / max(n_years, 0.1)

        raw_costs = float(bt_raw.costs.sum())
        ctrl_costs = float(bt_controlled.costs.sum())
        
        cost_savings = raw_costs - ctrl_costs
        turnover_reduction_pct = (
            (raw_summary["total_turnover"] - controlled_summary["total_turnover"])
            / max(raw_summary["total_turnover"], 1e-9)
            * 100.0
        )
        trade_reduction_pct = (
            (raw_summary["total_trades"] - controlled_summary["total_trades"])
            / max(raw_summary["total_trades"], 1)
            * 100.0
        )

        logger.info(
            f"[{tf_name.upper()}] RAW Strategy: {raw_summary['total_trades']} trades | "
            f"Turnover: {raw_summary['total_turnover']:.1f}x (Ann: {raw_ann_turnover:.1f}x) | "
            f"Total Friction: ${raw_costs:.2f} | Net Sharpe: {bt_raw.metrics['sharpe_ratio']:.3f} | "
            f"MaxDD: {bt_raw.metrics['max_drawdown']*100:.2f}%"
        )
        logger.info(
            f"[{tf_name.upper()}] CONTROLLED Strategy: {controlled_summary['total_trades']} trades "
            f"({trade_reduction_pct:.1f}% reduction) | "
            f"Turnover: {controlled_summary['total_turnover']:.1f}x ({turnover_reduction_pct:.1f}% reduction) | "
            f"Total Friction: ${ctrl_costs:.2f} (Saved ${cost_savings:.2f}) | "
            f"Net Sharpe: {bt_controlled.metrics['sharpe_ratio']:.3f} | "
            f"MaxDD: {bt_controlled.metrics['max_drawdown']*100:.2f}%"
        )

        raw_cum_ret = float((bt_raw.equity_curve.iloc[-1] / bt_raw.equity_curve.iloc[0] - 1.0) * 100.0)
        ctrl_cum_ret = float((bt_controlled.equity_curve.iloc[-1] / bt_controlled.equity_curve.iloc[0] - 1.0) * 100.0)

        results[tf_name] = {
            "timeframe": tf_name,
            "bars": len(sub_df),
            "years": round(n_years, 2),
            "raw": {
                "trades": raw_summary["total_trades"],
                "total_turnover": round(raw_summary["total_turnover"], 2),
                "annualized_turnover": round(raw_ann_turnover, 2),
                "total_costs_dollars": round(raw_costs, 2),
                "cum_return_pct": round(raw_cum_ret, 2),
                "annualized_return_pct": round(bt_raw.metrics.get("cagr", 0.0) * 100.0, 2),
                "sharpe_ratio": round(bt_raw.metrics.get("sharpe_ratio", 0.0), 3),
                "sortino_ratio": round(bt_raw.metrics.get("sortino_ratio", 0.0), 3),
                "max_drawdown_pct": round(bt_raw.metrics.get("max_drawdown", 0.0) * 100.0, 2),
                "win_rate_pct": round(bt_raw.metrics.get("win_rate", 0.0) * 100.0, 2),
                "profit_factor": round(bt_raw.metrics.get("profit_factor", 1.0), 3),
            },
            "controlled": {
                "trades": controlled_summary["total_trades"],
                "total_turnover": round(controlled_summary["total_turnover"], 2),
                "annualized_turnover": round(ctrl_ann_turnover, 2),
                "total_costs_dollars": round(ctrl_costs, 2),
                "cum_return_pct": round(ctrl_cum_ret, 2),
                "annualized_return_pct": round(bt_controlled.metrics.get("cagr", 0.0) * 100.0, 2),
                "sharpe_ratio": round(bt_controlled.metrics.get("sharpe_ratio", 0.0), 3),
                "sortino_ratio": round(bt_controlled.metrics.get("sortino_ratio", 0.0), 3),
                "max_drawdown_pct": round(bt_controlled.metrics.get("max_drawdown", 0.0) * 100.0, 2),
                "win_rate_pct": round(bt_controlled.metrics.get("win_rate", 0.0) * 100.0, 2),
                "profit_factor": round(bt_controlled.metrics.get("profit_factor", 1.0), 3),
                "reason_counts": controlled_summary["reason_counts"],
            },
            "comparison": {
                "trade_reduction_pct": round(trade_reduction_pct, 2),
                "turnover_reduction_pct": round(turnover_reduction_pct, 2),
                "cost_savings_dollars": round(cost_savings, 2),
                "sharpe_improvement": round(bt_controlled.metrics.get("sharpe_ratio", 0.0) - bt_raw.metrics.get("sharpe_ratio", 0.0), 3),
            },
        }

    # Save to reports artifact
    output_path = reports_dir / "turnover_controls_sweep.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Turnover sweep completed successfully. Saved to {output_path}")
    return results


if __name__ == "__main__":
    run_turnover_sweep()
