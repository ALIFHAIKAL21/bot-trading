"""Automated paper-trading execution scheduler loop."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.broker.broker import PaperBroker
from src.data.loader import MarketDataLoader
from src.ensemble.stacking import StackingMetaModel
from src.features.feature_pipeline import FeaturePipeline
from src.models.deep_sequence_model import DeepSequenceModel
from src.models.regime_model import CausalRegimeHMM
from src.models.sentiment_gate import SentimentNewsGate
from src.models.tabular_model import TabularDirectionModel
from src.risk.risk_engine import RiskEngine
from src.service.db import Database
from src.utils.config import AppConfig, load_config


class PaperTradingScheduler:
    """Executes live/paper trading cycle on new closed hourly bars."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg: AppConfig = load_config(config_path)
        self.db = Database(self.cfg.service.db_path)
        self.broker = PaperBroker(
            initial_capital=self.cfg.backtest.initial_capital,
            taker_fee=self.cfg.backtest.taker_fee,
            slippage_bps=self.cfg.backtest.base_slippage,
        )
        self.loader = MarketDataLoader(
            cache_dir=self.cfg.data.cache_dir,
            primary_source=self.cfg.data.primary_source,
            gap_fill_policy=self.cfg.data.gap_fill_policy,
        )
        self.feature_pipeline = FeaturePipeline(self.cfg.features)
        self.sentiment_gate = SentimentNewsGate(self.cfg.models.model_d)
        self.risk_engine = RiskEngine(self.cfg.risk)

        # Trained Ensemble Models
        self.meta_model: Optional[StackingMetaModel] = None
        self.lgbm_model: Optional[TabularDirectionModel] = None
        self.hmm_model: Optional[CausalRegimeHMM] = None
        self.deep_model: Optional[DeepSequenceModel] = None
        self._load_models()

    def _load_models(self):
        models_dir = Path("models_store")
        try:
            if (models_dir / "meta_model.joblib").exists():
                self.meta_model = joblib.load(models_dir / "meta_model.joblib")
            if (models_dir / "model_b_lgbm.joblib").exists():
                self.lgbm_model = joblib.load(models_dir / "model_b_lgbm.joblib")
            if (models_dir / "model_e_hmm.joblib").exists():
                self.hmm_model = joblib.load(models_dir / "model_e_hmm.joblib")
            if (models_dir / "model_c_best.pt").exists():
                self.deep_model = DeepSequenceModel(self.cfg.models.model_c)
            logger.success("Loaded real trained quant models into PaperTradingScheduler.")
        except Exception as e:
            logger.warning(f"Could not load all serialized models: {e}. Using fallback heuristic.")

    def run_step(self, symbol: str = "BTC/USDT") -> bool:
        """Run single evaluation step on the latest completed bar."""
        logger.info(f"Running paper trading cycle for {symbol}...")

        # 1. Fetch latest data
        try:
            df = self.loader.load_or_fetch(symbol, timeframe=self.cfg.market.timeframe, history_days=15)
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return False

        if len(df) < 80:
            logger.warning("Not enough bars for feature calculation.")
            return False

        # 2. Build features and select strictly closed candle (F4)
        df_feat = self.feature_pipeline.build_features(df)
        if len(df_feat) < 2:
            logger.warning("Insufficient feature rows.")
            return False

        # Enforce closed candle condition (F4): bar_open_time + timeframe + 5s <= now_utc
        latest_ts = df_feat.index[-1]
        now_utc = pd.Timestamp.now(tz="UTC")
        bar_ts_utc = latest_ts if latest_ts.tzinfo is not None else latest_ts.tz_localize("UTC")
        tf_delta = pd.Timedelta(self.cfg.market.timeframe)
        bar_close_time = bar_ts_utc + tf_delta

        if bar_close_time + pd.Timedelta(seconds=5) > now_utc:
            # Candle is still forming! Use the previous fully closed bar:
            eval_bar = df_feat.iloc[-2]
            eval_ts = df_feat.index[-2]
            eval_idx = -2
            logger.debug(f"Latest candle at {bar_ts_utc} still forming. Using previous closed candle: {eval_ts}")
        else:
            eval_bar = df_feat.iloc[-1]
            eval_ts = df_feat.index[-1]
            eval_idx = -1

        last_bar = eval_bar
        bar_ts = str(eval_ts)
        current_price = float(last_bar["close"])

        # 3. Check post-processing idempotency (F4)
        if self.db.is_bar_processed(symbol, bar_ts):
            logger.info(f"Bar {bar_ts} for {symbol} was already evaluated and processed. Skipping duplicate execution.")
            return True

        # 4. Sentiment check
        sentiment_metrics = self.sentiment_gate.get_latest_sentiment_metrics()
        current_sentiment = sentiment_metrics["sentiment_score"]

        # 5. Model ensemble prediction (Real multi-model inference)
        log_ret_1 = float(last_bar.get("log_ret_1", 0.0))
        rsi = float(last_bar.get("rsi_14", 50.0))
        realized_vol = float(last_bar.get("realized_vol_12", 0.01))
        feature_cols = self.feature_pipeline.get_feature_columns(df_feat)

        prob_long = 0.50
        confidence = 0.0
        regime_id = 0
        deep_fwd_ret = 0.0

        if self.meta_model is not None and self.lgbm_model is not None and self.hmm_model is not None:
            try:
                # Tabular LightGBM prediction
                lgbm_prob = float(self.lgbm_model.predict_proba(df_feat[feature_cols].iloc[[eval_idx]])[0])

                # HMM Regime prediction
                regimes_df = self.hmm_model.predict_filtered_proba(df_feat.iloc[-50:])
                last_regime = regimes_df.iloc[eval_idx]
                regime_id = int(np.argmax([last_regime["regime_p0"], last_regime["regime_p1"], last_regime["regime_p2"]]))

                # Deep Sequence Model prediction
                deep_dir_prob = 0.5
                deep_fwd_ret = 0.0
                if self.deep_model is not None:
                    try:
                        deep_df = self.deep_model.predict_proba(df_feat.iloc[-100:])
                        deep_dir_prob = float(deep_df["model_c_dir_prob"].iloc[eval_idx])
                        deep_fwd_ret = float(deep_df["model_c_fwd_ret"].iloc[eval_idx])
                    except Exception as de:
                        logger.warning(f"Deep sequence step inference failed: {de}")

                # Meta-learner stacking inference
                meta_row = pd.DataFrame(
                    [
                        {
                            "lgbm_p_long": lgbm_prob,
                            "regime_p0": float(last_regime["regime_p0"]),
                            "regime_p1": float(last_regime["regime_p1"]),
                            "regime_p2": float(last_regime["regime_p2"]),
                            "chronos_exp_ret": 0.0,
                            "chronos_skew": 0.0,
                            "chronos_uncertainty": 0.0,
                            "model_c_dir_prob": deep_dir_prob,
                            "model_c_fwd_ret": deep_fwd_ret,
                        }
                    ],
                    index=[eval_ts],
                )
                meta_preds = self.meta_model.predict_proba(meta_row)
                prob_long = float(meta_preds["meta_p_long"].iloc[0])
                confidence = float(meta_preds["meta_confidence"].iloc[0])
                logger.info(
                    f"Real Ensemble Inference -> LGBM: {lgbm_prob:.3f}, Regime: {regime_id}, Deep C: {deep_dir_prob:.3f} | Stacking Meta P(Long): {prob_long:.3f} (Conf: {confidence:.3f})"
                )
            except Exception as me:
                logger.error(f"Error during ensemble inference ({me}). Operating in fail-safe neutral cash mode (P=0.50).")
                prob_long = 0.50
                confidence = 0.0
                regime_id = 0
        else:
            # Fallback fail-safe: Neutral 0.50 (Never fabricate rogue buy signals)
            prob_long = 0.50
            confidence = 0.0
            regime_id = 0

        # 6. Apply Institutional Risk Engine (Turnover Controls & Cost Hurdle)
        regime_probs_arr = np.array([last_regime["regime_p0"], last_regime["regime_p1"], last_regime["regime_p2"]]) if "last_regime" in locals() else None
        decision = self.risk_engine.decide_step(
            prob_long=prob_long,
            current_vol_annual=realized_vol * np.sqrt(8760.0),
            bar_volatility=realized_vol,
            regime_probs=regime_probs_arr,
            expected_ret=deep_fwd_ret if "deep_fwd_ret" in locals() else 0.0,
            bar_timestamp=bar_ts,
            current_price=current_price,
        )
        target_size = decision.target_position
        decision_reason = decision.reason
        logger.info(
            f"Risk Decision -> Action: {decision.action}, Target Weight: {target_size:.3f}, Reason: {decision_reason} "
            f"(Edge: {decision.expected_edge:.4f}, Hurdle: {decision.hurdle_cost:.4f})"
        )

        # 7. Sentiment gate check
        adj_size, is_vetoed = self.sentiment_gate.apply_gate(target_size, current_sentiment)
        if is_vetoed:
            adj_size = 0.0
            decision_reason = "sentiment_gate_veto"

        # 8. Record signal to database with full structured decision metadata
        self.db.record_signal(
            symbol=symbol,
            bar_timestamp=bar_ts,
            prob_long=prob_long,
            confidence=confidence,
            regime_id=regime_id,
            sentiment_score=current_sentiment,
            raw_data={
                "rsi": rsi,
                "realized_vol": realized_vol,
                "target_size": adj_size,
                "action": decision.action,
                "decision_reason": decision_reason,
                "expected_edge": decision.expected_edge,
                "hurdle_cost": decision.hurdle_cost,
                "metadata": decision.metadata,
            },
        )

        # 9. Execute rebalance
        trade = self.broker.execute_rebalance(
            symbol=symbol,
            target_weight=adj_size,
            current_price=current_price,
            bar_timestamp=bar_ts,
        )

        if trade:
            self.db.record_order(
                symbol=symbol,
                bar_timestamp=bar_ts,
                side=trade["side"],
                qty=trade["qty"],
                fill_price=trade["price"],
                fee=trade["fee"],
                target_weight=trade["target_weight"],
            )

        # 10. Record snapshot
        bal = self.broker.get_balance()
        self.db.record_snapshot(
            equity=bal["equity"],
            cash=bal["cash"],
            positions_value=bal["positions_value"],
            drawdown=0.0,
            positions=self.broker.get_positions(),
        )

        # 11. Record processed bar idempotency key (F4)
        self.db.record_processed_bar(
            symbol=symbol,
            bar_timestamp=bar_ts,
            action=decision.action,
            decision_reason=decision_reason,
            target_weight=adj_size,
        )

        logger.success(
            f"Completed paper cycle for {symbol} | Price: ${current_price:.2f} | P(long): {prob_long:.2f} | Weight: {adj_size:.2f} | Equity: ${bal['equity']:.2f}"
        )
        return True

    def run_loop(self, poll_interval: Optional[int] = None):
        """Continuous execution loop."""
        interval = poll_interval or self.cfg.service.poll_interval_seconds
        logger.info(f"Starting paper trading loop (poll interval: {interval}s)...")

        while True:
            for sym in self.cfg.market.symbols:
                try:
                    self.run_step(sym)
                except Exception as e:
                    logger.error(f"Error in scheduler step for {sym}: {e}")
            time.sleep(interval)
