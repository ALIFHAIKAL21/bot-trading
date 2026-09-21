"""Unified 24/7 Bot Controller for Web UI Interactive Management.

Enables full control directly from Streamlit dashboard:
- Start / Stop 24/7 background worker with 1 click.
- Switch between 1m Scalping, 5m Scalping, and 1h Swing.
- Run instant simulation / replay on demand.
- Thread-safe state tracking (heartbeat, active positions, logs).
"""

import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.broker.broker import PaperBroker
from src.data.loader import MarketDataLoader
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_engine import RiskEngine
from src.service.db import Database
from src.utils.config import RiskConfig, load_config

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_risk_config_for_timeframe(timeframe: str = "5m") -> RiskConfig:
    """Return calibrated RiskConfig for 1m, 5m, or 1h."""
    if timeframe == "1m":
        return RiskConfig(
            entry_threshold=0.515,
            exit_threshold=0.485,
            min_holding_bars=3,            # 3 minutes
            cooldown_bars=1,               # 1 minute
            max_daily_trades=50,
            round_trip_cost=0.0020,        # 20 bps
            expected_edge_hurdle_multiplier=1.0,
            trade_horizon_bars=5,          # 5 minutes
            target_annual_vol=0.35,
            kelly_fraction=0.30,
            max_position_pct=0.35,
            dust_rebalance_threshold=0.02,
            hard_stop_loss_pct=0.015,       # 1.5% scalping SL
        )
    elif timeframe == "5m":
        return RiskConfig(
            entry_threshold=0.52,
            exit_threshold=0.48,
            min_holding_bars=2,            # 10 minutes
            cooldown_bars=1,               # 5 minutes
            max_daily_trades=24,
            round_trip_cost=0.0025,        # 25 bps
            expected_edge_hurdle_multiplier=1.0,
            trade_horizon_bars=6,          # 30 minutes
            target_annual_vol=0.30,
            kelly_fraction=0.35,
            max_position_pct=0.40,
            dust_rebalance_threshold=0.03,
            hard_stop_loss_pct=0.02,        # 2% scalping SL
        )
    else:
        # 1h Swing
        return RiskConfig(
            entry_threshold=0.54,
            exit_threshold=0.48,
            min_holding_bars=3,            # 3 hours
            cooldown_bars=2,               # 2 hours
            max_daily_trades=6,
            round_trip_cost=0.0030,        # 30 bps
            expected_edge_hurdle_multiplier=2.0,
            trade_horizon_bars=12,         # 12 hours
            target_annual_vol=0.25,
            kelly_fraction=0.25,
            max_position_pct=0.40,
            dust_rebalance_threshold=0.05,
            hard_stop_loss_pct=0.03,        # 3% swing SL
        )


class BotController:
    """Singleton controller managing background 24/7 worker threads."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.is_running = False
        self.should_stop = False
        self.thread: Optional[threading.Thread] = None
        self.timeframe = "5m"
        self.symbol = "BTC/USDT"
        self.last_heartbeat = None
        self.last_status_msg = "Bot initialized (Idle)"
        self.last_bar_evaluated = None
        self.last_prob = 0.50
        self.last_action = "FLAT"
        self.last_reason = "Initialized"

    def get_status(self) -> Dict:
        return {
            "is_running": self.is_running,
            "timeframe": self.timeframe,
            "symbol": self.symbol,
            "last_heartbeat": self.last_heartbeat.strftime("%Y-%m-%d %H:%M:%S UTC") if self.last_heartbeat else "Never",
            "status_msg": self.last_status_msg,
            "last_bar": self.last_bar_evaluated or "None",
            "last_prob": self.last_prob,
            "last_action": self.last_action,
            "last_reason": self.last_reason,
        }

    def start(self, timeframe: str = "5m", symbol: str = "BTC/USDT"):
        with self._lock:
            if self.is_running:
                logger.info("Bot is already running.")
                return True

            self.timeframe = timeframe
            self.symbol = symbol
            self.should_stop = False
            self.is_running = True
            self.last_heartbeat = datetime.now(timezone.utc)
            self.last_status_msg = f"Starting 24/7 {timeframe} worker for {symbol}..."
            self.thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.thread.start()
            logger.info(f"BotController started background thread for {symbol} ({timeframe}).")
            return True

    def stop(self):
        with self._lock:
            if not self.is_running:
                return True
            self.should_stop = True
            self.last_status_msg = "Stopping bot worker..."
            logger.info("BotController signaling stop to worker thread.")
            return True

    def switch_timeframe(self, new_timeframe: str, symbol: Optional[str] = None):
        """Cleanly stop the current worker and restart with new timeframe."""
        with self._lock:
            self.should_stop = True
            self.is_running = False
        time.sleep(0.5)
        return self.start(timeframe=new_timeframe, symbol=symbol or self.symbol)

    def _worker_loop(self):
        fatal_e = None
        try:
            logger.info(f"Worker thread active for {self.symbol} ({self.timeframe}).")
            self.last_heartbeat = datetime.now(timezone.utc)
            self.last_status_msg = f"Connecting models and data feeds for {self.symbol} ({self.timeframe})..."
            cfg = load_config(str(ROOT_DIR / "config" / "config.yaml"))
            db = Database(str(ROOT_DIR / cfg.service.db_path))
            broker = PaperBroker(initial_capital=10000.0, taker_fee=0.0010, slippage_bps=0.0005)
            risk_cfg = get_risk_config_for_timeframe(self.timeframe)
            risk_engine = RiskEngine(risk_cfg)
            loader = MarketDataLoader(cache_dir=str(ROOT_DIR / "data" / "cache"))
            pipeline = FeaturePipeline(cfg.features)

            models_dir = ROOT_DIR / "models_store"
            lgbm_model = joblib.load(models_dir / "model_b_lgbm.joblib")
            hmm_model = joblib.load(models_dir / "model_e_hmm.joblib") if (models_dir / "model_e_hmm.joblib").exists() else None

            tf_minutes = 1 if self.timeframe == "1m" else (5 if self.timeframe == "5m" else 60)
            poll_sec = 5 if self.timeframe == "1m" else (10 if self.timeframe == "5m" else 30)

            # Volatility annualization factor:
            # 1m: 365*24*60 = 525,600
            # 5m: 365*24*12 = 105,120
            # 1h: 365*24 = 8,760
            ann_factor = np.sqrt(525600.0 if self.timeframe == "1m" else (105120.0 if self.timeframe == "5m" else 8760.0))

            while not self.should_stop:
                try:
                    self.last_heartbeat = datetime.now(timezone.utc)
                    self.last_status_msg = f"Listening for closed {self.timeframe} candles on {self.symbol}..."

                    # Fetch recent candles
                    days = 1 if self.timeframe == "1m" else (3 if self.timeframe == "5m" else 14)
                    df_raw = loader.load_or_fetch(self.symbol, timeframe=self.timeframe, history_days=days, force_refresh=True)
                    if df_raw.empty:
                        time.sleep(poll_sec)
                        continue

                    df_feat = pipeline.build_features(df_raw)
                    feature_cols = pipeline.get_feature_columns(df_feat)

                    latest_bar_ts = df_feat.index[-1]
                    if latest_bar_ts.tzinfo is None:
                        latest_bar_ts = latest_bar_ts.replace(tzinfo=timezone.utc)

                    candle_close_utc = latest_bar_ts + timedelta(minutes=tf_minutes)
                    now_utc = datetime.now(timezone.utc)

                    # Check if forming or closed
                    if candle_close_utc + timedelta(seconds=5) > now_utc:
                        eval_idx = -2
                        eval_bar = df_feat.iloc[-2]
                        eval_ts = str(df_feat.index[-2])
                    else:
                        eval_idx = -1
                        eval_bar = df_feat.iloc[-1]
                        eval_ts = str(df_feat.index[-1])

                    self.last_bar_evaluated = eval_ts[:19]

                    # Check idempotency
                    if db.is_bar_processed(self.symbol, eval_ts):
                        self.last_status_msg = f"Candle {eval_ts[:19]} processed. Waiting for next close..."
                        time.sleep(poll_sec)
                        continue

                    current_price = float(eval_bar["close"])
                    vol_bar = float(eval_bar.get("realized_vol_12", 0.003))
                    vol_annual = vol_bar * ann_factor

                    # AI Inference
                    p_long = float(lgbm_model.predict_proba(df_feat[feature_cols].iloc[[eval_idx]])[0])
                    self.last_prob = p_long

                    regime_probs = None
                    regime_id = 0
                    if hmm_model is not None:
                        try:
                            r_df = hmm_model.predict_filtered_proba(df_feat.iloc[-50:])
                            last_r = r_df.iloc[eval_idx]
                            regime_probs = np.array([last_r["regime_p0"], last_r["regime_p1"], last_r["regime_p2"]])
                            regime_id = int(np.argmax(regime_probs))
                        except Exception:
                            pass

                    decision = risk_engine.decide_step(
                        prob_long=p_long,
                        current_vol_annual=vol_annual,
                        bar_volatility=vol_bar,
                        regime_probs=regime_probs,
                        bar_timestamp=eval_ts,
                        current_price=current_price,
                    )

                    self.last_action = decision.action
                    self.last_reason = decision.reason
                    self.last_status_msg = f"Evaluated {eval_ts[:19]} | Action: {decision.action} | P(Long): {p_long:.3f}"

                    # Record signal
                    db.record_signal(
                        symbol=self.symbol,
                        bar_timestamp=eval_ts,
                        prob_long=p_long,
                        confidence=abs(p_long - 0.5) * 2.0,
                        regime_id=regime_id,
                        sentiment_score=0.0,
                        raw_data={
                            "action": decision.action,
                            "target_size": decision.target_position,
                            "decision_reason": decision.reason,
                            "expected_edge": decision.expected_edge,
                            "hurdle_cost": decision.hurdle_cost,
                            "timeframe": self.timeframe,
                        },
                    )

                    # Execute order if action is BUY or SELL
                    if decision.action in ("BUY", "SELL"):
                        trade = broker.execute_rebalance(
                            symbol=self.symbol,
                            target_weight=decision.target_position,
                            current_price=current_price,
                            bar_timestamp=eval_ts,
                        )
                        if trade:
                            db.record_order(
                                symbol=self.symbol,
                                bar_timestamp=eval_ts,
                                side=trade["side"],
                                qty=trade["qty"],
                                fill_price=trade["price"],
                                fee=trade["fee"],
                                target_weight=trade["target_weight"],
                            )
                            logger.success(f"24/7 Bot Executed {trade['side']} {self.symbol} @ ${trade['price']:,.2f}")

                    bal = broker.get_balance()
                    db.record_snapshot(
                        equity=bal["equity"],
                        cash=bal["cash"],
                        positions_value=bal["positions_value"],
                        drawdown=0.0,
                        positions=broker.get_positions(),
                    )

                    db.record_processed_bar(
                        symbol=self.symbol,
                        bar_timestamp=eval_ts,
                        action=decision.action,
                        decision_reason=decision.reason,
                        target_weight=decision.target_position,
                    )

                except Exception as e:
                    logger.error(f"Error in 24/7 bot loop: {e}")
                    self.last_status_msg = f"Error: {e}"

                time.sleep(poll_sec)

        except Exception as fatal_e:
            logger.error(f"Fatal worker exception: {fatal_e}")
            self.last_status_msg = f"Fatal error: {fatal_e}"
        finally:
            self.is_running = False
            if not fatal_e:
                self.last_status_msg = "Bot paused by user."
            logger.info("Worker thread exited cleanly.")


def run_interactive_replay(timeframe: str = "5m", symbol: str = "BTC/USDT", n_bars: int = 300) -> Dict:
    """Run an instant historical replay directly from the UI and return execution stats."""
    cfg = load_config(str(ROOT_DIR / "config" / "config.yaml"))
    db = Database(str(ROOT_DIR / cfg.service.db_path))
    loader = MarketDataLoader(cache_dir=str(ROOT_DIR / "data" / "cache"))
    pipeline = FeaturePipeline(cfg.features)

    # Fetch data
    days = 2 if timeframe == "1m" else (7 if timeframe == "5m" else 30)
    df_raw = loader.load_or_fetch(symbol, timeframe=timeframe, history_days=days)
    if len(df_raw) < 72:
        return {"error": "Not enough candle data to evaluate features"}

    df_feat = pipeline.build_features(df_raw)
    feature_cols = pipeline.get_feature_columns(df_feat)
    df_eval = df_feat.iloc[-n_bars:].copy()

    models_dir = ROOT_DIR / "models_store"
    lgbm_model = joblib.load(models_dir / "model_b_lgbm.joblib")
    hmm_model = joblib.load(models_dir / "model_e_hmm.joblib") if (models_dir / "model_e_hmm.joblib").exists() else None

    probs = lgbm_model.predict_proba(df_eval[feature_cols])
    regimes_df = None
    if hmm_model is not None:
        try:
            regimes_df = hmm_model.predict_filtered_proba(df_eval)
        except Exception:
            pass

    broker = PaperBroker(initial_capital=10000.0, taker_fee=0.0010, slippage_bps=0.0005)
    risk_cfg = get_risk_config_for_timeframe(timeframe)
    risk_engine = RiskEngine(risk_cfg)

    ann_factor = np.sqrt(525600.0 if timeframe == "1m" else (105120.0 if timeframe == "5m" else 8760.0))
    trades_executed = []

    for i in range(len(df_eval)):
        row = df_eval.iloc[i]
        bar_ts = str(df_eval.index[i])
        current_price = float(row["close"])
        p_long = float(probs[i])
        vol_bar = float(row.get("realized_vol_12", 0.003))
        vol_annual = vol_bar * ann_factor

        regime_probs = None
        regime_id = 0
        if regimes_df is not None and i < len(regimes_df):
            r_row = regimes_df.iloc[i]
            regime_probs = np.array([r_row["regime_p0"], r_row["regime_p1"], r_row["regime_p2"]])
            regime_id = int(np.argmax(regime_probs))

        decision = risk_engine.decide_step(
            prob_long=p_long,
            current_vol_annual=vol_annual,
            bar_volatility=vol_bar,
            regime_probs=regime_probs,
            bar_timestamp=bar_ts,
            current_price=current_price,
        )

        db.record_signal(
            symbol=symbol,
            bar_timestamp=bar_ts,
            prob_long=p_long,
            confidence=abs(p_long - 0.5) * 2.0,
            regime_id=regime_id,
            sentiment_score=0.0,
            raw_data={
                "action": decision.action,
                "target_size": decision.target_position,
                "decision_reason": decision.reason,
                "expected_edge": decision.expected_edge,
                "hurdle_cost": decision.hurdle_cost,
                "timeframe": timeframe,
            },
        )

        if decision.action in ("BUY", "SELL"):
            trade = broker.execute_rebalance(
                symbol=symbol,
                target_weight=decision.target_position,
                current_price=current_price,
                bar_timestamp=bar_ts,
            )
            if trade:
                db.record_order(
                    symbol=symbol,
                    bar_timestamp=bar_ts,
                    side=trade["side"],
                    qty=trade["qty"],
                    fill_price=trade["price"],
                    fee=trade["fee"],
                    target_weight=trade["target_weight"],
                )
                trades_executed.append(trade)

        bal = broker.get_balance()
        db.record_snapshot(
            equity=bal["equity"],
            cash=bal["cash"],
            positions_value=bal["positions_value"],
            drawdown=0.0,
            positions=broker.get_positions(),
        )

        db.record_processed_bar(
            symbol=symbol,
            bar_timestamp=bar_ts,
            action=decision.action,
            decision_reason=decision.reason,
            target_weight=decision.target_position,
        )

    final_bal = broker.get_balance()
    return {
        "trades_count": len(trades_executed),
        "initial_capital": 10000.0,
        "final_equity": final_bal["equity"],
        "return_pct": (final_bal["equity"] / 10000.0 - 1.0) * 100.0,
        "timeframe": timeframe,
        "bars": len(df_eval),
    }
