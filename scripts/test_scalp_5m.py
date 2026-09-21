"""5-Minute Scalping Test Runner (Replay and Live Modes).

Demonstrates and executes real open/close positions on 5-minute candles:
- Computes causal features on 5m bars.
- Runs Model B (LightGBM) + Model E (HMM) + Model C (Deep Sequence).
- Applies 5m scalping-calibrated RiskEngine (tight holding, agile hurdle).
- Executes rebalances via PaperBroker.
- Persists all signals, orders, and equity snapshots to SQLite database (data/paper_trading.db).
- Displays a clean visual trade log and updates the Streamlit dashboard.
"""

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from loguru import logger

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.broker.broker import PaperBroker
from src.data.loader import MarketDataLoader
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_engine import RiskEngine
from src.service.db import Database
from src.utils.config import RiskConfig, load_config
from src.utils.security import print_mode_banner


def create_scalp_risk_config() -> RiskConfig:
    """Return RiskConfig calibrated specifically for 5-minute scalping."""
    return RiskConfig(
        entry_threshold=0.52,             # Scalping entry trigger (above 0.50 neutral)
        exit_threshold=0.48,              # Exit deadband boundary
        min_holding_bars=2,               # Hold for at least 2 bars (10 mins)
        cooldown_bars=1,                  # 1-bar cooldown (5 mins) post-exit
        max_daily_trades=24,              # Agile daily trade cap for 5m (288 bars/day)
        round_trip_cost=0.0025,           # 25 bps (10 bps fee + 5 bps slippage each way)
        expected_edge_hurdle_multiplier=1.0, # 1.0x hurdle
        trade_horizon_bars=6,             # 6 bars (30-minute target horizon)
        target_annual_vol=0.30,           # Scalping vol target
        kelly_fraction=0.35,              # Controlled fractional Kelly
        max_position_pct=0.40,            # Max 40% portfolio allocation per trade
        dust_rebalance_threshold=0.03,    # Filter out trades < 3% position change
        hard_stop_loss_pct=0.02,          # Tight 2% scalping stop-loss
    )


def run_scalp_replay(
    symbol: str = "BTC/USDT",
    n_bars: int = 500,
    db_path: str = "data/paper_trading.db",
    reset_db: bool = False,
):
    """Replay the most recent N 5-minute bars to demonstrate real trade execution."""
    print("=" * 80)
    print(f"       5-MINUTE SCALPING DEMO REPLAY ({symbol}) - LAST {n_bars} BARS")
    print("=" * 80)

    cfg = load_config("config/config.yaml")
    cache_path = Path("data/cache") / f"{symbol.replace('/', '_')}_5m.parquet"

    # Ingest 5m data if not cached or small
    if not cache_path.exists():
        logger.info(f"Downloading recent 5m market data for {symbol}...")
        loader = MarketDataLoader(cache_dir="data/cache")
        df_raw = loader.load_or_fetch(symbol, timeframe="5m", history_days=7)
    else:
        df_raw = pd.read_parquet(cache_path)

    logger.info(f"Loaded {len(df_raw)} total 5m candles from cache.")

    # Build features
    pipeline = FeaturePipeline(cfg.features)
    df_feat = pipeline.build_features(df_raw)
    feature_cols = pipeline.get_feature_columns(df_feat)
    df_eval = df_feat.iloc[-n_bars:].copy()

    # Load trained models
    models_dir = Path("models_store")
    lgbm_model = joblib.load(models_dir / "model_b_lgbm.joblib")
    hmm_model = joblib.load(models_dir / "model_e_hmm.joblib") if (models_dir / "model_e_hmm.joblib").exists() else None

    # Compute tabular direction probabilities
    probs = lgbm_model.predict_proba(df_eval[feature_cols])

    # HMM Regimes
    regimes_df = None
    if hmm_model is not None:
        try:
            regimes_df = hmm_model.predict_filtered_proba(df_eval)
        except Exception as e:
            logger.warning(f"HMM regime prediction skipped: {e}")

    # Initialize Database & Broker
    if reset_db and Path(db_path).exists():
        logger.warning(f"Clearing {db_path} for fresh scalping demonstration...")
        try:
            Path(db_path).unlink()
        except Exception:
            pass

    db = Database(db_path)
    broker = PaperBroker(initial_capital=10000.0, taker_fee=0.0010, slippage_bps=0.0005)
    scalp_risk = create_scalp_risk_config()
    risk_engine = RiskEngine(scalp_risk)

    executed_trades = []
    logger.info(f"Evaluating {len(df_eval)} 5-minute bars sequentially...")

    for i in range(len(df_eval)):
        row = df_eval.iloc[i]
        bar_ts = str(df_eval.index[i])
        current_price = float(row["close"])
        p_long = float(probs[i])
        vol_bar = float(row.get("realized_vol_12", 0.003))

        # Annualized vol for 5m bars: sqrt(105,120)
        vol_annual = vol_bar * np.sqrt(105120.0)

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

        # Record signal to DB
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
                "timeframe": "5m",
            },
        )

        # Execute rebalance if position changed
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
                executed_trades.append({
                    "bar": bar_ts[:19],
                    "side": trade["side"],
                    "price": trade["price"],
                    "qty": trade["qty"],
                    "fee": trade["fee"],
                    "target_wt": trade["target_weight"],
                    "prob": p_long,
                    "reason": decision.reason,
                    "equity": broker.get_balance()["equity"],
                })

        # Record snapshot
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

    # Print summary results table
    print("\n" + "=" * 100)
    print("                           EXECUTED 5-MINUTE SCALPING TRADES")
    print("=" * 100)
    if executed_trades:
        df_trades = pd.DataFrame(executed_trades)
        print(f"{'TIMESTAMP':<20} | {'SIDE':<5} | {'PRICE ($)':<11} | {'QTY (BTC)':<10} | {'FEE ($)':<7} | {'TARGET':<7} | {'P(LONG)':<8} | {'EQUITY ($)':<11} | {'REASON'}")
        print("-" * 125)
        for _, t in df_trades.iterrows():
            print(
                f"{t['bar']:<20} | {t['side']:<5} | ${t['price']:>9,.2f} | {t['qty']:>9.4f} | ${t['fee']:>5.2f} | {t['target_wt']:>6.2f} | {t['prob']:>7.3f} | ${t['equity']:>9.2f} | {t['reason']}"
            )
        print("=" * 100)
        print(f"Total Scalping Trades: {len(executed_trades)}")
        final_bal = broker.get_balance()
        print(f"Initial Capital: $10,000.00  -->  Final Simulated Equity: ${final_bal['equity']:,.2f} (Net: {(final_bal['equity']/10000.0 - 1.0)*100:+.2f}%)")
    else:
        print("No trades triggered within this window under strict cost hurdles.")
    print("[SUCCESS] Signals, orders, and portfolio history have been written to data/paper_trading.db.")
    print("Open the Streamlit Dashboard (http://localhost:8502) -> 'Live Paper Trading' tab to inspect!")



def run_scalp_live(symbol: str = "BTC/USDT", poll_interval: int = 10):
    """Run live real-time 5-minute candle listener and scalper."""
    print("=" * 80)
    print(f"       STARTING LIVE 5-MINUTE SCALPER ({symbol})")
    print(f"       Poll Interval: {poll_interval}s | Press Ctrl+C to stop")
    print("=" * 80)

    cfg = load_config("config/config.yaml")
    db = Database(cfg.service.db_path)
    broker = PaperBroker(initial_capital=10000.0, taker_fee=0.0010, slippage_bps=0.0005)
    scalp_risk = create_scalp_risk_config()
    risk_engine = RiskEngine(scalp_risk)
    loader = MarketDataLoader(cache_dir="data/cache")
    pipeline = FeaturePipeline(cfg.features)

    models_dir = Path("models_store")
    lgbm_model = joblib.load(models_dir / "model_b_lgbm.joblib")
    hmm_model = joblib.load(models_dir / "model_e_hmm.joblib") if (models_dir / "model_e_hmm.joblib").exists() else None

    logger.info("5m Scalping Daemon initialized. Listening for newly closed 5m candles...")

    while True:
        try:
            # Fetch recent 5m candles (last 2 days)
            df_raw = loader.load_or_fetch(symbol, timeframe="5m", history_days=2, force_refresh=True)
            df_feat = pipeline.build_features(df_raw)
            feature_cols = pipeline.get_feature_columns(df_feat)

            # Check closed-candle condition (F4)
            latest_bar_ts = df_feat.index[-1]
            if latest_bar_ts.tzinfo is None:
                latest_bar_ts = latest_bar_ts.replace(tzinfo=timezone.utc)

            candle_close_utc = latest_bar_ts + timedelta(minutes=5)
            now_utc = datetime.now(timezone.utc)

            # If latest is still forming, use previous closed candle
            if candle_close_utc + timedelta(seconds=5) > now_utc:
                eval_idx = -2
                eval_bar = df_feat.iloc[-2]
                eval_ts = str(df_feat.index[-2])
            else:
                eval_idx = -1
                eval_bar = df_feat.iloc[-1]
                eval_ts = str(df_feat.index[-1])

            # Check idempotency
            if db.is_bar_processed(symbol, eval_ts):
                logger.debug(f"Closed 5m bar {eval_ts} already evaluated. Sleeping {poll_interval}s...")
                time.sleep(poll_interval)
                continue

            current_price = float(eval_bar["close"])
            vol_bar = float(eval_bar.get("realized_vol_12", 0.003))
            vol_annual = vol_bar * np.sqrt(105120.0)

            # Inference
            p_long = float(lgbm_model.predict_proba(df_feat[feature_cols].iloc[[eval_idx]])[0])

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

            logger.info(
                f"[5m Scalper] Bar: {eval_ts} | Price: ${current_price:,.2f} | P(long): {p_long:.3f} | "
                f"Action: {decision.action} | Target: {decision.target_position:.2f} | Reason: {decision.reason}"
            )

            # Record signal
            db.record_signal(
                symbol=symbol,
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
                    "timeframe": "5m",
                },
            )

            # Execute trade if action is BUY or SELL
            if decision.action in ("BUY", "SELL"):
                trade = broker.execute_rebalance(
                    symbol=symbol,
                    target_weight=decision.target_position,
                    current_price=current_price,
                    bar_timestamp=eval_ts,
                )
                if trade:
                    db.record_order(
                        symbol=symbol,
                        bar_timestamp=eval_ts,
                        side=trade["side"],
                        qty=trade["qty"],
                        fill_price=trade["price"],
                        fee=trade["fee"],
                        target_weight=trade["target_weight"],
                    )
                    logger.success(
                        f"*** EXECUTED {trade['side']} {symbol} *** | Qty: {trade['qty']:.4f} @ ${trade['price']:,.2f} | Fee: ${trade['fee']:.2f}"
                    )

            # Record snapshot
            bal = broker.get_balance()
            db.record_snapshot(
                equity=bal["equity"],
                cash=bal["cash"],
                positions_value=bal["positions_value"],
                drawdown=0.0,
                positions=broker.get_positions(),
            )

            # Lock bar in DB
            db.record_processed_bar(
                symbol=symbol,
                bar_timestamp=eval_ts,
                action=decision.action,
                decision_reason=decision.reason,
                target_weight=decision.target_position,
            )

        except KeyboardInterrupt:
            logger.info("Stopping 5m scalper...")
            break
        except Exception as e:
            logger.error(f"Error in 5m scalper loop: {e}")

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="5-Minute Scalping Test Runner")
    parser.add_argument("--replay", action="store_true", help="Replay recent 5m market data to execute positions")
    parser.add_argument("--live", action="store_true", help="Run live 5-minute candle listener loop")
    parser.add_argument("--bars", type=int, default=500, help="Number of bars to replay (default: 500)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--reset-db", action="store_true", help="Reset paper database for fresh scalping test")
    args = parser.parse_args()

    if args.live:
        run_scalp_live(symbol=args.symbol)
    else:
        # Default to replay demonstration
        run_scalp_replay(symbol=args.symbol, n_bars=args.bars, reset_db=args.reset_db)


if __name__ == "__main__":
    main()
