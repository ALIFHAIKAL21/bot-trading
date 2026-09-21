from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from loguru import logger

from src.broker.broker import PaperBroker
from src.service.db import Database
from src.utils.config import load_config
from src.utils.security import (
    verify_api_token,
    determine_execution_mode,
    print_mode_banner,
    get_active_models,
    print_active_models_banner,
)

# Global configuration and state
cfg = load_config("config/config.yaml")
db = Database(cfg.service.db_path)
broker = PaperBroker(initial_capital=cfg.backtest.initial_capital)
execution_mode = determine_execution_mode(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup and shutdown lifecycle management."""
    print_mode_banner(execution_mode)
    print_active_models_banner(cfg, context="SERVICE STARTUP")
    logger.info(f"Paper-trading service bound strictly to {cfg.service.host}:{cfg.service.port}")
    yield
    logger.info("Shutting down paper-trading service.")


app = FastAPI(
    title="Multi-Model Quant Trading Service",
    description="Institutional-grade paper trading and live monitoring REST API",
    version="1.0.0",
    lifespan=lifespan,
)


class TradeStepRequest(BaseModel):
    symbol: str = "BTC/USDT"
    bar_timestamp: Optional[str] = None
    close_price: float
    target_weight: float
    prob_long: float
    confidence: float
    regime_id: int
    sentiment_score: float = 0.0


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "paper-trading",
        "execution_mode": execution_mode,
        "live_execution_enabled": cfg.service.live_execution_enabled,
        "active_models": get_active_models(cfg),
        "database": str(db.db_path),
        "auth_required_for_post": True,
    }


@app.get("/portfolio")
def get_portfolio():
    balance = broker.get_balance()
    positions = broker.get_positions()
    return {
        "balance": balance,
        "positions": positions,
    }


@app.get("/positions")
def get_positions():
    return broker.get_positions()


@app.get("/orders")
def get_orders(limit: int = 50):
    return db.get_recent_orders(limit=limit)


@app.get("/signals/latest")
def get_latest_signals(limit: int = 20):
    return db.get_latest_signals(limit=limit)


@app.get("/history")
def get_history(limit: int = 200):
    return db.get_portfolio_history(limit=limit)


@app.post("/trade/step", dependencies=[Depends(verify_api_token)])
def execute_trade_step(req: TradeStepRequest):
    """Execute a single rebalance step with restart-safety and idempotency."""
    import datetime

    ts = req.bar_timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Idempotency check: has this bar already been processed?
    if db.has_order_for_bar(req.symbol, ts):
        return {
            "status": "skipped",
            "message": f"Bar {ts} for {req.symbol} already executed. Idempotency enforced.",
        }

    # Record signal
    db.record_signal(
        symbol=req.symbol,
        bar_timestamp=ts,
        prob_long=req.prob_long,
        confidence=req.confidence,
        regime_id=req.regime_id,
        sentiment_score=req.sentiment_score,
    )

    # Execute order via broker
    trade = broker.execute_rebalance(
        symbol=req.symbol,
        target_weight=req.target_weight,
        current_price=req.close_price,
        bar_timestamp=ts,
    )

    if trade:
        db.record_order(
            symbol=req.symbol,
            bar_timestamp=ts,
            side=trade["side"],
            qty=trade["qty"],
            fill_price=trade["price"],
            fee=trade["fee"],
            target_weight=trade["target_weight"],
        )

    # Record portfolio snapshot
    bal = broker.get_balance()
    db.record_snapshot(
        equity=bal["equity"],
        cash=bal["cash"],
        positions_value=bal["positions_value"],
        drawdown=0.0,
        positions=broker.get_positions(),
    )

    return {
        "status": "executed",
        "trade": trade,
        "portfolio": broker.get_balance(),
    }
