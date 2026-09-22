"""Unit tests for ProSniperEngine."""

import numpy as np
import pandas as pd
import pytest

from src.risk.pro_sniper import ProSniperEngine, SniperSignal


def create_sample_df(n_bars: int = 60, trend: str = "bull") -> pd.DataFrame:
    """Generate realistic OHLCV bars for testing."""
    dates = pd.date_range("2026-09-20 00:00:00", periods=n_bars, freq="5min", tz="UTC")
    base_price = 80000.0

    if trend == "bull":
        prices = [base_price + i * 50.0 + np.random.uniform(-10, 10) for i in range(n_bars)]
    elif trend == "bear":
        prices = [base_price - i * 50.0 + np.random.uniform(-10, 10) for i in range(n_bars)]
    else:
        # Chop
        prices = [base_price + np.random.uniform(-20, 20) for i in range(n_bars)]

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["open"] = [p - 15.0 if trend == "bull" else p + 15.0 for p in prices]
    df["high"] = [max(o, c) + 20.0 for o, c in zip(df["open"], df["close"])]
    df["low"] = [min(o, c) - 20.0 for o, c in zip(df["open"], df["close"])]
    df["volume"] = [100.0 + np.random.uniform(10, 50) for _ in range(n_bars)]
    return df


def test_confluence_scoring_bull_vs_bear():
    engine = ProSniperEngine(min_confluence_score=65.0)
    df_bull = create_sample_df(60, trend="bull")
    df_bear = create_sample_df(60, trend="bear")

    score_bull = engine.compute_confluence_score(df_bull, eval_idx=-1, prob_long=0.55)
    score_bear = engine.compute_confluence_score(df_bear, eval_idx=-1, prob_long=0.40)

    assert score_bull["total_score"] > score_bear["total_score"]
    assert score_bull["trend_score"] >= 15.0
    assert score_bear["trend_score"] == 0.0


def test_entry_and_take_profit_cycle():
    engine = ProSniperEngine(
        tp_atr_mult=1.5,
        sl_atr_mult=1.5,
        breakeven_atr_trigger=0.7,
        min_confluence_score=60.0,
    )
    df = create_sample_df(60, trend="bull")

    # Step 1: Entry
    sig_entry = engine.evaluate_step(
        df=df,
        eval_idx=-1,
        prob_long=0.55,
        current_price=85000.0,
        bar_timestamp="2026-09-21 12:00:00",
    )
    assert sig_entry.action == "BUY"
    assert engine.in_position is True
    assert sig_entry.take_profit is not None
    assert sig_entry.stop_loss is not None
    assert sig_entry.take_profit > 85000.0
    assert sig_entry.stop_loss < 85000.0

    # Step 2: Price rises to TP
    tp_price = sig_entry.take_profit + 5.0
    sig_tp = engine.evaluate_step(
        df=df,
        eval_idx=-1,
        prob_long=0.50,
        current_price=tp_price,
        bar_timestamp="2026-09-21 12:05:00",
    )
    assert sig_tp.action == "SELL"
    assert "TP Hit" in sig_tp.reason
    assert engine.in_position is False


def test_trailing_breakeven_activation():
    engine = ProSniperEngine(
        tp_atr_mult=2.0,
        sl_atr_mult=1.5,
        breakeven_atr_trigger=0.7,
        min_confluence_score=60.0,
    )
    df = create_sample_df(60, trend="bull")

    # Entry @ 80,000
    engine.evaluate_step(
        df=df, eval_idx=-1, prob_long=0.55, current_price=80000.0, bar_timestamp="2026-09-21 12:00:00"
    )
    initial_sl = engine.stop_loss_price
    atr = engine.entry_atr

    # Price moves up by +0.8x ATR (triggers breakeven)
    price_move = 80000.0 + 0.8 * atr
    sig_hold = engine.evaluate_step(
        df=df, eval_idx=-1, prob_long=0.52, current_price=price_move, bar_timestamp="2026-09-21 12:05:00"
    )
    assert sig_hold.action == "HOLD"
    assert engine.trailing_stage == 1
    # SL should now be above entry price (entry * 1.0015)
    assert engine.stop_loss_price > 80000.0
    assert engine.stop_loss_price > initial_sl


def test_bearish_chop_stays_flat():
    engine = ProSniperEngine(min_confluence_score=65.0)
    df_chop = create_sample_df(60, trend="chop")

    sig = engine.evaluate_step(
        df=df_chop,
        eval_idx=-1,
        prob_long=0.45,
        current_price=80000.0,
        bar_timestamp="2026-09-21 12:00:00",
    )
    assert sig.action == "FLAT"
    assert engine.in_position is False
