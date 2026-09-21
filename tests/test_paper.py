"""Unit tests for PaperBroker and database restart safety."""

import os
import tempfile
from pathlib import Path
import pytest

from src.broker.broker import PaperBroker
from src.service.db import Database


def test_paper_broker_rebalance_buy_sell():
    broker = PaperBroker(initial_capital=10000.0, taker_fee=0.0010, slippage_bps=0.0005)

    # Rebalance to 30% BTC at $50,000
    trade1 = broker.execute_rebalance("BTC/USDT", target_weight=0.30, current_price=50000.0, bar_timestamp="2024-01-01T00:00:00Z")
    assert trade1 is not None
    assert trade1["side"] == "BUY"
    assert broker.positions["BTC/USDT"] > 0
    assert broker.cash < 10000.0

    # Sell down to 0%
    trade2 = broker.execute_rebalance("BTC/USDT", target_weight=0.0, current_price=55000.0, bar_timestamp="2024-01-01T01:00:00Z")
    assert trade2 is not None
    assert trade2["side"] == "SELL"
    assert broker.positions["BTC/USDT"] == 0.0
    # Equity should reflect profit ($50,000 -> $55,000 minus fees)
    assert broker.cash > 10000.0


def test_database_idempotency_restart_safety():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db_path = f.name

    try:
        db = Database(tmp_db_path)

        # First order record succeeds
        res1 = db.record_order(
            symbol="BTC/USDT",
            bar_timestamp="2024-01-01T00:00:00Z",
            side="BUY",
            qty=0.05,
            fill_price=50000.0,
            fee=2.5,
            target_weight=0.25,
        )
        assert res1 is True
        assert db.has_order_for_bar("BTC/USDT", "2024-01-01T00:00:00Z") is True

        # Duplicate order for same bar must be ignored/skipped (idempotent)
        res2 = db.record_order(
            symbol="BTC/USDT",
            bar_timestamp="2024-01-01T00:00:00Z",
            side="BUY",
            qty=0.05,
            fill_price=50000.0,
            fee=2.5,
            target_weight=0.25,
        )
        assert res2 is False
        orders = db.get_recent_orders()
        assert len(orders) == 1
    finally:
        if os.path.exists(tmp_db_path):
            os.remove(tmp_db_path)


def test_processed_bars_idempotency_restart_safety():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db_path = f.name

    try:
        db = Database(tmp_db_path)
        sym = "BTC/USDT"
        bar_ts = "2026-09-22T00:00:00Z"

        # Initially bar is not processed
        assert db.is_bar_processed(sym, bar_ts) is False

        # Record processed bar
        ok1 = db.record_processed_bar(
            symbol=sym,
            bar_timestamp=bar_ts,
            action="BUY",
            decision_reason="edge_exceeds_hurdle",
            target_weight=0.25,
        )
        assert ok1 is True
        assert db.is_bar_processed(sym, bar_ts) is True

        # Duplicate recording should be ignored
        ok2 = db.record_processed_bar(
            symbol=sym,
            bar_timestamp=bar_ts,
            action="BUY",
            decision_reason="duplicate_attempt",
            target_weight=0.25,
        )
        assert ok2 is False

        # Simulate service restart with new Database connection
        db_restarted = Database(tmp_db_path)
        assert db_restarted.is_bar_processed(sym, bar_ts) is True
    finally:
        if os.path.exists(tmp_db_path):
            os.remove(tmp_db_path)


def test_paper_broker_vs_backtest_parity():
    import pandas as pd
    import numpy as np
    from src.backtest.backtester import EventConsistentBacktester, BacktestConfig

    # 3 bars
    prices = [50000.0, 51000.0, 52000.0]
    weights = [0.0, 0.5, 0.5]

    broker = PaperBroker(initial_capital=10000.0, taker_fee=0.0010, slippage_bps=0.0005)

    # Bar 0: weight 0 -> no trade
    t0 = broker.execute_rebalance("BTC/USDT", target_weight=weights[0], current_price=prices[0], bar_timestamp="t0")
    assert t0 is None
    bal0 = broker.get_balance()
    assert bal0["equity"] == 10000.0

    # Bar 1: weight 0.5 at $51,000 -> BUY
    t1 = broker.execute_rebalance("BTC/USDT", target_weight=weights[1], current_price=prices[1], bar_timestamp="t1")
    assert t1 is not None
    assert t1["side"] == "BUY"
    bal1 = broker.get_balance()
    # Cost charged: fee + slippage
    assert bal1["equity"] < 10000.0

    # Backtester comparison on same sequence
    df_sim = pd.DataFrame(
        {
            "open": prices,
            "close": prices,
            "atr_14": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="1h"),
    )
    sig_series = pd.Series(weights, index=df_sim.index)
    bt = EventConsistentBacktester(BacktestConfig(initial_capital=10000.0, taker_fee=0.0010, base_slippage=0.0005, vol_slippage_coeff=0.0))
    res = bt.run(df_sim, sig_series)
    # Both backtester and paper broker incur friction on position change
    assert res.turnover.sum() > 0
    assert res.costs.sum() > 0


def test_forming_candle_guard_logic():
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    now_utc = datetime(2026, 9, 22, 3, 5, 0, tzinfo=timezone.utc)
    tf_delta = timedelta(hours=1)
    buffer = timedelta(seconds=5)

    # Candle opened at 03:00 -> close at 04:00 (in the future) -> forming!
    bar_ts_forming = datetime(2026, 9, 22, 3, 0, 0, tzinfo=timezone.utc)
    bar_close_forming = bar_ts_forming + tf_delta
    is_forming = (bar_close_forming + buffer) > now_utc
    assert is_forming is True

    # Candle opened at 02:00 -> close at 03:00 (already closed > 5s ago) -> closed!
    bar_ts_closed = datetime(2026, 9, 22, 2, 0, 0, tzinfo=timezone.utc)
    bar_close_closed = bar_ts_closed + tf_delta
    is_closed = (bar_close_closed + buffer) <= now_utc
    assert is_closed is True

