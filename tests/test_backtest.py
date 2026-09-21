"""Unit tests for event-consistent backtester mechanics and transaction costs."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtester import EventConsistentBacktester
from src.utils.config import BacktestConfig


def test_next_bar_execution_timing():
    """Verify that a signal generated at bar t executes on bar t+1."""
    timestamps = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    # Prices: 100, 100, 110, 120, 120
    closes = np.array([100.0, 100.0, 110.0, 120.0, 120.0])
    opens = np.array([100.0, 100.0, 100.0, 110.0, 120.0])

    df = pd.DataFrame({"open": opens, "close": closes, "atr_14": 1.0}, index=timestamps)

    # Buy signal generated at bar 1 (close of bar 1 = 100)
    # Positions should become active at bar 2 (where price goes from 100 to 110)
    signals = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=timestamps)

    backtester = EventConsistentBacktester(BacktestConfig(taker_fee=0.0, base_slippage=0.0, vol_slippage_coeff=0.0))
    res = backtester.run(df, signals)

    # At bar 1: position was 0.0 (return = 0)
    assert res.positions.iloc[1] == 0.0
    # At bar 2: position is 1.0 (captures return from 100 to 110 = +10%)
    assert res.positions.iloc[2] == 1.0
    assert np.isclose(res.returns.iloc[2], 0.10, atol=1e-4)


def test_transaction_cost_deduction():
    """Verify that taker fees and slippage reduce net equity on position changes."""
    timestamps = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    # Flat price: 100, 100, 100, 100
    closes = np.full(4, 100.0)
    df = pd.DataFrame({"open": closes, "close": closes, "atr_14": 0.0}, index=timestamps)

    # Trade: Enter at bar 0 -> active bar 1. Exit at bar 1 -> flat bar 2.
    signals = pd.Series([1.0, 0.0, 0.0, 0.0], index=timestamps)

    fee = 0.0010  # 10 bps
    slippage = 0.0005  # 5 bps
    backtester = EventConsistentBacktester(
        BacktestConfig(initial_capital=10000.0, taker_fee=fee, base_slippage=slippage, vol_slippage_coeff=0.0)
    )
    res = backtester.run(df, signals)

    # Turnover occurred entering at bar 1 (delta = 1.0) and exiting at bar 2 (delta = 1.0)
    # Total friction = 2 * (fee + slippage) = 2 * 0.0015 = 0.0030 (0.3%)
    expected_final_equity = 10000.0 * (1.0 - 0.0015) * (1.0 - 0.0015)
    assert np.isclose(res.equity_curve.iloc[-1], expected_final_equity, atol=1.0)
    assert res.equity_curve.iloc[-1] < 10000.0


def test_oracle_lookahead_impossibility():
    """Verify that future price manipulations cannot leak into past signals or execution."""
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    closes = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 105.0, 104.0, 106.0, 105.0, 108.0])
    opens = np.array([99.0, 100.0, 102.0, 101.0, 103.0, 102.0, 105.0, 104.0, 106.0, 105.0])

    df = pd.DataFrame({"open": opens, "close": closes, "atr_14": 1.0}, index=timestamps)

    # Signal at bar 4 (close of bar 4 = 102.0)
    # Even if an oracle knew that bar 5 would jump 100%, the signal at bar 4 ONLY executes at bar 5 OPEN
    signals = pd.Series(0.0, index=timestamps)
    signals.iloc[4] = 1.0  # Buy signal generated at close of bar 4

    backtester = EventConsistentBacktester(BacktestConfig(taker_fee=0.0, base_slippage=0.0, vol_slippage_coeff=0.0))
    res = backtester.run(df, signals)

    # Position at bar 4 must be 0.0
    assert res.positions.iloc[4] == 0.0
    # Position only becomes active at bar 5
    assert res.positions.iloc[5] == 1.0

    # Modifying future bars 6..9 has zero effect on returns/positions at bars 0..5
    df_manipulated = df.copy()
    df_manipulated.iloc[6:, df_manipulated.columns.get_loc("close")] *= 50.0  # Massive future pump

    res_manip = backtester.run(df_manipulated, signals)
    assert np.allclose(res.positions.iloc[:6].values, res_manip.positions.iloc[:6].values)
    assert np.allclose(res.returns.iloc[:6].values, res_manip.returns.iloc[:6].values)


def test_deflated_sharpe_ratio_penalizes_trials():
    """Verify that Deflated Sharpe Ratio (DSR) decreases as number of trials increases."""
    from src.backtest.metrics import calculate_deflated_sharpe_ratio
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 1000)
    sharpe_hat = float(np.mean(returns) / np.std(returns) * np.sqrt(8760.0))

    dsr_1 = calculate_deflated_sharpe_ratio(sharpe_hat=sharpe_hat, n_trials=1, returns=returns)
    dsr_10 = calculate_deflated_sharpe_ratio(sharpe_hat=sharpe_hat, n_trials=10, returns=returns)
    dsr_100 = calculate_deflated_sharpe_ratio(sharpe_hat=sharpe_hat, n_trials=100, returns=returns)

    assert dsr_1 >= dsr_10 >= dsr_100
    assert 0.0 <= dsr_100 <= 1.0
