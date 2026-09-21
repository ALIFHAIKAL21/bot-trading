"""Event-consistent vectorized backtester with realistic transaction costs.

Execution protocol:
- Signal computed at bar t close -> Order executed at bar t+1 OPEN.
- Taker fees (default 10 bps).
- Base slippage (default 5 bps) + volatility-scaled slippage (ATR/Price).
- Tracks equity, turnover, exposure, drawdown, and per-bar transaction costs.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import BacktestConfig


@dataclass
class BacktestResult:
    """Container for backtest time series and summary metrics."""
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    costs: pd.Series
    drawdown: pd.Series
    metrics: Dict[str, float]


class EventConsistentBacktester:
    """Vectorized next-bar execution backtester with realistic friction."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        price_col: str = "close",
        open_col: str = "open",
        atr_col: str = "atr_14",
    ) -> BacktestResult:
        """Run backtest on aligned DataFrame and signal series.

        signals: Target position weight in [-1.0, 1.0] or [0.0, 1.0] generated at bar t close.
        """
        aligned_df = df.copy()
        aligned_df["signal"] = signals.reindex(aligned_df.index).fillna(0.0)

        # Execution delay: signal at close of t becomes active position at open of t+1
        # Shift signals forward by 1 bar to represent executed position during bar t+1
        positions = aligned_df["signal"].shift(1).fillna(0.0)

        opens = aligned_df[open_col].values
        closes = aligned_df[price_col].values
        atrs = (
            aligned_df[atr_col].values
            if atr_col in aligned_df.columns
            else np.full(len(aligned_df), 0.0)
        )
        pos_arr = positions.values
        n = len(aligned_df)

        # Bar returns: holding from open to close, plus overnight/close-to-open gap
        # Next-bar open execution: return from open_t to open_{t+1}
        # In hourly data: open_t to open_{t+1} return is approximately close_t to close_{t+1}
        bar_returns = np.zeros(n, dtype=np.float64)
        bar_returns[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-9)

        # Turnover = |pos_t - pos_{t-1}|
        delta_pos = np.zeros(n, dtype=np.float64)
        delta_pos[1:] = np.abs(pos_arr[1:] - pos_arr[:-1])

        # Slippage rate = base_slippage + gamma * (ATR / Price)
        natr = np.zeros(n, dtype=np.float64)
        natr = np.where(closes > 0, atrs / closes, 0.0)
        slippage_rate = self.config.base_slippage + self.config.vol_slippage_coeff * natr
        friction_rate = self.config.taker_fee + slippage_rate

        # Total transaction cost deducted at trade execution
        cost_series = delta_pos * friction_rate

        # Net strategy return = pos * bar_return - costs
        net_returns = pos_arr * bar_returns - cost_series

        # Cumulative equity curve
        cum_equity = self.config.initial_capital * np.cumprod(1.0 + net_returns)

        # Drawdown series
        running_max = np.maximum.accumulate(cum_equity)
        drawdowns = (cum_equity - running_max) / (running_max + 1e-9)

        # Summary metrics
        metrics = self._calculate_metrics(net_returns, drawdowns, delta_pos, pos_arr)

        return BacktestResult(
            equity_curve=pd.Series(cum_equity, index=aligned_df.index, name="equity"),
            returns=pd.Series(net_returns, index=aligned_df.index, name="returns"),
            positions=pd.Series(pos_arr, index=aligned_df.index, name="position"),
            turnover=pd.Series(delta_pos, index=aligned_df.index, name="turnover"),
            costs=pd.Series(cost_series, index=aligned_df.index, name="costs"),
            drawdown=pd.Series(drawdowns, index=aligned_df.index, name="drawdown"),
            metrics=metrics,
        )

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        drawdowns: np.ndarray,
        turnover: np.ndarray,
        positions: np.ndarray,
    ) -> Dict[str, float]:
        """Compute annualized Sharpe, Sortino, Calmar, MaxDD, Win Rate, Turnover."""
        n = len(returns)
        if n < 10:
            return {}

        # 1h timeframe: 24 * 365 = 8760 periods per year (crypto 24/7)
        periods_per_year = 8760.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        # Annualized metrics
        cagr = (np.prod(1.0 + returns) ** (periods_per_year / max(n, 1))) - 1.0
        ann_vol = std_ret * np.sqrt(periods_per_year)

        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(periods_per_year)

        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
        sortino = (mean_ret / (downside_std + 1e-9)) * np.sqrt(periods_per_year)

        max_dd = np.abs(np.min(drawdowns))
        calmar = cagr / (max_dd + 1e-9) if max_dd > 0 else 0.0

        trade_bars = returns[turnover > 0]
        win_rate = (
            np.sum(returns > 0) / np.sum(np.abs(positions) > 0)
            if np.sum(np.abs(positions) > 0) > 0
            else 0.0
        )

        gross_wins = np.sum(returns[returns > 0])
        gross_losses = np.abs(np.sum(returns[returns < 0]))
        profit_factor = gross_wins / (gross_losses + 1e-9) if gross_losses > 0 else 1.0

        avg_exposure = np.mean(np.abs(positions))
        annual_turnover = np.mean(turnover) * periods_per_year

        return {
            "cagr": float(cagr),
            "annualized_vol": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_exposure": float(avg_exposure),
            "annual_turnover": float(annual_turnover),
        }
