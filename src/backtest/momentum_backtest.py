"""
Event-Driven Backtest Engine for Momentum Setups (Point 3: RR 1:1 / 1:2)
Simulates bar-by-bar execution with institutional friction (spread, commission, slippage).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class TradeResult:
    trade_id: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    risk_amount: float
    confidence: float
    pnl_gross: float
    friction: float
    pnl_net: float
    r_net: float
    exit_reason: str  # "TP_HIT", "SL_HIT", "TIMEOUT"
    bars_held: int


@dataclass
class BacktestSummary:
    initial_equity: float
    final_equity: float
    net_pnl: float
    net_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    total_friction: float
    expectancy_r: float
    trades: List[TradeResult] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)


class MomentumBacktestEngine:
    """
    Event-driven backtester evaluating momentum setups with full cost realism.
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        risk_pct: float = 0.01,
        spread_pip: float = 0.75,
        commission_per_lot: float = 3.50,
        slippage_pip: float = 0.30,
        contract_size: float = 100.0,
        pip_size: float = 0.10,
    ):
        self.initial_equity = initial_equity
        self.risk_pct = risk_pct
        self.spread_price = spread_pip * pip_size
        self.slippage_price = slippage_pip * pip_size
        self.commission_per_lot = commission_per_lot
        self.contract_size = contract_size

    def _calculate_lot_size(self, equity: float, sl_distance: float) -> Tuple[float, float]:
        """Calculates standard lot size and risk capital based on fixed risk fraction."""
        if equity <= 0 or sl_distance <= 0:
            return 0.01, 100.0
        target_risk = equity * self.risk_pct
        loss_per_lot = sl_distance * self.contract_size
        raw_lot = target_risk / (loss_per_lot + 1e-9)
        lot = np.clip(np.floor(raw_lot * 100.0) / 100.0, 0.01, 20.0)
        actual_risk = lot * loss_per_lot
        return round(float(lot), 2), round(float(actual_risk), 2)

    def _calculate_friction(self, lot_size: float) -> float:
        """Total round-trip friction cost (spread + slippage + commission)."""
        spread_cost = self.spread_price * lot_size * self.contract_size
        slippage_cost = self.slippage_price * lot_size * self.contract_size * 2.0  # entry and exit
        comm_cost = self.commission_per_lot * lot_size
        return round(float(spread_cost + slippage_cost + comm_cost), 2)

    def run(
        self,
        df_bars: pd.DataFrame,
        df_setups: pd.DataFrame,
        target_rr: int = 2,
        confidence_tau: float = 0.33,
        max_holding_bars: int = 48,
    ) -> BacktestSummary:
        """
        Executes sequential event-driven simulation.
        
        Args:
            df_bars: Clean price bars (e.g. 1H bars for the test period)
            df_setups: Labeled setups with 'prob_win' predicted by MomentumClassifier
            target_rr: 1 for RR 1:1, 2 for RR 1:2
            confidence_tau: Minimum P(Win) required to trigger entry
            max_holding_bars: Maximum duration before timeout exit
        """
        # Map setups by bar timestamp for O(1) lookup
        setups_by_time = {}
        for _, row in df_setups.iterrows():
            ts = pd.to_datetime(row["timestamp_utc"], utc=True)
            # Filter by confidence gate
            if row.get("prob_win", 1.0) >= confidence_tau:
                setups_by_time[ts] = row

        equity = self.initial_equity
        equity_curve = [equity]
        timestamps = []
        trades: List[TradeResult] = []

        active_trade = None
        trade_id = 0

        highs = df_bars["high"].values
        lows = df_bars["low"].values
        closes = df_bars["close"].values
        times = pd.to_datetime(df_bars["timestamp_utc"], utc=True).tolist()
        n_bars = len(df_bars)

        for i in range(n_bars):
            ts = times[i]
            timestamps.append(ts)
            c_high = highs[i]
            c_low = lows[i]
            c_close = closes[i]

            # 1. Manage Active Trade
            if active_trade is not None:
                d = active_trade["direction"]
                ep = active_trade["entry_price"]
                sl = active_trade["sl_price"]
                tp = active_trade["tp_price"]
                lot = active_trade["lot_size"]
                risk_amt = active_trade["risk_amount"]
                holding = i - active_trade["entry_bar_idx"]

                exit_reason = None
                exit_price = 0.0

                if d == "BUY":
                    sl_hit = c_low <= sl
                    tp_hit = c_high >= tp
                    if sl_hit and tp_hit:
                        exit_reason = "SL_HIT"  # Conservative execution
                        exit_price = sl
                    elif tp_hit:
                        exit_reason = "TP_HIT"
                        exit_price = tp
                    elif sl_hit:
                        exit_reason = "SL_HIT"
                        exit_price = sl
                    elif holding >= max_holding_bars:
                        exit_reason = "TIMEOUT"
                        exit_price = c_close
                else:  # SELL
                    sl_hit = c_high >= sl
                    tp_hit = c_low <= tp
                    if sl_hit and tp_hit:
                        exit_reason = "SL_HIT"
                        exit_price = sl
                    elif tp_hit:
                        exit_reason = "TP_HIT"
                        exit_price = tp
                    elif sl_hit:
                        exit_reason = "SL_HIT"
                        exit_price = sl
                    elif holding >= max_holding_bars:
                        exit_reason = "TIMEOUT"
                        exit_price = c_close

                if exit_reason is not None:
                    # Calculate PnL
                    if d == "BUY":
                        gross_pnl = (exit_price - ep) * lot * self.contract_size
                    else:
                        gross_pnl = (ep - exit_price) * lot * self.contract_size

                    friction = self._calculate_friction(lot)
                    net_pnl = round(gross_pnl - friction, 2)
                    r_net = round(net_pnl / (risk_amt + 1e-9), 3)

                    trade = TradeResult(
                        trade_id=trade_id,
                        entry_time=active_trade["entry_time"],
                        exit_time=ts,
                        direction=d,
                        entry_price=ep,
                        exit_price=exit_price,
                        sl_price=sl,
                        tp_price=tp,
                        lot_size=lot,
                        risk_amount=risk_amt,
                        confidence=active_trade["confidence"],
                        pnl_gross=round(gross_pnl, 2),
                        friction=friction,
                        pnl_net=net_pnl,
                        r_net=r_net,
                        exit_reason=exit_reason,
                        bars_held=holding,
                    )
                    trades.append(trade)
                    trade_id += 1
                    equity += net_pnl
                    active_trade = None

            # 2. Check for New Setup Entry
            if active_trade is None and ts in setups_by_time:
                setup = setups_by_time[ts]
                d = setup["direction"]
                ep = setup["entry_price"]
                sl = setup["sl_price"]
                tp = setup["tp2_price"] if target_rr == 2 else setup["tp1_price"]
                conf = setup.get("prob_win", 0.5)

                sl_dist = abs(ep - sl)
                lot, risk_amt = self._calculate_lot_size(equity, sl_dist)

                if lot >= 0.01:
                    active_trade = {
                        "direction": d,
                        "entry_price": ep,
                        "sl_price": sl,
                        "tp_price": tp,
                        "lot_size": lot,
                        "risk_amount": risk_amt,
                        "confidence": conf,
                        "entry_bar_idx": i,
                        "entry_time": ts,
                    }

            equity_curve.append(equity)

        # Performance Metrics Compilation
        total_trades = len(trades)
        wins = [t for t in trades if t.pnl_net > 0]
        losses = [t for t in trades if t.pnl_net < 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        gross_profits = sum(t.pnl_net for t in wins)
        gross_losses = abs(sum(t.pnl_net for t in losses))
        profit_factor = round(gross_profits / (gross_losses + 1e-9), 3) if gross_losses > 0 else 999.0

        net_pnl = round(equity - self.initial_equity, 2)
        net_ret = round((net_pnl / self.initial_equity) * 100.0, 2)
        total_fric = round(sum(t.friction for t in trades), 2)
        expectancy_r = round(float(np.mean([t.r_net for t in trades])), 3) if total_trades > 0 else 0.0

        # Drawdown calculation
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        max_dd_pct = round(float(abs(dd.min()) * 100.0), 2)
        max_dd_usd = round(float(np.max(peak - eq_arr)), 2)

        # Sharpe & Sortino (annualized from bar returns)
        bar_returns = np.diff(eq_arr) / (eq_arr[:-1] + 1e-9)
        std_ret = np.std(bar_returns)
        sharpe = round(float((np.mean(bar_returns) / (std_ret + 1e-9)) * np.sqrt(8760.0)), 2) if std_ret > 0 else 0.0

        downside_returns = bar_returns[bar_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
        sortino = round(float((np.mean(bar_returns) / downside_std) * np.sqrt(8760.0)), 2)

        return BacktestSummary(
            initial_equity=self.initial_equity,
            final_equity=round(equity, 2),
            net_pnl=net_pnl,
            net_return_pct=net_ret,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=round(win_rate * 100.0, 2),
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_usd=max_dd_usd,
            total_friction=total_fric,
            expectancy_r=expectancy_r,
            trades=trades,
            equity_curve=equity_curve,
            timestamps=timestamps,
        )
