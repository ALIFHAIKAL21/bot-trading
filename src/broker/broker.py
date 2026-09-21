"""Broker interfaces: PaperBroker and disabled CcxtBroker stub."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger


class Broker(ABC):
    """Abstract Broker Interface."""

    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Return dict with 'cash', 'equity', and 'positions_value'."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Return dict of symbol -> quantity."""
        pass

    @abstractmethod
    def execute_rebalance(
        self,
        symbol: str,
        target_weight: float,
        current_price: float,
        bar_timestamp: str,
    ) -> Optional[Dict]:
        """Rebalance symbol position to target weight of equity."""
        pass


class PaperBroker(Broker):
    """Realistic paper-trading broker tracking simulated cash, holdings, and fees."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        taker_fee: float = 0.0010,
        slippage_bps: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> qty
        self.entry_prices: Dict[str, float] = {}
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps
        self.trades: List[Dict] = []
        self.last_prices: Dict[str, float] = {}

    def get_balance(self) -> Dict[str, float]:
        pos_val = sum(qty * self.last_prices.get(sym, 0.0) for sym, qty in self.positions.items())
        equity = self.cash + pos_val
        return {
            "cash": float(self.cash),
            "equity": float(equity),
            "positions_value": float(pos_val),
        }

    def get_positions(self) -> Dict[str, float]:
        return {k: v for k, v in self.positions.items() if abs(v) > 1e-6}

    def execute_rebalance(
        self,
        symbol: str,
        target_weight: float,
        current_price: float,
        bar_timestamp: str,
    ) -> Optional[Dict]:
        """Rebalance to target weight [0.0, 1.0] of current portfolio equity."""
        self.last_prices[symbol] = current_price
        balance = self.get_balance()
        equity = balance["equity"]

        target_val = equity * target_weight
        current_qty = self.positions.get(symbol, 0.0)
        current_val = current_qty * current_price

        diff_val = target_val - current_val

        # Minimum trade threshold to avoid dust orders
        if abs(diff_val) < 10.0:
            return None

        side = "BUY" if diff_val > 0 else "SELL"
        # Apply slippage
        exec_price = (
            current_price * (1.0 + self.slippage_bps)
            if side == "BUY"
            else current_price * (1.0 - self.slippage_bps)
        )

        trade_qty = abs(diff_val) / exec_price
        gross_cost = trade_qty * exec_price
        fee = gross_cost * self.taker_fee

        if side == "BUY":
            total_outlay = gross_cost + fee
            if total_outlay > self.cash:
                # Cap to available cash
                gross_cost = max(self.cash - fee, 0.0)
                trade_qty = gross_cost / exec_price
                fee = gross_cost * self.taker_fee
                total_outlay = gross_cost + fee

            if trade_qty <= 0:
                return None

            self.cash -= total_outlay
            new_qty = current_qty + trade_qty
            # Update weighted average entry price
            old_qty = current_qty
            old_entry = self.entry_prices.get(symbol, current_price)
            self.entry_prices[symbol] = (
                (old_qty * old_entry + trade_qty * exec_price) / new_qty if new_qty > 0 else 0.0
            )
            self.positions[symbol] = new_qty
        else:
            # Sell
            trade_qty = min(trade_qty, current_qty)
            if trade_qty <= 0:
                return None

            net_proceeds = (trade_qty * exec_price) - fee
            self.cash += net_proceeds
            self.positions[symbol] = current_qty - trade_qty
            if self.positions[symbol] <= 1e-6:
                self.positions[symbol] = 0.0
                self.entry_prices.pop(symbol, None)

        trade_record = {
            "symbol": symbol,
            "bar_timestamp": bar_timestamp,
            "side": side,
            "qty": float(trade_qty),
            "price": float(exec_price),
            "fee": float(fee),
            "target_weight": float(target_weight),
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.trades.append(trade_record)
        logger.info(
            f"PaperBroker {side} {symbol} | qty: {trade_qty:.4f} @ ${exec_price:.2f} | fee: ${fee:.2f} | equity: ${self.get_balance()['equity']:.2f}"
        )
        return trade_record


class CcxtBroker(Broker):
    """Live/Testnet exchange broker stub. Disabled by default."""

    def __init__(self, live_execution_enabled: bool = False):
        self.live_execution_enabled = live_execution_enabled
        if not self.live_execution_enabled:
            logger.info("CcxtBroker initialized in SAFE mode (live_execution_enabled=False).")

    def get_balance(self) -> Dict[str, float]:
        if not self.live_execution_enabled:
            raise RuntimeError("Live execution is disabled. Enable in config.yaml with caution.")
        return {"cash": 0.0, "equity": 0.0, "positions_value": 0.0}

    def get_positions(self) -> Dict[str, float]:
        if not self.live_execution_enabled:
            raise RuntimeError("Live execution is disabled. Enable in config.yaml with caution.")
        return {}

    def execute_rebalance(
        self, symbol: str, target_weight: float, current_price: float, bar_timestamp: str
    ) -> Optional[Dict]:
        if not self.live_execution_enabled:
            raise RuntimeError("Live execution is disabled. Enable in config.yaml with caution.")
        return None
