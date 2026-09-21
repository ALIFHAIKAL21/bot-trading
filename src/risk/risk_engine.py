"""Institutional Risk Engine with Cost-Aware Decisioning & Turnover Controls.

Audit Defect Corrections:
- F5: Cost-unaware trading and "Buy at P=0.457" fixed.
  * Enter long ONLY if calibrated P(long) >= entry_threshold AND expected edge E[R] > k * round_trip_cost.
  * Continuous position sizing strictly yields 0.0 when P <= exit_threshold or P < entry_threshold on entry.
  * Structured decision reason metadata emitted for every single signal evaluation.
- F10: Turnover controls implemented to eradicate excessive churn and fee bleeding.
  * Entry/Exit hysteresis band (p_in > p_out).
  * Minimum holding period (N_min bars) to prevent rapid bar-to-bar whipsawing.
  * Exit cooldown (N_cooldown bars) after closing a position.
  * Daily trade cap (max trades within rolling 24-hour window).
  * Dust order filter (suppress rebalances with < 5% target change).
  * Hard stop loss override on adverse open position movement.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import RiskConfig


@dataclass
class RiskDecision:
    """Structured decision metadata for an individual signal bar."""
    action: str  # "BUY", "SELL", "HOLD", "FLAT"
    target_position: float  # In [0.0, max_position_pct]
    prev_position: float
    prob_long: float
    expected_edge: float
    hurdle_cost: float
    reason: str  # Machine and human-readable decision reason
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskEngine:
    """Institutional-grade risk management and turnover-controlled position sizing engine."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.reset_state()

    def reset_state(self) -> None:
        """Reset internal state tracking for fresh backtests or live runs."""
        self.current_position: float = 0.0
        self.holding_bars: int = 0
        self.cooldown_remaining: int = 0
        self.entry_price: Optional[float] = None
        self.trade_timestamps: List[pd.Timestamp] = []
        self.last_decisions: List[RiskDecision] = []

    def calculate_expected_edge(
        self,
        prob_long: float,
        bar_volatility: float,
        expected_ret: Optional[float] = None,
    ) -> float:
        """Calculate expected net return edge per trade.
        
        If an explicit forward return forecast (e.g. from sequence or regression model)
        is available, that estimate is used.
        Otherwise, computes probability-weighted barrier payoff:
        E[R] = p * (tp_multiplier * vol) - (1 - p) * (sl_multiplier * vol).
        """
        if expected_ret is not None and not np.isnan(expected_ret) and expected_ret != 0.0:
            return float(expected_ret)

        base_vol = bar_volatility if (bar_volatility > 0 and not np.isnan(bar_volatility)) else (self.config.target_annual_vol / np.sqrt(8760.0))
        horizon_mult = np.sqrt(max(self.config.trade_horizon_bars, 1))
        vol = base_vol * horizon_mult
        tp = self.config.tp_multiplier * vol
        sl = self.config.sl_multiplier * vol
        expected_edge = prob_long * tp - (1.0 - prob_long) * sl
        return float(expected_edge)

    def calculate_position_size(
        self,
        prob_long: float,
        current_vol_annual: float,
        regime_probs: Optional[np.ndarray] = None,
        win_loss_ratio: float = 1.33,
        is_entry: bool = False,
    ) -> float:
        """Calculate continuous target position size in [0.0, max_position_pct].
        
        Enforces:
        - Strict zero sizing for probabilities below exit threshold.
        - Strict zero sizing for probabilities below entry threshold when initiating entry.
        - Fractional Kelly scaling.
        - Volatility targeting scale.
        - Regime derisking (flat in crisis regime).
        """
        # Hard Rule: if entering a new position, probability must strictly exceed entry threshold
        if is_entry and prob_long < self.config.entry_threshold:
            return 0.0

        # Hard Rule: any probability at or below exit threshold (or < 0.50) cannot justify a long position
        if prob_long <= self.config.exit_threshold or prob_long < 0.50:
            return 0.0

        # 1. Fractional Kelly Sizing
        p = np.clip(prob_long, 0.01, 0.99)
        b = max(win_loss_ratio, 0.5)
        raw_kelly = (p * b - (1.0 - p)) / b
        kelly_size = max(raw_kelly, 0.0) * self.config.kelly_fraction

        if kelly_size <= 0:
            return 0.0

        # 2. Volatility Targeting Scale
        # target_vol / realized_vol, capped at 1.5x
        if current_vol_annual > 0:
            vol_scale = min(self.config.target_annual_vol / current_vol_annual, 1.5)
        else:
            vol_scale = 1.0

        # 3. Regime Scaling
        # State 0: Normal (1.0), State 1: High Vol (0.5), State 2: Crisis (0.0)
        regime_scale = 1.0
        if regime_probs is not None and len(regime_probs) >= 3:
            regime_scale = (
                1.0 * regime_probs[0] + 0.5 * regime_probs[1] + 0.0 * regime_probs[2]
            )

        # Combined position size
        target_size = kelly_size * vol_scale * regime_scale

        # 4. Hard Allocation Cap
        target_size = min(target_size, self.config.max_position_pct)
        target_size = min(target_size, self.config.max_leverage)

        return float(np.clip(target_size, 0.0, self.config.max_position_pct))

    def decide_step(
        self,
        prob_long: float,
        current_vol_annual: float,
        bar_volatility: Optional[float] = None,
        regime_probs: Optional[np.ndarray] = None,
        expected_ret: Optional[float] = None,
        bar_timestamp: Optional[pd.Timestamp] = None,
        current_price: Optional[float] = None,
    ) -> RiskDecision:
        """Evaluate a single bar signal through cost hurdle and turnover state machine."""
        bar_vol = bar_volatility if (bar_volatility is not None and bar_volatility > 0) else (current_vol_annual / np.sqrt(8760.0))
        expected_edge = self.calculate_expected_edge(prob_long, bar_vol, expected_ret)
        hurdle_cost = self.config.expected_edge_hurdle_multiplier * self.config.round_trip_cost

        meta: Dict[str, Any] = {
            "prob_long": float(prob_long),
            "expected_edge": float(expected_edge),
            "hurdle_cost": float(hurdle_cost),
            "current_vol_annual": float(current_vol_annual),
            "holding_bars": self.holding_bars,
            "cooldown_remaining": self.cooldown_remaining,
        }

        # Prune trade timestamp window (rolling 24 hours)
        ts_obj = None
        if bar_timestamp is not None:
            ts_obj = pd.to_datetime(bar_timestamp)
            cutoff = ts_obj - pd.Timedelta(hours=24)
            self.trade_timestamps = [pd.to_datetime(ts) for ts in self.trade_timestamps if pd.to_datetime(ts) >= cutoff]
            meta["daily_trades_count"] = len(self.trade_timestamps)

        # 1. Cooldown Handling
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.current_position == 0.0:
                decision = RiskDecision(
                    action="FLAT",
                    target_position=0.0,
                    prev_position=0.0,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="cooldown_active",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision

        # 2. Case: Currently FLAT (Evaluating for Entry)
        if self.current_position == 0.0:
            # Check daily trade cap
            if len(self.trade_timestamps) >= self.config.max_daily_trades:
                decision = RiskDecision(
                    action="FLAT",
                    target_position=0.0,
                    prev_position=0.0,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="daily_trade_cap_reached",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision

            # Check directional entry threshold
            if prob_long < self.config.entry_threshold:
                decision = RiskDecision(
                    action="FLAT",
                    target_position=0.0,
                    prev_position=0.0,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="below_entry_threshold",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision

            # Check crisis regime
            if regime_probs is not None and len(regime_probs) >= 3 and regime_probs[2] >= 0.70:
                decision = RiskDecision(
                    action="FLAT",
                    target_position=0.0,
                    prev_position=0.0,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="crisis_regime_blocked",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision

            # Check expected edge hurdle vs round-trip transaction costs
            if expected_edge <= hurdle_cost:
                decision = RiskDecision(
                    action="FLAT",
                    target_position=0.0,
                    prev_position=0.0,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="edge_below_cost_hurdle",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision

            # Passed all hurdles -> Compute sizing and enter
            target_size = self.calculate_position_size(
                prob_long=prob_long,
                current_vol_annual=current_vol_annual,
                regime_probs=regime_probs,
                is_entry=True,
            )

            if target_size <= 0.0:
                decision = RiskDecision(
                    action="FLAT",
                    target_position=0.0,
                    prev_position=0.0,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="kelly_size_zero",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision

            # Execute entry
            self.current_position = target_size
            self.holding_bars = 0
            self.entry_price = current_price
            if bar_timestamp is not None:
                self.trade_timestamps.append(bar_timestamp)

            decision = RiskDecision(
                action="BUY",
                target_position=target_size,
                prev_position=0.0,
                prob_long=prob_long,
                expected_edge=expected_edge,
                hurdle_cost=hurdle_cost,
                reason="entry_hurdle_passed",
                metadata=meta,
            )
            self.last_decisions.append(decision)
            return decision

        # 3. Case: Currently LONG (Evaluating for Hold / Exit / Rebalance)
        self.holding_bars += 1
        prev_pos = self.current_position

        # Hard stop-loss check
        if current_price is not None and self.entry_price is not None and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            meta["unrealized_pnl"] = float(unrealized_pnl)
            if unrealized_pnl <= -self.config.hard_stop_loss_pct:
                self.current_position = 0.0
                self.cooldown_remaining = self.config.cooldown_bars
                self.holding_bars = 0
                self.entry_price = None
                if bar_timestamp is not None:
                    self.trade_timestamps.append(bar_timestamp)
                decision = RiskDecision(
                    action="SELL",
                    target_position=0.0,
                    prev_position=prev_pos,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="hard_stop_loss_triggered",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision

        # Crisis regime emergency exit
        if regime_probs is not None and len(regime_probs) >= 3 and regime_probs[2] >= 0.70:
            self.current_position = 0.0
            self.cooldown_remaining = self.config.cooldown_bars
            self.holding_bars = 0
            self.entry_price = None
            if bar_timestamp is not None:
                self.trade_timestamps.append(bar_timestamp)
            decision = RiskDecision(
                action="SELL",
                target_position=0.0,
                prev_position=prev_pos,
                prob_long=prob_long,
                expected_edge=expected_edge,
                hurdle_cost=hurdle_cost,
                reason="crisis_regime_exit",
                metadata=meta,
            )
            self.last_decisions.append(decision)
            return decision

        # Exit threshold check
        if prob_long <= self.config.exit_threshold:
            if self.holding_bars < self.config.min_holding_bars:
                # Suppress early exit to satisfy minimum holding period
                decision = RiskDecision(
                    action="HOLD",
                    target_position=prev_pos,
                    prev_position=prev_pos,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="min_holding_bars_active",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision
            else:
                # Normal threshold exit
                self.current_position = 0.0
                self.cooldown_remaining = self.config.cooldown_bars
                self.holding_bars = 0
                self.entry_price = None
                if bar_timestamp is not None:
                    self.trade_timestamps.append(bar_timestamp)
                decision = RiskDecision(
                    action="SELL",
                    target_position=0.0,
                    prev_position=prev_pos,
                    prob_long=prob_long,
                    expected_edge=expected_edge,
                    hurdle_cost=hurdle_cost,
                    reason="exit_threshold_triggered",
                    metadata=meta,
                )
                self.last_decisions.append(decision)
                return decision

        # Hysteresis deadband: Between exit_threshold and entry_threshold
        # Maintain position; do not churn or resize
        if prob_long < self.config.entry_threshold:
            decision = RiskDecision(
                action="HOLD",
                target_position=prev_pos,
                prev_position=prev_pos,
                prob_long=prob_long,
                expected_edge=expected_edge,
                hurdle_cost=hurdle_cost,
                reason="hysteresis_band_hold",
                metadata=meta,
            )
            self.last_decisions.append(decision)
            return decision

        # Probability >= entry_threshold: Position adjustment / dust filter check
        new_target = self.calculate_position_size(
            prob_long=prob_long,
            current_vol_annual=current_vol_annual,
            regime_probs=regime_probs,
            is_entry=False,
        )

        if abs(new_target - prev_pos) < self.config.dust_rebalance_threshold:
            # Maintain position; suppress dust rebalance
            decision = RiskDecision(
                action="HOLD",
                target_position=prev_pos,
                prev_position=prev_pos,
                prob_long=prob_long,
                expected_edge=expected_edge,
                hurdle_cost=hurdle_cost,
                reason="dust_rebalance_suppressed",
                metadata=meta,
            )
            self.last_decisions.append(decision)
            return decision

        # Update position size smoothly
        self.current_position = new_target
        decision = RiskDecision(
            action="HOLD",
            target_position=new_target,
            prev_position=prev_pos,
            prob_long=prob_long,
            expected_edge=expected_edge,
            hurdle_cost=hurdle_cost,
            reason="rebalance_position_adjusted",
            metadata=meta,
        )
        self.last_decisions.append(decision)
        return decision

    def generate_positions_vectorized(
        self,
        meta_probs: pd.Series,
        realized_vols: pd.Series,
        regime_df: Optional[pd.DataFrame] = None,
        expected_rets: Optional[pd.Series] = None,
        close_prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Simulate sequential positions applying full cost hurdle and turnover controls."""
        self.reset_state()
        n = len(meta_probs)
        positions = np.zeros(n, dtype=np.float64)

        probs = meta_probs.values
        ann_vols = realized_vols.values * np.sqrt(8760.0)
        bar_vols = realized_vols.values
        reg_probs = regime_df.values if regime_df is not None else None
        rets = expected_rets.values if expected_rets is not None else None
        prices = close_prices.values if close_prices is not None else None
        timestamps = meta_probs.index

        for i in range(n):
            p = float(probs[i])
            vol_ann = float(ann_vols[i]) if not np.isnan(ann_vols[i]) else self.config.target_annual_vol
            vol_bar = float(bar_vols[i]) if not np.isnan(bar_vols[i]) else (vol_ann / np.sqrt(8760.0))
            r_prob = reg_probs[i] if reg_probs is not None else None
            exp_ret = float(rets[i]) if rets is not None else None
            price = float(prices[i]) if prices is not None else None
            ts = timestamps[i] if isinstance(timestamps[i], pd.Timestamp) else None

            decision = self.decide_step(
                prob_long=p,
                current_vol_annual=vol_ann,
                bar_volatility=vol_bar,
                regime_probs=r_prob,
                expected_ret=exp_ret,
                bar_timestamp=ts,
                current_price=price,
            )
            positions[i] = decision.target_position

        return pd.Series(positions, index=meta_probs.index, name="target_position")

    def get_decision_summary(self) -> Dict[str, Any]:
        """Aggregate diagnostic statistics from recorded decisions."""
        if not self.last_decisions:
            return {"total_bars": 0}

        actions = [d.action for d in self.last_decisions]
        reasons = [d.reason for d in self.last_decisions]
        
        reason_counts = pd.Series(reasons).value_counts().to_dict()
        action_counts = pd.Series(actions).value_counts().to_dict()
        
        # Calculate turnover and trade counts
        positions = [d.target_position for d in self.last_decisions]
        deltas = np.diff([0.0] + positions)
        non_zero_trades = np.count_nonzero(deltas)
        total_turnover = float(np.sum(np.abs(deltas)))

        return {
            "total_bars": len(self.last_decisions),
            "total_trades": non_zero_trades,
            "total_turnover": total_turnover,
            "action_counts": action_counts,
            "reason_counts": reason_counts,
        }
