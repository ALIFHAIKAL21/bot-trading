"""Unit tests for RiskEngine position sizing and constraints."""

import numpy as np
import pytest

from src.risk.risk_engine import RiskEngine
from src.utils.config import RiskConfig


def test_risk_engine_hard_limits():
    risk = RiskEngine(RiskConfig(max_leverage=1.0, max_position_pct=0.40))
    # High confidence should never exceed max_position_pct (0.40)
    size = risk.calculate_position_size(prob_long=0.99, current_vol_annual=0.15)
    assert size <= 0.40
    assert size > 0.0


def test_risk_engine_bearish_prob_is_zero():
    risk = RiskEngine(RiskConfig(max_position_pct=0.40))
    # Probability below 0.50 (e.g. 0.35) should yield zero spot long size
    size = risk.calculate_position_size(prob_long=0.35, current_vol_annual=0.20)
    assert size == 0.0


def test_risk_engine_crisis_regime_derisking():
    risk = RiskEngine(RiskConfig(max_position_pct=0.40))
    # Regime: 100% in State 2 (Crisis) -> scale should be 0.0
    crisis_regime = np.array([0.0, 0.0, 1.0])
    size = risk.calculate_position_size(
        prob_long=0.80,
        current_vol_annual=0.20,
        regime_probs=crisis_regime,
    )
    assert size == 0.0


def test_risk_engine_volatility_scaling():
    risk = RiskEngine(RiskConfig(target_annual_vol=0.20, max_position_pct=0.40))
    # When vol is double target (0.40 vs 0.20), position size should be halved compared to low vol
    size_normal = risk.calculate_position_size(prob_long=0.70, current_vol_annual=0.20)
    size_high_vol = risk.calculate_position_size(prob_long=0.70, current_vol_annual=0.40)
    assert size_high_vol < size_normal


def test_risk_engine_p_0457_never_buys():
    """Verify F5 audit defect: P=0.457 must NEVER produce a BUY order."""
    risk = RiskEngine(RiskConfig(entry_threshold=0.54, exit_threshold=0.48))
    
    # 1. Raw sizing must be exactly 0.0
    size = risk.calculate_position_size(prob_long=0.457, current_vol_annual=0.30, is_entry=True)
    assert size == 0.0
    
    size_non_entry = risk.calculate_position_size(prob_long=0.457, current_vol_annual=0.30, is_entry=False)
    assert size_non_entry == 0.0

    # 2. Stateful decide_step must reject entry with explicit reason
    decision = risk.decide_step(prob_long=0.457, current_vol_annual=0.30)
    assert decision.action == "FLAT"
    assert decision.target_position == 0.0
    assert decision.reason == "below_entry_threshold"


def test_risk_engine_expected_edge_cost_hurdle():
    """Verify F5: trades are rejected when expected edge <= k * round_trip_cost."""
    # Round-trip cost = 0.0030, k = 2.0 -> Hurdle = 0.0060 (60 bps)
    risk = RiskEngine(RiskConfig(
        entry_threshold=0.54,
        expected_edge_hurdle_multiplier=2.0,
        round_trip_cost=0.0030,
        tp_multiplier=2.0,
        sl_multiplier=1.5,
    ))
    
    # Case A: Probability is 0.55 (above 0.54), but volatility is very compressed (10 bps = 0.0010)
    # E[R] = 0.55 * (2.0 * 0.001) - 0.45 * (1.5 * 0.001) = 0.0011 - 0.000675 = 0.000425 (4.25 bps)
    # 4.25 bps <= 60 bps hurdle -> MUST BE REJECTED
    dec_low_vol = risk.decide_step(prob_long=0.55, current_vol_annual=0.05, bar_volatility=0.0010)
    assert dec_low_vol.action == "FLAT"
    assert dec_low_vol.target_position == 0.0
    assert dec_low_vol.reason == "edge_below_cost_hurdle"

    # Case B: Sufficient volatility (e.g. 100 bps = 0.010) with P=0.60
    # E[R] = 0.60 * 0.020 - 0.40 * 0.015 = 0.012 - 0.006 = 0.0060 (60 bps)
    # With P=0.65 -> E[R] = 0.65 * 0.020 - 0.35 * 0.015 = 0.013 - 0.00525 = 0.00775 > 0.0060 -> ACCEPTS
    dec_pass = risk.decide_step(prob_long=0.65, current_vol_annual=0.30, bar_volatility=0.010)
    assert dec_pass.action == "BUY"
    assert dec_pass.target_position > 0.0
    assert dec_pass.reason == "entry_hurdle_passed"


def test_risk_engine_turnover_hysteresis():
    """Verify F10: Hysteresis band suppresses flip-flopping between 0.49 and 0.53."""
    risk = RiskEngine(RiskConfig(entry_threshold=0.54, exit_threshold=0.48))
    
    # Initial entry at P=0.65
    dec1 = risk.decide_step(prob_long=0.65, current_vol_annual=0.30, bar_volatility=0.01)
    assert dec1.action == "BUY"
    initial_pos = dec1.target_position

    # Probability dips to 0.50 (between 0.48 and 0.54)
    # Should maintain position, not exit or churn
    dec2 = risk.decide_step(prob_long=0.50, current_vol_annual=0.30, bar_volatility=0.01)
    assert dec2.action == "HOLD"
    assert dec2.target_position == initial_pos


def test_risk_engine_turnover_min_holding_period():
    """Verify F10: Minimum holding period suppresses premature voluntary exits."""
    risk = RiskEngine(RiskConfig(
        entry_threshold=0.54,
        exit_threshold=0.48,
        min_holding_bars=3,
        cooldown_bars=2,
    ))
    
    # Bar 1: Enter position
    d1 = risk.decide_step(prob_long=0.65, current_vol_annual=0.30, bar_volatility=0.01)
    assert d1.action == "BUY"
    pos = d1.target_position

    # Bar 2: Signal plummets to 0.40 (below exit threshold), but holding_bars is 1 (< 3)
    d2 = risk.decide_step(prob_long=0.40, current_vol_annual=0.30, bar_volatility=0.01)
    assert d2.action == "HOLD"
    assert d2.target_position == pos
    assert d2.reason == "min_holding_bars_active"

    # Bar 3: Signal still 0.40, holding_bars is 2 (< 3)
    d3 = risk.decide_step(prob_long=0.40, current_vol_annual=0.30, bar_volatility=0.01)
    assert d3.action == "HOLD"
    assert d3.reason == "min_holding_bars_active"

    # Bar 4: holding_bars is now 3 (>= 3) -> Exit permitted!
    d4 = risk.decide_step(prob_long=0.40, current_vol_annual=0.30, bar_volatility=0.01)
    assert d4.action == "SELL"
    assert d4.target_position == 0.0
    assert d4.reason == "exit_threshold_triggered"


def test_risk_engine_turnover_cooldown():
    """Verify F10: Exit cooldown prevents immediate re-entry on subsequent bars."""
    risk = RiskEngine(RiskConfig(
        entry_threshold=0.54,
        exit_threshold=0.48,
        min_holding_bars=1,
        cooldown_bars=2,
    ))
    
    # Enter & exit
    risk.decide_step(prob_long=0.65, current_vol_annual=0.30, bar_volatility=0.01)
    d_exit = risk.decide_step(prob_long=0.40, current_vol_annual=0.30, bar_volatility=0.01)
    assert d_exit.action == "SELL"

    # Immediately on next bar, high probability signal appears
    # Cooldown counter is 2 -> Blocked!
    d_cool1 = risk.decide_step(prob_long=0.70, current_vol_annual=0.30, bar_volatility=0.01)
    assert d_cool1.action == "FLAT"
    assert d_cool1.target_position == 0.0
    assert d_cool1.reason == "cooldown_active"

    # Next bar: Cooldown counter was 1 -> Blocked!
    d_cool2 = risk.decide_step(prob_long=0.70, current_vol_annual=0.30, bar_volatility=0.01)
    assert d_cool2.action == "FLAT"
    assert d_cool2.reason == "cooldown_active"

    # Next bar: Cooldown expired -> Re-entry permitted!
    d_enter = risk.decide_step(prob_long=0.70, current_vol_annual=0.30, bar_volatility=0.01)
    assert d_enter.action == "BUY"
    assert d_enter.target_position > 0.0


def test_risk_engine_daily_trade_cap():
    """Verify F10: Daily trade cap blocks excessive trade churning in 24h window."""
    import pandas as pd
    risk = RiskEngine(RiskConfig(
        entry_threshold=0.54,
        exit_threshold=0.48,
        min_holding_bars=1,
        cooldown_bars=0,
        max_daily_trades=4,
    ))
    
    base_ts = pd.Timestamp("2026-09-20 00:00:00")
    
    # Perform 2 full round trips (4 trades: 2 buys, 2 sells)
    risk.decide_step(prob_long=0.65, current_vol_annual=0.30, bar_volatility=0.01, bar_timestamp=base_ts)
    risk.decide_step(prob_long=0.40, current_vol_annual=0.30, bar_volatility=0.01, bar_timestamp=base_ts + pd.Timedelta(hours=1))
    risk.decide_step(prob_long=0.65, current_vol_annual=0.30, bar_volatility=0.01, bar_timestamp=base_ts + pd.Timedelta(hours=2))
    risk.decide_step(prob_long=0.40, current_vol_annual=0.30, bar_volatility=0.01, bar_timestamp=base_ts + pd.Timedelta(hours=3))

    # 5th trade attempt at hour 4 (within 24h) should be capped
    d_capped = risk.decide_step(prob_long=0.65, current_vol_annual=0.30, bar_volatility=0.01, bar_timestamp=base_ts + pd.Timedelta(hours=4))
    assert d_capped.action == "FLAT"
    assert d_capped.target_position == 0.0
    assert d_capped.reason == "daily_trade_cap_reached"


def test_risk_engine_dust_rebalance_filter():
    """Verify F10: Small rebalancing adjustments (< 5%) are suppressed."""
    risk = RiskEngine(RiskConfig(
        entry_threshold=0.54,
        exit_threshold=0.48,
        dust_rebalance_threshold=0.05,
    ))
    
    # Enter at P=0.65 -> Passes hurdle, size ~ 0.096
    d1 = risk.decide_step(prob_long=0.65, current_vol_annual=0.25, bar_volatility=0.01)
    assert d1.action == "BUY"
    pos1 = d1.target_position

    # Probability shifts slightly to 0.66 -> Size change is ~ 0.004 (< 0.05)
    d2 = risk.decide_step(prob_long=0.66, current_vol_annual=0.25, bar_volatility=0.01)
    assert d2.action == "HOLD"
    assert d2.target_position == pos1  # Exactly unchanged
    assert d2.reason == "dust_rebalance_suppressed"
