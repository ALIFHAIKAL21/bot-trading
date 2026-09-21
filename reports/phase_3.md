# Phase 3 Report: Cost-Aware Decisioning & Turnover Controls

**Status**: COMPLETED  
**Execution Timestamp**: 2026-09-22T03:21:35+07:00  
**Verification**: Verified via `make verify` (39 unit/integration tests passing)  
**Manifests & Artifacts**: `reports/turnover_controls_sweep.json`, `reports/latest_run_manifest.json`

---

## 1. Executive Summary

Phase 3 resolved critical audit defects **F5** (Cost unawareness & "Buy at P=0.457" in live logs) and **F10** (Excessive turnover and fee bleeding). 

By implementing an institutional expected-edge hurdle ($\mathbb{E}[R] > k \times \text{round-trip cost}$) and strict turnover controls (hysteresis deadband, minimum holding period, exit cooldown, daily trade cap, and a dust rebalance filter), strategy turnover was reduced by **82.0%** and trade churn was slashed by **90.6%** on 1-hour BTC/USDT data. Crucially, net Sharpe ratio after all realistic fees and slippage **more than doubled from 2.093 to 4.355**, with maximum drawdown cut by more than half.

---

## 2. Root Cause Analysis & Defect Remediation

### Defect F5: "Buy at P=0.457" & Cost-Unaware Order Routing
- **Root Cause**:
  1. In `src/service/scheduler.py`, a silent `except Exception` catch triggered during live inference fell back to a heuristic formula:
     $$P(\text{long}) = 0.50 + 0.15 \tanh(\text{ret} \times 50) + 0.10 \left(\frac{50 - \text{RSI}}{50}\right) \approx 0.457$$
  2. `scheduler.py` passed this probability directly to `calculate_position_size(prob_long=0.457)`.
  3. In `calculate_position_size`, the Kelly criterion formula with win/loss ratio $b = 1.33$ was:
     $$f^* = \frac{p \cdot b - (1 - p)}{b} = \frac{0.457 \times 1.33 - 0.543}{1.33} = +0.0487 > 0$$
     Kelly sizing produced a positive target position ($1.2\%$) for **any** probability above $\frac{1}{1 + 1.33} = 0.429$, completely bypassing the configured `entry_threshold` ($0.54$)!
  4. The system placed real BUY orders with zero statistical edge and zero cost awareness.
- **Institutional Remediation**:
  1. **Dual Entry Hurdle**: A new long position is permitted **if and only if**:
     $$P(\text{long}) \ge \text{entry\_threshold} \quad (0.54) \quad \text{AND} \quad \mathbb{E}[R] > k \times \text{round-trip cost}$$
     Where round-trip cost is $2 \times (\text{fee} + \text{slippage}) = 30\text{ bps}$, and $k = 1.0$ (30 bps hurdle).
  2. **Strict Non-Negative Kelly Rule**: Any probability $P \le \text{exit\_threshold}$ ($0.48$) or $P < 0.50$ unconditionally yields target weight $0.0$.
  3. **Structured Decision Metadata**: Every signal bar produces a machine-readable `RiskDecision` recording `{action, target_position, prob_long, expected_edge, hurdle_cost, reason, metadata}`.

### Defect F10: Turnover Churn & Fee Bleeding
- **Institutional Turnover Controls**:
  1. **Entry/Exit Hysteresis Deadband**: $p_{\text{in}} = 0.54 > p_{\text{out}} = 0.48$. When in the deadband $[0.48, 0.54]$, existing positions are held with zero rebalancing or flip-flopping (`reason="hysteresis_band_hold"`).
  2. **Minimum Holding Period**: Positions must be held for at least $N_{\text{min}} = 3$ bars before voluntary exit (`reason="min_holding_bars_active"`), unless stopped out.
  3. **Exit Cooldown**: After closing a position, re-entry is blocked for $N_{\text{cooldown}} = 2$ bars (`reason="cooldown_active"`).
  4. **Daily Trade Cap**: Capped at $\le 6$ trades per rolling 24-hour window (`reason="daily_trade_cap_reached"`).
  5. **Dust Order Filter**: Position size changes $< 5\%$ are suppressed (`reason="dust_rebalance_suppressed"`).
  6. **Hard Stop Loss**: Open position drawdown $\le -3\%$ triggers immediate emergency liquidation regardless of holding period (`reason="hard_stop_loss_triggered"`).

---

## 3. Empirical Results: Validation Parameter Sweep

Conducted across 23,580 bars (2.69 years: Sept 2023 – June 2026) of `BTC/USDT` strictly quarantined within the `VALIDATION` partition via `HoldoutGuard`. The `LOCKED_TEST` partition (June 2026 – Sept 2026) remained sealed.

### Performance Summary Table
| Metric | 1H Raw (No Controls) | 1H Controlled (Phase 3) | Delta / Improvement | 4H Controlled | 1D Controlled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Trades** | 2,003 | **188** | **-90.61%** | 85 | 6 |
| **Total Turnover** | 35.11x | **6.31x** | **-82.03%** | 2.88x | 0.15x |
| **Annualized Turnover** | 13.04x / yr | **2.34x / yr** | **-10.70x / yr** | 1.07x / yr | 0.06x / yr |
| **Win Rate** | 52.01% | **61.36%** | **+9.35%** | 77.01% | 100.0% |
| **Profit Factor** | 1.300 | **2.628** | **+1.328** | 4.690 | 30.100 |
| **Net Sharpe Ratio** | 2.093 | **4.355** | **+2.262 (+108%)** | 5.177 | 4.945 |
| **Sortino Ratio** | 1.623 | **2.206** | **+0.583** | 3.080 | 20.466 |
| **Max Drawdown** | 0.17% | **0.08%** | **-53% (Halved)** | 0.10% | 0.01% |
| **Cumulative Net Return** | 2.24% | **3.50%** | **+1.26%** | 2.21% | 0.55% |
| **Annualized Net Return** | 0.82% / yr | **1.29% / yr** | **+0.47% / yr** | 3.30% / yr | 5.00% / yr |

---

## 4. Signal Action & Decision Reason Distribution (1H Controlled)

Across all 23,580 evaluated validation bars:
- `below_entry_threshold`: **22,374 bars** (94.9%) — Signal rejected cleanly below 0.54 hurdle.
- `dust_rebalance_suppressed`: **238 bars** (1.0%) — Tiny adjustments $< 5\%$ eliminated.
- `hysteresis_band_hold`: **216 bars** (0.9%) — Position maintained in deadband without churn.
- `edge_below_cost_hurdle`: **188 bars** (0.8%) — Signals $\ge 0.54$ blocked because expected payoff could not beat round-trip fees.
- `cooldown_active`: **180 bars** (0.8%) — Immediate re-entries blocked post-exit.
- `crisis_regime_blocked`: **176 bars** (0.7%) — Bullish entries blocked during high-volatility crisis states.
- `entry_hurdle_passed`: **90 entries** (0.4%) — High-conviction trades that satisfied every hurdle.
- `exit_threshold_triggered`: **64 exits** (0.3%) — Disciplined exits below 0.48 after holding period.
- `crisis_regime_exit`: **26 exits** (0.1%) — Immediate de-risking on crisis regime detection.
- `min_holding_bars_active`: **20 bars** (0.1%) — Premature whipsaw exits suppressed.
- `rebalance_position_adjusted`: **8 bars** — Meaningful position adjustments ($\ge 5\%$).

---

## 5. Acceptance Criteria Checklist
- [x] Implemented expected-edge hurdle ($P \ge \text{threshold}$ AND $\mathbb{E}[R] > k \times \text{round-trip cost}$).
- [x] Completely eradicated "Buy at P=0.457" defect (tested and proved via unit test).
- [x] Structured decision metadata emitted for every signal and stored in database.
- [x] Implemented turnover controls: entry/exit hysteresis, minimum holding period, exit cooldown, daily trade cap, dust order filter.
- [x] Parameter sweep across timeframes (1h, 4h, 1d) on VALIDATION data saved to `reports/turnover_controls_sweep.json`.
- [x] Turnover reduced by 82.0%, trades reduced by 90.6%, Net Sharpe improved from 2.093 to 4.355.
- [x] Comprehensive unit tests added to `tests/test_risk.py` (11/11 passed).
- [x] `make verify` passes (39/39 tests passed).
