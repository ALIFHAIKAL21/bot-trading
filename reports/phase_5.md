# Phase 5 Report: Trustworthy Evaluation, Baselines & Locked Out-of-Sample Scorecard

**Status**: COMPLETED  
**Execution Timestamp**: 2026-09-22T03:26:35+07:00  
**Verification**: Verified via `make verify` (44 unit/integration tests passing)  
**Manifests & Artifacts**: `reports/final_report.json`, `reports/summary.md`, `reports/backtest_results.json`

---

## 1. Executive Summary & Institutional Verdict

Phase 5 conducted the final, sealed evaluation of the refactored multi-model quant trading pipeline, resolving audit defects **F11** (Ablation testing on active trades) and **F12** (Missing and unfair baselines).

Following strict quarantine discipline, the `LOCKED_TEST` partition (June 4, 2026 to Sept 21, 2026, 2,628 bars) was unlocked once via `scripts/final_report.py`. 

### Institutional Verdict: **NO-GO FOR LIVE CAPITAL DEPLOYMENT**
- **Rationale**: On the strictly quarantined, untouched out-of-sample period (`LOCKED_TEST`), the strategy achieved an after-cost Net Sharpe Ratio of **-0.121** (95% CI: [-4.96, +2.72]) with a net return of **-0.01%** across 33 trades. While turnover controls successfully preserved capital (limiting maximum drawdown to **0.18%** compared to **13.38%** for Buy-and-Hold), the model lacked statistical directional alpha during this trending regime to overcome round-trip transaction friction.
- **Honest Recommendation**: **DO NOT DEPLOY LIVE REAL CAPITAL**. Continue paper-trading and research on multi-timeframe regime features.

---

## 2. Fair Baselines & Benchmark Comparison (F12)

Evaluated on the 2,628 hourly bars of `BTC/USDT` in `LOCKED_TEST`:

| Strategy / Benchmark | Return (Cum) | Net Sharpe | Sortino | Max DD | Win Rate | Profit Factor | Turnover |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Model B (Turnover Controlled)** | **-0.01%** | **-0.121** | **-0.050** | **0.18%** | **45.07%** | **0.973** | **1.25x** |
| **Meta Stacking Model** | 0.00% | 0.000 | 0.000 | 0.00% | 0.00% | 1.000 | 0.00x |
| **Buy-and-Hold Benchmark (100% Spot)** | +36.71% | 2.792 | 3.869 | 13.38% | 50.48% | 1.100 | 3.33x |
| **Time-Shuffled Signal Baseline** | -0.56% | -6.461 | -3.571 | 0.65% | 22.54% | 0.365 | 18.79x |
| **Monte Carlo Random (Mean / 95th)** | -87.30% | -22.590 | - | - | - | - | - |

---

## 3. Statistical Significance & Deflated Sharpe Ratio (DSR)

- **Circular Block Bootstrap (1,000 resamples, 24-bar blocks)**:
  - 95% Confidence Interval on Sharpe: `[-4.96, +2.72]`.
  - The confidence interval spans zero, confirming that positive out-of-sample performance cannot be statistically claimed at the 95% confidence level on this test slice.
- **Deflated Sharpe Ratio (DSR)**:
  - Calculated with $N=15$ total historical hyperparameter and model trials: $\text{DSR} = 0.000$.
  - Adjusts for non-normal return distributions (skewness and excess kurtosis) and multiple testing bias (Bailey & López de Prado, 2014).

---

## 4. Signal Action & Risk Decision Breakdown

Across the 2,628 bars of `LOCKED_TEST`:
- `below_entry_threshold`: **2,471 bars** (94.0%) — Sub-threshold signals successfully blocked.
- `cooldown_active`: **32 bars** — Re-entry suppressed during post-exit cooldown.
- `edge_below_cost_hurdle`: **31 bars** — Marginal signals rejected because expected payoff $< 30\text{ bps}$.
- `dust_rebalance_suppressed`: **26 bars** — Micro adjustments $< 5\%$ eliminated.
- `hysteresis_band_hold`: **21 bars** — Position held stably in the $[0.48, 0.54]$ deadband.
- `entry_hurdle_passed`: **16 entries** — Trades executed that satisfied all hurdle criteria.
- `exit_threshold_triggered`: **11 exits** — Disciplined exits below 0.48.
- `crisis_regime_blocked`: **7 bars** — High volatility regime blocks.
- `min_holding_bars_active`: **7 bars** — Premature noise exits suppressed.
- `crisis_regime_exit`: **5 exits** — Immediate risk-off liquidation upon crisis regime transition.

---

## 5. Acceptance Criteria Checklist
- [x] Event-consistent backtest engine verified with next-bar execution delay and realistic transaction costs.
- [x] Oracle look-ahead impossibility unit test passing in `tests/test_backtest.py`.
- [x] Fair baselines constructed and evaluated (Buy-and-Hold, 200-run Monte Carlo random, time-shuffled signal).
- [x] Deflated Sharpe Ratio (DSR) and circular block bootstrap 95% CIs computed.
- [x] Quarantined `LOCKED_TEST` partition unlocked strictly once via `scripts/final_report.py`.
- [x] `reports/final_report.json` and `reports/summary.md` emitted with un-manipulated empirical numbers.
- [x] Explicit, honest institutional verdict rendered (**NO-GO FOR LIVE**).
- [x] `make verify` passes (44/44 tests passed).
