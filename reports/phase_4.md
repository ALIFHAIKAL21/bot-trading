# Phase 4 Report: Adaptivity Layer & Empirical Ablation Study

**Status**: COMPLETED  
**Execution Timestamp**: 2026-09-22T03:24:00+07:00  
**Verification**: Verified via `make verify` (44 unit/integration tests passing)  
**Manifests & Artifacts**: `reports/adaptivity_ablation.json`, `reports/latest_run_manifest.json`

---

## 1. Executive Summary

Phase 4 built and evaluated the **Institutional Adaptivity Layer (A1–A7)**, designed to monitor market drift, dynamically re-weight models, safely promote challenger models, and protect capital via circuit breakers.

Per the institutional guideline (*"Conduct ablation study of adaptivity modules on validation set; leave OFF by default anything that does not improve after-cost performance"*), we conducted an empirical ablation study across 23,580 bars of quarantined `VALIDATION` data. The study yielded a profound quantitative insight:
- **Baseline Static Equal-Weight + Turnover Controls**: Achieved a Net Sharpe ratio of **4.298** with +6.46% return and 0.19% max drawdown across 395 trades.
- **Online Dynamic Reweighting (Hedge/EWA)** & **Tight Circuit Breakers**: Shrinking probabilities toward an uninformative prior (0.50) attenuated valid signal amplitudes into the hysteresis deadband ($< 0.54$), and tight circuit breaker thresholds ($0.28$) tripped prematurely on normal binomial market variance.
- **Institutional Decision**: Adaptivity monitoring modules (`DriftMonitor`, `ModelRegistry`, `RollingRetrainer`) are implemented and active in **MONITOR & SHADOW MODE ONLY** by default. Active trade execution retains the proven equal-weight stacking ensemble to prevent false-alarm strategy paralysis.

---

## 2. Implemented Adaptivity Modules (A1–A7)

### A1: Scheduled Rolling & Drift-Driven Retraining (`src/adaptive/retrainer.py`)
- **Purge / Embargo Discipline**: Enforces a strict 12-bar embargo between training windows and out-of-sample evaluation slices, completely eliminating label leakage.
- **Triggers**: Scheduled intervals (e.g. 720 bars / 30 days) and automated triggers when `DriftMonitor` detects critical distribution shifts.

### A2: Online Causal Reweighting (`src/adaptive/online_weights.py`)
- **Hedge / Exponentially Weighted Average (EWA)**: Maintains discounted cumulative Brier loss per model:
  $$L_m(t) = \lambda \cdot L_m(t-1) + (p_{m, t} - y_t)^2$$
- **Causality Guarantee**: Weights for bar $t$ are updated strictly after ground truth $y_t$ is resolved post-embargo. Enforces convex combination ($\sum w_i = 1.0, w_i \ge 0.05$).

### A3: Feature & Performance Drift Monitor (`src/adaptive/drift_monitor.py`)
- **Population Stability Index (PSI)**: Quantile-binned distribution shift detection against baseline reference distributions:
  $$\text{PSI} = \sum_{i=1}^K (P_i - Q_i) \ln\left(\frac{P_i}{Q_i}\right)$$
- **Alert Levels**: `NORMAL` ($\text{PSI} < 0.10$), `WARNING` ($0.10 \le \text{PSI} < 0.25$), `CRITICAL` ($\text{PSI} \ge 0.25$).
- **Rolling Performance Monitor**: Tracks moving Brier loss over a 168-bar evaluation window.

### A4: Auto-Degrade Circuit Breakers (`src/adaptive/model_registry.py`)
- If a champion model's rolling Brier loss exceeds threshold or IC drops below zero, the circuit breaker trips.
- Tripped models are automatically de-weighted to $0.0$ in active execution and moved to `SHADOW` recovery mode.

### A5: Champion / Challenger Model Registry (`src/adaptive/model_registry.py`)
- Versioned model tracking (`CHAMPION`, `CHALLENGER`, `SHADOW`, `DEGRADED`).
- Challenger models run in shadow mode and are promoted to `CHAMPION` only after achieving statistically superior Brier loss over a multi-day test window.
- Instant rollback capability preserves the previous champion snapshot.

### A6 & A7: Dynamic Regime Sizing & Volatility-Scaled Friction
- Position sizing scales dynamically based on HMM forward probabilities: full sizing in State 0 (Low Vol), halved in State 1 (High Vol), zeroed in State 2 (Crisis).
- Execution slippage dynamically scales with $\text{ATR}_{14} / \text{Price}$.

---

## 3. Empirical Ablation Results (`reports/adaptivity_ablation.json`)

Evaluated on 23,580 bars of `BTC/USDT` within the `VALIDATION` partition:

| Configuration | Trades | Turnover | Net Return | Net Sharpe | Max DD | Win Rate | Profit Factor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Baseline (Static Equal-Weight + Phase 3 Controls)** | **395** | **19.04x** | **+6.46%** | **4.298** | **0.19%** | **59.51%** | **2.091** |
| **2. + Online Weights (Hedge/EWA)** | 4 | 0.12x | +0.02% | 0.245 | 0.03% | 62.50% | 1.364 |
| **3. + Auto-Degrade Circuit Breakers** | 0 | 0.00x | +0.00% | 0.000 | 0.00% | 0.00% | 1.000 |
| **4. Full Adaptive Layer (Online + Breakers + Regime)** | 0 | 0.00x | +0.00% | 0.000 | 0.00% | 0.00% | 1.000 |

### Quantitative Diagnoses & Findings:
1. **The Signal Attenuation Effect (Variant 2)**:
   In financial markets where signal-to-noise ratio is low ($P \in [0.52, 0.58]$), blending model predictions with a prior (0.50) attenuates probabilities below the required $0.54$ entry hurdle ($0.55 \times 0.7 + 0.50 \times 0.3 = 0.535 < 0.54$). This caused 99% of profitable entries to be suppressed into the hysteresis deadband.
2. **The Binomial False-Alarm Paradox (Variant 3 & 4)**:
   For a binary outcome with 50% base rate, random variance over a short 24-bar window regularly causes empirical Brier score to reach $(0.55 - 0)^2 = 0.3025$. A static threshold of $0.28$ interprets ordinary short-term market chop as structural model failure, tripping the breaker at bar 24 and paralyzing the bot for the remainder of the backtest.
3. **Institutional Remedy**:
   - Keep dynamic re-weighting and aggressive circuit breakers in **SHADOW / MONITOR-ONLY mode by default** in production configuration.
   - Maintain the static equal-weight ensemble ($w_B=0.5, w_C=0.5$) with Phase 3 cost and turnover controls as the primary active executor.
   - Retain `DriftMonitor` and `ModelRegistry` as observability and alerting systems to inform discretionary oversight and scheduled retrains.

---

## 4. Acceptance Criteria Checklist
- [x] A1: Scheduled rolling / expanding retraining module with causal embargo implemented.
- [x] A2: Online Hedge / EWA dynamic re-weighting module implemented.
- [x] A3: Feature drift monitor (PSI, KS-test) and performance degradation tracker implemented.
- [x] A4: Auto-degrade circuit breakers with shadow recovery implemented.
- [x] A5: Champion / Challenger registry with shadow evaluation and rollback implemented.
- [x] A6 & A7: Dynamic regime-scaled sizing and volatility slippage implemented.
- [x] Ablation study executed on VALIDATION partition; documented in `reports/adaptivity_ablation.json`.
- [x] Institutional rule enforced: underperforming adaptive mechanisms left OFF/SHADOW by default.
- [x] Unit test suite created in `tests/test_adaptivity.py` (5/5 tests passed).
- [x] `make verify` passes (44/44 tests passed).
