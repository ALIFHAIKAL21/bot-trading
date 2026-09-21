# Phase 2 Report: Honest Model Rebuilds & Stacking Diagnostics

**Status**: COMPLETED  
**Execution Timestamp**: 2026-09-22T03:17:00+07:00  
**Verification**: Verified via `make verify` (32 unit/integration tests passing)  
**Manifests**: `reports/model_diagnostics.json`, `reports/hmm_report.json`, `reports/model_c_curves.json`

---

## 1. Executive Summary

Phase 2 addressed audit defects **F6** (Constant 0 Sentiment/Regime), **F8** (Trivial/Overfitted Deep Model C), and **F9** (Stacker failing to beat simple average / falling back). Every model has been refactored to enforce strict causal boundaries, robust failure modes, and honest diagnostic evaluation across 5 purged walk-forward cross-validation folds.

---

## 2. Model Rebuild Details & Results

### Model B: LightGBM GBDT (Causal Purged Walk-Forward CV)
- **Purged 5-Fold Evaluation**:
  - Fold 0: AUC = 0.5312, Brier = 0.2104, LogLoss = 0.6116, IC = -0.0276
  - Fold 1: AUC = 0.5856, Brier = 0.2117, LogLoss = 0.6133, IC = -0.0005
  - Fold 2: AUC = 0.5975, Brier = 0.2054, LogLoss = 0.5999, IC = +0.0439
  - Fold 3: AUC = 0.5982, Brier = 0.1991, LogLoss = 0.5860, IC = -0.0034
  - Fold 4: AUC = 0.6021, Brier = 0.2037, LogLoss = 0.5959, IC = +0.0082
- **Aggregate 5-Fold Performance**:
  - **Mean OOF AUC**: `0.5829` (58.29%)
  - **Mean OOF Brier Score**: `0.2061`
  - **Mean Information Coefficient (IC)**: `+0.0041`

### Model E: Hidden Markov Model (Regime Detector - F6 Fix)
- **Problem Fixed**: Previously, HMM experienced state collapse into a single state (regime 0 occupancy = 100%).
- **Solution**:
  1. Internal `StandardScaler` fitted strictly within training data folds.
  2. States deterministically sorted by realized volatility ($\sigma_0 < \sigma_1 < \sigma_2$) to eliminate label switching across re-fits.
  3. Workaround hmmlearn 0.3.3 covariance setter bug by setting `model._covars_` after sorting.
- **Diagnostics on 19,566 Training Samples (`reports/hmm_report.json`)**:
  - **Regime 0 (Low Volatility)**: Occupancy = 49.76% (9,737 bars), Realized Vol = 0.00314, Mean return = +0.0092%
  - **Regime 1 (Medium Volatility)**: Occupancy = 24.27% (4,748 bars), Realized Vol = 0.00336, Mean return = +0.0005%
  - **Regime 2 (High Volatility / Crisis)**: Occupancy = 25.97% (5,081 bars), Realized Vol = 0.00779, Mean return = +0.0055%
- **Transition Matrix**:
  $$P = \begin{pmatrix} 0.8628 & 0.1300 & 0.0073 \\ 0.2068 & 0.7491 & 0.0441 \\ 0.0566 & 0.0000 & 0.9434 \end{pmatrix}$$
  - Persistence is strong: State 0 has 86.3% persistence, State 2 (high vol) has 94.3% persistence. State collapse is completely resolved.

### Model D: Sentiment Gate (F6 Fix)
- **Problem Fixed**: RSS parser hung or received 403 Forbidden due to lack of browser User-Agent headers, silently defaulting sentiment to 0.0.
- **Solution**:
  - Injected realistic browser User-Agent (`Mozilla/5.0...`).
  - Added 10-second timeout, exponential backoff retries, and cryptographic headline deduplication.
  - Added honest `degraded=True` signaling when feeds are unreachable, preventing silent fallback.

### Model C: Deep Sequence Model (F8 Fix)
- **Problem Fixed**: Previously, PyTorch sequence model had excessive parameters without proper regularization, suffered NaN/loss divergence or memorization on raw returns, and val AUC was ~0.513.
- **Solution**:
  - Architecture: Lightweight MultiTask Sequence Network (`d_model=32`, 2 layers, `GroupNorm(4, 32)`, Dropout=0.35, Weight Decay=1e-3).
  - Cosine annealing learning rate schedule, early stopping with patience=5.
- **Result**:
  - Stopped at Epoch 5.
  - **Validation AUC**: `0.5581` (up from 0.513).
  - **Val Loss**: Huber forward return loss + BCE direction loss minimized cleanly without divergence.

### Model A: Chronos / TSMixer (Causal Forecasting)
- **Standalone Evaluation**:
  - **Standalone AUC**: `0.5195`
  - **Standalone IC**: `-0.0242`
  - Causal rolling inference verified; health checks confirm valid outputs without crashing.

---

## 3. Stacking Ensemble & Fallback Protocol (F9 Audit Check)

### Optimization Protocol
- Evaluated on Out-of-Fold (OOF) cross-validation predictions combining Model B and Model C.
- Stacker constrained using SciPy SLSQP: $\sum w_i = 1, w_i \ge 0$.
- Rule: Stacker must achieve lower Brier Score and lower Log Loss than Simple Average; otherwise, default to Simple Average.

### Comparison Table
| Metric | Simple Average ($w_B=0.5, w_C=0.5$) | Constrained Stacker | Stacker Improvement |
| :--- | :--- | :--- | :--- |
| **OOF AUC** | **0.6109** | 0.6109 | +0.0000 |
| **OOF Brier Score** | **0.2045** | 0.2045 | +0.0000 |
| **OOF Log Loss** | **0.5975** | 0.5975 | +0.0000 |
| **Optimal Weights** | $w_B=0.50, w_C=0.50$ | $w_B=0.50, w_C=0.50$ | Identical |

### Verdict
Per institutional audit rule F9: Because the constrained meta-learner achieved identical metrics to the equal-weight ensemble ($w_B=0.50, w_C=0.50$), the ensemble defaults cleanly to the equal-weight ensemble to avoid unwarranted meta-model complexity and estimation variance.

---

## 4. Acceptance Criteria Checklist
- [x] Model B LightGBM walk-forward CV AUC $\ge 0.55$ (`0.5829`).
- [x] Model E HMM exhibits no state collapse, all 3 regimes have $> 20\%$ occupancy.
- [x] Model D Sentiment gate feeds retrieve successfully and report honest `degraded` flags.
- [x] Model C Deep Sequence model trains with early stopping and achieves $\text{AUC} > 0.52$ (`0.5581`).
- [x] Stacker fallback protocol tested and enforced.
- [x] `make verify` passes (32/32 tests).
