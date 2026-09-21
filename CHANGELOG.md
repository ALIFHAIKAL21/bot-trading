# Changelog

All notable behavior changes, security enhancements, defect corrections, and quant modifications to this project are documented here with explicit rationales.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Phase 0: Safety & Reproducibility] - 2026-09-22

### Added
- **API Token Authentication**: Added `verify_api_token` dependency requiring `X-API-Token` header or `Authorization: Bearer <token>` on all state-modifying FastAPI routes (`/trade/step`). Configurable via `API_TOKEN` environment variable.
- **Dual Confirmation Live Trading Interlock (F14)**: Live trading now strictly requires **BOTH** `service.live_execution_enabled: true` in `config.yaml` AND the environment variable `CONFIRM_LIVE=YES`. In the absence of `CONFIRM_LIVE=YES`, the system logs a loud safety warning and automatically falls back to `PAPER` mode.
- **Binance Testnet Sandbox Default (F14)**: Live execution defaults to Binance Testnet unless explicitly set via `BINANCE_USE_TESTNET=false`.
- **Withdrawal Permission Rejection (F14)**: Added `validate_api_key_safety` which actively checks API key capabilities and aborts startup immediately if withdrawal permissions are detected.
- **Mode & Active Models Banners (F13, F14)**: Added prominent ASCII banners logged at startup, during backtests, and in paper scheduling:
  - `PAPER MODE`: Virtual simulation, real funds safe.
  - `TESTNET MODE`: Binance sandbox, real funds safe.
  - `LIVE MODE`: Production exchange, real funds at risk.
  - `ACTIVE MODELS AUDIT`: Explicit audit table showing active models (Model A, B, C, D, E, Meta-Learner) with zero silent skipping.
- **Run Manifest Generation**: Automated capture of SHA256 config checksum, git commit hash, seeds, data provenance, hardware, and active models into timestamped `reports/run_manifest_{timestamp}.json` and `reports/latest_run_manifest.json`.
- **Institutional Verification Target (`make verify`)**: Master verification script `scripts/verify.py` integrating environment checks, full pytest test suite, and run manifest emission.

### Changed
- `src/service/app.py`:
  - Added lifespan management displaying mode and active model banners on boot.
  - Protected `POST /trade/step` with `Depends(verify_api_token)`.
  - Added `execution_mode`, `active_models`, and `auth_required_for_post` to `GET /health` endpoint.
- `.env.example`:
  - Added `API_TOKEN`, `CONFIRM_LIVE`, and `BINANCE_USE_TESTNET` configuration templates with clear security documentation.
- `Makefile`:
  - Added `verify` target pointing to `python scripts/verify.py`.

### Security
- Closed unauthenticated access vector on FastAPI service.
- Prevented accidental live-order routing through dual-gate environment verification.

---

## [Phase 1: Data, Features & Leakage Eradication] - 2026-09-22

### Added
- **CCXT Binance Vision Public Fetcher (F3)**: Upgraded data pipeline to use Binance Vision public API (`https://data-api.binance.vision/api/v3`), bypassing Indonesian ISP / Telkomsel transparent proxy hijacking. Paginates 3 full years (26,280 1h bars) in `float64` precision.
- **Zero-Volume & Gap Detection (F3)**: Added automated tagging (`is_zero_volume`, `is_gap_filled`) and exclusion from volume feature baseline distribution. Emits `reports/data_quality.json` and `reports/data_manifest.json`.
- **Feature Whitelist Enforcement & Leakage Assertion (F1)**: Defined 31 explicit alpha features in `config.yaml`. Added `validate_feature_columns()` raising loud `LeakageViolationError` if any column matches `r'(label|weight|target|fwd|forward)'` or is absent from whitelist.
- **Hold-Out Quarantine Discipline**: Created `src/data/holdout_guard.py` partitioning data into VALIDATION (90%, 23,652 bars) and `LOCKED_TEST` (10%, 2,628 bars). Any access to `LOCKED_TEST` outside `scripts/final_report.py` raises `HoldoutViolationError`.
- **Causality & Leakage Verification Tests**: Added comprehensive test suite in `tests/test_leakage.py` covering truncate-future exact match, label-shuffle collapse, HTF shift causality, whitelist enforcement, and holdout guard.

### Changed
- `src/features/feature_pipeline.py`:
  - Enforced strict whitelist validation.
  - Excluded zero-volume/gap bars when computing rolling volume z-scores.
- `src/models/tabular_model.py`:
  - Completely purged `sample_weight` from feature lists and removed fallback weight padding hack.
- `config/config.yaml`:
  - Added explicit `features.whitelist` containing 31 causal alpha features.

### Empirical Quant Findings (F1, F2, F7)
- **Sample Weight Leakage (F1)**: Eradicated `sample_weight` from LightGBM inputs. CV AUC adjusted slightly from 0.5837 to 0.5829 (-0.0008), but **Brier score improved from 0.2159 to 0.2061** and fold variance dropped from 0.0397 to 0.0265.
- **HTF 1-Day Trend (F2)**: Removing `htf_1d_trend` reduced mean validation AUC by -0.0059 (0.5829 to 0.5770). Truncate-future tests confirmed 0 look-ahead bias across all daily boundaries; the feature's rank 1 importance reflects real crypto daily momentum.
- **"500k Dataset" Label (F7)**: Documented that the 527,040 rows in previous deep training correspond to 1-minute raw bars, not 1-hour strategy bars (which total 26,280 bars over 3 years).

---

## [Phase 2: Honest Model Rebuilds & Stacking Diagnostics] - 2026-09-22

### Added
- **Model E (HMM) Volatility State Alignment (F6)**: Fitted `StandardScaler` strictly inside training folds; sorted hidden states deterministically by realized volatility ($\sigma_0 < \sigma_1 < \sigma_2$) to eliminate label switching across re-fits. Solved state collapse: Regime 0 (49.8%), Regime 1 (24.3%), Regime 2 (26.0%). Emits `reports/hmm_report.json`.
- **Model D (Sentiment Gate) Network Hardening (F6)**: Configured realistic browser `User-Agent` headers, 10s timeout, retry logic, and headline deduplication. Replaced silent 0.0 fallback with honest `degraded=True` signaling.
- **Model C (Deep Sequence) Regularized Architecture (F8)**: Rebuilt network with lightweight MultiTask sequence architecture (`d_model=32`, 2 layers, `GroupNorm(4, 32)`, Dropout=0.35, Weight Decay=1e-3). Early stopping with patience=5 improved validation AUC from 0.513 to 0.5581.
- **Constrained SLSQP Stacker with Fallback Protocol (F9)**: Stacker constrained to non-negative convex combinations ($\sum w_i = 1, w_i \ge 0$). Verified protocol: defaults to equal weights ($w_B=0.5, w_C=0.5$) when meta-learner does not strictly outperform simple average on Brier score or net Sharpe.
- **Model A (Chronos) Standalone Diagnostics (F13)**: Evaluated standalone predictive metrics on validation set (AUC = 0.5195, IC = -0.0242).

---

## [Phase 3: Cost-Aware Decisioning & Turnover Controls] - 2026-09-22

### Added
- **Institutional Expected-Edge Hurdle (F5)**: Enforced rule that a new long position is permitted **if and only if** $P(\text{long}) \ge \text{entry\_threshold}$ AND expected trade edge $\mathbb{E}[R] > k \times \text{round-trip cost}$ ($k=1.0$, 30 bps hurdle).
- **Structured Decision Metadata (F5)**: Every signal evaluation produces a `RiskDecision` recording action (`BUY`, `SELL`, `HOLD`, `FLAT`), target weight, previous weight, expected edge, cost hurdle, and explicit machine-readable reason (e.g. `below_entry_threshold`, `edge_below_cost_hurdle`, `hysteresis_band_hold`, `dust_rebalance_suppressed`).
- **Turnover Controls (F10)**:
  - **Entry/Exit Hysteresis Deadband**: $p_{\text{in}} = 0.54 > p_{\text{out}} = 0.48$. Suppresses churn in the deadband $[0.48, 0.54]$.
  - **Minimum Holding Period**: Enforces minimum holding of $N_{\text{min}} = 3$ bars before voluntary exit.
  - **Exit Cooldown**: Enforces $N_{\text{cooldown}} = 2$ bars post-exit before re-entry.
  - **Daily Trade Cap**: Limits maximum executions to $\le 6$ trades per rolling 24-hour window.
  - **Dust Order Filter**: Suppresses position rebalances where delta is $< 5\%$.
  - **Hard Stop Loss Override**: Triggers emergency liquidation if open position drawdown $\le -3\%$.
- **Validation Timeframe Sweep (`scripts/sweep_turnover_timeframes.py`)**: Evaluated 1h, 4h, and 1d on 23,580 validation bars (Sept 2023 – June 2026) quarantined via `HoldoutGuard`. Emits `reports/turnover_controls_sweep.json`.
- **Unit Tests (`tests/test_risk.py`)**: Added 7 new unit tests verifying $P=0.457$ never buys, expected edge hurdle, hysteresis band, min holding, cooldown, trade cap, and dust filter (11/11 passed).

### Fixed
- **"Buy at P=0.457" Defect (F5)**: Identified root cause: fallback heuristic in `scheduler.py` generated $P \approx 0.457$, and fractional Kelly formula with $b=1.33$ yielded positive size for any $p > 0.429$, bypassing entry thresholds. Fixed by hard-gating `calculate_position_size` and `decide_step` to require $P \ge \text{entry\_threshold}$ and $P > 0.50$.

### Empirical Quant Findings (F5, F10)
- **1-Hour BTC/USDT Performance**:
  - Trade churn slashed by **90.61%** (from 2,003 trades to 188 trades).
  - Annualized turnover reduced by **82.03%** (from 13.04x to 2.34x per year).
  - Net Sharpe ratio after realistic fees and slippage **more than doubled from 2.093 to 4.355 (+2.262)**.
  - Win rate improved from 52.01% to 61.36% (+9.35 percentage points).
  - Max Drawdown cut by more than half from 0.17% to 0.08%.

---

## [Phase 4: Adaptivity Layer & Empirical Ablation Study] - 2026-09-22

### Added
- **Online Causal Model Reweighting (A2)**: Implemented `OnlineWeightManager` executing the Hedge / Exponentially Weighted Average (EWA) algorithm updating model weights strictly post-embargo.
- **Feature & Performance Drift Monitor (A3)**: Implemented `DriftMonitor` calculating Population Stability Index (PSI) per feature against training baselines and tracking rolling Brier loss.
- **Champion / Challenger Model Registry (A4, A5)**: Implemented `ModelRegistry` with auto-degrade circuit breakers, shadow evaluation, and automated rollback snapshots.
- **Scheduled & Drift Retrainer (A1)**: Implemented `RollingRetrainer` with strict 12-bar causal embargo slicing.
- **Adaptivity Unit Tests (`tests/test_adaptivity.py`)**: Added 5 comprehensive unit tests (5/5 passed). Full test suite now contains 44 passed tests.
- **Validation Ablation Study (`scripts/eval_adaptivity_ablation.py`)**: Evaluated static vs dynamic reweighting vs circuit breakers across 23,580 validation bars (`reports/adaptivity_ablation.json`).

### Empirical Quant Findings & Architectural Decision
- **Signal Attenuation & False Alarm Diagnosis**: Dynamic re-weighting towards an uninformative prior attenuated probabilities into the deadband, while tight circuit breakers ($0.28$) tripped on normal binomial market noise.
- **Default Configuration**: Per institutional guidelines, adaptivity modules are deployed in **MONITOR & SHADOW MODE ONLY** by default, keeping the static equal-weight ensemble with Phase 3 turnover controls as the primary active executor (Net Sharpe 4.298).

---

## [Phase 5: Trustworthy Evaluation, Baselines & Locked Test Scorecard] - 2026-09-22

### Added
- **Oracle Look-Ahead Impossibility Unit Test**: Implemented unit test in `tests/test_backtest.py` mathematically proving that manipulating future bars has zero effect on past signals, and verifying next-bar open execution shifts.
- **Fair Quantitative Baselines (F12)**: Integrated Buy-and-Hold, 200-simulation turnover-matched Monte Carlo random trading agent, and Time-Shuffled signal baseline.
- **Deflated Sharpe Ratio (DSR) & Bootstrap CIs**: Implemented DSR penalizing for 15 cumulative trials and 1,000-sample circular block bootstrap 95% confidence intervals on Sharpe ratio.
- **Sealed Out-of-Sample Scorecard (`scripts/final_report.py`)**: Unlocked the quarantined `LOCKED_TEST` partition (June 4, 2026 – Sept 21, 2026, 2,628 bars) under strict single-execution discipline. Emits `reports/final_report.json` and `reports/summary.md`.

### Empirical Findings & Institutional Verdict
- **Unseen Out-of-Sample Scorecard (`LOCKED_TEST`)**:
  - Primary Model B Strategy: Cumulative Net Return = -0.01%, Net Sharpe = -0.121 (95% CI: [-4.96, +2.72]), Max Drawdown = 0.18%, Trades = 33, Turnover = 1.25x.
  - Buy-and-Hold: Cumulative Return = +36.71%, Net Sharpe = 2.792, Max Drawdown = 13.38%.
  - Time-Shuffled Baseline: Return = -0.56%, Net Sharpe = -6.461.
  - Monte Carlo Random Baseline: Mean Return = -87.30%, Mean Sharpe = -22.590.
- **Institutional Verdict**: Explicitly rendered **NO-GO FOR LIVE DEPLOYMENT**. While turnover controls preserved capital flawlessly (0.18% max drawdown vs 13.38% for BTC), the strategy did not demonstrate positive statistical alpha after friction during this bull regime.

---

## [Phase 6: Paper-Trading Correctness & Live Parity] - 2026-09-22

### Added
- **Closed-Candle Execution Guard (F4)**: Added timestamp assertion `bar_close_utc + 5s <= now_utc` in `src/service/scheduler.py`. Forming candles are discarded or shifted to the previous closed candle (`iloc[-2]`), preventing mid-bar signal flickering.
- **SQLite Post-Processing Idempotency Ledger (F4)**: Implemented `processed_bars` table schema with unique `(symbol, bar_timestamp)` constraint. Records completion key with action, decision reason, and target weight, preventing duplicate order generation on scheduler restarts.
- **Fail-Safe Neutral Cash Mode (F5, F6)**: When models raise errors or become degraded, the scheduler defaults to $P(\text{long}) = 0.50$, Confidence = $0.0$, and Target Weight = $0.0$, eliminating rogue buy orders.
- **Paper Execution Tests (`tests/test_paper.py`)**: Added restart idempotency safety test, forming candle skip test, and paper broker vs backtester parity test. Total test suite expanded to 47 passing tests.
- **Streamlit Dashboard Overhaul (F15)**: Fixed PyArrow type conversion exception (`ArrowInvalid: Could not convert 'N/A' to double`). Added prominent `PAPER` / `TESTNET` / `LIVE` mode badges, explicit institutional `NO-GO FOR LIVE` banner, true percentage return axes, and adaptivity/drift telemetry.

---

## [Phase 7: Documentation & Final Acceptance Verdict] - 2026-09-22

### Added
- **Final System Documentation**:
  - `README.md`: Comprehensive before/after metrics table, step-by-step reproduction instructions, known limitations, and institutional NO-GO scorecard.
  - `ROADMAP.md`: Strategic research roadmap outlining multi-asset universe expansion, funding rate arbitrage, cross-sectional alpha, and reinforcement learning execution.
  - `DECISIONS.md`: Fully documented architectural and mathematical decisions covering Phases 0 through 7.
- **Unattended 7-Day Paper-Testing Runbook**: Standardized SOP for unattended operational testing, monitoring, daily health checks, and failure recovery.
- **Master Verification (`make verify`)**: 100% test pass rate across all 47 unit/integration tests with automated cryptographic run manifest generation.

