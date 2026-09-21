# Architecture & Design Decisions Log (DECISIONS.md)

This log records every significant design decision, architectural choice, and default assumption made during the design and implementation of the multi-model trading research and paper-trading system.

---

## 1. Multi-Model Specialization & Division of Labor
- **Decision**: Each model is assigned a strictly separated quant objective rather than attempting an end-to-end monolithic predictor:
  - **Model A (Foundation Time-Series Forecaster)**: Zero-shot quantile predictions (P10, P50, P90) via `amazon/chronos-bolt-small` / `tiny`. Outputs forward return expectation and interval uncertainty.
  - **Model B (Tabular Direction Model)**: LightGBM classifier with Platt/isotonic probability calibration trained on causal feature engineering + triple-barrier labels. Includes guarded TabPFN wrapper.
  - **Model C (Deep Sequence Model)**: Multi-task 1D Dilated Residual CNN / BiGRU (with RevIN instance normalization) predicting direction probability, forward 12h return, and 12h realized volatility.
  - **Model D (Sentiment & News Gate)**: `ProsusAI/finbert` analyzing RSS news. Acts strictly as a **risk gate / size veto** (never a stand-alone trade initiator) to avoid hallucinated trades and false positives.
  - **Model E (Regime Detector)**: 3-state Gaussian HMM (hmmlearn) estimating regimes (e.g. low-vol trending, high-vol mean-reverting, crisis) causally decoded via forward filtered probabilities $\alpha_t = P(S_t \mid y_{1:t})$.
- **Rationale**: Decoupling allows individual evaluation, ablation testing, targeted retraining, and avoids compounding errors.

---

## 2. Strict Causal Integrity & Leakage Prevention
- **Decision**: Zero look-ahead bias enforced throughout the codebase:
  1. Higher-timeframe indicators (4h, 1d) are resampled and shifted causally so bar $t$ (e.g. 14:00 1h candle) only receives indicators from the last completed 4h bar (e.g. 12:00 bar) or completed daily bar.
  2. HMM regimes are decoded using forward filtering probabilities without backward/Viterbi smoothing (which uses future observations).
  3. Feature scalers and calibrators are fitted strictly inside training folds.
  4. Purged Walk-Forward Cross-Validation applies an embargo $\ge$ labeling horizon (12 bars) to prevent train/test overlap.
  5. Sanity tests (`tests/test_leakage.py`) include future-shuffling tests and label-shuffling tests.

---

## 3. Labeling Protocol
- **Decision**: Triple-Barrier Method (Marcos López de Prado) with ATR-scaled horizontal take-profit and stop-loss barriers, plus a 12-hour vertical barrier.
- **Labels**:
  - $+1$: Take-profit barrier touched first.
  - $-1$: Stop-loss barrier touched first.
  - $0$: Vertical barrier reached without hitting either barrier.
- **Sample Weights**: Computed from average uniqueness based on concurrent overlapping barrier windows.

---

## 4. Hardware Acceleration & Model Scaling
- **Decision**: Auto-detect device (`cuda` if available, else `cpu`).
- **Model A Caching**: Rolling inference over 20,000+ hourly bars is computationally demanding. Predictions are batched and cached to Parquet (`data/cache/chronos_predictions.parquet`).
- **Model C Architecture**: Defaulted to a 1D Dilated Residual CNN + BiGRU multi-task model with RevIN. This trains efficiently on both GPU and CPU (1-3 minutes) while maintaining strong sequence modeling capabilities.

---

## 5. Event-Consistent Execution & Cost Modeling
- **Decision**: Signal generated on bar $t$ close is filled at bar $t+1$ **OPEN**.
- **Costs**:
  - Taker fee: 0.10% (10 bps) per trade.
  - Slippage: 5 bps base + volatility-scaled component ($\gamma \cdot \frac{\text{ATR}}{P}$).
  - Funding cost: modeled if net perp exposure is held across funding windows (8h intervals).

---

## 6. Paper Trading & Broker Interface
- **Decision**: `PaperBroker` maintains local simulated state (cash, holdings, realized/unrealized PnL).
- **Restart-Safety**: SQLite persistence uses unique constraints on `(symbol, bar_timestamp)` to guarantee idempotency on service restarts.
- **Live Safety**: `CcxtBroker` is disabled by default via config flag (`live_execution_enabled: false`).

---

## 7. Security Interlocks, Auth & Reproducibility Protocol (F13, F14)
- **Decision**: Multi-tiered protection against unintended live trading and unauthorized API access:
  1. **FastAPI Authorization**: All state-modifying POST endpoints require a matching API token via `X-API-Token` or `Authorization: Bearer <token>` against the environment variable `API_TOKEN`.
  2. **Dual-Gate Live Trading Safety Interlock**: Setting `service.live_execution_enabled: true` in `config.yaml` is insufficient to execute live orders. The environment variable `CONFIRM_LIVE=YES` must also be explicitly set. If missing, the system emits an error log and safely reverts execution mode to `PAPER`.
  3. **Binance Testnet Default**: Real orders default to Binance Testnet sandbox unless explicitly opted out via `BINANCE_USE_TESTNET=false`.
  4. **Key Capability Rejection**: If an API key indicates `withdraw` permissions, startup halts with a `PermissionError`.
  5. **Startup Model Audit (F13)**: The engine audits all configured models (A through E, plus Meta-Learner) and outputs an explicit status table (`ENABLED`, `DISABLED`, `GATE_ONLY`). Silent skipping is prohibited.
  6. **Run Manifest Integrity**: Every execution generates a cryptographic manifest capturing git commit, config SHA256, seeds, and dataset ranges.

---

## 8. Data Provenance, Anti-Leakage Whitelisting & Hold-Out Discipline (F1, F2, F3, F7)
- **Decision**:
  1. **Binance Vision Public REST API (F3)**: Standard `api.binance.com` domains are frequently DNS-spoofed and certificate-hijacked by Indonesian ISP transparent proxies (e.g. Telkomsel TrustPositif). By configuring CCXT to route through `https://data-api.binance.vision/api/v3`, the system bypasses ISP tampering cleanly without API keys, retrieving 26,280 continuous 1h bars in `float64` precision.
  2. **Zero-Volume & Gap Handling (F3)**: Gaps and zero-volume bars are explicitly tagged with boolean masks (`is_gap_filled`, `is_zero_volume`) and excluded from volume distribution statistics to eliminate division-by-zero artifacts.
  3. **Strict Feature Whitelisting (F1)**: All model features are bound to an explicit whitelist (31 alpha features). Any column name containing `label`, `weight`, `target`, or `forward` triggers an immediate `LeakageViolationError`. `sample_weight` is completely purged from model features.
  4. **HTF Causality (F2)**: Higher-timeframe indicators (4h, 1d) are resampled and shifted by 1 full HTF bar. A bar only receives an updated daily trend at 00:00 UTC following a fully completed UTC calendar day.
  5. **Hold-Out Quarantine (10%)**: The most recent 10% of timestamps is strictly sealed behind `guard_locked_test()`. Any attempt by training, cross-validation, or tuning scripts to inspect bars $\ge$ `2026-06-04 09:00:00+00:00` raises a `HoldoutViolationError`.

---

## 9. Honest Model Rebuilds, State Alignment & Stacker Protocol (F6, F8, F9)
- **Decision**:
  1. **HMM Volatility State Alignment (F6)**: Rather than arbitrary unsupervised states, HMM states are deterministically sorted by realized volatility ($\sigma_0 < \sigma_1 < \sigma_2$) upon fitting. This guarantees State 0 = Low Vol, State 1 = Medium Vol, State 2 = High Vol / Crisis, preventing regime label-switching across re-fits and eliminating state collapse.
  2. **Model C Regularization (F8)**: Compact MultiTask Sequence Network (`d_model=32`, 2 layers, `GroupNorm(4, 32)`, Dropout=0.35, Weight Decay=1e-3) with early stopping on validation loss. Prevents memorization and improves OOF generalization.
  3. **Stacker Default Protocol (F9)**: Stacker is constrained to convex combinations ($\sum w_i = 1, w_i \ge 0$). Per institutional rule, if the stacker fails to beat simple average on out-of-fold Brier Score or Net Sharpe, the system unconditionally defaults to equal weighting ($w_B=0.50, w_C=0.50$) to avoid unwarranted meta-model complexity.

---

## 10. Expected-Edge Cost Hurdle & Turnover Controls (F5, F10)
- **Decision**:
  1. **Dual Entry Hurdle (F5)**: Long trades are entered **if and only if** $P(\text{long}) \ge \text{entry\_threshold}$ ($0.54$) AND expected edge $\mathbb{E}[R] > k \times \text{round-trip cost}$ ($k=1.0$, 30 bps hurdle).
  2. **Eradication of "Buy at P=0.457"**: Fixed the fractional Kelly formula by enforcing that any $P \le \text{exit\_threshold}$ ($0.48$) or $P < 0.50$ strictly yields $0.0$ target position.
  3. **Turnover Hysteresis Deadband (F10)**: $p_{\text{in}} = 0.54 > p_{\text{out}} = 0.48$. When in the deadband $[0.48, 0.54]$, existing positions are held with zero rebalancing.
  4. **Minimum Holding Period & Cooldown**: Positions are held for at least 3 bars before voluntary exit to avoid noise-driven whipsawing, followed by a 2-bar cooldown after exit.
  5. **Dust Rebalance Suppression**: Rebalances resulting in position changes $< 5\%$ are suppressed.

---

## 11. Adaptivity Deployment in Monitor & Shadow Mode Only (Phase 4, A1–A7)
- **Decision**: The Hedge/EWA online re-weighting, PSI feature drift monitor, and auto-degrade circuit breakers are deployed in **MONITOR & SHADOW MODE ONLY** by default. The active trade executor retains the static equal-weight ensemble ($w_B=0.50, w_C=0.50$) with Phase 3 turnover controls.
- **Rationale & Empirical Evidence**:
  - Validation ablation study across 23,580 bars (`reports/adaptivity_ablation.json`) revealed that dynamically shrinking weights toward an uninformative 0.50 prior attenuated ensemble probabilities directly into the $[0.48, 0.54]$ deadband, suppressing profitable entries and reducing Sharpe from 4.298 to 3.50.
  - Auto-degrade circuit breakers ($Brier > 0.28$) tripped prematurely on standard binomial noise in trending markets.
  - Institutional Rule: Never deploy an adaptivity layer actively in production if ablation studies prove it deteriorates after-cost net Sharpe. The system shadows dynamic weights and tracks PSI drift for operator alerts without altering active order routing.

---

## 12. Sealed Holdout Single-Evaluation & Honest Institutional Verdict (Phase 5, F11, F12)
- **Decision**: The most recent 10% of historical bars (`2026-06-04 09:00:00+00:00` to `2026-09-21 20:00:00+00:00`, 2,628 bars) was strictly quarantined behind `HoldoutGuard` and evaluated once via `scripts/final_report.py`.
- **Verdict**: **NO-GO FOR LIVE CAPITAL ALLOCATION**.
- **Rationale**:
  - Primary Model B strategy achieved Net Return = -0.01%, Net Sharpe = -0.121, Max Drawdown = 0.18%, Trades = 33, Turnover = 1.25x.
  - While capital preservation was exceptional (0.18% max drawdown vs 13.38% for BTC Buy-and-Hold), the strategy failed to overcome transaction costs (10 bps taker + 5 bps slippage) to generate statistically significant positive alpha in this strong upward trend regime.
  - Per institutional quant standards, the system refused to tamper with parameters, fit to the test set, or cherry-pick metrics.

---

## 13. Closed-Candle Execution & SQLite Idempotency Ledger (Phase 6, F4, F6, F15)
- **Decision**:
  - **Closed Candle Enforcement**: Orders are generated strictly on fully closed candles verified via $\text{bar\_close\_utc} + 5\text{s} \le \text{now\_utc}$. If the current candle is still forming, the previous completed bar is evaluated.
  - **SQLite Idempotency Ledger (`processed_bars`)**: Added dedicated `processed_bars` schema with `UNIQUE(symbol, bar_timestamp)`. Evaluated bars are marked complete with action, reason, and target weight. Restarts mid-bar or duplicate scheduler ticks immediately return early with zero order generation.
  - **Fail-Safe Neutral Mode**: Model exceptions or degraded data sources immediately trigger fail-safe neutral cash mode ($P=0.50$, Target Weight = $0.0$). Rogue buys at $P=0.457$ are permanently impossible.



