# Multi-Model Quantitative Trading Research & Paper-Trading Platform

An institutional-grade, multi-model quantitative research, adaptivity, and paper-trading platform built with strict causality, cost-aware decisioning, and absolute metric honesty.

---

## Institutional Deployment Verdict: **NO-GO FOR LIVE TRADING**

```
========================================================================================
                          INSTITUTIONAL COMMITTEE VERDICT: NO-GO
========================================================================================
Status:                 REJECTED FOR LIVE CAPITAL ALLOCATION
Primary Strategy:       Model B LightGBM Directional + Institutional Turnover Controls
Evaluation Partition:   Quarantined Out-of-Sample Holdout (2026-06-04 to 2026-09-21, 2,628 bars)
Net Cumulative Return:  -0.01% (after 10 bps taker fee + 5 bps base slippage + ATR impact)
Net Annualized Sharpe:  -0.1212 (95% Bootstrap CI: [-4.96, +2.72])
Deflated Sharpe (DSR):  0.0000 (p-value adjusted for 15 cumulative trials)
Capital Preservation:   EXCEPTIONAL: Max Drawdown 0.18% vs Buy & Hold 13.38%
Core Defect:            Alpha signal edge failed to hurdle realistic taker exchange friction
========================================================================================
```

> [!WARNING]
> **Strict Metric Honesty & Anti-Fabrication Commitment**:
> All statistics, tables, and performance metrics in this repository are derived directly from machine-executable runs and sealed JSON reports (`reports/final_report.json`, `reports/turnover_controls_sweep.json`, `reports/adaptivity_ablation.json`). No backtest results are cherry-picked or fabricated.

---

## 1. System Architecture

```
                               ┌────────────────────────┐
                               │   Market Data Layer    │
                               │ Binance Vision Public  │
                               │ (Anti-ISP Spoofing)    │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Causal Feature Engine  │
                               │ 31 Whitelisted Alphas  │
                               │ Zero Future Leakage    │
                               └───────────┬────────────┘
                                           │
              ┌──────────────┬─────────────┼──────────────┬──────────────┐
              ▼              ▼             ▼              ▼              ▼
        ┌───────────┐  ┌───────────┐ ┌───────────┐  ┌───────────┐  ┌───────────┐
        │  Model A  │  │  Model B  │ │  Model C  │  │  Model D  │  │  Model E  │
        │ChronosBolt│  │ LightGBM  │ │Deep Seq   │  │FinBERT RSS│  │Causal HMM │
        │Zero-Shot  │  │Calibrated │ │GroupNorm  │  │Degraded   │  │Sorted Vol │
        │Forecaster │  │Directional│ │Multi-Task │  │Risk Gate  │  │3 Regimes  │
        └─────┬─────┘  └─────┬─────┘ └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │             │              │              │
              └──────────────┴───────┬─────┴──────────────┴──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Stacking Meta-Model      │
                      │  Convex Constrained SLSQP   │
                      │ Default: Simple Avg (w=0.5) │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Turnover Risk Engine (F5)  │
                      │ • P(long) >= 0.54 Hurdle    │
                      │ • E[R] > Cost (30 bps)      │
                      │ • Deadband [0.48, 0.54]     │
                      │ • Min Hold (3b) / Cooldown  │
                      │ • Daily Cap (<= 6 trades)   │
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      ▼                             ▼
        ┌───────────────────────────┐ ┌───────────────────────────┐
        │ Event-Consistent Backtest │ │   Paper Trading Service   │
        │ Next-Bar Open Execution   │ │ FastAPI (Auth Protected)  │
        │ 10 bps Taker + Slippage   │ │ SQLite Idempotency Ledger │
        │ DSR + Bootstrap CIs       │ │ Closed-Candle Only (F4)   │
        └───────────────────────────┘ └───────────────────────────┘
```

---

## 2. Before vs After: Audit Remediation Summary

| Defect / Component | Legacy Status (Audit Finding) | Refactored Status | Empirical Evidence |
| :--- | :--- | :--- | :--- |
| **F1: Target Leakage** | `sample_weight` fed into tabular model features | Purged entirely; regex assertion enforces whitelist | Brier score improved 0.2159 $\rightarrow$ 0.2061; fold variance -33% |
| **F2: HTF Leakage** | Suspected look-ahead in 4h/1d resample | Causal 1-bar lag verified; truncate-future test passes | Truncate-future diff = 0.00000; AUC drop = -0.0059 without trend |
| **F3: Data Quality** | Indonesian ISP Telkomsel proxy spoofing / gaps | Switched to Binance Vision API; zero-vol tagging | Continuous 26,280 1h bars (3 years); 0 division-by-zero errors |
| **F4: Execution Leakage**| Mid-bar order execution; duplicate rebalances | Closed candle only (`ts + tf + 5s <= now`); SQLite ledger | `processed_bars` table guarantees restart idempotency |
| **F5: "Buy at P=0.457"** | Bought whenever Kelly positive ($p > 0.429$) | Gated by $P \ge 0.54$ AND $\mathbb{E}[R] > 30\text{ bps}$ hurdle | Never buys below 0.50; all decisions logged with machine reasons |
| **F6: HMM Collapse** | State collapse into regime 0; RSS silent crash | Sorted states by volatility ($\sigma_0 < \sigma_1 < \sigma_2$); degraded flag | Regimes: 49.8% Low, 24.3% Med, 26.0% High; RSS degraded safe |
| **F7: Dataset Claims** | Ambiguity between 500k 1m bars and 1h bars | Documented 527k 1m bars for deep seq vs 26k 1h bars | Label hygiene strictly segregated by timeframe |
| **F8: Deep Overfitting**| Model C overparameterized; memorized training | Lightweight MultiTask with GroupNorm, Dropout 0.35 | Val AUC improved 0.5130 $\rightarrow$ 0.5581 with early stopping |
| **F9: Stacker Failure** | Stacker unconstrained; failed against average | Constrained SLSQP default to equal-weight ($w=0.50$) | Simple average preserved net Sharpe without arbitrary weights |
| **F10: Churn & Costs** | 2,003 trades on validation; 13.0x turnover | Hysteresis deadband, min holding (3b), exit cooldown | Trades slashed by **90.6%** (188); Sharpe: **2.09 $\rightarrow$ 4.35** |
| **F11: Evaluation Rigor**| Optimistic assumptions; no cost drag modeling | Next-bar open fill, 10 bps fee, ATR dynamic slippage | Backtest strictly event-consistent; oracle test passes |
| **F12: Weak Baselines** | No statistical baselines or trial penalties | Buy & Hold, 200-run Monte Carlo, Time-Shuffled, DSR | Monte Carlo Luck Sharpe: -17.92; DSR: 0.00 |
| **F13: Silent Skips** | Models silently skipped when dependencies failed | Startup active model audit table; explicit degraded flags | No silent skipping; loud log banner at boot |
| **F14: Live Safety** | Config-only toggle risked accidental live trading | Dual-confirmation (`CONFIRM_LIVE=YES`), testnet default | API withdrawal permission rejection active |
| **F15: Dashboard Bugs** | PyArrow crashed on `"N/A"`; misleading axes | Fixed types; honest % return axes; NO-GO banner | Interactive Streamlit running stably with live telemetry |

---

## 3. Out-of-Sample Scorecard (`LOCKED_TEST`)

The sealed holdout partition comprises the most recent 10% of historical bars (**2,628 hours / ~3.6 months**, `2026-06-04 09:00:00+00:00` to `2026-09-21 20:00:00+00:00`).

| Metric | Primary Model B + Controls | Buy & Hold (BTC) | Time-Shuffled Baseline | Monte Carlo Random (200 runs) |
| :--- | :---: | :---: | :---: | :---: |
| **Cumulative Return** | **-0.01%** | +36.71% | -0.56% | -87.30% (mean) |
| **Annualized Sharpe** | **-0.12** | +2.79 | -6.46 | -22.59 (mean) |
| **Sharpe 95% Bootstrap CI** | **[-4.96, +2.72]** | N/A | N/A | Max Luck: -17.92 |
| **Deflated Sharpe (DSR)** | **0.000** | N/A | N/A | N/A |
| **Max Drawdown** | **0.18%** | 13.38% | 0.65% | N/A |
| **Win Rate** | **45.07%** | 50.48% | 22.54% | N/A |
| **Profit Factor** | **0.97** | 1.10 | 0.37 | N/A |
| **Total Completed Trades** | **33** | 1 | 33 | Matched |
| **Annualized Turnover** | **4.17x** | 3.33x | 18.79x | Matched |

### Quant Analysis of the Result
1. **Capital Preservation**: The risk engine successfully protected capital, experiencing a maximum drawdown of only **0.18%** while Bitcoin suffered a 13.38% drawdown during the period.
2. **Transaction Drag**: The strategy generated 33 trades across 3.6 months. After 10 bps taker fees and 5 bps slippage, gross edge was erased, yielding net return of -0.01%.
3. **Verdict Rationale**: Deploying capital into a negative after-cost Sharpe system violates fiduciary risk standards. The system remains strictly in **PAPER TRADING** mode.

---

## 4. Quickstart & Complete Reproduction

### 4.1 Environment Setup
```powershell
# Create Python 3.10 virtual environment
uv venv .venv --python 3.10
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

### 4.2 Automated Master Verification (`make verify`)
Executes safety checks, full 47-test pytest suite, and generates a cryptographic run manifest:
```powershell
.venv\Scripts\python.exe scripts/verify.py
```

### 4.3 Step-by-Step Pipeline Reproduction

```powershell
# 1. Ingest clean 3-year historical dataset from Binance Vision (anti-spoofing)
.venv\Scripts\python.exe scripts/download_data.py

# 2. Train regularized deep sequence model (Model C)
.venv\Scripts\python.exe scripts/train_deep.py

# 3. Execute validation set backtesting against baselines
.venv\Scripts\python.exe scripts/backtest.py

# 4. Run turnover controls sweep across timeframes (1h, 4h, 1d)
.venv\Scripts\python.exe scripts/sweep_turnover_timeframes.py

# 5. Run adaptivity layer ablation study (Hedge, PSI, circuit breakers)
.venv\Scripts\python.exe scripts/eval_adaptivity_ablation.py

# 6. Execute sealed locked holdout evaluation (Generates final scorecard)
.venv\Scripts\python.exe scripts/final_report.py
```

### 4.4 Paper Trading Service & Dashboard
```powershell
# Start continuous paper-trading scheduler loop (1-hour closed candles)
.venv\Scripts\python.exe run.py --paper-loop

# Launch interactive Streamlit analytics dashboard
.venv\Scripts\python.exe run.py --dashboard
```
Dashboard is accessible locally at `http://localhost:8502`.

---

## 5. Adaptivity Layer Architecture (A1–A7)

- **A1 (Rolling Retraining)**: Scheduled rolling/expanding retrainer with causal 12-bar embargo.
- **A2 (Online Reweighting)**: Multi-expert Hedge / Exponentially Weighted Average (EWA) updating model weights post-embargo.
- **A3 (Drift Monitoring)**: Population Stability Index (PSI) per feature against training baselines + rolling Brier loss tracker.
- **A4 (Circuit Breakers)**: Automated zero-weighting on models exceeding Brier degradation limits.
- **A5 (Model Registry)**: Champion / challenger versioning with automated rollback.
- **A6 (Regime-Aware Sizing)**: Dynamic Kelly scaling conditioned on HMM volatility regime.
- **A7 (Dynamic Slippage)**: Volatility-scaled execution friction ($\text{fee} + \text{base} + \gamma \frac{\text{ATR}}{P}$).

> [!NOTE]
> **Operational Status**: In accordance with the validation ablation study (`reports/adaptivity_ablation.json`), dynamic Hedge reweighting and circuit breakers are deployed in **MONITOR & SHADOW MODE ONLY**. The primary executor runs the robust static equal-weight ensemble ($w_B=0.50, w_C=0.50$) to avoid signal attenuation into the deadband.

---

## 6. Standard Operating Procedure: 7-Day Unattended Paper Testing

To validate operational stability before any future strategy re-assessment:

1. **Pre-Flight Initialization**:
   - Ensure `data/paper_trading.db` is initialized: `.venv\Scripts\python.exe -c "from src.service.db import Database; Database('data/paper_trading.db')"`
   - Ensure environment variables in `.env`: `CONFIRM_LIVE=NO`, `BINANCE_USE_TESTNET=true`.
2. **Launch Background Process**:
   ```powershell
   Start-Process -NoNewWindow -FilePath ".venv\Scripts\python.exe" -ArgumentList "run.py --paper-loop" -RedirectStandardOutput "logs\paper_stdout.log" -RedirectStandardError "logs\paper_stderr.log"
   ```
3. **Daily Health Check Procedure**:
   - Check process liveness: `Get-Process python`
   - Inspect database order log: `SELECT * FROM orders ORDER BY id DESC LIMIT 5;`
   - Inspect idempotency key log: `SELECT COUNT(*) FROM processed_bars;`
   - Verify zero unhandled exceptions: `Select-String -Path logs\paper_stderr.log -Pattern "ERROR|Traceback"`
4. **Failure Recovery**:
   - In event of network disconnect or machine reboot, simply restart the process. The `processed_bars` table prevents duplicate orders for any candle already completed.

---

## 7. Known Limitations & Quantitative Risks

1. **Cost Hurdle in Low-Volatility Chop**: With 15 bps round-trip friction, 1-hour bar directional trading requires significant momentum. During sideways regimes, the expected edge hurdle correctly halts trading, leaving cash unallocated.
2. **Single-Asset Trend Vulnerability**: Directional `BTC/USDT` trend strategies cannot generate returns when Bitcoin enters an extended accumulation/distribution range. Multi-asset cross-sectional relative value is required (see [ROADMAP.md](file:///c:/Ngoding/bot_trading/ROADMAP.md)).
3. **Sentiment Feed Latency**: Free public RSS feeds introduce 5-15 minute reporting delays. Model D is constrained to a negative veto gate and flagged degraded during network anomalies.
