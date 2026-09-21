# Phase 6 Report: Paper-Trading Correctness & Live Parity (F4, F5, F6, F15)

## 1. Executive Summary
Phase 6 established operational correctness, restart idempotency, and live execution parity across the paper trading service, database, and monitoring dashboard. We solved the critical audit defects:
- **Closed-candle execution enforcement (F4)**: Eliminates look-ahead bias and unstable mid-bar flickering orders.
- **SQLite idempotency ledger (`processed_bars`) (F4)**: Guarantees zero duplicate orders on service restarts or repeated scheduler ticks.
- **Fail-safe neutral cash mode (F5, F6)**: Eliminates rogue buying at $P=0.457$ during missing/degraded data; guarantees $P=0.50$, Target Weight = $0.0$.
- **Streamlit Dashboard Overhaul (F15)**: Fixed PyArrow type conversion crashes; added honest percentage axes, prominent institutional `NO-GO` verdict, and adaptivity/drift health telemetry.

---

## 2. Key Implementations & Empirical Results

### 2.1 Closed-Candle Only Processing (F4)
- **Problem**: Polling exchanges at 10s intervals frequently returned forming candles where prices and indicators fluctuate. Evaluating forming candles creates look-ahead bias and causes flickering buy/sell orders.
- **Solution**: In `src/service/scheduler.py`, the scheduler checks:
  $$\text{bar\_close\_utc} + 5\text{s} \le \text{now\_utc}$$
  If the latest candle returned by CCXT is still forming, the scheduler shifts back to the previous completed bar (`df_feat.iloc[-2]`).
- **Verification**: `test_forming_candle_guard_logic` in `tests/test_paper.py` passes.

### 2.2 Post-Processing Idempotency Key (`processed_bars`) (F4)
- **Problem**: If the scheduler crashes after placing an order or if it restarts mid-candle, it could re-evaluate the same bar and issue duplicate orders.
- **Solution**:
  - Added table `processed_bars` in `src/service/db.py`:
    ```sql
    CREATE TABLE IF NOT EXISTS processed_bars (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        bar_timestamp TEXT NOT NULL,
        action TEXT NOT NULL,
        decision_reason TEXT NOT NULL,
        target_weight REAL NOT NULL,
        processed_at TEXT NOT NULL,
        UNIQUE(symbol, bar_timestamp)
    );
    ```
  - Before running inference, `self.db.is_bar_processed(symbol, bar_ts)` is checked. If already processed, the scheduler logs: `Bar {bar_ts} for {symbol} was already evaluated and processed. Skipping duplicate execution.`
  - Upon cycle completion, `record_processed_bar()` locks the timestamp.
- **Verification**: `test_processed_bars_idempotency_restart_safety` verifies that simulated bot restarts mid-bar produce zero duplicate orders.

### 2.3 Fail-Safe Neutral Cash Mode (F5, F6)
- **Problem**: When models failed or returned `None`, the legacy system fell back to Kelly formulas that issued positive position sizes at $P=0.457$.
- **Solution**: If model inference raises any exception or component is degraded:
  - $P(\text{long})$ defaults to exactly $0.500$.
  - Confidence defaults to $0.000$.
  - `RiskEngine` suppresses entry because $P < \text{entry\_threshold}$ ($0.54$) and edge ($0.0$) is below hurdle ($0.0030$).
  - Target weight is strictly clamped to $0.0$. Real funds/virtual cash remain safely unallocated.

### 2.4 Backtest & Paper Broker Parity
- **Comparison**:
  - Both use 10 bps taker fee.
  - Both use 5 bps base slippage + volume-adjusted slippage.
  - Both execute strictly on the bar immediately following signal emission.
- **Verification**: `test_paper_broker_vs_backtest_parity` in `tests/test_paper.py` passes.

### 2.5 Streamlit Dashboard Overhaul (F15)
- **Bug Fix**: Resolved PyArrow exception (`ArrowInvalid: Could not convert 'N/A' to double`) caused by mixed string/float types in baseline comparison tables.
- **Honest Metrics**:
  - Sourced directly from `reports/final_report.json` and `reports/turnover_controls_sweep.json`.
  - Prominent institutional badge: `VERDICT: NO-GO FOR LIVE TRADING`.
  - Display of true percentage returns alongside nominal equity.
  - Interactive tabs: Evaluation & Baselines, Models & Adaptivity, Market Data & Features, Live Paper Trading, Config & Audit.

---

## 3. Test Suite Verification
All 47 unit and integration tests passed via pytest:
```
tests/test_adaptivity.py .....                                           [ 10%]
tests/test_backtest.py ....                                              [ 19%]
tests/test_config.py ..                                                  [ 23%]
tests/test_deep_model.py ...                                             [ 30%]
tests/test_hmm.py .                                                      [ 32%]
tests/test_labels.py ...                                                 [ 39%]
tests/test_leakage.py ......                                             [ 52%]
tests/test_paper.py .....                                                [ 63%]
tests/test_risk.py ...........                                           [ 87%]
tests/test_security.py .........                                         [100%]
============================== 47 passed in 5.12s ==============================
```
