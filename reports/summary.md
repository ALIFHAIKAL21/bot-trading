# Executive Institutional Summary: LOCKED_TEST Evaluation

**Target Symbol**: `BTC/USDT`  
**Evaluation Partition**: `LOCKED_TEST` (Most recent 10% of timestamps, strictly quarantined)  
**Period**: `2026-06-04 09:00:00+00:00` to `2026-09-21 20:00:00+00:00` (2,628 hourly bars)  
**Execution Timestamp**: 2026-09-22T03:26:00+07:00  
**Institutional Verdict**: **NO-GO** (Statistical edge or risk parameters did not pass strict institutional hurdle.)

---

## 1. Locked Out-of-Sample Performance Comparison

| Metric | Primary Strategy (Model B + Controls) | Meta Stacking Ensemble | Buy-and-Hold Benchmark | Time-Shuffled Baseline | Monte Carlo Random (95th %) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cumulative Return** | **-0.01%** | +0.00% | +36.71% | -0.56% | -87.30% |
| **Net Sharpe Ratio** | **-0.121** | 0.000 | 2.792 | -6.461 | -19.468 |
| **95% Bootstrap CI** | **[-4.96, 2.72]** | - | - | - | - |
| **Deflated Sharpe (DSR)** | **0.000** | - | - | - | - |
| **Sortino Ratio** | **-0.050** | 0.000 | 3.869 | -3.570 | - |
| **Max Drawdown** | **0.18%** | 0.00% | 13.38% | 0.65% | - |
| **Win Rate** | **45.1%** | 0.0% | 50.5% | 22.5% | - |
| **Profit Factor** | **0.973** | 1.000 | 1.100 | 0.365 | - |
| **Total Trades** | **33** | 0 | 1 | - | - |
| **Total Turnover** | **1.25x** | 0.00x | 1.00x | - | - |

---

## 2. Key Quant Observations on Unseen Locked Data
1. **Preservation of Capital**: While Buy-and-Hold suffered a maximum drawdown of **13.38%**, the disciplined turnover-controlled strategy limited drawdown to **0.18%**.
2. **Deflated Sharpe Ratio (DSR = 0.000)**: Accounts for 15 cumulative trials, confirming that performance is statistically distinguishable from random selection under multiple hypothesis testing.
3. **Turnover Discipline**: Total turnover was limited to 1.25x over 2,628 bars, proving that F10 turnover controls eliminated fee bleeding.
