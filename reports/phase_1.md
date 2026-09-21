# Phase 1 Report: Data, Features & Leakage Eradication (F1, F2, F3, F7)

**Date**: 2026-09-22  
**Status**: COMPLETE (Passed 32/32 tests)  
**Artifacts Emitted**: `reports/data_quality.json`, `reports/data_manifest.json`, `reports/leakage_cv_comparison.json`

---

## 1. Executive Summary
Phase 1 systematically eliminates data defects, look-ahead leakage, and contaminated evaluation practices identified in audit findings F1, F2, F3, and F7:
- **Data Sourcing (F3)**: Upgraded from yfinance (truncated to ~730 days / 17,513 bars in float32) to **CCXT Binance Vision public REST API** paginating **26,280 bars (3.0 full years)** in **float64** precision.
- **Leakage Eradication (F1)**: Purged `sample_weight` from feature inputs and model schemas. Implemented an explicit feature whitelist in `config.yaml` with automated `LeakageViolationError` assertions failing loudly on forbidden regex patterns.
- **Higher-Timeframe Causality (F2)**: Verified that 4h and 1d resampled indicators are causally shifted so bars only observe completed HTF candles. Tested CV performance with vs without `htf_1d_trend`.
- **Dataset Label Truth (F7)**: Deconstructed the "500k dataset" claim. Documented that 527,040 bars represents 1-minute raw archives rather than 1-hour strategy bars, correcting the label.
- **Hold-Out Quarantine**: Sealed the most recent 10% of timestamps (2,628 bars) behind `guard_locked_test()`.

---

## 2. Real Empirical Metrics: Before vs After

### A. Data Quality & Provenance (F3)
| Dimension | Before Phase 1 (yfinance) | After Phase 1 (CCXT Binance) | Evidence / Source |
| :--- | :--- | :--- | :--- |
| **Primary Data Source** | yfinance fallback | **CCXT Binance Vision Public REST** | `reports/data_manifest.json` |
| **History Length (BTC/USDT)** | 17,513 bars (~730 days) | **26,280 bars (1,095 days / 3.0 years)** | `reports/data_quality.json` |
| **History Length (ETH/USDT)** | 17,513 bars (~730 days) | **26,280 bars (1,095 days / 3.0 years)** | `reports/data_quality.json` |
| **Numeric Precision** | float32 | **float64** | Verified in Parquet schemas |
| **Zero-Volume Bars** | Unflagged | **0 bars (0.00%)** | `reports/data_quality.json` |
| **Data Gaps** | Unflagged | **0 missing bars** | Continuous date range |
| **Date Coverage** | 2024 to 2026 | **2023-09-22 21:00 to 2026-09-21 20:00 UTC** | Fully continuous 3-year span |

### B. Leakage & HTF Feature Ablation (F1, F2)
Empirical 5-fold Purged Walk-Forward Cross-Validation results on 23,580 clean validation bars:

| Feature Configuration | Total Features | Mean AUC | AUC Std Dev | Mean Brier Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Leaked (with `sample_weight`)** | 32 | 0.5837 | +/- 0.0397 | 0.2159 | Leaked future overlap weight |
| **2. Clean (Strict Whitelist + `htf_1d_trend`)** | 31 | **0.5829** | **+/- 0.0265** | **0.2061** | **Calibrated, honest, stable** |
| **3. Clean Ablation (Excluding `htf_1d_trend`)** | 30 | 0.5770 | +/- 0.0284 | 0.2065 | AUC drops by -0.0059 |

**Key Findings**:
1. **Removing `sample_weight` Leakage**: While raw AUC slightly adjusted from 0.5837 to 0.5829 (-0.0008), the **Brier score improved significantly from 0.2159 to 0.2061**, and fold-to-fold AUC variance dropped from **0.0397 to 0.0265**. This confirms that `sample_weight` artificially distorted probability calibration without contributing real generalization.
2. **HTF 1-Day Trend Feature (F2)**: Removing `htf_1d_trend` drops mean validation AUC by **-0.0059** (from 0.5829 to 0.5770). Rigorous truncate-future testing confirms that `htf_1d_trend` strictly updates at the 00:00 UTC boundary following a completed day. Its top-rank importance stems from crypto's authentic medium-term momentum, not look-ahead leakage.

### C. Per-Fold Walk-Forward AUC Breakdown
```text
Fold 0:
  - Leaked: AUC = 0.5111 | Brier = 0.2451
  - Clean:  AUC = 0.5312 | Brier = 0.2104  (+0.0201 AUC, -0.0347 Brier)
Fold 1:
  - Leaked: AUC = 0.6203 | Brier = 0.2127
  - Clean:  AUC = 0.5856 | Brier = 0.2117
Fold 2:
  - Leaked: AUC = 0.6090 | Brier = 0.2105
  - Clean:  AUC = 0.5975 | Brier = 0.2054
Fold 3:
  - Leaked: AUC = 0.5717 | Brier = 0.2065
  - Clean:  AUC = 0.5982 | Brier = 0.1991
Fold 4:
  - Leaked: AUC = 0.6061 | Brier = 0.2046
  - Clean:  AUC = 0.6021 | Brier = 0.2037
```

---

## 3. Dataset Transparency Audit (F7)
- **Previous Claim**: "500k Dataset for Model C".
- **Factual Reality**: The 527,040 rows correspond to 12 months of **1-minute (`1m`)** high-frequency candles for BTC/USDT downloaded from `data.binance.vision`.
- **Correction**: The system's primary strategy timeframe is **1-hour (`1h`)** with 26,280 bars over 3 years. Training 1-hour horizon models on 1-minute overlapping sliding windows creates 60x auto-correlated sample inflation. Model C evaluation is strictly aligned to non-inflated purged sequences.

---

## 4. Hold-Out Discipline
- **Partitioning**:
  - `VALIDATION POOL`: First 90% (23,652 bars, 2023-09-22 21:00 to 2026-06-04 08:00 UTC).
  - `LOCKED_TEST`: Last 10% (2,628 bars, 2026-06-04 09:00 to 2026-09-21 20:00 UTC).
- **Code Enforcement**: Attempting to pass bars $\ge$ `2026-06-04 09:00:00+00:00` into any training or CV pipeline raises `HoldoutViolationError`. Only `scripts/final_report.py` with `ALLOW_LOCKED_TEST_ACCESS=1` is authorized.

---

## 5. Acceptance Checklist for Phase 1
- [x] CCXT Binance loader paginates 3 full years (26,280 bars) in float64.
- [x] Zero-volume bars and gaps flagged and excluded from volume features.
- [x] `reports/data_quality.json` and `reports/data_manifest.json` emitted.
- [x] Strict feature whitelist enforced; loud exception on label/weight columns.
- [x] `sample_weight` eradicated from model feature inputs.
- [x] HTF resample verified causal via truncate-future test.
- [x] Empirical CV comparison executed with real numbers.
- [x] "500k dataset" label corrected and explained.
- [x] Holdout partitioned into sealed `LOCKED_TEST` with code guard.
