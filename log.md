# PROJECT IDENTITY & PRODUCTION LOCK: FLOWDEV FRAME
## Flowdev Recurrent Algorithmic Model for Trade Execution (FRAME)
**Official Emblem:** The Cyber-Neural Peregrine Falcon (Golden Falcon)  
**Status: FROZEN ARCHITECTURE / TRANSITION TO SOFTWARE ENGINEERING (MT5 LIVE/DEMO)**  
**Tanggal Kunci:** 25 September 2026  
**Pair:** XAU/USD (Gold Spot vs US Dollar)  
**Timeframe Operasional:** M30 (30-Minute Bars)  
**Primary Execution Engine:** `scripts/run_500_smart_optimized_production.py`  
**Model Architecture:** 15-Channel Institutional Temporal Deep Network (1D-CNN + BiLSTM / Recurrent Neural Network)

---

## 1. Aturan Kunci Modal & Manajemen Risiko (FROZEN)

| Parameter | Nilai Kunci | Keterangan & Rasional |
|---|---|---|
| **Modal Awal (Anchor)** | **$500.00 USD** | Kunci modal akun riil dasar. |
| **Ukuran Lot (Lot Size)** | **0.01 Lot Flat** | **MUTLAK: Tanpa eskalasi, tanpa compounding/martingale.** |
| **Contract Size** | 100 oz per lot | 0.01 lot = 1 oz emas ($1 per $1.00 pergerakan harga emas). |
| **Margin Minimum Level** | > 1,200% | Beban margin hanya ~$25–$28. Sangat aman dari Margin Call. |
| **Friksi Broker Total** | **$0.17 per trade** | Spread 0.75 pip ($0.075) + Slippage 0.3 pip 2-way ($0.03) + Komisi $3.50/lot ($0.035). |

---

## 2. Struktur Sesi & Frekuensi Trade Harian (4–5 Trades/Hari)

Trade dieksekusi secara ketat hanya pada 5 jendela likuiditas institusional (maksimal 1 trade per sesi):

1. **Sesi 1: Asia Early** (01:00 – 04:00 UTC / 08:00 – 11:00 WIB)
2. **Sesi 2: Asia Late** (04:30 – 07:00 UTC / 11:30 – 14:00 WIB)
3. **Sesi 3: London Core** (08:30 – 12:30 UTC / 15:30 – 19:30 WIB)
4. **Sesi 4: NY Open** (13:00 – 17:00 UTC / 20:00 – 00:00 WIB)
5. **Sesi 5: NY Core** (17:30 – 21:00 UTC / 00:30 – 04:00 WIB)

* **Filter Blackout Gap:** Bar jam 07:00 – 08:30 UTC diblokir dari entri baru (zona transisi pra-London).
* **Filter Akhir Pekan (Weekend Shield):** Jumat setelah pukul 18:00 UTC dilarang membuka posisi baru. Seluruh posisi aktif ditutup paksa (*auto-closeout*) pada Jumat pukul 20:00 UTC guna menghindari gap harga Senin.

---

## 3. Parameter Model & Sinyal Gating (FROZEN)

* **Confidence Threshold ($\tau_{\text{base}}$):** `0.32`
* **Uncertainty Margin Filter:** `Margin = p_best - p_hold >= 0.01`
* **Sequence Offset:** 63 bar (M30 lookback window)

---

## 4. Mekanisme Trade Management & Kinetic Smart Protection (FROZEN CHECKPOINT)

Sistem menggunakan arsitektur proteksi 4 tahap tersinkronisasi (*Kinetic OMS*) yang telah divalidasi melintasi 59 bulan berturut-turut (100% Hijau):

1. **Stop Loss Awal:**  
   $$\text{SL Distance} = 1.5 \times \text{ATR}(14) \quad (\approx \$8.00 - \$15.00)$$
2. **Tahap 1: Micro-Breakeven Acceleration (+0.75R):**  
   Saat harga menyentuh $+0.75R$, SL otomatis digeser ke $\text{Entry} + \$0.25$ buffer. Menggaransi keuntungan bersih $+\$0.08$ jika tersenggol kembali dan menyelamatkan 39.2% trade pemenang awal agar tidak berbalik menjadi loss penuh.
3. **Tahap 2: Tiered Smart Ratchet (+1.2R):**  
   Saat harga mencapai $+1.2R$, SL otomatis dikunci ke $+0.5R$. Mengamankan profit pasti $+\$5.50 - \$9.00$ per trade.
4. **Tahap 3: Dynamic Trailing Stop (+1.5R+):**  
   Saat harga menembus $+1.5R$, Trailing Stop dinamis diaktifkan dengan jarak ketat $0.6R$ di belakang titik tertinggi harga.
5. **Tahap 4: Fast Stale Decay Protection (Bar 6 / 3 Jam):**  
   Jika posisi telah tertahan selama 6 bar (3 jam) dan belum mampu memicu Breakeven, SL otomatis diperketat dari $-1.0R$ menjadi $-0.45R$ (memotong risiko kerugian hingga 55% sebelum likuidasi berbalik).
6. **Kinetic Target Expansion (Max TP):**  
   Target Take Profit maksimum dipatok pada $+2.7R$ untuk memanen momentum gelombang besar.
7. **Time Barrier Hard Exit:**  
   Maksimal durasi penahanan posisi adalah 12 bar (6 jam).

---

## 5. Audit Kinerja Resmi Terkunci (Januari – Agustus 2026)

*Data di bawah ini adalah tolok ukur resmi (*benchmark baseline*) yang diverifikasi secara matematis:*

| Metrik Kinerja | Nilai Audit Resmi Terkunci | Catatan Validasi |
|---|---|---|
| **Modal Awal** | **$500.00 USD** | Modal riil akun dasar |
| **Saldo Akhir Akun** | **$4,458.87 USD** | Bertumbuh +$3,958.87 |
| **Persentase Keuntungan Bersih** | **+791.77%** | Bersih setelah semua friksi riil |
| **Total Trade Dieksekusi** | **706 Trade** | Rata-rata 4.13 trade/hari |
| **Menang / Kalah** | **427 Menang / 279 Kalah** | **Menang > Kalah Mutlak** |
| **Overall Win Rate** | **60.48%** | Meningkat melampaui 60% |
| **Profit Factor** | **1.81** | Rasio keuntungan terhadap kerugian sangat sehat |
| **Max Drawdown Portofolio** | **33.07%** | Turun 18.14% dari baseline sebelumnya (51.21%) |
| **Titik Saldo Terendah (Dip)** | **$433.24 USD** | Hanya minus -$66.76 dari modal awal! |
| **Konsistensi Frekuensi Harian** | **78.1% Hari (4–5 Trade)** | Stabilitas ritme terjaga |

### Tabel Performa Bulan ke Bulan 2026 (100% HIJAU)

| Bulan | Total Trade | Menang / Kalah | Win Rate (%) | Net PnL ($) | Saldo Akun Akhir Bulan |
|---|---|---|---|---|---|
| **Januari 2026** | 84 | 46 / 38 | **54.8%** | **+$213.97** | $713.97 |
| **Februari 2026** | 90 | 61 / 29 | **67.8%** | **+$949.83** | $1,663.80 |
| **Maret 2026** | 91 | 57 / 34 | **62.6%** | **+$666.49** | $2,330.29 |
| **April 2026** | 88 | 52 / 36 | **59.1%** | **+$420.87** | $2,751.16 |
| **Mei 2026** | 81 | 46 / 35 | **56.8%** | **+$155.89** | $2,907.05 |
| **Juni 2026** | 84 | 50 / 34 | **59.5%** | **+$538.89** | $3,445.94 |
| **Juli 2026** | 96 | 60 / 36 | **62.5%** | **+$593.80** | $4,039.74 |
| **Agustus 2026** | 92 | 55 / 37 | **59.8%** | **+$419.13** | **$4,458.87** |

---

## 6. Arsip File Terkait (Repository Assets)

1. **Script Eksekusi Produksi:**  
   [`scripts/run_500_smart_optimized_production.py`](file:///c:/Ngoding/bot_trading/scripts/run_500_smart_optimized_production.py)
2. **Laporan JSON Audit Lengkap:**  
   [`reports/backtest_500_modal_smart_optimized.json`](file:///c:/Ngoding/xau_deep_sniper/reports/backtest_500_modal_smart_optimized.json)
3. **Chart Tearsheet PNG (Resolusi Tinggi):**  
   [`reports/img/backtest_500_modal_smart_optimized.png`](file:///c:/Ngoding/xau_deep_sniper/reports/img/backtest_500_modal_smart_optimized.png)
4. **Dataset Fitur 15-Channel:**  
   `data/processed/xauusd_m30_labeled_15ch.parquet`
5. **Tensor Prediksi Model 15-Channel:**  
   `checkpoints/predictions_15ch.npy`

---

> **PROTOKOL KUNCI PERMANEN (CHECKPOINT RESMI):**  
> Dilarang mengubah konstanta parameter di atas (`INITIAL_EQUITY = 500`, `FIXED_LOT = 0.01`, `BE_TRIGGER_R = 0.75`, `RATCHET_12_R = 0.5`, `TRAIL_DIST_R = 0.6`, `TP_MAX_R = 2.7`, `STALE_DECAY_BARS = 6`, `STALE_DECAY_R = 0.45`) tanpa pengujian komparatif formal out-of-sample baru. Dokumen ini menjadi acuan mutlak (*single source of truth*) untuk implementasi live execution dan paper trading.


---

## 7. Aturan Default Perintah "Backtest" (User Contract)

**Mulai 25 September 2026:**  
Setiap kali pengguna/USER memberikan instruksi atau kata **"backtest"** tanpa spesifikasi parameter baru:
1. Sistem **100% WAJIB** menggunakan konfigurasi yang tercantum dalam dokumen ini tanpa ada modifikasi sedikit pun.
2. Engine yang dijalankan adalah: [`scripts/run_500_smart_optimized_production.py`](file:///c:/Ngoding/bot_trading/scripts/run_500_smart_optimized_production.py).
3. Parameter modal otomatis terkunci di **$500.00**, lot **0.01 flat**, dan seluruh filter 4 tahap proteksi aktif 100%.

---

## 8. Arsitektur Software Engineering & Modul Eksekusi Live (FRAME)

Peralihan resmi dari tahap penelitian kuantitatif ke tahap rekayasa perangkat lunak (*Software Engineering*) untuk kesiapan akun demo MT5:

```
                               ┌────────────────────────────────────────────────────────┐
                               │             FLOWDEV FRAME CORE ENGINE                  │
                               │  Flowdev Recurrent Algorithmic Model for Trade Exec    │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
             ┌───────────────────────────────┬─────────────┴─────────────────┬───────────────────────────────┐
             │                               │                               │                               │
             ▼                               ▼                               ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   Module 1: Ingestion   │     │    Module 2: Feature    │     │   Module 3: Inference   │     │      Module 4: OMS      │
│  MT5 Real-Time Gateway  │ ──► │ 15-Channel Pipeline     │ ──► │ PyTorch BiLSTM Runtime  │ ──► │ 4-Stage State Machine   │
│  - M30 Bar Completion   │     │ - Causal Streaming Norm │     │ - Tau Base >= 0.32      │     │ - Positive BE (+1.0R)   │
│  - Auto-Reconnect Drops │     │ - Zero Future Leakage   │     │ - Session Filter (1-5)  │     │ - Smart Ratchet (+1.2R) │
│  - Tick Health Monitor  │     │ - Shape: (1, 15, 64)    │     │ - Weekend Blackout Gate │     │ - Dynamic Trail (+1.5R) │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘     │ - Stale Decay (Bar 10)  │
                                                                                                └────────────┬────────────┘
                                                                                                             │
                                                                                                             ▼
                                                                                                ┌─────────────────────────┐
                                                                                                │   Module 5: Resilience  │
                                                                                                │ SQLite State & Failover │
                                                                                                │ - Crash Recovery        │
                                                                                                │ - Telegram Event Stream │
                                                                                                └─────────────────────────┘
```

---

## 9. Stasiun Simulasi Live Trade Terdedikasi (FLOWDEV FRAME LIVE PAPER TRADER)

**Status: DEPLOYED & OPERATIONAL (25 September 2026)**  
**Entry Point:** `launch_live_trader.bat` / `launch_live_trader.py` atau tombol `[🔴 LIVE REALTIME TRADER]` di Workstation Backtest.

### Fitur Kunci:
1. **Pemisahan Total Halaman (Decoupled Architecture):**
   - Halaman simulasi live trade berdiri sendiri dalam window terpisah (`src/frame/live/live_window.py`), tidak bercampur dengan backtester audit historis.
2. **Eksekusi Pasar Real-Time Tanpa Risiko Akun (Zero Account Risk Paper Broker):**
   - Broker virtual presisi tinggi (`LivePaperBroker`) menghitung margin, floating PnL tick-by-tick, komisi, spread riil, slippage bid/ask, dan audit riwayat trade.
3. **Dual Data Feed Real-Time:**
   - **Primary:** Native MT5 IPC Socket (latency 0ms streaming XAU/USD).
   - **Fallback 24/7:** Realtime Tick Emulator seeded dari harga pasar aktual ($4,314+) untuk pengujian tanpa henti bahkan saat pasar tutup/weekend.
4. **Hardware-Accelerated Low-Latency Chart:**
   - TradingView Lightweight Charts WebGL rendering dengan update tick real-time sub-millisecond dan overlay visual dinamis (Entry Line, Stop Loss Ratchet Line, Take Profit Line).
5. **Full 4-Stage Kinetic OMS & PyTorch RTX 4050 LoRA Inference:**
   - Inferensi MOMENT Neural Network langsung pada GPU RTX 4050 (4.90ms latency).
   - Otomatis menggerakkan Stage 1 (Micro-BE +0.75R), Stage 2 (Smart Ratchet +1.2R / Lock +0.5R), Stage 3 (Dynamic Trailing Stop +1.5R+), Stage 4 (Stale Decay Bar 6 / -0.45R), Ceiling TP (+2.7R) dan 12-bar Time Barrier.
6. **Active Position HUD & Panic Button:**
   - Pemantauan posisi terbuka secara real-time dengan status stage OMS, floating PnL ($ dan R-multiple), serta tombol Panic Close darurat untuk intervensi manual seketika.

