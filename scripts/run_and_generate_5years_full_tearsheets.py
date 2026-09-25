"""
XAU_DEEP_SNIPER - Full Year (Jan - Des) Backtests for 2021 to 2025
Generates 5 dedicated, ultra-clean, Indonesian trader tearsheets (one PNG per year).
"""

import sys, os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
brain_artifact_dir = pathlib.Path(r"C:\Users\haika\.gemini\antigravity-ide\brain\f18a7bdb-34cf-4c7d-86ed-374ce8cd5082")

df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')

print("Memuat dataset dan prediksi...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# Hitung ATR(14)
high, low, close = df["high"].values, df["low"].values, df["close"].values
tr = np.zeros(len(df))
tr[0] = high[0] - low[0]
for i in range(1, len(df)):
    tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
period = 14
atr = np.zeros(len(df))
atr[:period] = np.mean(tr[:period])
multiplier = 2.0 / (period + 1)
for i in range(period, len(df)):
    atr[i] = tr[i] * multiplier + atr[i - 1] * (1 - multiplier)

# Parameter Eksekusi Tier Terakhir (Tier 3 Optimized Sniper)
TAU = 0.355
INITIAL_EQUITY = 10_000.0
BASE_RISK = 0.01  # 1% per trade
SPREAD_PRICE = 0.75 * 0.10
SLIPPAGE_PRICE = 0.3 * 0.10
COMMISSION = 3.50
CONTRACT_SIZE = 100.0
SL_ATR_MULT = 1.5
POS_A_PCT = 0.40
POS_B_PCT = 0.60
TP1_R = 1.0
TP2_R = 2.5
TRAIL_AFTER_R = 1.5
TRAIL_DIST_R = 0.8
MAX_DAILY = 10

def calc_friction(lot):
    return round(SPREAD_PRICE * lot * CONTRACT_SIZE + SLIPPAGE_PRICE * lot * CONTRACT_SIZE * 2 + COMMISSION * lot, 2)

# Konfigurasi tahun yang diuji
test_years = [
    {"year": 2021, "start": "2021-09-23 12:30:00", "end": "2021-12-31 23:59:59", "label": "Tahun 2021 (Sep - Des)*", "is_partial": True},
    {"year": 2022, "start": "2022-01-01 00:00:00", "end": "2022-12-31 23:59:59", "label": "Tahun 2022 (Januari - Desember)", "is_partial": False},
    {"year": 2023, "start": "2023-01-01 00:00:00", "end": "2023-12-31 23:59:59", "label": "Tahun 2023 (Januari - Desember)", "is_partial": False},
    {"year": 2024, "start": "2024-01-01 00:00:00", "end": "2024-12-31 23:59:59", "label": "Tahun 2024 (Januari - Desember)", "is_partial": False},
    {"year": 2025, "start": "2025-01-01 00:00:00", "end": "2025-12-31 23:59:59", "label": "Tahun 2025 (Januari - Desember)", "is_partial": False},
]

annual_summaries = []

for y_cfg in test_years:
    year = y_cfg["year"]
    st_ts = pd.to_datetime(y_cfg["start"], utc=True)
    end_ts = pd.to_datetime(y_cfg["end"], utc=True)
    
    print(f"\nMenjalankan backtest {y_cfg['label']}...")
    
    n_bars = len(df)
    seq_offset = 63
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    equity_timestamps = [st_ts]
    trades = []
    trade_counter = 0
    open_position = None
    
    current_day = None
    daily_count = 0
    
    for bar_idx in range(n_bars):
        row = df.iloc[bar_idx]
        ts = row["timestamp_utc"]
        bar_open, bar_high, bar_low, bar_close = row["open"], row["high"], row["low"], row["close"]
        is_blackout = bool(row.get("is_news_blackout", False))
        is_eligible = bool(row.get("entry_eligible", True))
        
        in_date_range = (ts >= st_ts) and (ts <= end_ts)
        
        day_str = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]
        if day_str != current_day:
            current_day = day_str
            daily_count = 0
            
        # Posisi Terbuka
        if open_position is not None:
            bars_held = bar_idx - open_position["entry_bar_idx"]
            d = open_position["direction"]
            ep = open_position["entry_price"]
            sl_dist = open_position["sl_dist"]
            lot_a = open_position["lot_a"]
            lot_b = open_position["lot_b"]
            dow = ts.dayofweek if hasattr(ts, 'dayofweek') else pd.Timestamp(ts).dayofweek
            hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour

            is_friday_close = (dow == 4 and hour >= 20) or dow in [5, 6]
            is_time_barrier = (bars_held >= 16)

            if is_friday_close or is_time_barrier:
                exit_price = bar_close
                if open_position["pos_a_open"]:
                    pnl_a_g = (exit_price - ep) * lot_a * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * lot_a * CONTRACT_SIZE
                    open_position["pnl_net_accum"] += (pnl_a_g - calc_friction(lot_a))
                    open_position["pos_a_open"] = False
                if open_position["pos_b_open"]:
                    pnl_b_g = (exit_price - ep) * lot_b * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * lot_b * CONTRACT_SIZE
                    open_position["pnl_net_accum"] += (pnl_b_g - calc_friction(lot_b))
                    open_position["pos_b_open"] = False

                net_p = round(open_position["pnl_net_accum"], 2)
                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                trades.append({
                    'id': trade_counter, 'direction': d, 'entry_ts': open_position["entry_timestamp"],
                    'exit_ts': ts, 'pnl': net_p, 'win': 1 if net_p > 0 else 0, 'month': m_str,
                    'reason': 'Jumat Malam (20:00)' if is_friday_close else 'Batas Waktu (8 Jam)'
                })
                trade_counter += 1
                equity += net_p
                open_position = None

            else:
                if d == "BUY":
                    if not open_position["tp1_hit"] and bar_low <= open_position["cur_sl_a"]:
                        pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * CONTRACT_SIZE
                        net_p = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                        m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                        trades.append({
                            'id': trade_counter, 'direction': d, 'entry_ts': open_position["entry_timestamp"],
                            'exit_ts': ts, 'pnl': net_p, 'win': 0, 'month': m_str, 'reason': 'Stop Loss Kena'
                        })
                        trade_counter += 1
                        equity += net_p
                        open_position = None
                    else:
                        if open_position["pos_a_open"] and bar_high >= open_position["tp1_price"]:
                            pnl_a_g = (open_position["tp1_price"] - ep) * lot_a * CONTRACT_SIZE
                            open_position["pnl_net_accum"] += (pnl_a_g - calc_friction(lot_a))
                            open_position["pos_a_open"] = False
                            open_position["tp1_hit"] = True
                            f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                            open_position["cur_sl_b"] = max(open_position["cur_sl_b"], round(ep + f_p, 2))

                        if open_position["pos_b_open"]:
                            if bar_high >= open_position["tp2_price"]:
                                pnl_b_g = (open_position["tp2_price"] - ep) * lot_b * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_b_g - calc_friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                trades.append({
                                    'id': trade_counter, 'direction': d, 'entry_ts': open_position["entry_timestamp"],
                                    'exit_ts': ts, 'pnl': net_p, 'win': 1, 'month': m_str, 'reason': 'Target TP2 (+2.5R)'
                                })
                                trade_counter += 1
                                equity += net_p
                                open_position = None
                            elif bar_low <= open_position["cur_sl_b"]:
                                pnl_b_g = (open_position["cur_sl_b"] - ep) * lot_b * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_b_g - calc_friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                trades.append({
                                    'id': trade_counter, 'direction': d, 'entry_ts': open_position["entry_timestamp"],
                                    'exit_ts': ts, 'pnl': net_p, 'win': 1 if net_p > 0 else 0, 'month': m_str,
                                    'reason': 'Breakeven / Trailing'
                                })
                                trade_counter += 1
                                equity += net_p
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (bar_high - ep) / sl_dist
                                if r_gain >= TRAIL_AFTER_R:
                                    trail_sl = round(bar_high - (TRAIL_DIST_R * sl_dist), 2)
                                    open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)
                else:  # SELL
                    if not open_position["tp1_hit"] and bar_high >= open_position["cur_sl_a"]:
                        pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * CONTRACT_SIZE
                        net_p = round(pnl_g - calc_friction(open_position["total_lot"]), 2)
                        m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                        trades.append({
                            'id': trade_counter, 'direction': d, 'entry_ts': open_position["entry_timestamp"],
                            'exit_ts': ts, 'pnl': net_p, 'win': 0, 'month': m_str, 'reason': 'Stop Loss Kena'
                        })
                        trade_counter += 1
                        equity += net_p
                        open_position = None
                    else:
                        if open_position["pos_a_open"] and bar_low <= open_position["tp1_price"]:
                            pnl_a_g = (ep - open_position["tp1_price"]) * lot_a * CONTRACT_SIZE
                            open_position["pnl_net_accum"] += (pnl_a_g - calc_friction(lot_a))
                            open_position["pos_a_open"] = False
                            open_position["tp1_hit"] = True
                            f_p = SPREAD_PRICE + (COMMISSION / CONTRACT_SIZE)
                            open_position["cur_sl_b"] = min(open_position["cur_sl_b"], round(ep - f_p, 2))

                        if open_position["pos_b_open"]:
                            if bar_low <= open_position["tp2_price"]:
                                pnl_b_g = (ep - open_position["tp2_price"]) * lot_b * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_b_g - calc_friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                trades.append({
                                    'id': trade_counter, 'direction': d, 'entry_ts': open_position["entry_timestamp"],
                                    'exit_ts': ts, 'pnl': net_p, 'win': 1, 'month': m_str, 'reason': 'Target TP2 (+2.5R)'
                                })
                                trade_counter += 1
                                equity += net_p
                                open_position = None
                            elif bar_high >= open_position["cur_sl_b"]:
                                pnl_b_g = (ep - open_position["cur_sl_b"]) * lot_b * CONTRACT_SIZE
                                open_position["pnl_net_accum"] += (pnl_b_g - calc_friction(lot_b))
                                open_position["pos_b_open"] = False
                                net_p = round(open_position["pnl_net_accum"], 2)
                                m_str = ts.strftime('%Y-%m') if hasattr(ts, 'strftime') else str(ts)[:7]
                                trades.append({
                                    'id': trade_counter, 'direction': d, 'entry_ts': open_position["entry_timestamp"],
                                    'exit_ts': ts, 'pnl': net_p, 'win': 1 if net_p > 0 else 0, 'month': m_str,
                                    'reason': 'Breakeven / Trailing'
                                })
                                trade_counter += 1
                                equity += net_p
                                open_position = None
                            elif open_position["tp1_hit"]:
                                r_gain = (ep - bar_low) / sl_dist
                                if r_gain >= TRAIL_AFTER_R:
                                    trail_sl = round(bar_low + (TRAIL_DIST_R * sl_dist), 2)
                                    open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)

        if in_date_range:
            equity_curve.append(equity)
            equity_timestamps.append(ts)

        # Cek Entri Baru
        pred_idx = bar_idx - seq_offset
        if in_date_range and pred_idx >= 0 and pred_idx < len(preds) and open_position is None and daily_count < MAX_DAILY and is_eligible and not is_blackout:
            dow = ts.dayofweek if hasattr(ts, 'dayofweek') else pd.Timestamp(ts).dayofweek
            hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour
            minute = ts.minute if hasattr(ts, 'minute') else pd.Timestamp(ts).minute

            friday_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
            london_quarantine = (hour == 7 or (hour == 8 and minute < 30))

            if not friday_freeze and not london_quarantine:
                probs = preds[pred_idx]
                trade_probs = probs[1:]
                max_class_idx = int(np.argmax(trade_probs))
                action_class = max_class_idx + 1
                confidence = float(trade_probs[max_class_idx])
                p_hold = float(probs[0])

                if confidence >= TAU and confidence > p_hold:
                    d = "BUY" if action_class in [1, 2] else "SELL"
                    ob_zone = float(row.get("order_block_zone", 0.0))
                    ma_slope = float(row.get("ma_ribbon_slope", 0.0))

                    ok = True
                    if d == "BUY" and (ob_zone < 0 or ma_slope < -0.3): ok = False
                    if d == "SELL" and (ob_zone > 0 or ma_slope > 0.3): ok = False

                    if ok:
                        atr_val = float(atr[bar_idx]) if bar_idx < len(atr) else 5.0
                        sl_dist = round(atr_val * SL_ATR_MULT, 2)
                        if sl_dist > 0:
                            target_risk = equity * BASE_RISK
                            raw_lot = target_risk / (sl_dist * CONTRACT_SIZE)
                            lot_size = max(0.02, min(round(np.floor(raw_lot / 0.01) * 0.01, 2), 50.0))
                            actual_risk = lot_size * sl_dist * CONTRACT_SIZE

                            lot_a = max(0.01, round(lot_size * POS_A_PCT, 2))
                            lot_b = max(0.01, round(lot_size - lot_a, 2))

                            if d == "BUY":
                                ep = round(bar_close + SLIPPAGE_PRICE, 2)
                                sl_p = round(ep - sl_dist, 2)
                                tp1_p = round(ep + (TP1_R * sl_dist), 2)
                                tp2_p = round(ep + (TP2_R * sl_dist), 2)
                            else:
                                ep = round(bar_close - SLIPPAGE_PRICE, 2)
                                sl_p = round(ep + sl_dist, 2)
                                tp1_p = round(ep - (TP1_R * sl_dist), 2)
                                tp2_p = round(ep - (TP2_R * sl_dist), 2)

                            open_position = {
                                "direction": d, "entry_bar_idx": bar_idx, "entry_timestamp": ts,
                                "entry_price": ep, "sl_dist": sl_dist, "initial_sl": sl_p,
                                "cur_sl_a": sl_p, "cur_sl_b": sl_p, "tp1_price": tp1_p, "tp2_price": tp2_p,
                                "total_lot": lot_size, "lot_a": lot_a, "lot_b": lot_b,
                                "risk_amount": actual_risk, "confidence": confidence,
                                "pos_a_open": True, "pos_b_open": True, "tp1_hit": False, "tp2_hit": False,
                                "pnl_net_accum": 0.0
                            }
                            daily_count += 1

    # Olah Statistik Hasil
    tdf = pd.DataFrame(trades)
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = np.min((eq_arr - peak) / peak) * 100 if len(eq_arr) > 0 else 0
    total_trades = len(tdf)
    n_wins = tdf['win'].sum() if total_trades > 0 else 0
    n_losses = total_trades - n_wins
    wr = (n_wins / total_trades * 100) if total_trades > 0 else 0.0
    net_pnl = equity - INITIAL_EQUITY
    ret_pct = (net_pnl / INITIAL_EQUITY) * 100

    # Profit Factor
    if total_trades > 0:
        gross_profit = tdf[tdf['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(tdf[tdf['pnl'] < 0]['pnl'].sum())
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9
    else:
        profit_factor = 0.0

    # Monthly breakdown
    if y_cfg["is_partial"]:
        month_list = [f"{year}-09", f"{year}-10", f"{year}-11", f"{year}-12"]
    else:
        month_list = [f"{year}-{m:02d}" for m in range(1, 13)]

    monthly_stats = []
    c_eq = INITIAL_EQUITY
    green_m = 0
    for m in month_list:
        sub = tdf[tdf['month'] == m] if total_trades > 0 else pd.DataFrame()
        s_n = len(sub)
        if s_n > 0:
            sw = sub['win'].sum()
            spnl = sub['pnl'].sum()
            spct = (spnl / c_eq) * 100
            c_eq += spnl
            if spnl >= 0: green_m += 1
            monthly_stats.append({
                'month': m, 'name': m[-2:], 'trades': s_n, 'wins': sw, 'losses': s_n - sw,
                'wr': round(sw / s_n * 100, 1), 'pnl': spnl, 'pct': spct
            })
        else:
            monthly_stats.append({
                'month': m, 'name': m[-2:], 'trades': 0, 'wins': 0, 'losses': 0,
                'wr': 0.0, 'pnl': 0.0, 'pct': 0.0
            })

    annual_summaries.append({
        'year': year, 'label': y_cfg['label'], 'trades': total_trades, 'wins': n_wins,
        'losses': n_losses, 'wr': wr, 'pnl': net_pnl, 'ret': ret_pct, 'max_dd': max_dd,
        'pf': profit_factor, 'green_m': f"{green_m}/{len(month_list)}", 'm_stats': monthly_stats,
        'eq_curve': eq_arr, 'eq_dates': equity_timestamps, 'tdf': tdf
    })
    
    print(f"Hasil {year}: Trades={total_trades}, WR={wr:.1f}%, Net PnL=${net_pnl:+,.2f} ({ret_pct:+.2f}%), Max DD={max_dd:.2f}%, PF={profit_factor}")

print("\nSeluruh simulasi 5 tahun selesai. Membuat 5 tearsheet gambar PNG...")

# ------------------------------------------------------------------------------
# FUNGSI PEMBUAT TEARSHEET INDIVIDUAL (1 GAMBAR PER TAHUN)
# ------------------------------------------------------------------------------
month_names_id = {
    '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'Mei', '06': 'Jun',
    '07': 'Jul', '08': 'Agu', '09': 'Sep', '10': 'Okt', '11': 'Nov', '12': 'Des'
}

for item in annual_summaries:
    yr = item['year']
    yr_label = item['label']
    trades_n = item['trades']
    wr = item['wr']
    pnl = item['pnl']
    ret = item['ret']
    max_dd = item['max_dd']
    pf = item['pf']
    eq_curve = item['eq_curve']
    eq_dates = item['eq_dates']
    m_stats = item['m_stats']

    # Palet Warna Clean Konservatif
    BG_DARK = '#0b0f17'       # Obsidian Slate
    PANEL_BG = '#121824'      # Panel Gelap Elegan
    BORDER_COL = '#1e2636'    # Garis Pembatas Lembut
    TEXT_MAIN = '#f8fafc'     # Teks Utama Putih
    TEXT_MUTED = '#94a3b8'    # Teks Sekunder Abu-abu
    GREEN_COL = '#10b981'     # Hijau Emerald
    RED_COL = '#f43f5e'       # Merah Muted
    BLUE_COL = '#3b82f6'      # Biru Muted

    fig = plt.figure(figsize=(18, 11), dpi=150)
    fig.patch.set_facecolor(BG_DARK)

    # 3 Baris: Top KPI (15%), Tengah Grafik (45%), Bawah Tabel (40%)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1.3, 1.2], width_ratios=[1.2, 0.8],
                           left=0.05, right=0.95, top=0.92, bottom=0.06, hspace=0.35, wspace=0.22)

    # Title Minimalis & Clean
    title_text = f"LAPORAN TRADING EMAS (XAU/USD) — {yr_label.upper()}"
    fig.text(0.05, 0.960, title_text, fontsize=16, fontweight='bold', color=TEXT_MAIN, ha='left')
    fig.text(0.05, 0.938, f"Modal Awal: USD 10,000.00  |  Risiko: 1.0% per Trade (USD 100)  |  Maks 10 Entri/Hari  |  Friksi Broker Penuh", 
             fontsize=10, color=TEXT_MUTED, ha='left')

    # Badge Status
    status_str = "HASIL: PROFIT" if pnl >= 0 else "HASIL: DRAWDOWN"
    status_col = GREEN_COL if pnl >= 0 else RED_COL
    fig.text(0.95, 0.950, status_str, fontsize=11, fontweight='bold', color=status_col, ha='right',
             bbox=dict(boxstyle="round,pad=0.35", fc="#111827", ec=status_col, lw=1.2))

    # --- ROW 1: 4 KPI CARDS UTAMA ---
    gs_kpi = gs[0, :].subgridspec(1, 4, wspace=0.15)
    
    kpis = [
        ("TOTAL KEUNTUNGAN / KERUGIAN", f"{pnl:+,.2f} USD ({ret:+.2f}%)", "Hasil Bersih Setelah Biaya & Komisi", GREEN_COL if pnl >= 0 else RED_COL),
        ("AKURASI (WIN RATE)", f"{wr:.1f}%", f"{item['wins']} Menang dari {trades_n} Trade", BLUE_COL),
        ("PENURUNAN TERBESAR (MAX DD)", f"{max_dd:.2f}%", "Batas Maksimal Risiko Modal", GREEN_COL if abs(max_dd) < 10 else RED_COL),
        ("PROFIT FACTOR", f"{pf:.2f}", "Rasio Laba Bersih vs Kerugian", GREEN_COL if pf >= 1.2 else (BLUE_COL if pf >= 1.0 else RED_COL)),
    ]

    for k_idx, (k_title, k_val, k_sub, k_col) in enumerate(kpis):
        ax_k = fig.add_subplot(gs_kpi[0, k_idx])
        ax_k.set_facecolor(PANEL_BG)
        for s in ax_k.spines.values():
            s.set_color(BORDER_COL)
            s.set_linewidth(1.0)
        ax_k.set_xticks([])
        ax_k.set_yticks([])
        ax_k.text(0.08, 0.74, k_title, fontsize=8.5, fontweight='bold', color=TEXT_MUTED, transform=ax_k.transAxes)
        ax_k.text(0.08, 0.40, k_val, fontsize=16.0, fontweight='bold', color=k_col, transform=ax_k.transAxes)
        ax_k.text(0.08, 0.16, k_sub, fontsize=8.2, color='#cbd5e1', transform=ax_k.transAxes)

    # --- ROW 2, KIRI: KURVA PERTUMBUHAN MODAL (EQUITY CURVE) ---
    ax_eq = fig.add_subplot(gs[1, 0])
    ax_eq.set_facecolor(PANEL_BG)
    for s in ax_eq.spines.values():
        s.set_color(BORDER_COL)

    # Plot kurva saldo
    ax_eq.plot(eq_dates, eq_curve, color=GREEN_COL if pnl >= 0 else RED_COL, linewidth=2.0, label='Saldo Akun (USD)')
    ax_eq.axhline(INITIAL_EQUITY, color=BORDER_COL, linestyle='--', linewidth=1.2, label='Modal Awal (USD 10,000)')
    ax_eq.fill_between(eq_dates, INITIAL_EQUITY, eq_curve, color=GREEN_COL if pnl >= 0 else RED_COL, alpha=0.10)

    ax_eq.set_title("Pertumbuhan Saldo Modal Akun (USD 10,000)", fontsize=11.5, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left')
    ax_eq.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_eq.grid(True, color=BORDER_COL, linestyle='--', alpha=0.6)
    ax_eq.tick_params(colors=TEXT_MUTED, labelsize=8.8)
    ax_eq.legend(loc='upper left', frameon=True, facecolor=BG_DARK, edgecolor=BORDER_COL, fontsize=8.5)

    # --- ROW 2, KANAN: PROFIT / LOSS PER BULAN (BAR CHART) ---
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.set_facecolor(PANEL_BG)
    for s in ax_bar.spines.values():
        s.set_color(BORDER_COL)

    m_labels = [month_names_id.get(m['name'], m['name']) for m in m_stats]
    m_pcts = [m['pct'] for m in m_stats]
    x_pos = np.arange(len(m_labels))

    bar_colors = [GREEN_COL if p >= 0 else RED_COL for p in m_pcts]
    bars = ax_bar.bar(x_pos, m_pcts, color=bar_colors, width=0.55, edgecolor='#ffffff', linewidth=0.5)

    ax_bar.axhline(0, color=BORDER_COL, linewidth=1.2)
    ax_bar.set_title("Hasil Bersih (%) Tiap Bulan", fontsize=11.5, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(m_labels, fontsize=9.0, color=TEXT_MAIN)
    ax_bar.grid(axis='y', color=BORDER_COL, linestyle='--', alpha=0.6)
    ax_bar.tick_params(colors=TEXT_MUTED, labelsize=8.8)
    ax_bar.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))

    # Nilai di atas batang
    for b in bars:
        h = b.get_height()
        va = 'bottom' if h >= 0 else 'top'
        y_off = 0.25 if h >= 0 else -0.45
        ax_bar.text(b.get_x() + b.get_width()/2, h + y_off, f"{h:+.1f}%",
                    ha='center', va=va, fontsize=8.0, fontweight='bold',
                    color=GREEN_COL if h >= 0 else RED_COL)

    # Sesuaikan limit Y
    min_val = min(m_pcts) if len(m_pcts) > 0 else -5
    max_val = max(m_pcts) if len(m_pcts) > 0 else 5
    ax_bar.set_ylim(min(min_val * 1.35, -2.5), max(max_val * 1.35, 3.5))

    # --- ROW 3, FULL WIDTH: TABEL RINCIAN PER BULAN ---
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.set_facecolor(PANEL_BG)
    for s in ax_tbl.spines.values():
        s.set_color(BORDER_COL)
    ax_tbl.set_xticks([])
    ax_tbl.set_yticks([])

    ax_tbl.text(0.015, 0.90, "TABEL RINCIAN HASIL TRADING BULAN KE BULAN",
                fontsize=11.0, fontweight='bold', color=TEXT_MAIN)

    headers = ["Bulan", "Jumlah Trade", "Trade Menang", "Trade Kalah", "Akurasi (Win Rate)", "Profit / Loss (USD)", "Persentase (%)", "Hasil"]
    table_rows = []
    
    for m in m_stats:
        m_name = f"{month_names_id.get(m['name'], m['name'])} {yr}"
        pnl_val = f"{m['pnl']:+,.2f} USD"
        pct_val = f"{m['pct']:+.2f}%"
        status = "PROFIT" if m['pnl'] > 0 else ("BE / FLAT" if m['pnl'] == 0 else "MINUS")
        table_rows.append([
            m_name, str(m['trades']), str(m['wins']), str(m['losses']),
            f"{m['wr']:.1f}%", pnl_val, pct_val, status
        ])

    # Baris total tahunan
    table_rows.append([
        f"TOTAL TAHUN {yr}", str(trades_n), str(item['wins']), str(item['losses']),
        f"{wr:.1f}%", f"{pnl:+,.2f} USD", f"{ret:+.2f}%", "PROFIT" if pnl >= 0 else "DRAWDOWN"
    ])

    tbl = ax_tbl.table(
        cellText=table_rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        bbox=[0.015, 0.05, 0.97, 0.76]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER_COL)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor('#1e293b')
            cell.set_text_props(weight='bold', color='#ffffff')
        else:
            is_total_row = (r == len(table_rows))
            if is_total_row:
                cell.set_facecolor('#162235')
                cell.set_text_props(weight='bold', color='#ffffff')
            else:
                cell.set_facecolor('#111827' if r % 2 == 0 else '#0d131f')

            if c in [5, 6]:
                txt = table_rows[r-1][c]
                cell.set_text_props(color=GREEN_COL if '+' in txt else RED_COL, weight='bold')
            elif c == 4:
                cell.set_text_props(color=BLUE_COL, weight='bold')
            elif c == 7:
                txt = table_rows[r-1][c]
                cell.set_text_props(color=GREEN_COL if 'PROFIT' in txt else ('#cbd5e1' if 'FLAT' in txt else RED_COL), weight='bold')
            else:
                cell.set_text_props(color='#cbd5e1')

    # Simpan Gambar
    out_file_proj = reports_dir / f"backtest_full_{yr}.png"
    out_file_brain = brain_artifact_dir / f"backtest_full_{yr}.png"
    
    plt.savefig(out_file_proj, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(out_file_brain, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Gambar untuk tahun {yr} berhasil disimpan ke: {out_file_brain.name}")

print("\nSeluruh 5 gambar PNG berhasil digenerate!")
