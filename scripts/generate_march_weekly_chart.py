"""
Generate Clean Visual Chart for March 2026 1-Week Simulation (Min 3 trades per day)
"""

import sys, os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
brain_artifact_dir = pathlib.Path(r"C:\Users\haika\.gemini\antigravity-ide\brain\f18a7bdb-34cf-4c7d-86ed-374ce8cd5082")

out_png_proj = reports_dir / "simulasi_maret_2026_1minggu.png"
out_png_brain = brain_artifact_dir / "simulasi_maret_2026_1minggu.png"

# Data Simulasi 1 Minggu
days = ['Senin (2 Mar)', 'Selasa (3 Mar)', 'Rabu (4 Mar)', 'Kamis (5 Mar)', 'Jumat (6 Mar)']
day_pnl = [+174.96, +240.17, +181.88, -191.76, +127.40]
day_trades = [3, 3, 3, 3, 3]
day_wins = [3, 2, 3, 0, 3]

# Cumulative equity progression over the 15 trades
equity_steps = [
    10000.0,
    10092.01, 10121.66, 10174.96, # Senin
    10348.81, 10270.13, 10415.13, # Selasa
    10456.86, 10504.66, 10597.01, # Rabu
    10506.21, 10417.59, 10405.25, # Kamis
    10461.11, 10510.31, 10532.65  # Jumat
]

trade_labels = [f"T{i}" for i in range(16)]

BG_DARK = '#090d16'
PANEL_BG = '#111726'
BORDER_COL = '#1e293b'
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#94a3b8'
GREEN_COL = '#10b981'
RED_COL = '#f43f5e'
BLUE_COL = '#3b82f6'

fig = plt.figure(figsize=(18, 11), dpi=150)
fig.patch.set_facecolor(BG_DARK)

gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1.25, 1.25], width_ratios=[1.15, 0.85],
                       left=0.05, right=0.95, top=0.92, bottom=0.06, hspace=0.35, wspace=0.22)

# Header
fig.text(0.05, 0.962, "SIMULASI TRADING 1 MINGGU (2 – 6 MARET 2026) — XAU/USD (EMAS)", fontsize=15.5, fontweight='bold', color=TEXT_MAIN, ha='left')
fig.text(0.05, 0.938, "Skenario Uji: Wajib Eksekusi 3 Trade Setiap Hari Pasar Buka (Total 15 Trade) | Modal Awal USD 10,000 | Risiko 1% per Trade", fontsize=10, color=TEXT_MUTED, ha='left')

fig.text(0.95, 0.950, "HASIL 1 MINGGU: PROFIT +5.33%", fontsize=11, fontweight='bold', color=GREEN_COL, ha='right',
         bbox=dict(boxstyle="round,pad=0.4", fc="#062e22", ec=GREEN_COL, lw=1.2))

# Row 1: 4 KPI Cards
gs_kpi = gs[0, :].subgridspec(1, 4, wspace=0.15)
kpis = [
    ("TOTAL PROFIT 1 MINGGU", "+532.65 USD (+5.33%)", "Keuntungan Bersih dari Modal USD 10,000", GREEN_COL),
    ("AKURASI (WIN RATE)", "73.3%", "11 Menang / 4 Kalah (15 Trade)", BLUE_COL),
    ("HARI PASAR PROFIT", "4 dari 5 Hari Hijau (80%)", "Senin, Selasa, Rabu, & Jumat Untung", GREEN_COL),
    ("PENURUNAN TERBESAR (MAX DD)", "-1.81%", "Terjadi di Hari Kamis (-191.76 USD)", GREEN_COL)
]

for idx, (t, v, s, c) in enumerate(kpis):
    ax_k = fig.add_subplot(gs_kpi[0, idx])
    ax_k.set_facecolor(PANEL_BG)
    for sp in ax_k.spines.values():
        sp.set_color(BORDER_COL)
    ax_k.set_xticks([])
    ax_k.set_yticks([])
    ax_k.text(0.08, 0.74, t, fontsize=8.5, fontweight='bold', color=TEXT_MUTED, transform=ax_k.transAxes)
    ax_k.text(0.08, 0.40, v, fontsize=16.0, fontweight='bold', color=c, transform=ax_k.transAxes)
    ax_k.text(0.08, 0.16, s, fontsize=8.2, color='#cbd5e1', transform=ax_k.transAxes)

# Row 2, Kiri: Kurva Pertumbuhan Saldo
ax_eq = fig.add_subplot(gs[1, 0])
ax_eq.set_facecolor(PANEL_BG)
for sp in ax_eq.spines.values():
    sp.set_color(BORDER_COL)

x_steps = np.arange(len(equity_steps))
ax_eq.plot(x_steps, equity_steps, color=GREEN_COL, linewidth=2.2, marker='o', markersize=4, label='Saldo Akun (USD)')
ax_eq.axhline(10000.0, color=BORDER_COL, linestyle='--', linewidth=1.2, label='Modal Awal (USD 10,000)')
ax_eq.fill_between(x_steps, 10000.0, equity_steps, color=GREEN_COL, alpha=0.10)

# Day dividers
for day_div in [3, 6, 9, 12]:
    ax_eq.axvline(day_div, color='#334155', linestyle=':', alpha=0.7)

ax_eq.set_title("Pertumbuhan Saldo Modal per Trade (Trade #1 s/d #15)", fontsize=11.5, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left')
ax_eq.set_xticks([0, 3, 6, 9, 12, 15])
ax_eq.set_xticklabels(['Awal', 'Senin End', 'Selasa End', 'Rabu End', 'Kamis End', 'Jumat End'], fontsize=9.0, color=TEXT_MAIN)
ax_eq.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} USD'))
ax_eq.grid(True, color=BORDER_COL, linestyle='--', alpha=0.6)
ax_eq.tick_params(colors=TEXT_MUTED, labelsize=8.8)
ax_eq.legend(loc='upper left', frameon=True, facecolor=BG_DARK, edgecolor=BORDER_COL, fontsize=8.5)

# Row 2, Kanan: Hasil Profit / Rugi per Hari
ax_bar = fig.add_subplot(gs[1, 1])
ax_bar.set_facecolor(PANEL_BG)
for sp in ax_bar.spines.values():
    sp.set_color(BORDER_COL)

x_d = np.arange(len(days))
bar_cols = [GREEN_COL if p >= 0 else RED_COL for p in day_pnl]
bars = ax_bar.bar(x_d, day_pnl, color=bar_cols, width=0.55, edgecolor='#ffffff', linewidth=0.5)
ax_bar.axhline(0, color=BORDER_COL, linewidth=1.2)

ax_bar.set_title("Hasil Bersih per Hari (Senin – Jumat)", fontsize=11.5, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left')
ax_bar.set_xticks(x_d)
ax_bar.set_xticklabels(['Senin\n(2 Mar)', 'Selasa\n(3 Mar)', 'Rabu\n(4 Mar)', 'Kamis\n(5 Mar)', 'Jumat\n(6 Mar)'], fontsize=9.0, color=TEXT_MAIN)
ax_bar.grid(axis='y', color=BORDER_COL, linestyle='--', alpha=0.6)
ax_bar.tick_params(colors=TEXT_MUTED, labelsize=8.8)
ax_bar.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:+,.0f} USD'))

for b in bars:
    h = b.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_off = 10 if h >= 0 else -18
    ax_bar.text(b.get_x() + b.get_width()/2, h + y_off, f"{h:+,.1f} USD",
                ha='center', va=va, fontsize=8.5, fontweight='bold',
                color=GREEN_COL if h >= 0 else RED_COL)

ax_bar.set_ylim(-260, 320)

# Row 3: Tabel Rekapitulasi Hari ke Hari
ax_tbl = fig.add_subplot(gs[2, :])
ax_tbl.set_facecolor(PANEL_BG)
for sp in ax_tbl.spines.values():
    sp.set_color(BORDER_COL)
ax_tbl.set_xticks([])
ax_tbl.set_yticks([])

ax_tbl.text(0.015, 0.90, "TABEL REKAPITULASI HASIL HARIAN (SENIN – JUMAT)", fontsize=11.0, fontweight='bold', color=TEXT_MAIN)

tbl_headers = ["Hari / Tanggal", "Jumlah Trade", "Menang (Win)", "Kalah (Loss)", "Akurasi (Win Rate)", "Profit / Rugi Bersih", "Persentase Modal", "Status Harian"]
tbl_rows = [
    ["Senin, 02 Maret 2026", "3 Trade", "3 Trade", "0 Trade", "100.0%", "+174.96 USD", "+1.75%", "PROFIT"],
    ["Selasa, 03 Maret 2026", "3 Trade", "2 Trade", "1 Trade", "66.7%", "+240.17 USD", "+2.40%", "PROFIT"],
    ["Rabu, 04 Maret 2026", "3 Trade", "3 Trade", "0 Trade", "100.0%", "+181.88 USD", "+1.82%", "PROFIT"],
    ["Kamis, 05 Maret 2026", "3 Trade", "0 Trade", "3 Trade", "0.0%", "-191.76 USD", "-1.92%", "DRAWDOWN"],
    ["Jumat, 06 Maret 2026", "3 Trade", "3 Trade", "0 Trade", "100.0%", "+127.40 USD", "+1.27%", "PROFIT"],
    ["TOTAL 1 MINGGU", "15 Trade", "11 Trade", "4 Trade", "73.3%", "+532.65 USD", "+5.33%", "PROFIT MINGGUAN"]
]

tbl = ax_tbl.table(
    cellText=tbl_rows,
    colLabels=tbl_headers,
    loc='center',
    cellLoc='center',
    bbox=[0.015, 0.08, 0.97, 0.72]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.2)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(BORDER_COL)
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(weight='bold', color='#ffffff')
        cell.set_height(0.18)
    else:
        is_tot = (r == len(tbl_rows))
        if is_tot:
            cell.set_facecolor('#162235')
            cell.set_text_props(weight='bold', color='#ffffff')
        else:
            cell.set_facecolor('#111827' if r % 2 == 0 else '#0d131f')

        if c in [5, 6]:
            val_txt = tbl_rows[r-1][c]
            cell.set_text_props(color=GREEN_COL if '+' in val_txt else RED_COL, weight='bold')
        elif c == 4:
            cell.set_text_props(color=BLUE_COL, weight='bold')
        elif c == 7:
            val_txt = tbl_rows[r-1][c]
            cell.set_text_props(color=GREEN_COL if 'PROFIT' in val_txt else RED_COL, weight='bold')
        else:
            cell.set_text_props(color='#cbd5e1')

plt.savefig(out_png_proj, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(out_png_brain, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print(f"Chart simulasi 1 minggu berhasil digenerate:\n  {out_png_brain}")
