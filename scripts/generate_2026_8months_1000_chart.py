"""
Generate Clean Visual Chart for Jan - Aug 2026 Simulation ($1,000 Capital, 3 Trades/Day)
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

out_png_proj = reports_dir / "simulasi_jan_agu_2026_modal_1000.png"
out_png_brain = brain_artifact_dir / "simulasi_jan_agu_2026_modal_1000.png"

# Monthly Data
months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu']
m_pnl = [-439.59, 1244.77, 1486.25, -180.68, -191.88, 463.16, 627.01, 254.72]
m_trades = [63, 60, 66, 63, 63, 66, 69, 63]
m_wins = [30, 43, 46, 33, 31, 38, 45, 36]
m_wr = [47.6, 71.7, 69.7, 52.4, 49.2, 57.6, 65.2, 57.1]
m_balances = [560.41, 1805.18, 3291.43, 3110.75, 2918.87, 3382.03, 4009.04, 4263.76]

# Matplotlib Styling
BG_DARK = '#090d16'
PANEL_BG = '#111726'
BORDER_COL = '#1e293b'
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#94a3b8'
GREEN_COL = '#10b981'
RED_COL = '#f43f5e'
BLUE_COL = '#3b82f6'

fig = plt.figure(figsize=(19, 12), dpi=150)
fig.patch.set_facecolor(BG_DARK)

gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1.25, 1.25], width_ratios=[1.15, 0.85],
                       left=0.05, right=0.95, top=0.92, bottom=0.06, hspace=0.35, wspace=0.22)

# Title
fig.text(0.05, 0.962, "SIMULASI TRADING XAU/USD (EMAS) — JANUARI S/D AGUSTUS 2026", fontsize=16, fontweight='bold', color=TEXT_MAIN, ha='left')
fig.text(0.05, 0.938, "Modal Awal: USD 1,000.00  |  3 Trade per Hari Pasar Buka (Total 513 Trade)  |  Lot Retail 0.02 (0.01 Pos A + 0.01 Pos B)", fontsize=10, color=TEXT_MUTED, ha='left')

fig.text(0.95, 0.950, "HASIL 8 BULAN: PROFIT +326.38%", fontsize=11, fontweight='bold', color=GREEN_COL, ha='right',
         bbox=dict(boxstyle="round,pad=0.4", fc="#062e22", ec=GREEN_COL, lw=1.2))

# Row 1: 4 KPI Cards
gs_kpi = gs[0, :].subgridspec(1, 4, wspace=0.15)
kpis = [
    ("MODAL AWAL AKUN", "USD 1,000.00", "Ukuran Akun Retail Standar", TEXT_MAIN),
    ("SALDO AKHIR (AGUSTUS)", "USD 4,263.76", "+3,263.76 USD (+326.38% Net Profit)", GREEN_COL),
    ("AKURASI (WIN RATE)", "58.9%", "302 Menang / 211 Kalah (513 Trade)", BLUE_COL),
    ("KONSISTENSI BULANAN", "5 dari 8 Bulan Profit", "Puncak: Feb (+1.24k) & Mar (+1.48k)", GREEN_COL)
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

# Row 2, Kiri: Kurva Pertumbuhan Saldo Akun
ax_eq = fig.add_subplot(gs[1, 0])
ax_eq.set_facecolor(PANEL_BG)
for sp in ax_eq.spines.values():
    sp.set_color(BORDER_COL)

x_months = np.arange(len(months_labels) + 1)
balances_curve = [1000.0] + m_balances

ax_eq.plot(x_months, balances_curve, color=GREEN_COL, linewidth=2.5, marker='o', markersize=5, label='Saldo Akun (USD)')
ax_eq.axhline(1000.0, color=BORDER_COL, linestyle='--', linewidth=1.2, label='Modal Awal (USD 1,000)')
ax_eq.fill_between(x_months, 1000.0, balances_curve, color=GREEN_COL, alpha=0.10)

ax_eq.set_title("Pertumbuhan Saldo Modal Akun dari USD 1,000 ke USD 4,263", fontsize=11.5, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left')
ax_eq.set_xticks(x_months)
ax_eq.set_xticklabels(['Awal'] + months_labels, fontsize=9.0, color=TEXT_MAIN)
ax_eq.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} USD'))
ax_eq.grid(True, color=BORDER_COL, linestyle='--', alpha=0.6)
ax_eq.tick_params(colors=TEXT_MUTED, labelsize=8.8)
ax_eq.legend(loc='upper left', frameon=True, facecolor=BG_DARK, edgecolor=BORDER_COL, fontsize=8.5)

# Row 2, Kanan: Hasil Bersih per Bulan (Bar Chart)
ax_bar = fig.add_subplot(gs[1, 1])
ax_bar.set_facecolor(PANEL_BG)
for sp in ax_bar.spines.values():
    sp.set_color(BORDER_COL)

x_m = np.arange(len(months_labels))
bar_cols = [GREEN_COL if p >= 0 else RED_COL for p in m_pnl]
bars = ax_bar.bar(x_m, m_pnl, color=bar_cols, width=0.55, edgecolor='#ffffff', linewidth=0.5)
ax_bar.axhline(0, color=BORDER_COL, linewidth=1.2)

ax_bar.set_title("Keuntungan / Kerugian Tiap Bulan (USD)", fontsize=11.5, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left')
ax_bar.set_xticks(x_m)
ax_bar.set_xticklabels(months_labels, fontsize=9.0, color=TEXT_MAIN)
ax_bar.grid(axis='y', color=BORDER_COL, linestyle='--', alpha=0.6)
ax_bar.tick_params(colors=TEXT_MUTED, labelsize=8.8)
ax_bar.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:+,.0f} USD'))

for b in bars:
    h = b.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_off = 45 if h >= 0 else -70
    ax_bar.text(b.get_x() + b.get_width()/2, h + y_off, f"{h:+,.0f} USD",
                ha='center', va=va, fontsize=8.0, fontweight='bold',
                color=GREEN_COL if h >= 0 else RED_COL)

ax_bar.set_ylim(-700, 1800)

# Row 3: Tabel Rekapitulasi Bulan ke Bulan
ax_tbl = fig.add_subplot(gs[2, :])
ax_tbl.set_facecolor(PANEL_BG)
for sp in ax_tbl.spines.values():
    sp.set_color(BORDER_COL)
ax_tbl.set_xticks([])
ax_tbl.set_yticks([])

ax_tbl.text(0.015, 0.90, "TABEL RINCIAN HASIL BULAN KE BULAN (JANUARI – AGUSTUS 2026)", fontsize=11.0, fontweight='bold', color=TEXT_MAIN)

tbl_headers = ["Bulan", "Total Trade", "Menang (Win)", "Kalah (Loss)", "Akurasi (Win Rate)", "Profit / Rugi Bersih", "Saldo Akun Akhir Bulan", "Status Bulan"]
tbl_rows = [
    ["Januari 2026", "63 Trade", "30 Trade", "33 Trade", "47.6%", "-439.59 USD", "560.41 USD", "DRAWDOWN"],
    ["Februari 2026", "60 Trade", "43 Trade", "17 Trade", "71.7%", "+1,244.77 USD", "1,805.18 USD", "PROFIT BESAR"],
    ["Maret 2026", "66 Trade", "46 Trade", "20 Trade", "69.7%", "+1,486.25 USD", "3,291.43 USD", "PROFIT BESAR"],
    ["April 2026", "63 Trade", "33 Trade", "30 Trade", "52.4%", "-180.68 USD", "3,110.75 USD", "DRAWDOWN KECIL"],
    ["Mei 2026", "63 Trade", "31 Trade", "32 Trade", "49.2%", "-191.88 USD", "2,918.87 USD", "DRAWDOWN KECIL"],
    ["Juni 2026", "66 Trade", "38 Trade", "28 Trade", "57.6%", "+463.16 USD", "3,382.03 USD", "PROFIT"],
    ["Juli 2026", "69 Trade", "45 Trade", "24 Trade", "65.2%", "+627.01 USD", "4,009.04 USD", "PROFIT"],
    ["Agustus 2026", "63 Trade", "36 Trade", "27 Trade", "57.1%", "+254.72 USD", "4,263.76 USD", "PROFIT"],
    ["TOTAL 8 BULAN", "513 Trade", "302 Trade", "211 Trade", "58.9%", "+3,263.76 USD", "4,263.76 USD", "PROFIT +326.38%"]
]

tbl = ax_tbl.table(
    cellText=tbl_rows,
    colLabels=tbl_headers,
    loc='center',
    cellLoc='center',
    bbox=[0.015, 0.05, 0.97, 0.77]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.0)

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

        if c in [5]:
            val_txt = tbl_rows[r-1][c]
            cell.set_text_props(color=GREEN_COL if '+' in val_txt else RED_COL, weight='bold')
        elif c == 4:
            cell.set_text_props(color=BLUE_COL, weight='bold')
        elif c in [6]:
            cell.set_text_props(color=TEXT_MAIN, weight='bold')
        elif c == 7:
            val_txt = tbl_rows[r-1][c]
            cell.set_text_props(color=GREEN_COL if 'PROFIT' in val_txt else RED_COL, weight='bold')
        else:
            cell.set_text_props(color='#cbd5e1')

plt.savefig(out_png_proj, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(out_png_brain, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print(f"Grafik 8 bulan modal $1,000 berhasil digenerate:\n  {out_png_brain}")
