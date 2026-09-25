"""
Generate Clean Visual Tearsheet Chart for $200 Capital Stress Test & Optimization
(Januari - Maret 2026)
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

out_png_proj = reports_dir / "stress_test_jan_mar_2026_modal_200.png"
out_png_brain = brain_artifact_dir / "stress_test_jan_mar_2026_modal_200.png"

# Color Palette
BG_DARK = '#090d16'
PANEL_BG = '#111726'
BORDER_COL = '#1e293b'
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#94a3b8'
GREEN_COL = '#10b981'
RED_COL = '#f43f5e'
BLUE_COL = '#3b82f6'
AMBER_COL = '#f59e0b'
PURPLE_COL = '#a855f7'

fig = plt.figure(figsize=(19, 12), dpi=150)
fig.patch.set_facecolor(BG_DARK)

gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1.25, 1.25], width_ratios=[1.15, 0.85],
                       left=0.05, right=0.95, top=0.92, bottom=0.06, hspace=0.35, wspace=0.22)

# Headers
fig.text(0.05, 0.962, "STRESS TEST & SIMULASI PROFIT OPTIMAL MODAL USD 200 (JANUARI – MARET 2026)", fontsize=16, fontweight='bold', color=TEXT_MAIN, ha='left')
fig.text(0.05, 0.938, "Analisis Ketahanan Akun Mikro $200 pada XAU/USD (Emas) | Evaluasi Brutal Drawdown vs Solusi Scaling Cerdas", fontsize=10, color=TEXT_MUTED, ha='left')

fig.text(0.95, 0.950, "HASIL TERBAIK: +2,844.4% (DARI USD 200 KE USD 5,888)", fontsize=11, fontweight='bold', color=GREEN_COL, ha='right',
         bbox=dict(boxstyle="round,pad=0.4", fc="#062e22", ec=GREEN_COL, lw=1.2))

# Row 1: 4 KPI Cards
gs_kpi = gs[0, :].subgridspec(1, 4, wspace=0.15)
kpis = [
    ("MODAL AWAL AKUN", "USD 200.00", "Ukuran Akun Mikro Retail", TEXT_MAIN),
    ("PROFIT MAKSIMAL (OPTIMAL)", "USD 5,888.79", "+2,844.4% Net Return (SL 1.0 ATR + Scale)", GREEN_COL),
    ("MODE AMAN KONSERVATIF", "USD 1,868.95", "+834.5% Return (Fixed 0.01 Lot, DD -2.8% Jan)", BLUE_COL),
    ("TEMUAN STRESS TEST BRUTAL", "MARGIN CALL (0.02 Lot)", "Ukuran Standar 0.02 Lot Gagal di Jan (-93.7%)", RED_COL)
]

for idx, (t, v, s, c) in enumerate(kpis):
    ax_k = fig.add_subplot(gs_kpi[0, idx])
    ax_k.set_facecolor(PANEL_BG)
    for sp in ax_k.spines.values():
        sp.set_color(BORDER_COL)
    ax_k.set_xticks([])
    ax_k.set_yticks([])
    ax_k.text(0.08, 0.74, t, fontsize=8.5, fontweight='bold', color=TEXT_MUTED, transform=ax_k.transAxes)
    ax_k.text(0.08, 0.40, v, fontsize=15.5, fontweight='bold', color=c, transform=ax_k.transAxes)
    ax_k.text(0.08, 0.16, s, fontsize=8.2, color='#cbd5e1', transform=ax_k.transAxes)

# Row 2, Kiri: Kurva Pertumbuhan Saldo
ax_eq = fig.add_subplot(gs[1, 0])
ax_eq.set_facecolor(PANEL_BG)
for sp in ax_eq.spines.values():
    sp.set_color(BORDER_COL)

# Data Kurva
months_labels = ['Awal', 'Januari', 'Februari', 'Maret']
x_m = np.arange(4)

# 1. Optimal Compounding ($200 -> $109.86 -> $1,299.38 -> $5,888.79)
curve_optimal = [200.0, 109.86, 1299.38, 5888.79]

# 2. Conservative Fixed 0.01 ($200 -> $194.39 -> $972.39 -> $1,868.95)
curve_safe = [200.0, 194.39, 972.39, 1868.95]

# 3. Unoptimized 0.02 Lot MC ($200 -> $12.58 Margin Call)
curve_mc = [200.0, 12.58, 12.58, 12.58]

ax_eq.plot(x_m, curve_optimal, color=GREEN_COL, linewidth=2.8, marker='o', markersize=6, label='Juara 1: Profit Maksimal (SL 1.0 ATR + Scale Up)')
ax_eq.plot(x_m, curve_safe, color=BLUE_COL, linewidth=2.2, marker='s', markersize=5, label='Juara 2: Konservatif Aman (SL 1.2 ATR + Fixed 0.01 Lot)')
ax_eq.plot(x_m, curve_mc, color=RED_COL, linewidth=1.8, linestyle='--', marker='x', markersize=6, label='Gagal: 0.02 Lot Standar (Margin Call di Jan)')
ax_eq.axhline(200.0, color=BORDER_COL, linestyle=':', linewidth=1.2)

ax_eq.fill_between(x_m, 200.0, curve_optimal, color=GREEN_COL, alpha=0.12)

ax_eq.set_title("Perbandingan Kurva Pertumbuhan Saldo Akun dari Modal USD 200", fontsize=11.5, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left')
ax_eq.set_xticks(x_m)
ax_eq.set_xticklabels(months_labels, fontsize=9.0, color=TEXT_MAIN)
ax_eq.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} USD'))
ax_eq.grid(True, color=BORDER_COL, linestyle='--', alpha=0.6)
ax_eq.tick_params(colors=TEXT_MUTED, labelsize=8.8)
ax_eq.legend(loc='upper left', frameon=True, facecolor=BG_DARK, edgecolor=BORDER_COL, fontsize=8.5)

# Row 2, Kanan: Bar Chart Hasil Bulan ke Bulan
ax_bar = fig.add_subplot(gs[1, 1])
ax_bar.set_facecolor(PANEL_BG)
for sp in ax_bar.spines.values():
    sp.set_color(BORDER_COL)

m_names = ['Januari 2026', 'Februari 2026', 'Maret 2026']
x_bar = np.arange(len(m_names))
w = 0.35

pnl_opt = [-90.14, 1189.52, 4589.41]
pnl_safe = [-5.61, 778.00, 896.56]

rects1 = ax_bar.bar(x_bar - w/2, pnl_opt, width=w, label='Mode Profit Maksimal', color=[RED_COL if p < 0 else GREEN_COL for p in pnl_opt], edgecolor='#ffffff', linewidth=0.5)
rects2 = ax_bar.bar(x_bar + w/2, pnl_safe, width=w, label='Mode Konservatif Aman', color=[AMBER_COL if p < 0 else BLUE_COL for p in pnl_safe], edgecolor='#ffffff', linewidth=0.5)

ax_bar.axhline(0, color=BORDER_COL, linewidth=1.2)
ax_bar.set_title("Hasil Bersih Bulan ke Bulan (Januari – Maret 2026)", fontsize=11.5, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left')
ax_bar.set_xticks(x_bar)
ax_bar.set_xticklabels(m_names, fontsize=9.0, color=TEXT_MAIN)
ax_bar.grid(axis='y', color=BORDER_COL, linestyle='--', alpha=0.6)
ax_bar.tick_params(colors=TEXT_MUTED, labelsize=8.8)
ax_bar.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:+,.0f} USD'))
ax_bar.legend(loc='upper left', frameon=True, facecolor=BG_DARK, edgecolor=BORDER_COL, fontsize=8.5)

# Value annotations
for r in rects1:
    h = r.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_off = 100 if h >= 0 else -180
    ax_bar.text(r.get_x() + r.get_width()/2, h + y_off, f"{h:+,.0f} USD", ha='center', va=va, fontsize=7.8, fontweight='bold', color=GREEN_COL if h >= 0 else RED_COL)

for r in rects2:
    h = r.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_off = 100 if h >= 0 else -180
    ax_bar.text(r.get_x() + r.get_width()/2, h + y_off, f"{h:+,.0f} USD", ha='center', va=va, fontsize=7.8, fontweight='bold', color=BLUE_COL if h >= 0 else AMBER_COL)

ax_bar.set_ylim(-600, 5600)

# Row 3: Tabel Perbandingan Stress Test & Rincian Strategi
ax_tbl = fig.add_subplot(gs[2, :])
ax_tbl.set_facecolor(PANEL_BG)
for sp in ax_tbl.spines.values():
    sp.set_color(BORDER_COL)
ax_tbl.set_xticks([])
ax_tbl.set_yticks([])

ax_tbl.text(0.015, 0.90, "TABEL PERBANDINGAN SKENARIO STRESS TEST & SOLUSI OPTIMAL MODAL USD 200", fontsize=11.0, fontweight='bold', color=TEXT_MAIN)

tbl_headers = ["Skenario & Strategi", "Konfigurasi Sizing & SL", "Drawdown Terendah (Jan)", "Saldo Akhir (Maret)", "Profit Bersih (USD)", "ROI (%)", "Status Ketahanan Akun"]
tbl_rows = [
    ["1. Standar Tanpa Optimasi", "0.02 Lot (Twin Order), SL 1.5 ATR", "$12.58 (-93.7% di hari ke-3)", "$12.58", "-$187.42 USD", "-93.7%", "MARGIN CALL (Akun Hangus)"],
    ["2. Standar 0.01 Lot Biasa", "0.01 Lot Flat, SL 1.5 ATR", "$3.22 (-98.4% akhir Jan)", "$3.22", "-$196.78 USD", "-98.4%", "MARGIN CALL (Akun Hangus)"],
    ["3. Akun Cent (Mikro 1%)", "Risiko Proporsional 1% (Akun Cent)", "$72.10 (-68.9% Jan)", "$767.69", "+$567.69 USD", "+283.9%", "SELAMAT (Pertumbuhan Stabil)"],
    ["4. Konservatif Safe Flat", "0.01 Lot Flat, SL 1.2 ATR, TP 1.8R", "$194.39 (Hanya -2.8% Jan!)", "$1,868.95", "+$1,668.95 USD", "+834.5%", "SELAMAT (Sangat Rendah Risiko)"],
    ["5. JUARA 1 PROFIT MAKSIMAL", "SL 1.0 ATR, TP 1.5R + Stepwise Scale", "$83.20 (Bertahan di Jan!)", "$5,888.79", "+$5,688.79 USD", "+2,844.4%", "SUKSES BESAR (Hampir 30x Lipat)"]
]

tbl = ax_tbl.table(
    cellText=tbl_rows,
    colLabels=tbl_headers,
    colWidths=[0.18, 0.23, 0.16, 0.11, 0.12, 0.07, 0.17],
    loc='center',
    cellLoc='center',
    bbox=[0.015, 0.05, 0.97, 0.77]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(BORDER_COL)
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(weight='bold', color='#ffffff')
        cell.set_height(0.18)
    else:
        is_winner = (r == len(tbl_rows))
        if is_winner:
            cell.set_facecolor('#0d2e24')
            cell.set_text_props(weight='bold', color='#ffffff')
        elif r <= 2:
            cell.set_facecolor('#2d1217')
        else:
            cell.set_facecolor('#111827' if r % 2 == 0 else '#0d131f')

        if c == 4:
            val_txt = tbl_rows[r-1][c]
            cell.set_text_props(color=GREEN_COL if '+' in val_txt else RED_COL, weight='bold')
        elif c == 3:
            cell.set_text_props(color=TEXT_MAIN, weight='bold')
        elif c == 5:
            cell.set_text_props(color=GREEN_COL if '+' in tbl_rows[r-1][c] else RED_COL, weight='bold')
        elif c == 6:
            val_txt = tbl_rows[r-1][c]
            if "MARGIN CALL" in val_txt:
                cell.set_text_props(color=RED_COL, weight='bold')
            elif "SUKSES" in val_txt:
                cell.set_text_props(color=GREEN_COL, weight='bold')
            else:
                cell.set_text_props(color=BLUE_COL, weight='bold')
        else:
            cell.set_text_props(color='#cbd5e1')

plt.savefig(out_png_proj, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(out_png_brain, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print(f"Grafik tearsheet $200 berhasil disimpan ke:\n  {out_png_brain}")
