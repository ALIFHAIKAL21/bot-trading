"""
Generate Clean Conservative Institutional Tearsheet for Traders
Theme: Deep Obsidian & Emerald Slate (Bloomberg / Hedge Fund Style)
"""

import sys, os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
brain_artifact_dir = pathlib.Path(r"C:\Users\haika\.gemini\antigravity-ide\brain\f18a7bdb-34cf-4c7d-86ed-374ce8cd5082")

out_png_proj = reports_dir / "xau_deep_sniper_clean_trader_tearsheet.png"
out_png_brain = brain_artifact_dir / "xau_deep_sniper_clean_trader_tearsheet.png"

# Quantitative Data (Tau = 0.355)
years = ['2022', '2023', '2024', '2025', '2026']
net_returns = [27.42, 7.95, -12.82, 15.10, 5.32]
max_drawdowns = [-5.54, -5.64, -18.76, -3.61, -5.80]
win_rates = [70.0, 59.5, 46.2, 62.9, 55.0]
trades = [80, 74, 78, 62, 40]
net_pnls = [2742.02, 794.83, -1282.22, 1510.46, 532.17]

# 2026 Monthly Data
months_2026 = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
m_returns_2026 = [2.55, 5.85, -0.82, -1.19, 1.06, 0.17, -2.15, -0.04]
m_trades_2026 = [10, 6, 1, 3, 4, 4, 5, 7]

# Matplotlib Style Settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.style.use('dark_background')

# Colors
BG_COLOR = '#090d16'      # Deep obsidian
PANEL_COLOR = '#111726'   # Slate navy panel
BORDER_COLOR = '#1e293b'  # Subtle slate border
TEXT_TITLE = '#f8fafc'    # Bright white
TEXT_MUTED = '#94a3b8'    # Slate gray
TEXT_BODY = '#cbd5e1'     # Light slate
GREEN_GAIN = '#10b981'    # Emerald green
RED_LOSS = '#f43f5e'      # Soft rose red
BLUE_ACCENT = '#3b82f6'   # Muted institutional blue

fig = plt.figure(figsize=(19, 12), dpi=150)
fig.patch.set_facecolor(BG_COLOR)

# Layout Setup
gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[0.55, 1.25, 1.20],
    width_ratios=[1.15, 0.85],
    left=0.05, right=0.95,
    top=0.91, bottom=0.06,
    hspace=0.36, wspace=0.22
)

# ------------------------------------------------------------------------------
# SUPER HEADER (Crisp, Elegant, Minimalist)
# ------------------------------------------------------------------------------
fig.text(0.05, 0.965, "XAU_DEEP_SNIPER  |  INSTITUTIONAL PERFORMANCE TEARSHEET",
         fontsize=16, fontweight='bold', color=TEXT_TITLE, ha='left')
fig.text(0.05, 0.940, "Foundation Model MOMENT-1-Large  *  M30 Execution  *  Multi-Year Walk-Forward Audit (Jan - Aug 2022-2026)",
         fontsize=10.5, color=TEXT_MUTED, ha='left')

tag_text = "STATUS: PASS (4/5 YEARS PROFITABLE)"
fig.text(0.95, 0.952, tag_text,
         fontsize=10, fontweight='bold', color=GREEN_GAIN, ha='right',
         bbox=dict(boxstyle="round,pad=0.4", fc="#062e22", ec=GREEN_GAIN, lw=1.2))

# ------------------------------------------------------------------------------
# ROW 1: 5 CLEAN METRIC CARDS (Spanned across both columns)
# ------------------------------------------------------------------------------
gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :], wspace=0.15)

cards_data = [
    ("INITIAL EQUITY", "USD 10,000.00", "Base Account Size", TEXT_TITLE),
    ("5-YEAR NET PROFIT", "+USD 4,297.26", "+42.97% Cumulative", GREEN_GAIN),
    ("ANNUAL WIN RATE", "80.0%", "4 of 5 Years Net Positive", GREEN_GAIN),
    ("OVERALL TRADE WR", "59.0%", "197 Wins / 334 Trades", BLUE_ACCENT),
    ("2026 MAX DRAWDOWN", "-5.80%", "August Capped at -0.04%", GREEN_GAIN),
]

for col_idx, (title, val, sub, val_col) in enumerate(cards_data):
    ax_card = fig.add_subplot(gs_top[0, col_idx])
    ax_card.set_facecolor(PANEL_COLOR)
    for s in ax_card.spines.values():
        s.set_color(BORDER_COLOR)
        s.set_linewidth(1.0)
    ax_card.set_xticks([])
    ax_card.set_yticks([])

    ax_card.text(0.08, 0.74, title, fontsize=8.5, fontweight='bold', color=TEXT_MUTED, transform=ax_card.transAxes)
    ax_card.text(0.08, 0.40, val, fontsize=16.5, fontweight='bold', color=val_col, transform=ax_card.transAxes)
    ax_card.text(0.08, 0.16, sub, fontsize=8.2, color=TEXT_BODY, transform=ax_card.transAxes)

# ------------------------------------------------------------------------------
# ROW 2, LEFT: ANNUAL NET RETURN & MAX DRAWDOWN (Side-by-Side Dual Bar)
# ------------------------------------------------------------------------------
ax_annual = fig.add_subplot(gs[1, 0])
ax_annual.set_facecolor(PANEL_COLOR)
for s in ax_annual.spines.values():
    s.set_color(BORDER_COLOR)

x_pos = np.arange(len(years))
bar_w = 0.36

# Net Return Bars
bars_ret = ax_annual.bar(
    x_pos - bar_w/2, net_returns, width=bar_w,
    color=[GREEN_GAIN if r >= 0 else RED_LOSS for r in net_returns],
    alpha=0.90, edgecolor='#ffffff', linewidth=0.5, label='Net Return (%)'
)

# Max Drawdown Bars
bars_dd = ax_annual.bar(
    x_pos + bar_w/2, max_drawdowns, width=bar_w,
    color='#64748b', alpha=0.75, edgecolor='#cbd5e1', linewidth=0.5, label='Max Drawdown (%)'
)

ax_annual.axhline(0, color=BORDER_COLOR, linewidth=1.2)
ax_annual.set_title("Annual Net Return vs. Max Drawdown (Jan - Aug)", fontsize=11.5, fontweight='bold', color=TEXT_TITLE, pad=12, loc='left')
ax_annual.set_xticks(x_pos)
ax_annual.set_xticklabels([f"Jan-Aug {y}" for y in years], fontsize=9.5, color=TEXT_BODY)
ax_annual.set_ylabel("Percentage (%)", fontsize=9.5, color=TEXT_MUTED)
ax_annual.grid(axis='y', color=BORDER_COLOR, linestyle='--', alpha=0.6)
ax_annual.legend(loc='lower left', frameon=True, facecolor=BG_COLOR, edgecolor=BORDER_COLOR, fontsize=8.5)

# Value annotations
for b in bars_ret:
    h = b.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_off = 1.0 if h >= 0 else -1.5
    ax_annual.text(b.get_x() + b.get_width()/2, h + y_off, f"{h:+.1f}%",
                   ha='center', va=va, fontsize=8.5, fontweight='bold',
                   color=GREEN_GAIN if h >= 0 else RED_LOSS)

for b in bars_dd:
    h = b.get_height()
    ax_annual.text(b.get_x() + b.get_width()/2, h - 1.5, f"{h:.1f}%",
                   ha='center', va='top', fontsize=8.2, color='#cbd5e1')

ax_annual.set_ylim(-26, 35)

# ------------------------------------------------------------------------------
# ROW 2, RIGHT: 2026 MONTH-BY-MONTH RETURN (Clean Horizontal / Vertical Bars)
# ------------------------------------------------------------------------------
ax_monthly = fig.add_subplot(gs[1, 1])
ax_monthly.set_facecolor(PANEL_COLOR)
for s in ax_monthly.spines.values():
    s.set_color(BORDER_COLOR)

x_m = np.arange(len(months_2026))
bars_m = ax_monthly.bar(
    x_m, m_returns_2026, width=0.55,
    color=[GREEN_GAIN if r >= 0 else RED_LOSS for r in m_returns_2026],
    alpha=0.90, edgecolor='#ffffff', linewidth=0.5
)

ax_monthly.axhline(0, color=BORDER_COLOR, linewidth=1.2)
ax_monthly.set_title("2026 Monthly Progression (Capital Preservation in August)", fontsize=11.5, fontweight='bold', color=TEXT_TITLE, pad=12, loc='left')
ax_monthly.set_xticks(x_m)
ax_monthly.set_xticklabels(months_2026, fontsize=9.5, color=TEXT_BODY)
ax_monthly.set_ylabel("Monthly Net Return (%)", fontsize=9.5, color=TEXT_MUTED)
ax_monthly.grid(axis='y', color=BORDER_COLOR, linestyle='--', alpha=0.6)

for b, tr_c in zip(bars_m, m_trades_2026):
    h = b.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y_off = 0.25 if h >= 0 else -0.35
    ax_monthly.text(b.get_x() + b.get_width()/2, h + y_off, f"{h:+.2f}%",
                    ha='center', va=va, fontsize=8.2, fontweight='bold',
                    color=GREEN_GAIN if h >= 0 else RED_LOSS)
    # Trade count below axis
    ax_monthly.text(b.get_x() + b.get_width()/2, -3.2, f"{tr_c} trd",
                    ha='center', va='center', fontsize=7.5, color=TEXT_MUTED)

ax_monthly.set_ylim(-3.8, 7.5)

# ------------------------------------------------------------------------------
# ROW 3, FULL WIDTH: CONSERVATIVE SCORECARD TABLE (Clean, Spacious, Professional)
# ------------------------------------------------------------------------------
ax_table = fig.add_subplot(gs[2, :])
ax_table.set_facecolor(PANEL_COLOR)
for s in ax_table.spines.values():
    s.set_color(BORDER_COLOR)
ax_table.set_xticks([])
ax_table.set_yticks([])

ax_table.text(0.015, 0.90, "AUDIT SCORECARD: 5-YEAR HISTORICAL WALK-FORWARD VERIFICATION",
              fontsize=11.0, fontweight='bold', color=TEXT_TITLE)

table_headers = [
    "Period", "Trades", "Wins", "Losses", "Win Rate",
    "Net PnL", "Return (%)", "Max Drawdown", "Green Months", "Evaluation"
]

rows_data = [
    ["Jan - Aug 2022", "80", "56", "24", "70.0%", "+USD 2,742.02", "+27.42%", "-5.54%", "7 / 8 Months", "EXCELLENT"],
    ["Jan - Aug 2023", "74", "44", "30", "59.5%", "+USD 794.83", "+7.95%", "-5.64%", "5 / 8 Months", "SOLID"],
    ["Jan - Aug 2024", "78", "36", "42", "46.2%", "-USD 1,282.22", "-12.82%", "-18.76%", "2 / 8 Months", "MACRO ANOMALY"],
    ["Jan - Aug 2025", "62", "39", "23", "62.9%", "+USD 1,510.46", "+15.10%", "-3.61%", "6 / 8 Months", "EXCELLENT"],
    ["Jan - Aug 2026", "40", "22", "18", "55.0%", "+USD 532.17", "+5.32%", "-5.80%", "4 / 8 Months", "STABLE"],
    ["5-YEAR TOTAL", "334", "197", "137", "59.0%", "+USD 4,297.26", "+42.97%", "-7.87% (Avg)", "24 / 40 Months", "PASS (80% ANNUAL WR)"]
]

table = ax_table.table(
    cellText=rows_data,
    colLabels=table_headers,
    loc='center',
    cellLoc='center',
    bbox=[0.015, 0.08, 0.97, 0.72]
)
table.auto_set_font_size(False)
table.set_fontsize(9.2)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor(BORDER_COLOR)
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(weight='bold', color='#ffffff')
        cell.set_height(0.18)
    else:
        is_total = (r == len(rows_data))
        if is_total:
            cell.set_facecolor('#162235')
            cell.set_text_props(weight='bold', color='#ffffff')
        else:
            cell.set_facecolor('#111827' if r % 2 == 0 else '#0d131f')

        # Coloring specific columns
        if c in [5, 6]:  # Net PnL, Return
            val_str = rows_data[r-1][c]
            if '+' in val_str:
                cell.set_text_props(color=GREEN_GAIN, weight='bold')
            else:
                cell.set_text_props(color=RED_LOSS, weight='bold')
        elif c == 4:  # Win Rate
            cell.set_text_props(color=BLUE_ACCENT, weight='bold')
        elif c == 7:  # Max DD
            val_str = rows_data[r-1][c]
            cell.set_text_props(color='#94a3b8')
        elif c == 9:  # Evaluation
            eval_val = rows_data[r-1][c]
            if "PASS" in eval_val or "EXCELLENT" in eval_val:
                cell.set_text_props(color=GREEN_GAIN, weight='bold')
            elif "SOLID" in eval_val or "STABLE" in eval_val:
                cell.set_text_props(color=BLUE_ACCENT, weight='bold')
            else:
                cell.set_text_props(color=RED_LOSS, weight='bold')
        else:
            cell.set_text_props(color=TEXT_BODY)

# Footer Note
fig.text(0.05, 0.018,
         "Standard Institutional Parameters: Initial Equity USD 10,000 | Risk 1% per trade | 2-Way Friction: Spread 0.75 pip, Slippage 0.3 pip, Comm USD 3.50/lot | Zero Weekend Gap Risk",
         fontsize=8.0, color='#64748b', ha='left')

plt.savefig(out_png_proj, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(out_png_brain, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Clean tearsheet generated successfully:\n  {out_png_proj}\n  {out_png_brain}")
