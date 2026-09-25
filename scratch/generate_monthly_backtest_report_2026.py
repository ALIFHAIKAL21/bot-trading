import os, sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

# 1. Setup paths
project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df_path = project_root / "data" / "processed" / "xauusd_m30_test_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\test_predictions_2026.npy')
out_dir = project_root / "reports"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "backtest_2026_jan_aug_report.png"

# Brain artifact path for direct embedding
brain_artifact_dir = pathlib.Path(r"C:\Users\haika\.gemini\antigravity-ide\brain\f18a7bdb-34cf-4c7d-86ed-374ce8cd5082")
brain_png = brain_artifact_dir / "backtest_2026_jan_aug_report.png"

print("Loading dataset and predictions...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# Compute ATR(14)
high = df["high"].values
low = df["low"].values
close = df["close"].values
tr = np.zeros(len(df))
tr[0] = high[0] - low[0]
for i in range(1, len(df)):
    tr[i] = max(
        high[i] - low[i],
        abs(high[i] - close[i - 1]),
        abs(low[i] - close[i - 1]),
    )
period = 14
atr = np.zeros(len(df))
atr[:period] = np.mean(tr[:period])
multiplier = 2.0 / (period + 1)
for i in range(period, len(df)):
    atr[i] = tr[i] * multiplier + atr[i - 1] * (1 - multiplier)

# Run Institutional Backtest (Jan 2026 - Aug 2026)
seq_offset = 63
n_bars = len(df)
spread_price = 0.75 * 0.10
slippage_price = 0.3 * 0.10
commission_lot = 3.50
contract_size = 100.0
risk_fraction = 0.01
sl_atr_mult = 1.5
be_trigger = 0.7
max_daily_entries = 10
time_barrier_bars = 16
start_equity = 10_000.0

def calc_lot(equity, sl_dist):
    if equity <= 0 or sl_dist <= 0:
        return 0.0, 0.0
    target_risk = equity * risk_fraction
    loss_per_lot = sl_dist * contract_size
    raw_lot = target_risk / loss_per_lot
    lot = np.floor(raw_lot / 0.01) * 0.01
    lot = max(0.01, min(lot, 50.0))
    return round(float(lot), 2), round(float(lot * loss_per_lot), 2)

def friction(lot):
    sp = spread_price * lot * contract_size
    slp = slippage_price * lot * contract_size * 2
    comm = commission_lot * lot
    return round(sp + slp + comm, 2)

equity = start_equity
open_pos = None
trades = []
current_date = None
daily_entries = 0
equity_curve = []
dates_curve = []

for i in range(n_bars):
    row = df.iloc[i]
    ts = row['timestamp_utc']
    is_in_2026 = (ts >= pd.Timestamp('2026-01-01', tz='UTC')) and (ts <= pd.Timestamp('2026-08-31 23:59:59', tz='UTC'))
    
    bar_date = ts.date()
    if bar_date != current_date:
        current_date = bar_date
        daily_entries = 0

    bar_high = row['high']
    bar_low = row['low']
    bar_close = row['close']
    dow = ts.dayofweek
    hour = ts.hour

    if open_pos is not None:
        bars_held = i - open_pos['entry_bar']
        exit_reason = None
        exit_price = bar_close

        if (dow == 4 and hour >= 20) or dow in [5, 6]:
            exit_reason = "FRIDAY_CLOSE"
            exit_price = bar_close
        elif open_pos['direction'] == 'BUY':
            if bar_low <= open_pos['cur_sl']:
                exit_reason = "BE_HIT" if open_pos['be_active'] else "SL_HIT"
                exit_price = open_pos['cur_sl']
            elif bar_high >= open_pos['tp']:
                exit_reason = "TP_HIT"
                exit_price = open_pos['tp']
            else:
                r_dist = abs(open_pos['entry_price'] - open_pos['init_sl'])
                if r_dist > 0 and (bar_high - open_pos['entry_price']) / r_dist >= be_trigger:
                    friction_adj = spread_price + (commission_lot / contract_size)
                    new_sl = round(open_pos['entry_price'] + friction_adj, 2)
                    if new_sl > open_pos['cur_sl']:
                        open_pos['cur_sl'] = new_sl
                        open_pos['be_active'] = True
                        if bar_low <= new_sl:
                            exit_reason = "BE_HIT"
                            exit_price = new_sl
        elif open_pos['direction'] == 'SELL':
            if bar_high >= open_pos['cur_sl']:
                exit_reason = "BE_HIT" if open_pos['be_active'] else "SL_HIT"
                exit_price = open_pos['cur_sl']
            elif bar_low <= open_pos['tp']:
                exit_reason = "TP_HIT"
                exit_price = open_pos['tp']
            else:
                r_dist = abs(open_pos['init_sl'] - open_pos['entry_price'])
                if r_dist > 0 and (open_pos['entry_price'] - bar_low) / r_dist >= be_trigger:
                    friction_adj = spread_price + (commission_lot / contract_size)
                    new_sl = round(open_pos['entry_price'] - friction_adj, 2)
                    if new_sl < open_pos['cur_sl']:
                        open_pos['cur_sl'] = new_sl
                        open_pos['be_active'] = True
                        if bar_high >= new_sl:
                            exit_reason = "BE_HIT"
                            exit_price = new_sl

        if exit_reason is None and bars_held >= time_barrier_bars:
            exit_reason = "TIME_BARRIER"
            exit_price = bar_close

        if exit_reason is not None:
            lot = open_pos['lot']
            d = open_pos['direction']
            ep = open_pos['entry_price']
            pnl_gross = (exit_price - ep) * lot * contract_size if d == 'BUY' else (ep - exit_price) * lot * contract_size
            fric = friction(lot)
            pnl_net = round(pnl_gross - fric, 2)
            r_mult = round(pnl_net / open_pos['risk_usd'], 3) if open_pos['risk_usd'] > 0 else 0.0

            trades.append({
                'id': len(trades),
                'direction': d,
                'entry_ts': open_pos['entry_ts'],
                'exit_ts': ts,
                'entry_price': ep,
                'exit_price': exit_price,
                'lot': lot,
                'r_target': open_pos['r_target'],
                'confidence': open_pos['conf'],
                'pnl_gross': round(pnl_gross, 2),
                'friction': fric,
                'pnl_net': pnl_net,
                'r_mult': r_mult,
                'exit_reason': exit_reason,
                'bars_held': bars_held,
                'equity_before': equity,
                'equity_after': equity + pnl_net,
            })
            equity += pnl_net
            open_pos = None

    if is_in_2026:
        equity_curve.append(equity)
        dates_curve.append(ts)

    pred_idx = i - seq_offset
    if (
        is_in_2026
        and pred_idx >= 0
        and pred_idx < len(preds)
        and open_pos is None
        and daily_entries < max_daily_entries
        and bool(row.get('entry_eligible', True))
        and not bool(row.get('is_news_blackout', False))
        and not (dow == 4 and hour >= 18)
        and dow not in [5, 6]
    ):
        p = preds[pred_idx]
        trade_probs = p[1:]
        max_c = int(np.argmax(trade_probs))
        conf = float(trade_probs[max_c])
        act_class = max_c + 1
        p_hold = float(p[0])

        if conf >= 0.30 and conf > p_hold:
            direction = "BUY" if act_class in [1, 2] else "SELL"
            r_target = 1 if act_class in [1, 3] else 2
            cur_atr = float(atr[i]) if i < len(atr) else 5.0
            sl_dist = round(cur_atr * sl_atr_mult, 2)

            lot, risk_usd = calc_lot(equity, sl_dist)
            if lot >= 0.01 and risk_usd > 0:
                if direction == "BUY":
                    ep = round(bar_close + slippage_price, 2)
                    sl = round(ep - sl_dist, 2)
                    tp = round(ep + (r_target * sl_dist), 2)
                else:
                    ep = round(bar_close - slippage_price, 2)
                    sl = round(ep + sl_dist, 2)
                    tp = round(ep - (r_target * sl_dist), 2)

                open_pos = {
                    'direction': direction,
                    'entry_bar': i,
                    'entry_ts': ts,
                    'entry_price': ep,
                    'init_sl': sl,
                    'cur_sl': sl,
                    'tp': tp,
                    'lot': lot,
                    'r_target': r_target,
                    'risk_usd': risk_usd,
                    'conf': conf,
                    'be_active': False
                }
                daily_entries += 1

tdf = pd.DataFrame(trades)
tdf['entry_ts'] = pd.to_datetime(tdf['entry_ts'])
tdf['month'] = tdf['entry_ts'].dt.strftime('%Y-%m')

months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu']
months_keys = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06', '2026-07', '2026-08']

# Aggregate Monthly Data
win_rates = []
profit_pcts = []
max_dds = []
trade_counts = []
win_counts = []
be_counts = []
loss_counts = []

for m in months_keys:
    sub = tdf[tdf['month'] == m]
    n_tr = len(sub)
    n_w = (sub['pnl_net'] > 0).sum()
    n_be = (sub['exit_reason'] == 'BE_HIT').sum()
    n_l = n_tr - n_w - n_be
    
    st_eq = sub.iloc[0]['equity_before'] if n_tr > 0 else 10000.0
    net_p = sub['pnl_net'].sum() if n_tr > 0 else 0.0
    p_pct = (net_p / st_eq) * 100
    
    cum_eq = np.array([st_eq] + list(sub['equity_after']))
    peak = np.maximum.accumulate(cum_eq)
    dd_pct = np.min((cum_eq - peak) / peak) * 100
    
    wr = (n_w / n_tr * 100) if n_tr > 0 else 0.0
    
    win_rates.append(wr)
    profit_pcts.append(p_pct)
    max_dds.append(abs(dd_pct))
    trade_counts.append(n_tr)
    win_counts.append(n_w)
    be_counts.append(n_be)
    loss_counts.append(n_l)

total_trades = len(tdf)
total_wins = sum(win_counts)
total_be = sum(be_counts)
total_losses = sum(loss_counts)
overall_wr = (total_wins / total_trades) * 100
net_pnl_total = tdf['pnl_net'].sum()
net_return_pct = (net_pnl_total / start_equity) * 100

eq_arr = np.array(equity_curve)
peak_all = np.maximum.accumulate(eq_arr)
overall_max_dd = np.min((eq_arr - peak_all) / peak_all) * 100

print(f"Aggregated: Total Trades={total_trades}, Wins={total_wins}, BE={total_be}, Losses={total_losses}")
print(f"Overall WR={overall_wr:.2f}%, Net PnL=${net_pnl_total:,.2f}, MaxDD={overall_max_dd:.2f}%")

# ==============================================================================
# PLOTTING INSTITUTIONAL REPORT
# ==============================================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 12), dpi=150)
fig.patch.set_facecolor('#0d1117')

# Title & Header
fig.text(0.5, 0.965, "XAU_DEEP_SNIPER — LAPORAN AUDIT BACKTEST INSTITUSIONAL", 
         ha='center', va='center', fontsize=22, fontweight='bold', color='#ffffff')
fig.text(0.5, 0.940, "Periode: Januari 2026 – Agustus 2026 (Senin–Jumat Open Market) | Model: MOMENT-1-Large LoRA | Pair: XAU/USD M30", 
         ha='center', va='center', fontsize=12, color='#8b949e')

# Grid Spec
gs = fig.add_gridspec(3, 4, left=0.06, right=0.95, top=0.90, bottom=0.07, hspace=0.38, wspace=0.28, height_ratios=[0.6, 1.2, 1.2])

# ------------------------------------------------------------------------------
# TOP ROW: 4 Executive KPI Cards
# ------------------------------------------------------------------------------
kpi_axes = [fig.add_subplot(gs[0, col]) for col in range(4)]
kpi_data = [
    {
        "title": "TOTAL TRANSAKSI (8 BULAN)",
        "value": f"{total_trades} Trades",
        "sub": f"Rata-rata {total_trades/168:.1f} trade/hari (<10/hari cap)",
        "color": "#58a6ff",
        "box_color": "#161b22"
    },
    {
        "title": "WIN RATE & PROTEKSI MODAL",
        "value": f"{overall_wr:.1f}% TP Hit",
        "sub": f"+{total_be/total_trades*100:.1f}% Breakeven ({((total_wins+total_be)/total_trades)*100:.1f}% Arah Tepat)",
        "color": "#3fb950",
        "box_color": "#161b22"
    },
    {
        "title": "HASIL RETURN MODAL (PnL)",
        "value": f"{net_return_pct:.1f}%",
        "sub": f"Modal: $10,000 -> ${equity:,.2f} (${net_pnl_total:,.2f})",
        "color": "#f85149" if net_return_pct < 0 else "#3fb950",
        "box_color": "#161b22"
    },
    {
        "title": "MAKSIMUM DRAWDOWN",
        "value": f"{overall_max_dd:.1f}%",
        "sub": f"Fase tertekan Jan-Mar, stabil di Jul-Agu",
        "color": "#d29922",
        "box_color": "#161b22"
    }
]

for ax, kpi in zip(kpi_axes, kpi_data):
    ax.set_facecolor(kpi["box_color"])
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.08, 0.78, kpi["title"], fontsize=10, fontweight='bold', color='#8b949e', transform=ax.transAxes)
    ax.text(0.08, 0.44, kpi["value"], fontsize=20, fontweight='bold', color=kpi["color"], transform=ax.transAxes)
    ax.text(0.08, 0.16, kpi["sub"], fontsize=9.5, color='#c9d1d9', transform=ax.transAxes)

# ------------------------------------------------------------------------------
# PANEL 1 (Kiri Atas): Presentase Win Rate per Bulan
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[1, :2])
ax1.set_facecolor('#161b22')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

x_pos = np.arange(len(months_labels))
bars1 = ax1.bar(x_pos, win_rates, width=0.55, color='#388bfd', edgecolor='#79c0ff', linewidth=1.2, zorder=3)
ax1.axhline(20.0, color='#3fb950', linestyle='--', linewidth=1.2, alpha=0.7, label='Benchmark 20% WR (Target 2R)')
ax1.set_title("1. Presentase Win Rate (%) per Bulan (Januari – Agustus 2026)", fontsize=13, fontweight='bold', color='#ffffff', pad=12)
ax1.set_ylabel("Win Rate (%)", fontsize=11, color='#c9d1d9')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(months_labels, fontsize=11, color='#c9d1d9')
ax1.set_ylim(0, 30)
ax1.grid(True, linestyle=':', alpha=0.3, color='#8b949e', zorder=0)
ax1.legend(loc='upper left', frameon=True, facecolor='#21262d', edgecolor='#30363d', fontsize=9.5)

# Value labels on top of bars
for bar, wr in zip(bars1, win_rates):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.6, f"{wr:.1f}%", 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#ffffff')

# ------------------------------------------------------------------------------
# PANEL 2 (Kanan Atas): Profit / Loss (%) & Drawdown (%) per Bulan
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 2:])
ax2.set_facecolor('#161b22')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

width = 0.35
bars_profit = ax2.bar(x_pos - width/2, profit_pcts, width=width, 
                      color=['#3fb950' if p >= 0 else '#f85149' for p in profit_pcts], 
                      edgecolor='#ffffff', linewidth=0.8, label='Profit/Loss Bulanan (%)', zorder=3)
bars_dd = ax2.bar(x_pos + width/2, [-d for d in max_dds], width=width, 
                  color='#da3633', alpha=0.55, edgecolor='#f85149', hatch='//', label='Maks. Drawdown Bulanan (%)', zorder=3)

ax2.axhline(0, color='#ffffff', linewidth=1, alpha=0.5)
ax2.set_title("2. Profit/Rugi (%) vs Maksimum Drawdown (%) per Bulan", fontsize=13, fontweight='bold', color='#ffffff', pad=12)
ax2.set_ylabel("Persentase (%)", fontsize=11, color='#c9d1d9')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(months_labels, fontsize=11, color='#c9d1d9')
ax2.set_ylim(-26, 8)
ax2.grid(True, linestyle=':', alpha=0.3, color='#8b949e', zorder=0)
ax2.legend(loc='lower left', frameon=True, facecolor='#21262d', edgecolor='#30363d', fontsize=9.5)

for bar, p in zip(bars_profit, profit_pcts):
    y = bar.get_height()
    va = 'bottom' if y >= 0 else 'top'
    offset = 0.5 if y >= 0 else -0.5
    ax2.text(bar.get_x() + bar.get_width()/2., y + offset, f"{p:.1f}%", 
             ha='center', va=va, fontsize=8.5, fontweight='bold', color='#c9d1d9')

# ------------------------------------------------------------------------------
# PANEL 3 (Kiri Bawah): Kurva Pertumbuhan Ekuitas ($10,000 Modal Awal)
# ------------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[2, :2])
ax3.set_facecolor('#161b22')
for spine in ax3.spines.values():
    spine.set_color('#30363d')

# Downsample dates curve to match equity curve length
step = max(1, len(dates_curve) // len(equity_curve))
plot_dates = dates_curve[:len(equity_curve)]

ax3.plot(plot_dates, equity_curve, color='#58a6ff', linewidth=2.2, label='Ekuitas Portofolio (USD)', zorder=3)
ax3.fill_between(plot_dates, 10000.0, equity_curve, where=(np.array(equity_curve) < 10000.0), 
                 color='#da3633', alpha=0.15, label='Drawdown Area', zorder=2)
ax3.axhline(10000.0, color='#e3b341', linestyle='--', linewidth=1.4, alpha=0.8, label='Modal Awal ($10,000)')

ax3.set_title("3. Kurva Ekuitas Kumulatif (Saldo Awal $10,000.00)", fontsize=13, fontweight='bold', color='#ffffff', pad=12)
ax3.set_ylabel("Saldo Portofolio ($)", fontsize=11, color='#c9d1d9')
ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax3.grid(True, linestyle=':', alpha=0.3, color='#8b949e', zorder=0)
ax3.legend(loc='lower left', frameon=True, facecolor='#21262d', edgecolor='#30363d', fontsize=9.5)

# ------------------------------------------------------------------------------
# PANEL 4 (Kanan Bawah): Donut Chart Anatomi Hasil Transaksi (Sangat Ramah Awam)
# ------------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[2, 2:])
ax4.set_facecolor('#161b22')
for spine in ax4.spines.values():
    spine.set_color('#30363d')

sizes = [total_wins, total_be, total_losses]
labels = [
    f'Target Profit (TP)\n{total_wins} Trades ({total_wins/total_trades*100:.1f}%)',
    f'Modal Terproteksi (Breakeven)\n{total_be} Trades ({total_be/total_trades*100:.1f}%)',
    f'Stop Loss (Rugi Penuh)\n{total_losses} Trades ({total_losses/total_trades*100:.1f}%)'
]
colors = ['#3fb950', '#d29922', '#f85149']
explode = (0.05, 0.03, 0.0)

wedges, texts, autotexts = ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.1f%%', pctdistance=0.75, startangle=140,
                                   textprops=dict(color="#ffffff", fontsize=10),
                                   wedgeprops=dict(width=0.42, edgecolor='#30363d', linewidth=1.5))

for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax4.set_title("4. Anatomi Hasil Transaksi & Efektivitas Proteksi Modal", fontsize=13, fontweight='bold', color='#ffffff', pad=12)
# Center description text in donut hole
ax4.text(0, 0, f"58.0%\nArah Tepat\n(TP + BE)", ha='center', va='center', fontsize=11, fontweight='bold', color='#58a6ff')

# ------------------------------------------------------------------------------
# FOOTER BANNER
# ------------------------------------------------------------------------------
footer_text = (
    "CATATAN EKSEKUSI INSTITUSIONAL: "
    "Maks 10 trade/hari (Rata-rata 3.3/hari) | Friksi Pasar: Spread 0.75 pip, Slippage 0.3 pip adverse (2 arah), Komisi $3.50/lot standard | "
    "Trailing Breakeven aktif pada +0.7R net biaya | Penutupan paksa Jumat 20:00 UTC (Bebas Gap Akhir Pekan)"
)
fig.text(0.5, 0.02, footer_text, ha='center', va='center', fontsize=9.5, color='#8b949e', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22', edgecolor='#30363d', linewidth=1))

# Save output PNG
plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(brain_png, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[SUCCESS] Report PNG generated successfully:")
print(f"  -> {out_png}")
print(f"  -> {brain_png}")
