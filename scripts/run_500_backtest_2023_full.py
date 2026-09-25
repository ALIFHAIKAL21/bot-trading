"""
Production Institutional Backtest: FULL YEAR 2023 (All 12 Months)
100% EXACT LOCKED PRODUCTION CONFIGURATION (log.md Contract)
Modal: $500.00 USD, Lot 0.01 Flat, 4-Stage Smart Protection
"""

import sys, os, pathlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
reports_dir = project_root / "reports"
reports_img_dir = reports_dir / "img"
reports_dir.mkdir(parents=True, exist_ok=True)
reports_img_dir.mkdir(parents=True, exist_ok=True)

brain_artifact_dir = pathlib.Path(r"C:\Users\haika\.gemini\antigravity-ide\brain\f18a7bdb-34cf-4c7d-86ed-374ce8cd5082")

df_path = project_root / "data" / "processed" / "xauusd_m30_labeled_15ch.parquet"
preds_path = project_root / "checkpoints" / "predictions_15ch.npy"

print("Loading 15-channel dataset and model predictions...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# Compute ATR(14)
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

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
ts_arr = df['timestamp_utc'].values.astype('datetime64[s]')
dt_series = df['timestamp_utc'].dt
dows = dt_series.dayofweek.values
hours = dt_series.hour.values
minutes = dt_series.minute.values
day_ints = dt_series.strftime('%Y%m%d').astype(int).values

# 100% LOCKED PARAMETERS (FROM LOG.MD)
INITIAL_EQUITY = 500.0
FIXED_LOT = 0.01
SPREAD_PRICE = 0.75 * 0.10
SLIPPAGE_PRICE = 0.3 * 0.10
COMMISSION_PER_LOT = 3.50
CONTRACT_SIZE = 100.0
friction_cost = round(SPREAD_PRICE * FIXED_LOT * CONTRACT_SIZE + SLIPPAGE_PRICE * FIXED_LOT * CONTRACT_SIZE * 2 + COMMISSION_PER_LOT * FIXED_LOT, 3)

SL_ATR_MULT = 1.5
TP_MAX_R = 2.7
BE_TRIGGER_R = 1.0
BE_BUFFER_PRICE = 0.25
RATCHET_12_R = 0.5
TRAIL_TRIGGER_R = 1.5
TRAIL_DIST_R = 0.6
STALE_DECAY_BARS = 10
STALE_DECAY_R = 0.6
TIME_BARRIER_BARS = 12
TAU_BASE = 0.32
MARGIN_MIN = 0.01
seq_offset = 63

# TARGET: FULL YEAR 2023 (Jan 1, 2023 to Dec 31, 2023)
st_dt = np.datetime64('2023-01-01T00:00:00')
en_dt = np.datetime64('2023-12-31T23:59:59')
mask_2023 = (ts_arr >= st_dt) & (ts_arr <= en_dt)
indices_2023 = np.where(mask_2023)[0]
unique_days = np.unique(day_ints[indices_2023])

day_to_indices = {}
for idx in indices_2023:
    d = day_ints[idx]
    if d not in day_to_indices:
        day_to_indices[d] = []
    day_to_indices[d].append(idx)

# Session definitions (Same 5 windows)
session_defs = [
    (1*60, 4*60, "Asia Early (01:00-04:00)"),
    (4*60 + 30, 7*60, "Asia Late (04:30-07:00)"),
    (8*60 + 30, 12*60 + 30, "London Core (08:30-12:30)"),
    (13*60, 17*60, "NY Open (13:00-17:00)"),
    (17*60 + 30, 21*60, "NY Core (17:30-21:00)")
]

def run_2023_full_backtest():
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    equity_timestamps = [ts_arr[indices_2023[0]]]
    trades = []
    
    for d_int in unique_days:
        day_idxs = day_to_indices[d_int]
        dow = dows[day_idxs[0]]
        if dow in [5, 6]:
            continue
            
        for w_st, w_en, s_name in session_defs:
            best_idx = None
            best_conf = -1.0
            best_act = None
            
            for b_idx in day_idxs:
                h = hours[b_idx]
                m = minutes[b_idx]
                if h == 7 or (h == 8 and m < 30):
                    continue
                if dow == 4 and h >= 18:
                    continue
                    
                t_val = h * 60 + m
                if w_st <= t_val <= w_en:
                    p_idx = b_idx - seq_offset
                    if 0 <= p_idx < len(preds):
                        probs = preds[p_idx]
                        p_hold = float(probs[0])
                        t_probs = probs[1:]
                        max_i = int(np.argmax(t_probs))
                        conf = float(t_probs[max_i])
                        margin = conf - p_hold
                        
                        if conf > best_conf and conf >= TAU_BASE and margin >= MARGIN_MIN:
                            best_conf = conf
                            best_idx = b_idx
                            best_act = max_i + 1
                            
            if best_idx is not None:
                d = "BUY" if best_act in [1, 2] else "SELL"
                atr_val = float(atr[best_idx]) if best_idx < len(atr) else 5.0
                sl_dist = round(atr_val * SL_ATR_MULT, 2)
                b_close = closes[best_idx]
                ts = ts_arr[best_idx]
                
                ep = round(b_close + SLIPPAGE_PRICE, 2) if d == "BUY" else round(b_close - SLIPPAGE_PRICE, 2)
                sl_p = round(ep - sl_dist, 2) if d == "BUY" else round(ep + sl_dist, 2)
                tp_max_p = round(ep + (TP_MAX_R * sl_dist), 2) if d == "BUY" else round(ep - (TP_MAX_R * sl_dist), 2)
                be_trigger_p = round(ep + (BE_TRIGGER_R * sl_dist), 2) if d == "BUY" else round(ep - (BE_TRIGGER_R * sl_dist), 2)
                
                cur_sl = sl_p
                be_activated = False
                pnl_net = 0.0
                exit_reason = None
                exit_price = None
                exit_ts = None
                
                for f_idx in range(best_idx + 1, min(best_idx + 16, len(df))):
                    f_high = highs[f_idx]
                    f_low = lows[f_idx]
                    f_close = closes[f_idx]
                    f_dow = dows[f_idx]
                    f_hour = hours[f_idx]
                    f_ts = ts_arr[f_idx]
                    
                    is_fri = (f_dow == 4 and f_hour >= 20) or f_dow in [5, 6]
                    is_tb = (f_idx - best_idx >= TIME_BARRIER_BARS)
                    
                    if is_fri or is_tb:
                        exit_price = f_close
                        exit_reason = "Friday Closeout" if is_fri else f"Time Barrier ({TIME_BARRIER_BARS//2}h)"
                        exit_ts = f_ts
                        pnl_raw = (exit_price - ep) * FIXED_LOT * CONTRACT_SIZE if d == "BUY" else (ep - exit_price) * FIXED_LOT * CONTRACT_SIZE
                        pnl_net = pnl_raw - friction_cost
                        break
                        
                    bars_held = f_idx - best_idx
                    if STALE_DECAY_BARS > 0 and bars_held >= STALE_DECAY_BARS and not be_activated:
                        decay_sl = round(ep - (STALE_DECAY_R * sl_dist), 2) if d == "BUY" else round(ep + (STALE_DECAY_R * sl_dist), 2)
                        if d == "BUY":
                            cur_sl = max(cur_sl, decay_sl)
                        else:
                            cur_sl = min(cur_sl, decay_sl)
                            
                    if d == "BUY":
                        if f_low <= cur_sl:
                            exit_price = cur_sl
                            exit_reason = "Protected Stop" if be_activated else "Stop Loss"
                            exit_ts = f_ts
                            pnl_raw = (cur_sl - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_high >= tp_max_p:
                            exit_price = tp_max_p
                            exit_reason = f"Max TP (+{TP_MAX_R}R)"
                            exit_ts = f_ts
                            pnl_raw = (tp_max_p - ep) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        else:
                            if not be_activated and f_high >= be_trigger_p:
                                be_activated = True
                                cur_sl = max(cur_sl, round(ep + BE_BUFFER_PRICE, 2))
                                
                            r_gain = (f_high - ep) / sl_dist
                            if RATCHET_12_R > 0 and r_gain >= 1.2:
                                r_sl = round(ep + (RATCHET_12_R * sl_dist), 2)
                                cur_sl = max(cur_sl, r_sl)
                                
                            if r_gain >= TRAIL_TRIGGER_R:
                                trail_sl = round(f_high - (TRAIL_DIST_R * sl_dist), 2)
                                cur_sl = max(cur_sl, trail_sl)
                    else: # SELL
                        if f_high >= cur_sl:
                            exit_price = cur_sl
                            exit_reason = "Protected Stop" if be_activated else "Stop Loss"
                            exit_ts = f_ts
                            pnl_raw = (ep - cur_sl) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        elif f_low <= tp_max_p:
                            exit_price = tp_max_p
                            exit_reason = f"Max TP (+{TP_MAX_R}R)"
                            exit_ts = f_ts
                            pnl_raw = (ep - tp_max_p) * FIXED_LOT * CONTRACT_SIZE
                            pnl_net = pnl_raw - friction_cost
                            break
                        else:
                            if not be_activated and f_low <= be_trigger_p:
                                be_activated = True
                                cur_sl = min(cur_sl, round(ep - BE_BUFFER_PRICE, 2))
                                
                            r_gain = (ep - f_low) / sl_dist
                            if RATCHET_12_R > 0 and r_gain >= 1.2:
                                r_sl = round(ep - (RATCHET_12_R * sl_dist), 2)
                                cur_sl = min(cur_sl, r_sl)
                                
                            if r_gain >= TRAIL_TRIGGER_R:
                                trail_sl = round(f_low + (TRAIL_DIST_R * sl_dist), 2)
                                cur_sl = min(cur_sl, trail_sl)
                
                tot = round(pnl_net, 2)
                equity += tot
                trades.append({
                    'time': str(ts),
                    'month': str(ts)[:7],
                    'dir': d,
                    'win': 1 if tot > 0 else 0,
                    'pnl': tot,
                    'lot': FIXED_LOT,
                    'conf': best_conf,
                    'session': s_name,
                    'exit_reason': exit_reason or 'Time Barrier',
                    'sl_dist': sl_dist
                })
                equity_curve.append(equity)
                equity_timestamps.append(ts)
                
    tdf = pd.DataFrame(trades)
    return tdf, equity_curve, equity_timestamps

print("Executing Locked Production Backtest for Full Year 2023...")
tdf, eq_curve, eq_ts = run_2023_full_backtest()

# Metrics Calculation
n_trades = len(tdf)
wins = tdf[tdf['win'] == 1]
losses = tdf[tdf['win'] == 0]
win_rate = len(wins) / n_trades * 100
gross_profit = wins['pnl'].sum()
gross_loss = abs(losses['pnl'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
net_pnl = eq_curve[-1] - INITIAL_EQUITY
return_pct = (eq_curve[-1] / INITIAL_EQUITY - 1) * 100

eq_arr = np.array(eq_curve)
peaks = np.maximum.accumulate(eq_arr)
dds = (peaks - eq_arr) / peaks * 100
max_dd = np.max(dds)
min_equity = np.min(eq_arr)

tdf['date'] = pd.to_datetime(tdf['time']).dt.date
total_trading_days = len(unique_days)
active_days = tdf['date'].nunique()
trades_per_day = n_trades / total_trading_days
cadence_dist = tdf.groupby('date').size()
days_4_5 = cadence_dist[cadence_dist.isin([4, 5])].count()
pct_4_5 = days_4_5 / active_days * 100

# Monthly metrics for all 12 months of 2023
months_2023 = [f"2023-{m:02d}" for m in range(1, 13)]
month_names_2023 = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']

monthly_summary = []
running_bal = INITIAL_EQUITY
for m_str in months_2023:
    grp = tdf[tdf['month'] == m_str]
    n_m_tr = len(grp)
    n_m_win = grp['win'].sum()
    m_wr = (n_m_win / n_m_tr * 100) if n_m_tr > 0 else 0
    m_pnl = grp['pnl'].sum()
    running_bal += m_pnl
    monthly_summary.append({
        'month': m_str,
        'trades': n_m_tr,
        'wins': int(n_m_win),
        'win_rate': round(m_wr, 1),
        'pnl': round(m_pnl, 2),
        'balance': round(running_bal, 2)
    })

print(f"\n=== FULL AUDIT RESULTS: YEAR 2023 ===")
print(f"Initial Equity: ${INITIAL_EQUITY:.2f}")
print(f"Final Balance: ${eq_curve[-1]:.2f}")
print(f"Net Profit: +${net_pnl:.2f} (+{return_pct:.2f}%)")
print(f"Total Trades: {n_trades} (Wins: {len(wins)}, Losses: {len(losses)})")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: {max_dd:.2f}%")
print(f"Lowest Equity Dip: ${min_equity:.2f}")
print(f"Avg Trades/Day: {trades_per_day:.2f} (Cadence 4-5 entries/day: {pct_4_5:.1f}%)")

print("\n--- 2023 Monthly Table ---")
mdf = pd.DataFrame(monthly_summary)
print(mdf.to_string(index=False))

# Save results JSON
out_json_path = reports_dir / "backtest_500_modal_2023_full.json"
results_dict = {
    'target_year': 2023,
    'initial_capital': INITIAL_EQUITY,
    'lot_size': FIXED_LOT,
    'final_equity': round(eq_curve[-1], 2),
    'net_profit': round(net_pnl, 2),
    'return_pct': round(return_pct, 2),
    'total_trades': n_trades,
    'wins': len(wins),
    'losses': len(losses),
    'win_rate_pct': round(win_rate, 2),
    'profit_factor': round(profit_factor, 2),
    'max_drawdown_pct': round(max_dd, 2),
    'min_equity': round(min_equity, 2),
    'avg_trades_per_day': round(trades_per_day, 2),
    'pct_days_4_to_5_trades': round(pct_4_5, 1),
    'monthly_summary': monthly_summary
}
with open(out_json_path, 'w') as f:
    json.dump(results_dict, f, indent=4)
print(f"\nSaved report JSON to: {out_json_path}")

# ================= PLOTTING HIGH-END TEARSHEET FOR 2023 =================
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 12), dpi=200)
gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 1.3, 1.1], hspace=0.38, wspace=0.25)
fig.patch.set_facecolor('#07090e')

c_blue = '#2979ff'
c_teal = '#00e676'
c_gold = '#ffd700'
c_red = '#ff1744'
c_gray = '#78909c'
c_card = '#10141d'

# 1. Equity Curve (gs[0, :2])
ax_eq = fig.add_subplot(gs[0, :2])
ax_eq.set_facecolor(c_card)
ax_eq.grid(True, linestyle='--', alpha=0.15, color='#ffffff')

eq_dates = pd.to_datetime(eq_ts)
ax_eq.plot(eq_dates, eq_curve, color=c_teal, linewidth=2.2, label=f'2023 Locked Production Equity (Final: ${eq_curve[-1]:,.2f})')
ax_eq.axhline(INITIAL_EQUITY, color='#ef5350', linestyle=':', alpha=0.7, label=f'Initial Modal ($500.00)')
ax_eq.fill_between(eq_dates, INITIAL_EQUITY, eq_curve, where=(np.array(eq_curve) >= INITIAL_EQUITY), color=c_teal, alpha=0.12)

ax_eq.text(0.03, 0.78, f"Initial: ${INITIAL_EQUITY:,.2f} | Final: ${eq_curve[-1]:,.2f}\nNet Profit: +${net_pnl:,.2f} (+{return_pct:,.1f}%)\nFixed 0.01 Lot Flat | Full Year 2023", 
           transform=ax_eq.transAxes, fontsize=10.5, fontweight='bold', color='#ffffff',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#161c28', edgecolor=c_teal, alpha=0.9))

ax_eq.set_title("EQUITY GROWTH CURVE: $500 CAPITAL (FIXED 0.01 LOT) — FULL YEAR 2023", fontsize=13, fontweight='heavy', color='#ffffff', pad=12)
ax_eq.set_ylabel("Account Balance ($ USD)", fontsize=10, color=c_gray)
ax_eq.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax_eq.legend(loc='lower right', framealpha=0.8, facecolor='#161c28', edgecolor='none', fontsize=9)

# 2. Executive Metric Cards Panel (gs[0, 2])
ax_card = fig.add_subplot(gs[0, 2])
ax_card.set_facecolor(c_card)
ax_card.axis('off')

card_text = (
    f"2023 FULL YEAR AUDIT SUMMARY\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"• Initial Capital : $500.00 USD\n"
    f"• Fixed Lot Size  : 0.01 Lot Flat (No Esc.)\n"
    f"• Final Balance   : ${eq_curve[-1]:,.2f} USD\n"
    f"• Total Net Profit: +${net_pnl:,.2f} (+{return_pct:,.1f}%)\n"
    f"• Total Trades    : {n_trades} (W:{len(wins)} / L:{len(losses)})\n"
    f"• Overall Win Rate: {win_rate:.2f}%\n"
    f"• Profit Factor   : {profit_factor:.2f}\n"
    f"• Max Drawdown    : {max_dd:.2f}%\n"
    f"• Lowest Dip      : ${min_equity:,.2f}\n"
    f"• Avg Trades/Day  : {trades_per_day:.2f} trades/day\n"
    f"• 4-5 Trades/Day  : {pct_4_5:.1f}% of active days\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"LOCKED PRODUCTION RULES:\n"
    f"1. Tiered Ratchet: Lock +0.5R at +1.2R\n"
    f"2. Dynamic Trailing: 0.6R Trail at 1.5R+\n"
    f"3. Stale Decay: Tighten to -0.6R at 10 bars\n"
    f"4. Max TP: +2.7R kinetic ceiling\n"
    f"5. Frictions: All Included ($0.17/trade)"
)

ax_card.text(0.06, 0.5, card_text, fontsize=9.2, fontfamily='monospace', color='#e0e0e0',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#161c28', edgecolor='#2979ff', alpha=0.9))

# 3. Monthly Net PnL Bar Chart (gs[1, :2])
ax_pnl = fig.add_subplot(gs[1, :2])
ax_pnl.set_facecolor(c_card)
ax_pnl.grid(True, linestyle='--', alpha=0.15, color='#ffffff')

pnl_vals = [m['pnl'] for m in monthly_summary]
bar_colors = [c_teal if p >= 0 else c_red for p in pnl_vals]
bars = ax_pnl.bar(month_names_2023, pnl_vals, color=bar_colors, width=0.55, edgecolor='#ffffff', linewidth=0.5, alpha=0.85)

for bar, val in zip(bars, pnl_vals):
    y_pos = bar.get_height() + 15 if val >= 0 else bar.get_height() - 35
    ax_pnl.text(bar.get_x() + bar.get_width()/2, y_pos, f"${val:+,.0f}", 
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#ffffff')

ax_pnl.set_title("2023 MONTHLY NET PNL ($ USD) — 12 MONTHS PERFORMANCE", fontsize=11, fontweight='bold', color='#ffffff', pad=10)
ax_pnl.set_ylabel("PnL ($ USD)", fontsize=9, color=c_gray)
ax_pnl.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:+,.0f}'))
ax_pnl.axhline(0, color='#ffffff', linestyle='-', linewidth=0.8, alpha=0.5)

# 4. Monthly Win Rate Chart (gs[1, 2])
ax_wr = fig.add_subplot(gs[1, 2])
ax_wr.set_facecolor(c_card)
ax_wr.grid(True, linestyle='--', alpha=0.15, color='#ffffff')

wr_vals = [m['win_rate'] for m in monthly_summary]
ax_wr.plot(month_names_2023, wr_vals, marker='o', markersize=5, color=c_gold, linewidth=1.8, label='Win Rate (%)')
ax_wr.axhline(50.0, color=c_red, linestyle='--', linewidth=1.2, label='50% Threshold')

for i, txt in enumerate(wr_vals):
    ax_wr.annotate(f"{txt:.0f}%", (month_names_2023[i], wr_vals[i] + 1.2), ha='center', fontsize=7.5, color='#ffffff', fontweight='bold')

ax_wr.set_title("2023 MONTHLY WIN RATE (%)", fontsize=11, fontweight='bold', color='#ffffff', pad=10)
ax_wr.set_ylabel("Win Rate (%)", fontsize=9, color=c_gray)
ax_wr.set_ylim(35, 80)
ax_wr.legend(loc='lower right', framealpha=0.8, facecolor='#161c28', edgecolor='none', fontsize=8)

# 5. Drawdown Profile (gs[2, :2])
ax_dd = fig.add_subplot(gs[2, :2])
ax_dd.set_facecolor(c_card)
ax_dd.grid(True, linestyle='--', alpha=0.15, color='#ffffff')

ax_dd.fill_between(eq_dates, 0, -dds, color=c_red, alpha=0.35, label=f'Underwater Drawdown (Max: {max_dd:.2f}%)')
ax_dd.plot(eq_dates, -dds, color=c_red, linewidth=1.0)
ax_dd.set_title(f"PORTFOLIO UNDERWATER DRAWDOWN PROFILE (PEAK-TO-TROUGH)", fontsize=11, fontweight='bold', color='#ffffff', pad=10)
ax_dd.set_ylabel("Drawdown (%)", fontsize=9, color=c_gray)
ax_dd.yaxis.set_major_formatter(ticker.PercentFormatter())
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax_dd.legend(loc='lower right', framealpha=0.8, facecolor='#161c28', edgecolor='none', fontsize=8)

# 6. Trade Cadence Distribution (gs[2, 2])
ax_cad = fig.add_subplot(gs[2, 2])
ax_cad.set_facecolor(c_card)
ax_cad.grid(True, linestyle='--', alpha=0.15, color='#ffffff')

cad_counts = cadence_dist.value_counts().sort_index()
cad_x = [f"{int(k)}" for k in cad_counts.index]
cad_colors = [c_teal if int(k) in [4, 5] else c_blue for k in cad_counts.index]
cad_bars = ax_cad.bar(cad_x, cad_counts.values, color=cad_colors, width=0.55, edgecolor='#ffffff', linewidth=0.5, alpha=0.85)

for bar, val in zip(cad_bars, cad_counts.values):
    pct = val / active_days * 100
    ax_cad.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val}\n({pct:.0f}%)", 
                ha='center', va='bottom', fontsize=8, color='#ffffff')

ax_cad.set_ylim(0, max(cad_counts.values) * 1.18)
ax_cad.set_title(f"DAILY TRADES DISTRIBUTION ({pct_4_5:.1f}% are 4-5/day)", fontsize=11, fontweight='bold', color='#ffffff', pad=10)
ax_cad.set_xlabel("Trades per Day", fontsize=9, color=c_gray)
ax_cad.set_ylabel("Number of Days", fontsize=9, color=c_gray)

out_png_path = reports_img_dir / "backtest_500_modal_2023_full.png"
plt.savefig(out_png_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

# Also save to brain artifact dir
brain_png_path = brain_artifact_dir / "backtest_500_modal_2023_full.png"
import shutil
shutil.copyfile(out_png_path, brain_png_path)

print(f"\nSaved 2023 chart tearsheet to: {out_png_path}")
print(f"Synced to brain artifact: {brain_png_path}")
