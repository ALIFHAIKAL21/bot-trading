import sys, os, pathlib
import numpy as np
import pandas as pd

# Load test data and predictions
project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df = pd.read_parquet(project_root / "data" / "processed" / "xauusd_m30_test_labeled.parquet")
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(r'c:\Ngoding\bot_trading\scratch\test_predictions_2026.npy')

# Compute ATR(14)
def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
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
    atr = np.zeros(len(df))
    atr[:period] = np.mean(tr[:period])
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(df)):
        atr[i] = tr[i] * multiplier + atr[i - 1] * (1 - multiplier)
    return atr

atr = compute_atr(df, 14)

def run_detailed_backtest(df, preds, atr, tau=0.30, max_daily_entries=10, be_trigger=0.7, default_r_target=None):
    seq_offset = 63
    n_bars = len(df)
    
    spread_price = 0.75 * 0.10
    slippage_price = 0.3 * 0.10
    commission_lot = 3.50
    contract_size = 100.0
    risk_fraction = 0.01
    sl_atr_mult = 1.5
    time_barrier_bars = 16

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

    equity = 10_000.0
    open_pos = None
    trades = []
    current_date = None
    daily_entries = 0
    equity_curve = [equity]

    for i in range(n_bars):
        row = df.iloc[i]
        ts = row['timestamp_utc']
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
                    exit_reason = "SL_HIT" if not open_pos['be_active'] else "BE_HIT"
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
                    exit_reason = "SL_HIT" if not open_pos['be_active'] else "BE_HIT"
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
                    'pnl_net': pnl_net,
                    'r_mult': r_mult,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                    'equity_before': equity,
                    'equity_after': equity + pnl_net,
                })
                equity += pnl_net
                open_pos = None

        pred_idx = i - seq_offset
        if (
            pred_idx >= 0
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

            if conf >= tau and conf > p_hold:
                direction = "BUY" if act_class in [1, 2] else "SELL"
                if default_r_target is not None:
                    r_target = default_r_target
                else:
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
        equity_curve.append(equity)

    tdf = pd.DataFrame(trades)
    return tdf, np.array(equity_curve)

# Run strict baseline
tdf, eq_curve = run_detailed_backtest(df, preds, atr, tau=0.30, max_daily_entries=10, be_trigger=0.7)
tdf['entry_ts'] = pd.to_datetime(tdf['entry_ts'])
tdf['month'] = tdf['entry_ts'].dt.strftime('%Y-%m')

# Filter Jan 2026 - Aug 2026
months_2026 = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06', '2026-07', '2026-08']
tdf_2026 = tdf[tdf['month'].isin(months_2026)].copy()

print(f"Total trades Jan-Aug 2026: {len(tdf_2026)}")

# Month-by-month calculation
monthly_stats = []
for m in months_2026:
    sub = tdf_2026[tdf_2026['month'] == m]
    if len(sub) == 0:
        continue
    n_trades = len(sub)
    n_wins = (sub['pnl_net'] > 0).sum()
    n_losses = (sub['pnl_net'] < 0).sum()
    n_be = (sub['exit_reason'] == 'BE_HIT').sum()
    n_tp = (sub['exit_reason'] == 'TP_HIT').sum()
    n_sl = (sub['exit_reason'] == 'SL_HIT').sum()
    
    start_eq = sub.iloc[0]['equity_before']
    end_eq = sub.iloc[-1]['equity_after']
    net_pnl = sub['pnl_net'].sum()
    pnl_pct = (net_pnl / start_eq) * 100
    
    # Calculate drawdown within month
    cum_eq = np.array([start_eq] + list(sub['equity_after']))
    peak = np.maximum.accumulate(cum_eq)
    dd_pct = np.min((cum_eq - peak) / peak) * 100
    
    wr = (n_wins / n_trades) * 100
    gross_win = sub[sub['pnl_net'] > 0]['pnl_net'].sum()
    gross_loss = abs(sub[sub['pnl_net'] < 0]['pnl_net'].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else 99.0
    
    monthly_stats.append({
        'Month': m,
        'Trades': n_trades,
        'Wins': n_wins,
        'Losses': n_losses,
        'Breakevens': n_be,
        'WinRatePct': wr,
        'StartEquity': start_eq,
        'EndEquity': end_eq,
        'NetPnL': net_pnl,
        'ProfitPct': pnl_pct,
        'MaxDrawdownPct': dd_pct,
        'ProfitFactor': pf
    })

mdf = pd.DataFrame(monthly_stats)
print(mdf.to_string(index=False))
