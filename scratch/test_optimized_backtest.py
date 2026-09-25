import os, sys, pathlib
import numpy as np
import pandas as pd

# Load test data and predictions
project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df = pd.read_parquet(project_root / "data" / "processed" / "xauusd_m30_test_labeled.parquet")
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(r'c:\Ngoding\bot_trading\scratch\test_predictions_2026.npy')

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

def run_partial_tp_backtest(
    df, preds, atr,
    tau=0.30,
    max_daily_entries=10,
    be_trigger=1.0,  # Move to BE only after reaching +1.0R
    trail_after_1r=True,
    start_equity=10_000.0,
    risk_fraction=0.01,
    sl_atr_mult=1.5
):
    seq_offset = 63
    n_bars = len(df)
    
    spread_price = 0.75 * 0.10
    slippage_price = 0.3 * 0.10
    commission_lot = 3.50
    contract_size = 100.0
    time_barrier_bars = 16

    def calc_lot(equity, sl_dist):
        if equity <= 0 or sl_dist <= 0:
            return 0.0, 0.0
        target_risk = equity * risk_fraction
        loss_per_lot = sl_dist * contract_size
        raw_lot = target_risk / loss_per_lot
        lot = np.floor(raw_lot / 0.01) * 0.01
        lot = max(0.02, min(lot, 50.0))  # at least 0.02 so each half is >= 0.01
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
            d = open_pos['direction']
            ep = open_pos['entry_price']
            sl_dist = open_pos['sl_dist']
            lot_half = open_pos['lot_half']

            # Check exit conditions
            # 1. Friday close
            is_friday_close = (dow == 4 and hour >= 20) or dow in [5, 6]
            is_time_barrier = (bars_held >= time_barrier_bars)

            if is_friday_close or is_time_barrier:
                # Close whatever is still open at bar_close
                exit_reason = "FRIDAY_CLOSE" if is_friday_close else "TIME_BARRIER"
                exit_price = bar_close

                # If pos_A still open
                if open_pos['pos_a_open']:
                    pnl_a_gross = (exit_price - ep) * lot_half * contract_size if d == 'BUY' else (ep - exit_price) * lot_half * contract_size
                    pnl_a_net = round(pnl_a_gross - friction(lot_half), 2)
                    open_pos['pnl_accum'] += pnl_a_net
                    open_pos['pos_a_open'] = False

                # If pos_B still open
                if open_pos['pos_b_open']:
                    pnl_b_gross = (exit_price - ep) * lot_half * contract_size if d == 'BUY' else (ep - exit_price) * lot_half * contract_size
                    pnl_b_net = round(pnl_b_gross - friction(lot_half), 2)
                    open_pos['pnl_accum'] += pnl_b_net
                    open_pos['pos_b_open'] = False

                # Record trade
                total_pnl = round(open_pos['pnl_accum'], 2)
                r_mult = round(total_pnl / open_pos['total_risk'], 3)
                trades.append({
                    'id': len(trades),
                    'direction': d,
                    'entry_ts': open_pos['entry_ts'],
                    'exit_ts': ts,
                    'pnl_net': total_pnl,
                    'r_mult': r_mult,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                    'pos_a_tp_hit': open_pos['tp1_hit'],
                    'pos_b_tp_hit': open_pos['tp2_hit'],
                    'equity_before': equity,
                    'equity_after': equity + total_pnl
                })
                equity += total_pnl
                open_pos = None

            else:
                # Normal bar-by-bar price evaluation
                if d == 'BUY':
                    # Check initial SL hit for both before BE
                    if not open_pos['tp1_hit'] and bar_low <= open_pos['cur_sl_a']:
                        # Both pos A and pos B hit initial SL
                        pnl_gross = (open_pos['cur_sl_a'] - ep) * (lot_half * 2) * contract_size
                        pnl_net = round(pnl_gross - friction(lot_half * 2), 2)
                        r_mult = round(pnl_net / open_pos['total_risk'], 3)
                        trades.append({
                            'id': len(trades),
                            'direction': d,
                            'entry_ts': open_pos['entry_ts'],
                            'exit_ts': ts,
                            'pnl_net': pnl_net,
                            'r_mult': r_mult,
                            'exit_reason': "SL_HIT",
                            'bars_held': bars_held,
                            'pos_a_tp_hit': False,
                            'pos_b_tp_hit': False,
                            'equity_before': equity,
                            'equity_after': equity + pnl_net
                        })
                        equity += pnl_net
                        open_pos = None

                    else:
                        # Check TP1 (+1.0R) hit for Pos A
                        if open_pos['pos_a_open'] and bar_high >= open_pos['tp1']:
                            # Pos A hits TP1 (+1.0R)
                            pnl_a_gross = (open_pos['tp1'] - ep) * lot_half * contract_size
                            pnl_a_net = round(pnl_a_gross - friction(lot_half), 2)
                            open_pos['pnl_accum'] += pnl_a_net
                            open_pos['pos_a_open'] = False
                            open_pos['tp1_hit'] = True

                            # MOVE Pos B SL to Breakeven (entry + friction cost)
                            friction_adj = spread_price + (commission_lot / contract_size)
                            new_sl_b = round(ep + friction_adj, 2)
                            open_pos['cur_sl_b'] = max(open_pos['cur_sl_b'], new_sl_b)

                        # Check Pos B if still open
                        if open_pos['pos_b_open']:
                            # Check TP2 (+2.0R) hit
                            if bar_high >= open_pos['tp2']:
                                pnl_b_gross = (open_pos['tp2'] - ep) * lot_half * contract_size
                                pnl_b_net = round(pnl_b_gross - friction(lot_half), 2)
                                open_pos['pnl_accum'] += pnl_b_net
                                open_pos['pos_b_open'] = False
                                open_pos['tp2_hit'] = True

                                total_pnl = round(open_pos['pnl_accum'], 2)
                                r_mult = round(total_pnl / open_pos['total_risk'], 3)
                                trades.append({
                                    'id': len(trades),
                                    'direction': d,
                                    'entry_ts': open_pos['entry_ts'],
                                    'exit_ts': ts,
                                    'pnl_net': total_pnl,
                                    'r_mult': r_mult,
                                    'exit_reason': "TP2_HIT",
                                    'bars_held': bars_held,
                                    'pos_a_tp_hit': True,
                                    'pos_b_tp_hit': True,
                                    'equity_before': equity,
                                    'equity_after': equity + total_pnl
                                })
                                equity += total_pnl
                                open_pos = None

                            elif bar_low <= open_pos['cur_sl_b']:
                                # Pos B stopped out at SL_B (which is BE if TP1 was hit, or initial SL)
                                exit_r = "BE_HIT" if open_pos['tp1_hit'] else "SL_HIT"
                                pnl_b_gross = (open_pos['cur_sl_b'] - ep) * lot_half * contract_size
                                pnl_b_net = round(pnl_b_gross - friction(lot_half), 2)
                                open_pos['pnl_accum'] += pnl_b_net
                                open_pos['pos_b_open'] = False

                                total_pnl = round(open_pos['pnl_accum'], 2)
                                r_mult = round(total_pnl / open_pos['total_risk'], 3)
                                trades.append({
                                    'id': len(trades),
                                    'direction': d,
                                    'entry_ts': open_pos['entry_ts'],
                                    'exit_ts': ts,
                                    'pnl_net': total_pnl,
                                    'r_mult': r_mult,
                                    'exit_reason': exit_r,
                                    'bars_held': bars_held,
                                    'pos_a_tp_hit': open_pos['tp1_hit'],
                                    'pos_b_tp_hit': False,
                                    'equity_before': equity,
                                    'equity_after': equity + total_pnl
                                })
                                equity += total_pnl
                                open_pos = None

                            elif trail_after_1r and open_pos['tp1_hit']:
                                # Trailing stop for Pos B: if price reaches +1.5R, move SL to +0.8R
                                r_gain = (bar_high - ep) / sl_dist
                                if r_gain >= 1.5:
                                    trail_sl = round(ep + (0.8 * sl_dist), 2)
                                    open_pos['cur_sl_b'] = max(open_pos['cur_sl_b'], trail_sl)

                elif d == 'SELL':
                    # Check initial SL hit for both before BE
                    if not open_pos['tp1_hit'] and bar_high >= open_pos['cur_sl_a']:
                        pnl_gross = (ep - open_pos['cur_sl_a']) * (lot_half * 2) * contract_size
                        pnl_net = round(pnl_gross - friction(lot_half * 2), 2)
                        r_mult = round(pnl_net / open_pos['total_risk'], 3)
                        trades.append({
                            'id': len(trades),
                            'direction': d,
                            'entry_ts': open_pos['entry_ts'],
                            'exit_ts': ts,
                            'pnl_net': pnl_net,
                            'r_mult': r_mult,
                            'exit_reason': "SL_HIT",
                            'bars_held': bars_held,
                            'pos_a_tp_hit': False,
                            'pos_b_tp_hit': False,
                            'equity_before': equity,
                            'equity_after': equity + pnl_net
                        })
                        equity += pnl_net
                        open_pos = None

                    else:
                        # Check TP1 (+1.0R) hit for Pos A
                        if open_pos['pos_a_open'] and bar_low <= open_pos['tp1']:
                            pnl_a_gross = (ep - open_pos['tp1']) * lot_half * contract_size
                            pnl_a_net = round(pnl_a_gross - friction(lot_half), 2)
                            open_pos['pnl_accum'] += pnl_a_net
                            open_pos['pos_a_open'] = False
                            open_pos['tp1_hit'] = True

                            friction_adj = spread_price + (commission_lot / contract_size)
                            new_sl_b = round(ep - friction_adj, 2)
                            open_pos['cur_sl_b'] = min(open_pos['cur_sl_b'], new_sl_b)

                        # Check Pos B
                        if open_pos['pos_b_open']:
                            if bar_low <= open_pos['tp2']:
                                pnl_b_gross = (ep - open_pos['tp2']) * lot_half * contract_size
                                pnl_b_net = round(pnl_b_gross - friction(lot_half), 2)
                                open_pos['pnl_accum'] += pnl_b_net
                                open_pos['pos_b_open'] = False
                                open_pos['tp2_hit'] = True

                                total_pnl = round(open_pos['pnl_accum'], 2)
                                r_mult = round(total_pnl / open_pos['total_risk'], 3)
                                trades.append({
                                    'id': len(trades),
                                    'direction': d,
                                    'entry_ts': open_pos['entry_ts'],
                                    'exit_ts': ts,
                                    'pnl_net': total_pnl,
                                    'r_mult': r_mult,
                                    'exit_reason': "TP2_HIT",
                                    'bars_held': bars_held,
                                    'pos_a_tp_hit': True,
                                    'pos_b_tp_hit': True,
                                    'equity_before': equity,
                                    'equity_after': equity + total_pnl
                                })
                                equity += total_pnl
                                open_pos = None

                            elif bar_high >= open_pos['cur_sl_b']:
                                exit_r = "BE_HIT" if open_pos['tp1_hit'] else "SL_HIT"
                                pnl_b_gross = (ep - open_pos['cur_sl_b']) * lot_half * contract_size
                                pnl_b_net = round(pnl_b_gross - friction(lot_half), 2)
                                open_pos['pnl_accum'] += pnl_b_net
                                open_pos['pos_b_open'] = False

                                total_pnl = round(open_pos['pnl_accum'], 2)
                                r_mult = round(total_pnl / open_pos['total_risk'], 3)
                                trades.append({
                                    'id': len(trades),
                                    'direction': d,
                                    'entry_ts': open_pos['entry_ts'],
                                    'exit_ts': ts,
                                    'pnl_net': total_pnl,
                                    'r_mult': r_mult,
                                    'exit_reason': exit_r,
                                    'bars_held': bars_held,
                                    'pos_a_tp_hit': open_pos['tp1_hit'],
                                    'pos_b_tp_hit': False,
                                    'equity_before': equity,
                                    'equity_after': equity + total_pnl
                                })
                                equity += total_pnl
                                open_pos = None

                            elif trail_after_1r and open_pos['tp1_hit']:
                                r_gain = (ep - bar_low) / sl_dist
                                if r_gain >= 1.5:
                                    trail_sl = round(ep - (0.8 * sl_dist), 2)
                                    open_pos['cur_sl_b'] = min(open_pos['cur_sl_b'], trail_sl)

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

            if conf >= tau and conf > p_hold:
                direction = "BUY" if act_class in [1, 2] else "SELL"
                cur_atr = float(atr[i]) if i < len(atr) else 5.0
                sl_dist = round(cur_atr * sl_atr_mult, 2)

                total_lot, total_risk = calc_lot(equity, sl_dist)
                lot_half = round(total_lot / 2.0, 2)
                lot_half = max(0.01, lot_half)

                if total_lot >= 0.02 and total_risk > 0:
                    if direction == "BUY":
                        ep = round(bar_close + slippage_price, 2)
                        init_sl = round(ep - sl_dist, 2)
                        tp1 = round(ep + (1.0 * sl_dist), 2)
                        tp2 = round(ep + (2.0 * sl_dist), 2)
                    else:
                        ep = round(bar_close - slippage_price, 2)
                        init_sl = round(ep + sl_dist, 2)
                        tp1 = round(ep - (1.0 * sl_dist), 2)
                        tp2 = round(ep - (2.0 * sl_dist), 2)

                    open_pos = {
                        'direction': direction,
                        'entry_bar': i,
                        'entry_ts': ts,
                        'entry_price': ep,
                        'sl_dist': sl_dist,
                        'total_lot': total_lot,
                        'lot_half': lot_half,
                        'total_risk': total_risk,
                        'cur_sl_a': init_sl,
                        'cur_sl_b': init_sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'pos_a_open': True,
                        'pos_b_open': True,
                        'tp1_hit': False,
                        'tp2_hit': False,
                        'pnl_accum': 0.0,
                    }
                    daily_entries += 1

    tdf = pd.DataFrame(trades)
    return tdf, np.array(equity_curve), dates_curve

# Let's test different Tau values with Partial TP & Trailing Stop!
print("=== TESTING PARTIAL TP (TWIN POSITIONS: TP1 1.0R, TP2 2.0R, TRAIL) ===")
for test_tau in [0.28, 0.30, 0.32, 0.34]:
    tdf, eq_curve, _ = run_partial_tp_backtest(df, preds, atr, tau=test_tau)
    tdf['entry_ts'] = pd.to_datetime(tdf['entry_ts'])
    tdf['month'] = tdf['entry_ts'].dt.strftime('%Y-%m')
    months_2026 = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06', '2026-07', '2026-08']
    tdf_2026 = tdf[tdf['month'].isin(months_2026)]

    n_trades = len(tdf_2026)
    n_wins = (tdf_2026['pnl_net'] > 0).sum()
    wr = (n_wins / n_trades) * 100 if n_trades > 0 else 0
    tot_pnl = tdf_2026['pnl_net'].sum()
    peak = np.maximum.accumulate(eq_curve)
    max_dd = np.min((eq_curve - peak) / peak) * 100 if len(eq_curve) > 0 else 0
    tp1_cnt = (tdf_2026['pos_a_tp_hit']).sum()
    tp2_cnt = (tdf_2026['pos_b_tp_hit']).sum()

    print(f"Tau={test_tau:.2f} | Trades={n_trades:3d} | WR={wr:5.1f}% | TP1 Hits={tp1_cnt:3d} | TP2 Hits={tp2_cnt:3d} | Net PnL=${tot_pnl:8.2f} | Max DD={max_dd:5.1f}%")
