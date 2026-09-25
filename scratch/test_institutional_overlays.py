import sys, os, pathlib
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
df = pd.read_parquet(project_root / "data" / "processed" / "xauusd_m30_test_labeled.parquet")
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(r'c:\Ngoding\bot_trading\scratch\test_predictions_2026.npy')

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

def run_experiment(
    df, preds, atr,
    tau=0.32,
    quarantine_london_open=True,  # No entry 07:00-08:30 UTC
    use_structural_filter=True,    # Avoid buying directly into supply or selling into demand
    tp1_r=1.0,
    tp2_r=2.5,
    pos_a_pct=0.40,
    pos_b_pct=0.60,
    trail_trigger_r=1.5,
    trail_dist_r=0.8
):
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
        lot = max(0.02, min(lot, 50.0))
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
    equity_curve = []

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
        minute = ts.minute

        # Check exit
        if open_pos is not None:
            bars_held = i - open_pos['entry_bar']
            d = open_pos['direction']
            ep = open_pos['entry_price']
            sl_dist = open_pos['sl_dist']
            lot_a = open_pos['lot_a']
            lot_b = open_pos['lot_b']

            is_friday_close = (dow == 4 and hour >= 20) or dow in [5, 6]
            is_time_barrier = (bars_held >= time_barrier_bars)

            if is_friday_close or is_time_barrier:
                exit_reason = "FRIDAY_CLOSE" if is_friday_close else "TIME_BARRIER"
                exit_price = bar_close

                if open_pos['pos_a_open']:
                    pnl_a = (exit_price - ep) * lot_a * contract_size if d == 'BUY' else (ep - exit_price) * lot_a * contract_size
                    open_pos['pnl_accum'] += (pnl_a - friction(lot_a))
                    open_pos['pos_a_open'] = False

                if open_pos['pos_b_open']:
                    pnl_b = (exit_price - ep) * lot_b * contract_size if d == 'BUY' else (ep - exit_price) * lot_b * contract_size
                    open_pos['pnl_accum'] += (pnl_b - friction(lot_b))
                    open_pos['pos_b_open'] = False

                tot_pnl = round(open_pos['pnl_accum'], 2)
                r_mult = round(tot_pnl / open_pos['total_risk'], 3)
                trades.append({
                    'month': open_pos['entry_ts'].strftime('%Y-%m'),
                    'direction': d,
                    'entry_ts': open_pos['entry_ts'],
                    'pnl_net': tot_pnl,
                    'r_mult': r_mult,
                    'exit_reason': exit_reason,
                    'tp1_hit': open_pos['tp1_hit'],
                    'tp2_hit': open_pos['tp2_hit'],
                })
                equity += tot_pnl
                open_pos = None

            else:
                if d == 'BUY':
                    # Check initial SL before TP1
                    if not open_pos['tp1_hit'] and bar_low <= open_pos['cur_sl_a']:
                        pnl_loss = (open_pos['cur_sl_a'] - ep) * open_pos['total_lot'] * contract_size
                        tot_net = round(pnl_loss - friction(open_pos['total_lot']), 2)
                        r_mult = round(tot_net / open_pos['total_risk'], 3)
                        trades.append({
                            'month': open_pos['entry_ts'].strftime('%Y-%m'),
                            'direction': d,
                            'entry_ts': open_pos['entry_ts'],
                            'pnl_net': tot_net,
                            'r_mult': r_mult,
                            'exit_reason': 'SL_HIT',
                            'tp1_hit': False,
                            'tp2_hit': False,
                        })
                        equity += tot_net
                        open_pos = None
                    else:
                        # Check TP1
                        if open_pos['pos_a_open'] and bar_high >= open_pos['tp1']:
                            pnl_a = (open_pos['tp1'] - ep) * lot_a * contract_size
                            open_pos['pnl_accum'] += (pnl_a - friction(lot_a))
                            open_pos['pos_a_open'] = False
                            open_pos['tp1_hit'] = True
                            # Move SL of Pos B to Breakeven
                            new_sl_b = round(ep + spread_price + (commission_lot / contract_size), 2)
                            open_pos['cur_sl_b'] = max(open_pos['cur_sl_b'], new_sl_b)

                        # Check Pos B
                        if open_pos['pos_b_open']:
                            if bar_high >= open_pos['tp2']:
                                pnl_b = (open_pos['tp2'] - ep) * lot_b * contract_size
                                open_pos['pnl_accum'] += (pnl_b - friction(lot_b))
                                open_pos['pos_b_open'] = False
                                open_pos['tp2_hit'] = True
                                tot_net = round(open_pos['pnl_accum'], 2)
                                r_mult = round(tot_net / open_pos['total_risk'], 3)
                                trades.append({
                                    'month': open_pos['entry_ts'].strftime('%Y-%m'),
                                    'direction': d,
                                    'entry_ts': open_pos['entry_ts'],
                                    'pnl_net': tot_net,
                                    'r_mult': r_mult,
                                    'exit_reason': 'TP2_HIT',
                                    'tp1_hit': True,
                                    'tp2_hit': True,
                                })
                                equity += tot_net
                                open_pos = None
                            elif bar_low <= open_pos['cur_sl_b']:
                                pnl_b = (open_pos['cur_sl_b'] - ep) * lot_b * contract_size
                                open_pos['pnl_accum'] += (pnl_b - friction(lot_b))
                                open_pos['pos_b_open'] = False
                                tot_net = round(open_pos['pnl_accum'], 2)
                                r_mult = round(tot_net / open_pos['total_risk'], 3)
                                trades.append({
                                    'month': open_pos['entry_ts'].strftime('%Y-%m'),
                                    'direction': d,
                                    'entry_ts': open_pos['entry_ts'],
                                    'pnl_net': tot_net,
                                    'r_mult': r_mult,
                                    'exit_reason': 'BE_HIT' if open_pos['tp1_hit'] else 'SL_HIT',
                                    'tp1_hit': open_pos['tp1_hit'],
                                    'tp2_hit': False,
                                })
                                equity += tot_net
                                open_pos = None
                            elif open_pos['tp1_hit']:
                                r_gain = (bar_high - ep) / sl_dist
                                if r_gain >= trail_trigger_r:
                                    trail_sl = round(ep + (trail_dist_r * sl_dist), 2)
                                    open_pos['cur_sl_b'] = max(open_pos['cur_sl_b'], trail_sl)

                elif d == 'SELL':
                    if not open_pos['tp1_hit'] and bar_high >= open_pos['cur_sl_a']:
                        pnl_loss = (ep - open_pos['cur_sl_a']) * open_pos['total_lot'] * contract_size
                        tot_net = round(pnl_loss - friction(open_pos['total_lot']), 2)
                        r_mult = round(tot_net / open_pos['total_risk'], 3)
                        trades.append({
                            'month': open_pos['entry_ts'].strftime('%Y-%m'),
                            'direction': d,
                            'entry_ts': open_pos['entry_ts'],
                            'pnl_net': tot_net,
                            'r_mult': r_mult,
                            'exit_reason': 'SL_HIT',
                            'tp1_hit': False,
                            'tp2_hit': False,
                        })
                        equity += tot_net
                        open_pos = None
                    else:
                        if open_pos['pos_a_open'] and bar_low <= open_pos['tp1']:
                            pnl_a = (ep - open_pos['tp1']) * lot_a * contract_size
                            open_pos['pnl_accum'] += (pnl_a - friction(lot_a))
                            open_pos['pos_a_open'] = False
                            open_pos['tp1_hit'] = True
                            new_sl_b = round(ep - (spread_price + commission_lot / contract_size), 2)
                            open_pos['cur_sl_b'] = min(open_pos['cur_sl_b'], new_sl_b)

                        if open_pos['pos_b_open']:
                            if bar_low <= open_pos['tp2']:
                                pnl_b = (ep - open_pos['tp2']) * lot_b * contract_size
                                open_pos['pnl_accum'] += (pnl_b - friction(lot_b))
                                open_pos['pos_b_open'] = False
                                open_pos['tp2_hit'] = True
                                tot_net = round(open_pos['pnl_accum'], 2)
                                r_mult = round(tot_net / open_pos['total_risk'], 3)
                                trades.append({
                                    'month': open_pos['entry_ts'].strftime('%Y-%m'),
                                    'direction': d,
                                    'entry_ts': open_pos['entry_ts'],
                                    'pnl_net': tot_net,
                                    'r_mult': r_mult,
                                    'exit_reason': 'TP2_HIT',
                                    'tp1_hit': True,
                                    'tp2_hit': True,
                                })
                                equity += tot_net
                                open_pos = None
                            elif bar_high >= open_pos['cur_sl_b']:
                                pnl_b = (ep - open_pos['cur_sl_b']) * lot_b * contract_size
                                open_pos['pnl_accum'] += (pnl_b - friction(lot_b))
                                open_pos['pos_b_open'] = False
                                tot_net = round(open_pos['pnl_accum'], 2)
                                r_mult = round(tot_net / open_pos['total_risk'], 3)
                                trades.append({
                                    'month': open_pos['entry_ts'].strftime('%Y-%m'),
                                    'direction': d,
                                    'entry_ts': open_pos['entry_ts'],
                                    'pnl_net': tot_net,
                                    'r_mult': r_mult,
                                    'exit_reason': 'BE_HIT' if open_pos['tp1_hit'] else 'SL_HIT',
                                    'tp1_hit': open_pos['tp1_hit'],
                                    'tp2_hit': False,
                                })
                                equity += tot_net
                                open_pos = None
                            elif open_pos['tp1_hit']:
                                r_gain = (ep - bar_low) / sl_dist
                                if r_gain >= trail_trigger_r:
                                    trail_sl = round(ep - (trail_dist_r * sl_dist), 2)
                                    open_pos['cur_sl_b'] = min(open_pos['cur_sl_b'], trail_sl)

        if is_in_2026:
            equity_curve.append(equity)

        # Check new entry
        pred_idx = i - seq_offset
        if (
            is_in_2026
            and pred_idx >= 0
            and pred_idx < len(preds)
            and open_pos is None
            and daily_entries < 10
            and bool(row.get('entry_eligible', True))
            and not bool(row.get('is_news_blackout', False))
            and not (dow == 4 and hour >= 18)
            and dow not in [5, 6]
        ):
            # Check London open quarantine (07:00 to 08:30 UTC)
            if quarantine_london_open and (hour == 7 or (hour == 8 and minute < 30)):
                continue

            p = preds[pred_idx]
            trade_probs = p[1:]
            max_c = int(np.argmax(trade_probs))
            conf = float(trade_probs[max_c])
            act_class = max_c + 1
            p_hold = float(p[0])

            if conf >= tau and conf > p_hold:
                direction = "BUY" if act_class in [1, 2] else "SELL"
                
                # Check structural filter
                if use_structural_filter:
                    ob_zone = float(row.get('order_block_zone', 0.0))
                    ma_slope = float(row.get('ma_ribbon_slope', 0.0))
                    
                    # Don't buy right inside supply zone (-1) or strong downtrend
                    if direction == "BUY" and (ob_zone < 0 or ma_slope < -0.3):
                        continue
                    # Don't sell right inside demand zone (+1) or strong uptrend
                    if direction == "SELL" and (ob_zone > 0 or ma_slope > 0.3):
                        continue

                cur_atr = float(atr[i]) if i < len(atr) else 5.0
                sl_dist = round(cur_atr * sl_atr_mult, 2)
                tot_lot, tot_risk = calc_lot(equity, sl_dist)
                
                lot_a = round(tot_lot * pos_a_pct, 2)
                lot_a = max(0.01, lot_a)
                lot_b = round(tot_lot - lot_a, 2)
                lot_b = max(0.01, lot_b)

                if tot_lot >= 0.02 and tot_risk > 0:
                    if direction == "BUY":
                        ep = round(bar_close + slippage_price, 2)
                        init_sl = round(ep - sl_dist, 2)
                        tp1 = round(ep + (tp1_r * sl_dist), 2)
                        tp2 = round(ep + (tp2_r * sl_dist), 2)
                    else:
                        ep = round(bar_close - slippage_price, 2)
                        init_sl = round(ep + sl_dist, 2)
                        tp1 = round(ep - (tp1_r * sl_dist), 2)
                        tp2 = round(ep - (tp2_r * sl_dist), 2)

                    open_pos = {
                        'direction': direction,
                        'entry_bar': i,
                        'entry_ts': ts,
                        'entry_price': ep,
                        'sl_dist': sl_dist,
                        'total_lot': tot_lot,
                        'lot_a': lot_a,
                        'lot_b': lot_b,
                        'total_risk': tot_risk,
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
    return tdf, np.array(equity_curve)

# Test configurations
print("=== TESTING INSTITUTIONAL OVERLAY COMBINATIONS ===")
configs = [
    {"name": "Current Baseline", "quar": False, "struct": False, "tp1": 1.0, "tp2": 2.0, "pa": 0.5, "pb": 0.5, "tau": 0.32},
    {"name": "London Quarantine Only", "quar": True, "struct": False, "tp1": 1.0, "tp2": 2.0, "pa": 0.5, "pb": 0.5, "tau": 0.32},
    {"name": "London Quar + Struct Filter", "quar": True, "struct": True, "tp1": 1.0, "tp2": 2.0, "pa": 0.5, "pb": 0.5, "tau": 0.32},
    {"name": "Quar + Struct + TP2=2.5R (40/60)", "quar": True, "struct": True, "tp1": 1.0, "tp2": 2.5, "pa": 0.4, "pb": 0.6, "tau": 0.32},
    {"name": "Quar + Struct + Tau=0.34", "quar": True, "struct": True, "tp1": 1.0, "tp2": 2.5, "pa": 0.4, "pb": 0.6, "tau": 0.34},
    {"name": "Quar + Struct + Tau=0.35", "quar": True, "struct": True, "tp1": 1.0, "tp2": 2.5, "pa": 0.4, "pb": 0.6, "tau": 0.35},
]

for cfg in configs:
    tdf, eq_curve = run_experiment(
        df, preds, atr,
        tau=cfg["tau"],
        quarantine_london_open=cfg["quar"],
        use_structural_filter=cfg["struct"],
        tp1_r=cfg["tp1"],
        tp2_r=cfg["tp2"],
        pos_a_pct=cfg["pa"],
        pos_b_pct=cfg["pb"]
    )
    if len(tdf) > 0:
        n_tr = len(tdf)
        wins = (tdf['pnl_net'] > 0).sum()
        wr = wins / n_tr * 100
        pnl = tdf['pnl_net'].sum()
        peak = np.maximum.accumulate(eq_curve)
        dd = np.min((eq_curve - peak) / peak) * 100 if len(eq_curve) > 0 else 0
        g_win = tdf[tdf['pnl_net'] > 0]['pnl_net'].sum()
        g_loss = abs(tdf[tdf['pnl_net'] < 0]['pnl_net'].sum())
        pf = g_win / g_loss if g_loss > 0 else 99.0
        
        # Monthly wins
        pos_months = 0
        for m, g in tdf.groupby('month'):
            if g['pnl_net'].sum() > 0:
                pos_months += 1
                
        print(f"{cfg['name']:32s} | N={n_tr:3d} | WR={wr:4.1f}% | PnL=${pnl:8.2f} ({(pnl/10000)*100:5.1f}%) | Max DD={dd:5.1f}% | PF={pf:4.2f} | Green Months: {pos_months}/8")
