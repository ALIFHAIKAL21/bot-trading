"""
Test Prime Liquidity Window and Post-Loss Cooldown on 2026
"""

import sys, os, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, r'c:\Ngoding\bot_trading')
sys.path.insert(0, r'c:\Ngoding\bot_trading\scripts')

from scripts.backtest_engine_adaptive import AdaptiveBacktestEngine, TradeRecord, BacktestResult

df_path = r'c:\Ngoding\xau_deep_sniper\data\processed\xauusd_m30_labeled.parquet'
preds_path = r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy'

df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

# ATR(14)
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

# Test various session windows & cooldowns
# Let's create an updated run function or subclass to evaluate:
class EnhancedEngine(AdaptiveBacktestEngine):
    def __init__(self, prime_session_only=True, post_loss_cooldown_bars=3, uncapped_runner=True, **kwargs):
        super().__init__(**kwargs)
        self.prime_session_only = prime_session_only
        self.post_loss_cooldown_bars = post_loss_cooldown_bars
        self.enable_uncapped_runner = uncapped_runner

    def run(self, df, predictions, atr_values, start_date=None, end_date=None):
        # We can implement prime session and cooldown directly
        # Prime session: 08:30 to 16:30 UTC
        # Cooldown: last_sl_bar_idx
        n_bars = len(df)
        seq_offset = 63
        equity = self.initial_equity
        equity_curve = [equity]
        timestamps_list = []
        trades = []
        trade_counter = 0
        open_position = None
        last_sl_bar_idx = -999

        weekly_starting_equity = equity
        all_time_peak = equity
        current_week_id = None
        is_weekly_tripped = False
        is_permanent_stopped = False
        circuit_breaker_trips = 0
        weekly_drawdowns = {}
        current_day = None
        daily_entries_count = 0
        consecutive_losses = 0

        ts_series = df["timestamp_utc"]
        st_ts = pd.to_datetime(start_date, utc=True) if start_date else ts_series.iloc[0]
        end_ts = pd.to_datetime(end_date, utc=True) if end_date else ts_series.iloc[-1]

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
                daily_entries_count = 0

            # Manage open position
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
                is_time_barrier = (bars_held >= self.time_barrier_bars)

                if is_friday_close or is_time_barrier:
                    exit_reason = "FRIDAY_CLOSE" if is_friday_close else "TIME_BARRIER"
                    exit_price = bar_close
                    if open_position["pos_a_open"]:
                        pnl_a_g = (exit_price - ep) * lot_a * self.contract_size if d == "BUY" else (ep - exit_price) * lot_a * self.contract_size
                        open_position["pnl_net_accum"] += (pnl_a_g - self._friction_per_trade(lot_a))
                        open_position["pos_a_open"] = False
                    if open_position["pos_b_open"]:
                        pnl_b_g = (exit_price - ep) * lot_b * self.contract_size if d == "BUY" else (ep - exit_price) * lot_b * self.contract_size
                        open_position["pnl_net_accum"] += (pnl_b_g - self._friction_per_trade(lot_b))
                        open_position["pos_b_open"] = False

                    total_net = round(open_position["pnl_net_accum"], 2)
                    r_mult = round(total_net / open_position["risk_amount"], 3) if open_position["risk_amount"] > 0 else 0.0
                    if total_net < 0:
                        last_sl_bar_idx = bar_idx
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                    trades.append(TradeRecord(
                        trade_id=trade_counter, direction=d, playbook="TREND_RUNNER",
                        entry_bar_idx=open_position["entry_bar_idx"], exit_bar_idx=bar_idx,
                        entry_timestamp=open_position["entry_timestamp"], exit_timestamp=ts,
                        entry_price=ep, exit_price=exit_price, sl_price=open_position["initial_sl"],
                        tp1_price=open_position["tp1_price"], tp2_price=open_position["tp2_price"],
                        lot_size=open_position["total_lot"], r_target=self.tp2_r_ratio,
                        confidence=open_position["confidence"],
                        pnl_gross=round(total_net + self._friction_per_trade(open_position["total_lot"]), 2),
                        friction_cost=self._friction_per_trade(open_position["total_lot"]),
                        pnl_net=total_net, r_multiple=r_mult, exit_reason=exit_reason,
                        was_breakeven_moved=open_position["tp1_hit"], bars_held=bars_held,
                        tp1_hit=open_position["tp1_hit"], tp2_hit=open_position["tp2_hit"],
                    ))
                    trade_counter += 1
                    equity += total_net
                    open_position = None

                else:
                    if d == "BUY":
                        if not open_position["tp1_hit"] and bar_low <= open_position["cur_sl_a"]:
                            pnl_g = (open_position["cur_sl_a"] - ep) * open_position["total_lot"] * self.contract_size
                            fric = self._friction_per_trade(open_position["total_lot"])
                            pnl_net = round(pnl_g - fric, 2)
                            r_mult = round(pnl_net / open_position["risk_amount"], 3)
                            last_sl_bar_idx = bar_idx
                            consecutive_losses += 1

                            trades.append(TradeRecord(
                                trade_id=trade_counter, direction=d, playbook="TREND_RUNNER",
                                entry_bar_idx=open_position["entry_bar_idx"], exit_bar_idx=bar_idx,
                                entry_timestamp=open_position["entry_timestamp"], exit_timestamp=ts,
                                entry_price=ep, exit_price=open_position["cur_sl_a"],
                                sl_price=open_position["initial_sl"], tp1_price=open_position["tp1_price"],
                                tp2_price=open_position["tp2_price"], lot_size=open_position["total_lot"],
                                r_target=self.tp2_r_ratio, confidence=open_position["confidence"],
                                pnl_gross=round(pnl_g, 2), friction_cost=fric, pnl_net=pnl_net,
                                r_multiple=r_mult, exit_reason="SL_HIT", was_breakeven_moved=False,
                                bars_held=bars_held, tp1_hit=False, tp2_hit=False,
                            ))
                            trade_counter += 1
                            equity += pnl_net
                            open_position = None

                        else:
                            if open_position["pos_a_open"] and bar_high >= open_position["tp1_price"]:
                                pnl_a_g = (open_position["tp1_price"] - ep) * lot_a * self.contract_size
                                open_position["pnl_net_accum"] += (pnl_a_g - self._friction_per_trade(lot_a))
                                open_position["pos_a_open"] = False
                                open_position["tp1_hit"] = True
                                f_price = self.spread_price + (self.commission_per_lot / self.contract_size)
                                new_sl_b = round(ep + f_price, 2)
                                open_position["cur_sl_b"] = max(open_position["cur_sl_b"], new_sl_b)

                            if open_position["pos_b_open"]:
                                if not self.enable_uncapped_runner and bar_high >= open_position["tp2_price"]:
                                    pnl_b_g = (open_position["tp2_price"] - ep) * lot_b * self.contract_size
                                    open_position["pnl_net_accum"] += (pnl_b_g - self._friction_per_trade(lot_b))
                                    open_position["pos_b_open"] = False
                                    open_position["tp2_hit"] = True
                                    total_net = round(open_position["pnl_net_accum"], 2)
                                    r_mult = round(total_net / open_position["risk_amount"], 3)
                                    consecutive_losses = 0

                                    trades.append(TradeRecord(
                                        trade_id=trade_counter, direction=d, playbook="TREND_RUNNER",
                                        entry_bar_idx=open_position["entry_bar_idx"], exit_bar_idx=bar_idx,
                                        entry_timestamp=open_position["entry_timestamp"], exit_timestamp=ts,
                                        entry_price=ep, exit_price=open_position["tp2_price"],
                                        sl_price=open_position["initial_sl"], tp1_price=open_position["tp1_price"],
                                        tp2_price=open_position["tp2_price"], lot_size=open_position["total_lot"],
                                        r_target=self.tp2_r_ratio, confidence=open_position["confidence"],
                                        pnl_gross=round(total_net + self._friction_per_trade(open_position["total_lot"]), 2),
                                        friction_cost=self._friction_per_trade(open_position["total_lot"]),
                                        pnl_net=total_net, r_multiple=r_mult, exit_reason="TP2_HIT",
                                        was_breakeven_moved=True, bars_held=bars_held, tp1_hit=True, tp2_hit=True,
                                    ))
                                    trade_counter += 1
                                    equity += total_net
                                    open_position = None

                                elif bar_low <= open_position["cur_sl_b"]:
                                    exit_r = "BE_HIT" if open_position["tp1_hit"] else "SL_HIT"
                                    pnl_b_g = (open_position["cur_sl_b"] - ep) * lot_b * self.contract_size
                                    open_position["pnl_net_accum"] += (pnl_b_g - self._friction_per_trade(lot_b))
                                    open_position["pos_b_open"] = False
                                    total_net = round(open_position["pnl_net_accum"], 2)
                                    r_mult = round(total_net / open_position["risk_amount"], 3)
                                    if total_net > 0: consecutive_losses = 0
                                    elif total_net < 0:
                                        last_sl_bar_idx = bar_idx
                                        consecutive_losses += 1

                                    trades.append(TradeRecord(
                                        trade_id=trade_counter, direction=d, playbook="TREND_RUNNER",
                                        entry_bar_idx=open_position["entry_bar_idx"], exit_bar_idx=bar_idx,
                                        entry_timestamp=open_position["entry_timestamp"], exit_timestamp=ts,
                                        entry_price=ep, exit_price=open_position["cur_sl_b"],
                                        sl_price=open_position["initial_sl"], tp1_price=open_position["tp1_price"],
                                        tp2_price=open_position["tp2_price"], lot_size=open_position["total_lot"],
                                        r_target=self.tp2_r_ratio, confidence=open_position["confidence"],
                                        pnl_gross=round(total_net + self._friction_per_trade(open_position["total_lot"]), 2),
                                        friction_cost=self._friction_per_trade(open_position["total_lot"]),
                                        pnl_net=total_net, r_multiple=r_mult, exit_reason=exit_r,
                                        was_breakeven_moved=open_position["tp1_hit"], bars_held=bars_held,
                                        tp1_hit=open_position["tp1_hit"], tp2_hit=False,
                                    ))
                                    trade_counter += 1
                                    equity += total_net
                                    open_position = None

                                elif open_position["tp1_hit"]:
                                    r_gain = (bar_high - ep) / sl_dist
                                    if r_gain >= self.trail_after_r:
                                        trail_sl = round(bar_high - (self.trail_distance_r * sl_dist), 2)
                                        open_position["cur_sl_b"] = max(open_position["cur_sl_b"], trail_sl)

                    else:  # SELL
                        if not open_position["tp1_hit"] and bar_high >= open_position["cur_sl_a"]:
                            pnl_g = (ep - open_position["cur_sl_a"]) * open_position["total_lot"] * self.contract_size
                            fric = self._friction_per_trade(open_position["total_lot"])
                            pnl_net = round(pnl_g - fric, 2)
                            r_mult = round(pnl_net / open_position["risk_amount"], 3)
                            last_sl_bar_idx = bar_idx
                            consecutive_losses += 1

                            trades.append(TradeRecord(
                                trade_id=trade_counter, direction=d, playbook="TREND_RUNNER",
                                entry_bar_idx=open_position["entry_bar_idx"], exit_bar_idx=bar_idx,
                                entry_timestamp=open_position["entry_timestamp"], exit_timestamp=ts,
                                entry_price=ep, exit_price=open_position["cur_sl_a"],
                                sl_price=open_position["initial_sl"], tp1_price=open_position["tp1_price"],
                                tp2_price=open_position["tp2_price"], lot_size=open_position["total_lot"],
                                r_target=self.tp2_r_ratio, confidence=open_position["confidence"],
                                pnl_gross=round(pnl_g, 2), friction_cost=fric, pnl_net=pnl_net,
                                r_multiple=r_mult, exit_reason="SL_HIT", was_breakeven_moved=False,
                                bars_held=bars_held, tp1_hit=False, tp2_hit=False,
                            ))
                            trade_counter += 1
                            equity += pnl_net
                            open_position = None

                        else:
                            if open_position["pos_a_open"] and bar_low <= open_position["tp1_price"]:
                                pnl_a_g = (ep - open_position["tp1_price"]) * lot_a * self.contract_size
                                open_position["pnl_net_accum"] += (pnl_a_g - self._friction_per_trade(lot_a))
                                open_position["pos_a_open"] = False
                                open_position["tp1_hit"] = True
                                f_price = self.spread_price + (self.commission_per_lot / self.contract_size)
                                new_sl_b = round(ep - f_price, 2)
                                open_position["cur_sl_b"] = min(open_position["cur_sl_b"], new_sl_b)

                            if open_position["pos_b_open"]:
                                if not self.enable_uncapped_runner and bar_low <= open_position["tp2_price"]:
                                    pnl_b_g = (ep - open_position["tp2_price"]) * lot_b * self.contract_size
                                    open_position["pnl_net_accum"] += (pnl_b_g - self._friction_per_trade(lot_b))
                                    open_position["pos_b_open"] = False
                                    open_position["tp2_hit"] = True
                                    total_net = round(open_position["pnl_net_accum"], 2)
                                    r_mult = round(total_net / open_position["risk_amount"], 3)
                                    consecutive_losses = 0

                                    trades.append(TradeRecord(
                                        trade_id=trade_counter, direction=d, playbook="TREND_RUNNER",
                                        entry_bar_idx=open_position["entry_bar_idx"], exit_bar_idx=bar_idx,
                                        entry_timestamp=open_position["entry_timestamp"], exit_timestamp=ts,
                                        entry_price=ep, exit_price=open_position["tp2_price"],
                                        sl_price=open_position["initial_sl"], tp1_price=open_position["tp1_price"],
                                        tp2_price=open_position["tp2_price"], lot_size=open_position["total_lot"],
                                        r_target=self.tp2_r_ratio, confidence=open_position["confidence"],
                                        pnl_gross=round(total_net + self._friction_per_trade(open_position["total_lot"]), 2),
                                        friction_cost=self._friction_per_trade(open_position["total_lot"]),
                                        pnl_net=total_net, r_multiple=r_mult, exit_reason="TP2_HIT",
                                        was_breakeven_moved=True, bars_held=bars_held, tp1_hit=True, tp2_hit=True,
                                    ))
                                    trade_counter += 1
                                    equity += total_net
                                    open_position = None

                                elif bar_high >= open_position["cur_sl_b"]:
                                    exit_r = "BE_HIT" if open_position["tp1_hit"] else "SL_HIT"
                                    pnl_b_g = (ep - open_position["cur_sl_b"]) * lot_b * self.contract_size
                                    open_position["pnl_net_accum"] += (pnl_b_g - self._friction_per_trade(lot_b))
                                    open_position["pos_b_open"] = False
                                    total_net = round(open_position["pnl_net_accum"], 2)
                                    r_mult = round(total_net / open_position["risk_amount"], 3)
                                    if total_net > 0: consecutive_losses = 0
                                    elif total_net < 0:
                                        last_sl_bar_idx = bar_idx
                                        consecutive_losses += 1

                                    trades.append(TradeRecord(
                                        trade_id=trade_counter, direction=d, playbook="TREND_RUNNER",
                                        entry_bar_idx=open_position["entry_bar_idx"], exit_bar_idx=bar_idx,
                                        entry_timestamp=open_position["entry_timestamp"], exit_timestamp=ts,
                                        entry_price=ep, exit_price=open_position["cur_sl_b"],
                                        sl_price=open_position["initial_sl"], tp1_price=open_position["tp1_price"],
                                        tp2_price=open_position["tp2_price"], lot_size=open_position["total_lot"],
                                        r_target=self.tp2_r_ratio, confidence=open_position["confidence"],
                                        pnl_gross=round(total_net + self._friction_per_trade(open_position["total_lot"]), 2),
                                        friction_cost=self._friction_per_trade(open_position["total_lot"]),
                                        pnl_net=total_net, r_multiple=r_mult, exit_reason=exit_r,
                                        was_breakeven_moved=open_position["tp1_hit"], bars_held=bars_held,
                                        tp1_hit=open_position["tp1_hit"], tp2_hit=False,
                                    ))
                                    trade_counter += 1
                                    equity += total_net
                                    open_position = None

                                elif open_position["tp1_hit"]:
                                    r_gain = (ep - bar_low) / sl_dist
                                    if r_gain >= self.trail_after_r:
                                        trail_sl = round(bar_low + (self.trail_distance_r * sl_dist), 2)
                                        open_position["cur_sl_b"] = min(open_position["cur_sl_b"], trail_sl)

            if in_date_range:
                equity_curve.append(equity)
                timestamps_list.append(ts)

            # Evaluate entry
            pred_idx = bar_idx - seq_offset
            if (
                in_date_range
                and pred_idx >= 0
                and pred_idx < len(predictions)
                and open_position is None
                and daily_entries_count < self.max_daily_entries
                and is_eligible
                and not is_blackout
            ):
                dow = ts.dayofweek if hasattr(ts, 'dayofweek') else pd.Timestamp(ts).dayofweek
                hour = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour
                minute = ts.minute if hasattr(ts, 'minute') else pd.Timestamp(ts).minute

                friday_freeze = (dow == 4 and hour >= 18) or dow in [5, 6]
                london_quarantine = (hour == 7 or (hour == 8 and minute < 30))

                # Post-loss cooldown
                in_cooldown = (bar_idx - last_sl_bar_idx) < self.post_loss_cooldown_bars

                # Prime Session: Only trade between 08:30 and 17:00 UTC
                prime_ok = (hour >= 8 and (hour > 8 or minute >= 30) and hour < 17) if self.prime_session_only else True

                if not friday_freeze and not london_quarantine and not in_cooldown and prime_ok:
                    probs = predictions[pred_idx]
                    trade_probs = probs[1:]
                    max_class_idx = int(np.argmax(trade_probs))
                    action_class = max_class_idx + 1
                    confidence = float(trade_probs[max_class_idx])
                    p_hold = float(probs[0])

                    if confidence >= self.confidence_tau and confidence > p_hold:
                        direction = "BUY" if action_class in [1, 2] else "SELL"
                        ob_zone = float(row.get("order_block_zone", 0.0))
                        ma_slope = float(row.get("ma_ribbon_slope", 0.0))

                        structural_ok = True
                        if direction == "BUY" and (ob_zone < 0 or ma_slope < -0.3):
                            structural_ok = False
                        if direction == "SELL" and (ob_zone > 0 or ma_slope > 0.3):
                            structural_ok = False

                        if structural_ok:
                            risk_frac = self.defensive_risk_fraction if (self.enable_defensive_scaling and consecutive_losses >= 2) else self.base_risk_fraction
                            atr_val = float(atr_values[bar_idx]) if bar_idx < len(atr_values) else 5.0
                            sl_distance = round(atr_val * self.sl_atr_multiplier, 2)
                            if sl_distance > 0:
                                lot_size, risk_amount = self._calculate_lot_size(equity, sl_distance, risk_frac)
                                lot_a = round(lot_size * self.pos_a_pct, 2)
                                lot_a = max(0.01, lot_a)
                                lot_b = round(lot_size - lot_a, 2)
                                lot_b = max(0.01, lot_b)

                                if lot_size >= 0.02 and risk_amount > 0:
                                    if direction == "BUY":
                                        ep = round(bar_close + self.slippage_price, 2)
                                        sl_price = round(ep - sl_distance, 2)
                                        tp1_p = round(ep + (self.tp1_r_ratio * sl_distance), 2)
                                        tp2_p = round(ep + (self.tp2_r_ratio * sl_distance), 2)
                                    else:
                                        ep = round(bar_close - self.slippage_price, 2)
                                        sl_price = round(ep + sl_distance, 2)
                                        tp1_p = round(ep - (self.tp1_r_ratio * sl_distance), 2)
                                        tp2_p = round(ep - (self.tp2_r_ratio * sl_distance), 2)

                                    open_position = {
                                        "direction": direction,
                                        "entry_bar_idx": bar_idx,
                                        "entry_timestamp": ts,
                                        "entry_price": ep,
                                        "sl_dist": sl_distance,
                                        "initial_sl": sl_price,
                                        "cur_sl_a": sl_price,
                                        "cur_sl_b": sl_price,
                                        "tp1_price": tp1_p,
                                        "tp2_price": tp2_p,
                                        "total_lot": lot_size,
                                        "lot_a": lot_a,
                                        "lot_b": lot_b,
                                        "risk_amount": risk_amount,
                                        "confidence": confidence,
                                        "pos_a_open": True,
                                        "pos_b_open": True,
                                        "tp1_hit": False,
                                        "tp2_hit": False,
                                        "pnl_net_accum": 0.0,
                                    }
                                    daily_entries_count += 1

        return self._compute_metrics(trades, np.array(equity_curve), timestamps_list, weekly_drawdowns, circuit_breaker_trips)


# Test 2026
engine = EnhancedEngine(
    initial_equity=10_000.0,
    base_risk_fraction=0.01,
    confidence_tau=0.34,
    prime_session_only=True,
    post_loss_cooldown_bars=3,
    enable_defensive_scaling=True,
    uncapped_runner=False,
)

res = engine.run(df=df, predictions=preds, atr_values=atr, start_date="2026-01-01", end_date="2026-08-31 23:59:59")
print(f"2026 Enhanced Results:")
print(f"Total Trades: {res.total_trades}")
print(f"Win Rate: {res.win_rate*100:.1f}%")
print(f"Net PnL: ${res.net_pnl:,.2f} ({res.net_pnl/10000*100:.2f}%)")
peak = np.maximum.accumulate(res.equity_curve)
max_dd = np.min((res.equity_curve - peak) / peak) * 100
print(f"Max DD: {max_dd:.2f}%, PF: {res.net_profit_factor:.2f}")

# Monthly Breakdown
tdf = pd.DataFrame([{
    'month': pd.to_datetime(t.entry_timestamp).strftime('%Y-%m'),
    'pnl': t.pnl_net,
    'win': 1 if t.pnl_net > 0 else 0
} for t in res.trades])

print("\n--- MONTHLY BREAKDOWN 2026 ---")
cur_eq = 10000.0
for m in [f"2026-{i:02d}" for i in range(1, 9)]:
    sub = tdf[tdf['month'] == m] if len(tdf) > 0 else pd.DataFrame()
    if len(sub) > 0:
        sw = sub['win'].sum()
        spnl = sub['pnl'].sum()
        spct = spnl / cur_eq * 100
        cur_eq += spnl
        print(f"{m} | Trades: {len(sub):2d} | Win: {sw:2d} ({sw/len(sub)*100:4.1f}%) | PnL: ${spnl:+7.2f} ({spct:+5.2f}%)")
    else:
        print(f"{m} | Trades:  0 | Win:  0 ( 0.0%) | PnL:   $0.00 (+0.00%)")
