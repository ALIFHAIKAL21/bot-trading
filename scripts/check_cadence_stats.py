from fast_session_picker import run_session_picker
import pandas as pd

for tau in [0.32, 0.33, 0.34, 0.35]:
    res = run_session_picker(n_windows=5, tau_min=tau, sizing="fixed_002")
    tdf = res["tdf"]
    cadence = tdf.groupby("date").size()
    c45 = cadence[cadence.isin([4, 5])].count()
    total_days = cadence.count()
    pct = c45 / total_days * 100
    print(f"Tau {tau:.2f} | Total Tr: {res['trades']} | Avg/day: {res['trades_per_day']:.2f} | Days with 4-5 entries: {c45}/{total_days} ({pct:.1f}%) | WR: {res['win_rate']:.1f}% | PnL: ${res['net_pnl']:.2f} | Max DD: {res['max_dd']:.1f}%")
