"""
Analyze monthly performance for candidate setups
"""
import sys, pathlib
import numpy as np
import pandas as pd

from fast_session_picker import run_session_picker

candidates = [
    ("5_win_tau32", 5, 0.32, "fixed_002"),
    ("5_win_tau35", 5, 0.35, "fixed_002"),
    ("5_win_tau38", 5, 0.38, "fixed_002"),
    ("4_win_tau32", 4, 0.32, "fixed_002"),
    ("4_win_tau35", 4, 0.35, "fixed_002"),
    ("4_win_tau38", 4, 0.38, "fixed_002"),
]

for label, w, tau, sz in candidates:
    print(f"\n=======================================================")
    print(f"CONFIGURATION: {w} Windows/Day | Tau = {tau} | {sz}")
    print(f"=======================================================")
    res = run_session_picker(n_windows=w, tau_min=tau, sizing=sz)
    tdf = res['tdf']
    
    print(f"{'Month':>7} | {'Trades':>6} | {'Wins':>5} | {'WR%':>5} | {'Net PnL':>10} | {'Cum PnL':>10} | {'Balance':>10}")
    print("-" * 65)
    
    cum_pnl = 0.0
    for m, grp in tdf.groupby('month'):
        n_t = len(grp)
        n_w = grp['win'].sum()
        wr = n_w / n_t * 100
        pnl = grp['pnl'].sum()
        cum_pnl += pnl
        bal = 1000.0 + cum_pnl
        print(f"{m:>7} | {n_t:6d} | {n_w:5d} | {wr:5.1f}% | ${pnl:9.2f} | ${cum_pnl:9.2f} | ${bal:9.2f}")
        
    print(f"TOTAL: {res['trades']} Trades ({res['trades_per_day']:.2f}/day) | WR: {res['win_rate']:.1f}% | PF: {res['pf']:.2f} | Net PnL: ${res['net_pnl']:.2f} | Max DD: {res['max_dd']:.2f}%")
