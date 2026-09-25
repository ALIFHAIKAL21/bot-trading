import sys, pathlib
import pandas as pd
import numpy as np

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
sys.path.insert(0, str(project_root))
sys.path.insert(0, r'c:\Ngoding\bot_trading')

from scripts.run_optimized_monthly_backtest_2026 import tdf_opt, res_opt

print("=== DEEP TRADER DIAGNOSTIC AUDIT ===")
tdf = tdf_opt.copy()

wins = tdf[tdf['pnl_net'] > 0]
losses = tdf[tdf['pnl_net'] < 0]
be = tdf[tdf['pnl_net'] == 0]

avg_win = wins['pnl_net'].mean()
avg_loss = abs(losses['pnl_net'].mean())
payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

win_rate = len(wins) / len(tdf)
loss_rate = len(losses) / len(tdf)
expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

print(f"Total Trades: {len(tdf)}")
print(f"Wins: {len(wins)} ({win_rate*100:.1f}%), Losses: {len(losses)} ({loss_rate*100:.1f}%), BE: {len(be)}")
print(f"Average Win: ${avg_win:.2f}")
print(f"Average Loss: ${avg_loss:.2f}")
print(f"Payoff Ratio (Reward : Risk achieved): {payoff_ratio:.2f} : 1")
print(f"Mathematical Expectancy per trade: ${expectancy:.2f}")

# Breakeven Win Rate formula: BE_WR = 1 / (1 + Payoff_Ratio)
be_wr = (1.0 / (1.0 + payoff_ratio)) * 100
print(f"Breakeven Win Rate Required: {be_wr:.1f}% vs Actual Win Rate: {win_rate*100:.1f}%")

# Let's inspect R-Multiples
print(f"\nR-Multiple Stats:")
print(f"Wins Avg R: {wins['r_mult'].mean():.3f} (Max: {wins['r_mult'].max():.3f}, Min: {wins['r_mult'].min():.3f})")
print(f"Losses Avg R: {losses['r_mult'].mean():.3f} (Max: {losses['r_mult'].max():.3f}, Min: {losses['r_mult'].min():.3f})")

# Look at trade categories
tp1_only = tdf[(tdf['tp1_hit'] == True) & (tdf['tp2_hit'] == False)]
tp1_and_tp2 = tdf[(tdf['tp1_hit'] == True) & (tdf['tp2_hit'] == True)]
neither = tdf[(tdf['tp1_hit'] == False) & (tdf['tp2_hit'] == False)]

print(f"\nTrade Categories:")
print(f"TP1 only (+0.5R banked, runner stopped at BE/trail): {len(tp1_only)} trades (Avg PnL: ${tp1_only['pnl_net'].mean():.2f}, Avg R: {tp1_only['r_mult'].mean():.2f})")
print(f"TP1 + TP2 (Full +1.5R): {len(tp1_and_tp2)} trades (Avg PnL: ${tp1_and_tp2['pnl_net'].mean():.2f}, Avg R: {tp1_and_tp2['r_mult'].mean():.2f})")
print(f"Full Losses (Initial SL hit): {len(neither)} trades (Avg PnL: ${neither['pnl_net'].mean():.2f}, Avg R: {neither['r_mult'].mean():.2f})")

# Look at friction impact
total_friction = sum(t.friction_cost for t in res_opt.trades)
print(f"\nTotal Friction (Spread + Slippage + Commission paid to broker): ${total_friction:,.2f}")
print(f"Net PnL: ${tdf['pnl_net'].sum():,.2f}")
print(f"Gross PnL (before broker took fees): ${tdf['pnl_net'].sum() + total_friction:,.2f}")

# Look at Directional Accuracy of MOMENT model:
# Did the market move in predicted direction?
print(f"\nDirectional Accuracy (did price reach at least +1.0R?):")
reached_1r = len(tp1_only) + len(tp1_and_tp2)
print(f"Reached +1.0R: {reached_1r} / {len(tdf)} = {reached_1r / len(tdf) * 100:.1f}%")

# By month breakdown of Gross vs Net
for m, g in tdf.groupby('month'):
    fric_m = sum(t.friction_cost for t in res_opt.trades if pd.Timestamp(t.entry_timestamp).strftime('%Y-%m') == m)
    net_m = g['pnl_net'].sum()
    gross_m = net_m + fric_m
    print(f"{m} | Trades: {len(g):2d} | Gross: ${gross_m:8.2f} | Friction: ${fric_m:7.2f} | Net: ${net_m:8.2f} | WR: {(g['pnl_net']>0).mean()*100:4.1f}%")
