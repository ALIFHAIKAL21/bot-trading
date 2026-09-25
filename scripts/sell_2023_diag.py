import sys, pathlib
import numpy as np
import pandas as pd

from sell_diagnostics import sells

loss_23 = sells[(sells['year'] == 2023) & (sells['win'] == 0)]
win_23 = sells[(sells['year'] == 2023) & (sells['win'] == 1)]

print(f"2023 Losing SELLs (N={len(loss_23)}):")
print(f"  h4_dist_ema200 mean: {loss_23['h4_dist_ema200'].mean():.2f}")
print(f"  h4_slope50 mean: {loss_23['h4_slope50'].mean():.2f}")
print(f"  h4_pct_range mean: {loss_23['h4_pct_range'].mean():.2f}")

print(f"\n2023 Winning SELLs (N={len(win_23)}):")
print(f"  h4_dist_ema200 mean: {win_23['h4_dist_ema200'].mean():.2f}")
print(f"  h4_slope50 mean: {win_23['h4_slope50'].mean():.2f}")
print(f"  h4_pct_range mean: {win_23['h4_pct_range'].mean():.2f}")
