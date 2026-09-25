"""
XAU_DEEP_SNIPER - Multi-Year Backtest Audit (Jan - Agu across 2022 - 2026)
==========================================================================
Evaluasi bar-by-bar periode Januari - Agustus pada seluruh tahun historis
tersedia (2022 s/d 2026) menggunakan arsitektur Institutional Sniper (Tier 3).
"""

import sys, os, pathlib, json
import numpy as np
import pandas as pd

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, r'c:\Ngoding\bot_trading')

try:
    from src.validation.backtest_engine import RealisticBacktestEngine
except ImportError:
    from validation.backtest_engine import RealisticBacktestEngine

df_path = project_root / "data" / "processed" / "xauusd_m30_labeled.parquet"
preds_path = pathlib.Path(r'c:\Ngoding\bot_trading\scratch\full_5year_predictions.npy')
reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

print("Loading 5-year dataset and cached predictions...")
df = pd.read_parquet(df_path)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
preds = np.load(preds_path)

print(f"Dataset bars: {len(df):,}, Predictions: {len(preds):,}")

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

years = [2022, 2023, 2024, 2025, 2026]
annual_summaries = []
all_monthly_records = []

for year in years:
    st_date = f"{year}-01-01"
    end_date = f"{year}-08-31 23:59:59"
    
    engine = RealisticBacktestEngine(
        initial_equity=10_000.0,
        risk_fraction=0.01,
        confidence_tau=0.34,
        sl_atr_multiplier=1.5,
        spread_pip=0.75,
        commission_per_lot=3.50,
        slippage_pip=0.3,
        time_barrier_bars=16,
        contract_size=100.0,
        use_partial_tp=True,
        pos_a_pct=0.40,
        pos_b_pct=0.60,
        tp1_r_ratio=1.0,
        tp2_r_ratio=2.5,
        be_trigger_r=1.0,
        trail_after_r=1.5,
        trail_distance_r=0.8,
        max_daily_entries=10,
        quarantine_london_open=True,
        use_structural_filter=True
    )
    
    res = engine.run(
        df=df,
        predictions=preds,
        atr_values=atr,
        start_date=st_date,
        end_date=end_date
    )
    
    trades = []
    for t in res.trades:
        trades.append({
            'id': t.trade_id,
            'direction': t.direction,
            'entry_ts': t.entry_timestamp,
            'exit_ts': t.exit_timestamp,
            'pnl_net': t.pnl_net,
            'r_mult': t.r_multiple,
            'exit_reason': t.exit_reason,
            'tp1_hit': t.tp1_hit,
            'tp2_hit': t.tp2_hit,
        })
    tdf = pd.DataFrame(trades)
    
    n_trades = len(tdf)
    if n_trades > 0:
        tdf['entry_ts'] = pd.to_datetime(tdf['entry_ts'])
        tdf['month'] = tdf['entry_ts'].dt.strftime('%Y-%m')
        
        n_wins = (tdf['pnl_net'] > 0).sum()
        wr = (n_wins / n_trades) * 100
        net_pnl = tdf['pnl_net'].sum()
        ret_pct = (net_pnl / 10_000.0) * 100
        
        peak = np.maximum.accumulate(res.equity_curve)
        max_dd = np.min((res.equity_curve - peak) / peak) * 100
        
        # Monthly breakdown
        cur_eq = 10_000.0
        months_in_year = [f"{year}-{m:02d}" for m in range(1, 9)]
        green_m = 0
        for m in months_in_year:
            sub = tdf[tdf['month'] == m]
            sub_n = len(sub)
            if sub_n > 0:
                sub_w = (sub['pnl_net'] > 0).sum()
                sub_wr = (sub_w / sub_n) * 100
                sub_pnl = sub['pnl_net'].sum()
                sub_pct = (sub_pnl / cur_eq) * 100
                cur_eq += sub_pnl
                if sub_pnl >= 0:
                    green_m += 1
                all_monthly_records.append({
                    'Year': year,
                    'Month': m,
                    'Trades': sub_n,
                    'Wins': sub_w,
                    'WinRatePct': round(sub_wr, 1),
                    'NetPnL': round(sub_pnl, 2),
                    'ProfitPct': round(sub_pct, 2)
                })
            else:
                all_monthly_records.append({
                    'Year': year,
                    'Month': m,
                    'Trades': 0,
                    'Wins': 0,
                    'WinRatePct': 0.0,
                    'NetPnL': 0.0,
                    'ProfitPct': 0.0
                })
        
        annual_summaries.append({
            'Year': f"Jan - Agu {year}",
            'Trades': n_trades,
            'Wins': n_wins,
            'Losses': n_trades - n_wins,
            'WinRatePct': round(wr, 1),
            'NetPnL': round(net_pnl, 2),
            'ReturnPct': round(ret_pct, 2),
            'MaxDrawdownPct': round(max_dd, 2),
            'ProfitFactor': round(res.net_profit_factor, 2),
            'GreenMonths': f"{green_m}/8"
        })
    else:
        annual_summaries.append({
            'Year': f"Jan - Agu {year}",
            'Trades': 0,
            'Wins': 0,
            'Losses': 0,
            'WinRatePct': 0.0,
            'NetPnL': 0.0,
            'ReturnPct': 0.0,
            'MaxDrawdownPct': 0.0,
            'ProfitFactor': 0.0,
            'GreenMonths': "0/8"
        })

print("\n=== MULTI-YEAR ANNUAL SCORECARD (JAN - AGU 2022 - 2026) ===")
summary_df = pd.DataFrame(annual_summaries)
print(summary_df.to_string(index=False))

# Save audit JSON
audit_multiyear = {
    "engine": "RealisticBacktestEngine (Tier 3 Institutional Sniper)",
    "years_evaluated": years,
    "annual_summaries": annual_summaries,
    "monthly_records": all_monthly_records
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

json_path = reports_dir / "backtest_multiyear_2022_2026_audit.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(audit_multiyear, f, indent=2, cls=NpEncoder)

print(f"\nSaved multi-year audit JSON to {json_path}")
