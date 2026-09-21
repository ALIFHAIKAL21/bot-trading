"""Interactive Streamlit dashboard for multi-model quant research, evaluation, and paper trading."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repository root is in sys.path for Streamlit Cloud and subfolder execution
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.service.db import Database
from src.utils.config import load_config
from src.utils.security import determine_execution_mode


# Page setup
st.set_page_config(
    page_title="Multi-Model Quant Trading System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .badge-nogo {
        background-color: #8B0000;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-paper {
        background-color: #1b5e20;
        color: #e8f5e9;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-testnet {
        background-color: #e65100;
        color: #fff3e0;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-live {
        background-color: #b71c1c;
        color: #ffebee;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load config and database anchored to root_dir
config_path = root_dir / "config" / "config.yaml"
cfg = load_config(str(config_path) if config_path.exists() else "config/config.yaml")
db_path = root_dir / cfg.service.db_path
db = Database(str(db_path))
reports_dir = root_dir / "reports"
cache_dir = root_dir / "data" / "cache"

# Determine mode and safety state using canonical security helper
mode_name = determine_execution_mode(cfg)



import threading

@st.cache_resource
def start_embedded_background_scalper():
    """Run 5-minute scalper daemon in a background thread inside Streamlit server."""
    from scripts.test_scalp_5m import run_scalp_live
    t = threading.Thread(target=run_scalp_live, kwargs={"symbol": "BTC/USDT", "poll_interval": 15}, daemon=True)
    t.start()
    return t

# Sidebar navigation & Status
st.sidebar.title("⚡ Quant Research MVP")
st.sidebar.caption(f"Environment: **{mode_name}** | Symbol: **{cfg.market.symbols[0]}**")

if mode_name == "LIVE":
    st.sidebar.error("🔴 MODE: LIVE TRADING (REAL FUNDS)")
elif mode_name == "TESTNET":
    st.sidebar.warning("🟡 MODE: BINANCE TESTNET (SANDBOX)")
else:
    st.sidebar.success("🟢 MODE: SIMULATED PAPER TRADING")

# Institutional Readiness Warning in Sidebar
st.sidebar.markdown(
    """
    <div style='margin-top: 10px; margin-bottom: 15px;'>
        <span class='badge-nogo'>VERDICT: NO-GO FOR LIVE</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Sealed evaluation on locked test set demonstrated lack of statistical edge after costs.")

# 24/7 Cloud Background Scalper Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 24/7 Cloud Scalper")
auto_scalp_env = os.getenv("STREAMLIT_AUTO_SCALP", "false").lower() == "true"
auto_scalp = st.sidebar.checkbox("Run 5m Scalper in Background", value=auto_scalp_env, help="Activates automated 5m trading listener inside Streamlit server")
if auto_scalp:
    thread = start_embedded_background_scalper()
    st.sidebar.success("🟢 24/7 Scalper: ACTIVE")
else:
    st.sidebar.info("⏸️ Scalper: STANDBY")

st.sidebar.markdown("---")

selected_tab = st.sidebar.radio(
    "Navigation",
    [
        "📊 Evaluation & Baselines",
        "🤖 Models & Adaptivity",
        "📈 Market Data & Features",
        "💼 Live Paper Trading",
        "⚙️ Config & Audit",
    ],
)


# Top Banner on Every Page
st.markdown(
    f"""
    <div style='display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; background: rgba(255, 255, 255, 0.03); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 20px;'>
        <div>
            <span style='font-size: 18px; font-weight: 600;'>Multi-Model Adaptive Trading System</span>
            <span style='margin-left: 12px; font-size: 13px; color: #888;'>Timeframe: {cfg.market.timeframe} | Target Horizon: {cfg.market.horizon} bars</span>
        </div>
        <div style='display: flex; gap: 10px; align-items: center;'>
            <span class='badge-{mode_name.lower()}'>MODE: {mode_name}</span>
            <span class='badge-nogo'>NO-GO FOR LIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# TAB 1: EVALUATION & BASELINES
# -------------------------------------------------------------
if selected_tab == "📊 Evaluation & Baselines":
    st.header("📊 Out-of-Sample Evaluation & Institutional Benchmarks")

    final_report_file = reports_dir / "final_report.json"
    sweep_file = reports_dir / "turnover_controls_sweep.json"
    equity_file = reports_dir / "equity_curves.csv"

    if final_report_file.exists():
        with open(final_report_file, "r") as f:
            final_data = json.load(f)

        dataset_meta = final_data.get("dataset", {})
        strategies = final_data.get("strategies", {})
        model_b = strategies.get("primary_model_b", {})
        bnh = strategies.get("buy_and_hold_benchmark", {})
        shuffled = strategies.get("time_shuffled_baseline", {})
        mc = strategies.get("monte_carlo_random_baseline", {})

        st.info(
            f"**Sealed Out-of-Sample Holdout Period:** `{dataset_meta.get('start_time')}` to `{dataset_meta.get('end_time')}` "
            f"({dataset_meta.get('n_bars', 0):,} bars / ~3.6 months). Holdout quarantine strictly enforced."
        )

        # Executive Metrics Cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Strategy Return", f"{model_b.get('cum_return_pct', 0.0):.2f}%", delta=f"{model_b.get('cum_return_pct', 0.0) - bnh.get('cum_return_pct', 0.0):.2f}% vs BTC")
        col2.metric("Strategy Sharpe", f"{model_b.get('sharpe_ratio', 0.0):.2f}", delta="Cost-drag negative")
        col3.metric("Max Drawdown", f"{model_b.get('max_drawdown', 0.0)*100:.2f}%", delta=f"vs BTC {bnh.get('max_drawdown', 0.0)*100:.1f}%")
        col4.metric("Total Trades", f"{model_b.get('total_trades', 0)}")
        col5.metric("Annual Turnover", f"{model_b.get('annual_turnover', 0.0):.2f}x")
        col6.metric("Deflated Sharpe (DSR)", f"{model_b.get('deflated_sharpe_ratio', 0.0):.3f}")

        # Comparison Table
        st.subheader("Performance vs Baselines (Locked Test)")
        bench_rows = [
            {
                "Strategy / Baseline": "Primary Model B + Controls",
                "Return (%)": f"{model_b.get('cum_return_pct', 0.0):.2f}%",
                "Sharpe": f"{model_b.get('sharpe_ratio', 0.0):.2f}",
                "Sortino": f"{model_b.get('sortino_ratio', 0.0):.2f}",
                "Max DD": f"{model_b.get('max_drawdown', 0.0)*100:.2f}%",
                "Win Rate": f"{model_b.get('win_rate', 0.0)*100:.1f}%",
                "Profit Factor": f"{model_b.get('profit_factor', 0.0):.2f}",
                "Trades": str(model_b.get("total_trades", 0)),
                "Turnover": f"{model_b.get('total_turnover', 0.0):.2f}x",
            },
            {
                "Strategy / Baseline": "Buy & Hold (BTC/USDT)",
                "Return (%)": f"{bnh.get('cum_return_pct', 0.0):.2f}%",
                "Sharpe": f"{bnh.get('sharpe_ratio', 0.0):.2f}",
                "Sortino": f"{bnh.get('sortino_ratio', 0.0):.2f}",
                "Max DD": f"{bnh.get('max_drawdown', 0.0)*100:.2f}%",
                "Win Rate": f"{bnh.get('win_rate', 0.0)*100:.1f}%",
                "Profit Factor": f"{bnh.get('profit_factor', 0.0):.2f}",
                "Trades": "1",
                "Turnover": f"{bnh.get('annual_turnover', 0.0):.2f}x",
            },
            {
                "Strategy / Baseline": "Time-Shuffled Signal",
                "Return (%)": f"{shuffled.get('cum_return_pct', 0.0):.2f}%",
                "Sharpe": f"{shuffled.get('sharpe_ratio', 0.0):.2f}",
                "Sortino": f"{shuffled.get('sortino_ratio', 0.0):.2f}",
                "Max DD": f"{shuffled.get('max_drawdown', 0.0)*100:.2f}%",
                "Win Rate": f"{shuffled.get('win_rate', 0.0)*100:.1f}%",
                "Profit Factor": f"{shuffled.get('profit_factor', 0.0):.2f}",
                "Trades": "N/A",
                "Turnover": f"{shuffled.get('annual_turnover', 0.0):.2f}x",
            },
            {
                "Strategy / Baseline": "Monte Carlo Random (200 runs)",
                "Return (%)": f"{mc.get('mc_cagr_mean', 0.0)*100:.1f}%",
                "Sharpe": f"{mc.get('mc_sharpe_mean', 0.0):.2f}",
                "Sortino": "N/A",
                "Max DD": "N/A",
                "Win Rate": "N/A",
                "Profit Factor": "N/A",
                "Trades": "Matched",
                "Turnover": "Matched",
            },
        ]
        st.dataframe(pd.DataFrame(bench_rows), width="stretch")

        # Two columns: Statistical CIs & Decision Breakdown
        c_left, c_right = st.columns(2)

        with c_left:
            st.subheader("Statistical Tests & Monte Carlo Luck Bounds")
            st.markdown(
                f"""
                - **95% Bootstrap Confidence Interval on Sharpe:** `[{model_b.get('sharpe_95ci_low', 0.0):.2f}, {model_b.get('sharpe_95ci_high', 0.0):.2f}]`
                - **Deflated Sharpe Ratio (DSR):** `{model_b.get('deflated_sharpe_ratio', 0.0):.3f}` (Adjusted for selection bias across 15 historical trials)
                - **Monte Carlo 95th Percentile Luck Sharpe:** `{mc.get('mc_sharpe_95th', 0.0):.2f}`
                - **Institutional Verdict:** **NO-GO**. The strategy successfully protected capital (Max DD of 0.18% vs BTC 13.38%), but was unable to overcome transaction frictions (10 bps taker + 5 bps slippage) to produce positive net alpha during this trending regime.
                """
            )

        with c_right:
            st.subheader("Risk Engine Decision Breakdown (Locked Test)")
            reasons = model_b.get("decision_reasons", {})
            if reasons:
                df_reasons = pd.DataFrame(
                    list(reasons.items()), columns=["Decision Reason", "Bar Count"]
                ).sort_values("Bar Count", ascending=True)
                fig_reasons = px.bar(
                    df_reasons,
                    x="Bar Count",
                    y="Decision Reason",
                    orientation="h",
                    title="Audit of 2,628 Bar Decisions",
                )
                fig_reasons.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_reasons, width="stretch")

        # Validation Sweep Results
        if sweep_file.exists():
            st.subheader("Turnover Control Sweep: Validation Set Impact (Before vs After)")
            with open(sweep_file, "r") as f:
                sweep_data = json.load(f)
            h1 = sweep_data.get("1h", {})
            comp = h1.get("comparison", {})
            raw = h1.get("raw", {})
            ctl = h1.get("controlled", {})

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Trade Reduction", f"-{comp.get('trade_reduction_pct', 0.0):.1f}%", f"{raw.get('trades')} -> {ctl.get('trades')} trades")
            col_s2.metric("Turnover Reduction", f"-{comp.get('turnover_reduction_pct', 0.0):.1f}%", f"{raw.get('annualized_turnover'):.1f}x -> {ctl.get('annualized_turnover'):.1f}x")
            col_s3.metric("Validation Net Sharpe", f"{ctl.get('sharpe_ratio', 0.0):.2f}", f"+{comp.get('sharpe_improvement', 0.0):.2f}")
            col_s4.metric("Validation Max DD", f"{ctl.get('max_drawdown_pct', 0.0):.2f}%", f"Halved from {raw.get('max_drawdown_pct', 0.0):.2f}%")

    else:
        st.warning("Locked holdout evaluation report not found. Run `python scripts/final_report.py`.")


# -------------------------------------------------------------
# TAB 2: MODELS & ADAPTIVITY
# -------------------------------------------------------------
elif selected_tab == "🤖 Models & Adaptivity":
    st.header("🤖 Specialized Model Diagnostics & Adaptivity Layer")

    feat_imp_file = reports_dir / "feature_importances.json"
    hmm_file = reports_dir / "hmm_report.json"
    curves_file = reports_dir / "model_c_500k_curves.json"
    ablation_file = reports_dir / "adaptivity_ablation.json"

    # Online Adaptivity & Health Overview
    st.subheader("Component Health & Degraded Flags (F6, A4)")
    health_cols = st.columns(5)
    health_cols[0].metric("Model A (Chronos)", "Active", "Uncertainty / Skewness")
    health_cols[1].metric("Model B (LightGBM)", "Active", "Split Gain Calibrated")
    health_cols[2].metric("Model C (Deep Seq)", "Active", "GroupNorm Regularized")
    health_cols[3].metric("Model D (Sentiment)", "Degraded (RSS)", "Safe Fallback: 0.0")
    health_cols[4].metric("Model E (HMM)", "Active", "3-State Volatility Sorted")

    # Adaptivity Ablation Results
    if ablation_file.exists():
        st.subheader("Adaptivity Module Ablation (Validation Period)")
        with open(ablation_file, "r") as f:
            abl_data = json.load(f)
        abl_df = pd.DataFrame(abl_data).T
        st.dataframe(abl_df, width="stretch")
        st.caption("Quant Finding: Dynamic Hedge reweighting shrank weights toward uninformative 0.5 prior, attenuating signals into deadband. Deployed in **MONITOR & SHADOW MODE ONLY**.")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("Model B: Top Predictive Feature Importances")
        if feat_imp_file.exists():
            with open(feat_imp_file, "r") as f:
                feat_imp = json.load(f)
            top_feats = dict(list(feat_imp.items())[:12])
            fig_imp = px.bar(
                x=list(top_feats.values()),
                y=list(top_feats.keys()),
                orientation="h",
                labels={"x": "Importance (Gain)", "y": "Feature"},
                title="Top 12 Whitelisted Alpha Features",
            )
            fig_imp.update_layout(yaxis={"autorange": "reversed"}, height=400)
            st.plotly_chart(fig_imp, width="stretch")

    with col_m2:
        st.subheader("Model E: HMM Regime Separation (F6)")
        if hmm_file.exists():
            with open(hmm_file, "r") as f:
                hmm_data = json.load(f)
            occ = hmm_data.get("state_occupancy_pct", {})
            vols = hmm_data.get("state_volatilities", [])

            st.write(
                f"- **State 0 (Low Vol / Trend):** {occ.get('regime_p0', 0):.1f}% occupancy (Vol: {vols[0]:.4f})\n"
                f"- **State 1 (Medium Vol / Normal):** {occ.get('regime_p1', 0):.1f}% occupancy (Vol: {vols[1]:.4f})\n"
                f"- **State 2 (High Vol / Crisis):** {occ.get('regime_p2', 0):.1f}% occupancy (Vol: {vols[2]:.4f})"
            )

            # Transition Matrix
            t_matrix = hmm_data.get("transition_matrix", [])
            if t_matrix:
                fig_hm = px.imshow(
                    t_matrix,
                    labels=dict(x="To Regime", y="From Regime", color="Probability"),
                    x=["Low Vol (0)", "Med Vol (1)", "High Vol (2)"],
                    y=["Low Vol (0)", "Med Vol (1)", "High Vol (2)"],
                    text_auto=".2f",
                    title="HMM Regime Transition Probabilities",
                )
                fig_hm.update_layout(height=280)
                st.plotly_chart(fig_hm, width="stretch")


# -------------------------------------------------------------
# TAB 3: MARKET DATA & FEATURES
# -------------------------------------------------------------
elif selected_tab == "📈 Market Data & Features":
    st.header("📈 Causal Market Data & Features Explorer")
    sym = st.selectbox("Select Asset", cfg.market.symbols)
    clean_sym = sym.replace("/", "_")
    parquet_path = cache_dir / f"{clean_sym}_1h.parquet"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        st.write(f"Loaded **{len(df):,}** 1-hour candles from `{parquet_path}`")

        recent_df = df.iloc[-300:]
        fig_candle = go.Figure(
            data=[
                go.Candlestick(
                    x=recent_df.index,
                    open=recent_df["open"],
                    high=recent_df["high"],
                    low=recent_df["low"],
                    close=recent_df["close"],
                    name="OHLC",
                )
            ]
        )
        fig_candle.update_layout(xaxis_rangeslider_visible=False, height=450, title=f"{sym} Recent Price Action")
        st.plotly_chart(fig_candle, width="stretch")

        st.subheader("Data Cleanliness & Provenance (F1, F3)")
        data_quality_file = reports_dir / "data_quality.json"
        if data_quality_file.exists():
            with open(data_quality_file, "r") as f:
                dq = json.load(f)
            st.json(dq)
    else:
        st.warning(f"No cached data for {sym}. Run `python scripts/download_data.py`.")


# -------------------------------------------------------------
# TAB 4: LIVE PAPER TRADING
# -------------------------------------------------------------
elif selected_tab == "💼 Live Paper Trading":
    st.header("💼 Live Paper Trading & Real-Time Scalper")

    # Top Action Bar: Refresh button & Live Status
    c_btn, c_stat = st.columns([1, 4])
    with c_btn:
        if st.button("🔄 Refresh Data", width="stretch"):
            st.rerun()

    # Query database
    history = db.get_portfolio_history(limit=500)
    signals = db.get_latest_signals(limit=25)
    orders = db.get_recent_orders(limit=25)

    current_equity = 10000.0
    current_cash = 10000.0
    if history:
        current_equity = float(history[-1].get("equity", 10000.0))
        current_cash = float(history[-1].get("cash", 10000.0))

    # Live status description
    with c_stat:
        if signals:
            last_sig = signals[0]
            raw_data = last_sig.get("raw_data")
            if isinstance(raw_data, str):
                try:
                    raw_dict = json.loads(raw_data)
                except Exception:
                    raw_dict = {}
            else:
                raw_dict = raw_data if isinstance(raw_data, dict) else {}

            action = raw_dict.get("action", "HOLD")
            reason = raw_dict.get("decision_reason", "monitoring")
            p_val = last_sig.get("prob_long", 0.5)
            ts_val = last_sig.get("bar_timestamp", "")[:19]
            st.info(
                f"🟢 **Scalper Active & Listening** | Last Evaluated Bar: `{ts_val} UTC` | "
                f"AI P(Long): **`{p_val:.3f}`** | Action: **`{action}`** | Reason: `{reason}`"
            )
        else:
            st.info("🟢 **Scalper Initialized** | Waiting for first closed candle...")

    # Summary balances
    pos_val = current_equity - current_cash
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("Simulated Equity", f"${current_equity:,.2f}", f"{(current_equity/10000.0 - 1.0)*100:.2f}%")
    b_col2.metric("Simulated Cash", f"${current_cash:,.2f}")
    b_col3.metric("Open Position Value", f"${pos_val:,.2f}", f"{(pos_val/current_equity)*100:.1f}% Allocation")
    b_col4.metric("Completed Trades", f"{len(orders)}")

    # Tables FIRST: Executed Orders & Recent Signals side-by-side
    col_ord, col_sig = st.columns(2)

    with col_ord:
        st.subheader("📋 Executed Orders (BUY / SELL)")
        if orders:
            ord_rows = []
            for o in orders:
                ord_rows.append(
                    {
                        "Timestamp": o.get("bar_timestamp", "")[:19],
                        "Side": o.get("side", ""),
                        "Qty (BTC)": f"{o.get('qty', 0.0):.4f}",
                        "Fill Price": f"${o.get('fill_price', 0.0):,.2f}",
                        "Fee": f"${o.get('fee', 0.0):.2f}",
                        "Target Wt": f"{o.get('target_weight', 0.0):.2f}",
                    }
                )
            st.dataframe(pd.DataFrame(ord_rows), width="stretch", height=280)
        else:
            st.info("No orders executed yet. Orders will appear here when a signal triggers entry.")

    with col_sig:
        st.subheader("📡 Recent Evaluated Signals (AI Predictions)")
        if signals:
            sig_rows = []
            for s in signals:
                raw_data = s.get("raw_data")
                if isinstance(raw_data, str):
                    try:
                        raw = json.loads(raw_data)
                    except Exception:
                        raw = {}
                else:
                    raw = raw_data if isinstance(raw_data, dict) else {}

                sig_rows.append(
                    {
                        "Timestamp": s.get("bar_timestamp", "")[:19],
                        "P(Long)": f"{s.get('prob_long', 0.5):.3f}",
                        "Action": raw.get("action", "HOLD"),
                        "Target Wt": f"{raw.get('target_size', 0.0):.2f}",
                        "Decision Reason": raw.get("decision_reason", "N/A"),
                        "Expected Edge": f"{raw.get('expected_edge', 0.0):.4f}",
                    }
                )
            st.dataframe(pd.DataFrame(sig_rows), width="stretch", height=280)
        else:
            st.info("No signals evaluated yet in current paper database.")

    # Portfolio Equity Curve with Trade Markers
    if history:
        st.subheader("📈 Paper Trading Cumulative Equity Curve")
        hist_df = pd.DataFrame(history)
        fig_hist = px.line(
            hist_df,
            x="timestamp",
            y="equity",
            title=f"Portfolio Simulated Equity (${current_equity:,.2f})",
            labels={"equity": "Equity ($)", "timestamp": "Time"},
        )
        fig_hist.update_traces(line=dict(color="#00bcd4", width=2.5))
        fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist, width="stretch")



# -------------------------------------------------------------
# TAB 5: CONFIG & AUDIT
# -------------------------------------------------------------
elif selected_tab == "⚙️ Config & Audit":
    st.header("⚙️ Configuration & System Integrity Audit")

    manifest_file = reports_dir / "latest_run_manifest.json"
    if manifest_file.exists():
        st.subheader("Latest Run Manifest (F14)")
        with open(manifest_file, "r") as f:
            manifest = json.load(f)
        st.json(manifest)

    st.subheader("Active YAML Configuration")
    config_file = root_dir / "config" / "config.yaml"
    if config_file.exists():

        with open(config_file, "r") as f:
            raw_yaml = f.read()
        st.code(raw_yaml, language="yaml")
