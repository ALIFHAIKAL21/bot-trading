"""Interactive Streamlit dashboard for multi-model quant research, evaluation, and paper trading."""

import json
import os
import sys
from datetime import datetime, timezone
import time
from pathlib import Path

# Ensure repository root is in sys.path for Streamlit Cloud and subfolder execution
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from src.data.loader import MarketDataLoader
from src.service.bot_controller import BotController, run_interactive_replay
from src.service.db import Database
from src.utils.config import load_config
from src.utils.security import determine_execution_mode

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


def build_live_execution_candlestick_chart(symbol: str, timeframe: str, orders: list) -> go.Figure:
    """Build interactive candlestick chart overlaid with EMA and actual BUY/SELL trade executions."""
    try:
        loader = MarketDataLoader(cache_dir=str(root_dir / "data" / "cache"))
        try:
            df = loader.load_or_fetch(symbol, timeframe=timeframe, history_days=2 if timeframe in ("1m", "5m") else 14)
        except Exception:
            df = pd.DataFrame()

        if df.empty or len(df) < 5:
            fig = go.Figure()
            fig.update_layout(
                title=f"📈 Menunggu pembentukan lilin {symbol} ({timeframe})...",
                height=420,
                template="plotly_dark",
            )
            return fig

        # Take last 80 bars for clean, responsive, readable display
        df_slice = df.iloc[-80:].copy()
        df_slice["ema12"] = df_slice["close"].ewm(span=12, adjust=False).mean()
        df_slice["ema26"] = df_slice["close"].ewm(span=26, adjust=False).mean()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.78, 0.22],
        )

        # 1. Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_slice.index,
                open=df_slice["open"],
                high=df_slice["high"],
                low=df_slice["low"],
                close=df_slice["close"],
                name=f"{symbol} OHLC",
                increasing_line_color="#00E676",
                decreasing_line_color="#FF5252",
            ),
            row=1, col=1,
        )

        # 2. EMAs
        fig.add_trace(
            go.Scatter(
                x=df_slice.index, y=df_slice["ema12"],
                mode="lines",
                line=dict(color="#29B6F6", width=1.5),
                name="EMA 12",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_slice.index, y=df_slice["ema26"],
                mode="lines",
                line=dict(color="#FFA726", width=1.5),
                name="EMA 26",
            ),
            row=1, col=1,
        )

        # 3. Overlay Executed Orders (BUY = Green ▲, SELL = Red ▼)
        if orders:
            min_ts = df_slice.index.min()
            max_ts = df_slice.index.max()
            buy_x, buy_y, buy_txt = [], [], []
            sell_x, sell_y, sell_txt = [], [], []

            for o in orders:
                try:
                    o_sym = o.get("symbol", "")
                    if o_sym and o_sym.upper() not in (symbol.upper(), "XAU/USD" if "XAU" in symbol.upper() else "BTC/USDT"):
                        continue

                    o_ts = pd.to_datetime(o.get("bar_timestamp", ""), utc=True)
                    fill_px = float(o.get("fill_price", 0.0))
                    qty = float(o.get("qty", 0.0))
                    side = str(o.get("side", "")).upper()

                    if min_ts <= o_ts <= max_ts:
                        if side == "BUY":
                            buy_x.append(o_ts)
                            buy_y.append(fill_px)
                            buy_txt.append(f"BUY @ ${fill_px:,.2f}<br>Qty: {qty:.4f}")
                        elif side == "SELL":
                            sell_x.append(o_ts)
                            sell_y.append(fill_px)
                            sell_txt.append(f"SELL @ ${fill_px:,.2f}<br>Qty: {qty:.4f}")
                except Exception:
                    continue

            if buy_x:
                fig.add_trace(
                    go.Scatter(
                        x=buy_x, y=buy_y,
                        mode="markers+text",
                        marker=dict(symbol="triangle-up", size=16, color="#00E676", line=dict(width=1.5, color="#ffffff")),
                        text=["▲ BUY"] * len(buy_x),
                        textposition="bottom center",
                        textfont=dict(color="#00E676", size=11, family="sans-serif"),
                        name="Bot BUY Fill",
                        hovertext=buy_txt,
                        hoverinfo="text+x",
                    ),
                    row=1, col=1,
                )

            if sell_x:
                fig.add_trace(
                    go.Scatter(
                        x=sell_x, y=sell_y,
                        mode="markers+text",
                        marker=dict(symbol="triangle-down", size=16, color="#FF1744", line=dict(width=1.5, color="#ffffff")),
                        text=["▼ SELL"] * len(sell_x),
                        textposition="top center",
                        textfont=dict(color="#FF1744", size=11, family="sans-serif"),
                        name="Bot SELL Fill",
                        hovertext=sell_txt,
                        hoverinfo="text+x",
                    ),
                    row=1, col=1,
                )

        # 4. Volume Bars in Row 2
        colors = ["#00E676" if c >= o else "#FF5252" for c, o in zip(df_slice["close"], df_slice["open"])]
        fig.add_trace(
            go.Bar(
                x=df_slice.index,
                y=df_slice["volume"],
                name="Volume",
                marker_color=colors,
                showlegend=False,
            ),
            row=2, col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=520,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_rangeslider_visible=False,
            title=f"📈 Real-Time Price Action & AI Execution Markers: {symbol} ({timeframe})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Vol", row=2, col=1)
        return fig
    except Exception as chart_err:
        fig = go.Figure()
        fig.update_layout(
            title=f"📈 Candlestick Live: Menyiapkan data lilin {symbol} ({timeframe})...",
            height=420,
            template="plotly_dark",
        )
        return fig


def render_tradingview_widget(symbol: str, timeframe: str):
    """Embed responsive, live-streaming TradingView chart."""
    tv_interval = "1" if timeframe == "1m" else ("5" if timeframe == "5m" else "60")
    if "XAU" in symbol.upper() or "GOLD" in symbol.upper():
        tv_sym = "OANDA:XAUUSD"
    elif "ETH" in symbol.upper():
        tv_sym = "BINANCE:ETHUSDT"
    elif "SOL" in symbol.upper():
        tv_sym = "BINANCE:SOLUSDT"
    else:
        tv_sym = "BINANCE:BTCUSDT"

    html_code = f"""
    <div class="tradingview-widget-container" style="height:550px;width:100%;">
      <div id="tradingview_live" style="height:calc(100% - 32px);width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "autosize": true,
        "symbol": "{tv_sym}",
        "interval": "{tv_interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "id",
        "toolbar_bg": "#131722",
        "enable_publishing": false,
        "hide_top_toolbar": false,
        "allow_symbol_change": true,
        "save_image": false,
        "container_id": "tradingview_live"
      }}
      );
      </script>
    </div>
    """
    components.html(html_code, height=560)


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

# 24/7 Cloud Background Scalper Toggle & Controller
bot_ctrl = BotController()
bot_status = bot_ctrl.get_status()

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 24/7 Cloud Engine")
if bot_status["is_running"]:
    strat_tag = "⚡ Pro Sniper (Trader Kakap)" if bot_status.get("strategy_mode") == "pro_sniper" else "🛡️ Institusional"
    score_tag = f"\n- Radar Confluence: `{bot_status.get('last_score', 0)}/100`" if bot_status.get("strategy_mode") == "pro_sniper" else ""
    st.sidebar.success(
        f"🟢 **STATUS: AKTIF (24/7)**\n\n"
        f"- Engine: **`{strat_tag}`**\n"
        f"- Timeframe: `{bot_status['timeframe']}`\n"
        f"- Pair: `{bot_status['symbol']}`"
        f"{score_tag}\n"
        f"- Heartbeat: `{bot_status['last_heartbeat']}`"
    )
    if st.sidebar.button("⏹️ Hentikan Bot", key="side_stop_btn", use_container_width=True):
        bot_ctrl.stop()
        st.rerun()
else:
    st.sidebar.info("⏸️ **STATUS: STANDBY (Idle)**")
    side_strategy = st.sidebar.selectbox(
        "Pilih Karakter Strategi:",
        ["⚡ Pro Sniper (Trader Kakap)", "🛡️ Institusional (Konservatif)"],
        index=0,
        key="side_strat_select",
    )
    side_strat_code = "pro_sniper" if "Pro Sniper" in side_strategy else "institutional"
    side_tf = st.sidebar.selectbox(
        "Pilih Timeframe Bot:",
        ["5m (Scalping Standar)", "1m (Ultra-Fast Scalp)", "1h (Swing Trading)"],
        index=0,
        key="side_tf_select",
    )
    side_tf_code = side_tf.split()[0]
    if st.sidebar.button("▶️ Aktifkan 24/7 Bot", type="primary", key="side_start_btn", use_container_width=True):
        bot_ctrl.start(timeframe=side_tf_code, symbol=cfg.market.symbols[0], strategy_mode=side_strat_code)
        st.rerun()

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
    st.header("💼 Live Paper Trading & Interactive Simulation Hub")

    # -----------------------------------------------------------------
    # PUSAT KENDALI SIMULASI & BOT (CLICK-BASED UI)
    # -----------------------------------------------------------------
    with st.expander("🎛️ **PUSAT KENDALI SIMULASI & BOT (KLIK DISINI UNTUK KONTROL)**", expanded=True):
        st.markdown(
            "Pilih mode strategi dan jalankan simulasi instan atau aktifkan bot 24/7 di cloud server tanpa terminal!"
        )

        hub_col1, hub_col2 = st.columns([1, 1.2])

        with hub_col1:
            st.markdown("##### ⚙️ 1. Pilihan Strategi & Timeframe")
            strategy_selection = st.radio(
                "Pilih Karakter & Gaya Trading:",
                [
                    "⚡ Pro Institutional Sniper (Trader Kelas Kakap: Confluence 0-100, Trailing BE, Target WR ~70%, 10+ trade/hari)",
                    "🛡️ Institusional Akademik (Konservatif: Ketat Cost-Hurdle, Jarang Open Posisi)",
                ],
                index=0,
                key="hub_strat_radio",
            )
            selected_strat = "pro_sniper" if "Pro Institutional Sniper" in strategy_selection else "institutional"

            tf_selection = st.radio(
                "Pilih Kecepatan & Horizon:",
                [
                    "⚡ Scalping 1m (Ultra-Fast) — Lilin 1 menit, SL ketat 1.5%, hold 3-5 candle",
                    "🚀 Scalping 5m (Standar) — Lilin 5 menit, SL 2.0%, hold 10-30 menit",
                    "🌊 Swing 1h (Multi-Hour) — Lilin 1 jam, SL 3.0%, hold 3-12 jam",
                ],
                index=1,
                key="hub_tf_radio",
            )
            selected_tf = "1m" if "1m" in tf_selection else ("5m" if "5m" in tf_selection else "1h")
            selected_symbol = st.selectbox("Pilih Pasangan Aset:", cfg.market.symbols, key="hub_symbol_select")

        with hub_col2:
            st.markdown("##### 🚀 2. Jalankan Aksi")
            action_tab_sim, action_tab_live = st.tabs(["⚡ Jalankan Simulasi Instan", "🤖 Bot 24/7 Cloud (Nonstop)"])

            with action_tab_sim:
                st.caption("Replay lilin harga nyata terbaru untuk melihat order beli/jual dan return secara instan.")
                n_bars = st.slider("Jumlah Lilin untuk Simulasi:", min_value=50, max_value=500, value=200, step=50, key="hub_slider_bars")
                if st.button("⚡ Mulai Simulasi Replay Sekarang", type="primary", use_container_width=True, key="hub_btn_sim"):
                    with st.spinner(f"Sedang mengunduh data lilin nyata dan menjalankan simulasi {selected_tf} ({selected_strat}) pada {selected_symbol}..."):
                        sim_res = run_interactive_replay(timeframe=selected_tf, symbol=selected_symbol, n_bars=n_bars, strategy_mode=selected_strat)
                    if "error" in sim_res:
                        st.error(f"Gagal: {sim_res['error']}")
                    else:
                        wr_display = f"{sim_res.get('win_rate', 0.0):.1f}% ({sim_res.get('wins', 0)}W / {sim_res.get('losses', 0)}L)" if 'win_rate' in sim_res else "-"
                        st.success(
                            f"✅ **Simulasi Selesai!** Lilin: {sim_res['bars']} | "
                            f"Order: **{sim_res['trades_count']} transaksi** | "
                            f"Win Rate: **{wr_display}** | "
                            f"Return: **{sim_res['return_pct']:+.2f}%** | "
                            f"Saldo: **${sim_res['final_equity']:,.2f}**"
                        )
                        st.rerun()

            with action_tab_live:
                st.caption("Jalankan bot trading otomatis 24/7 di server cloud Streamlit (laptop bebas dimatikan).")
                b_ctrl = BotController()
                b_stat = b_ctrl.get_status()

                col_btn_start, col_btn_stop = st.columns(2)
                with col_btn_start:
                    if st.button("▶️ Aktifkan Bot 24/7", type="primary", use_container_width=True, disabled=b_stat["is_running"], key="hub_btn_start_live"):
                        b_ctrl.start(timeframe=selected_tf, symbol=selected_symbol, strategy_mode=selected_strat)
                        st.success(f"Bot 24/7 ({selected_strat}) berhasil diaktifkan pada timeframe {selected_tf}!")
                        st.rerun()
                with col_btn_stop:
                    if st.button("⏹️ Hentikan Bot", use_container_width=True, disabled=not b_stat["is_running"], key="hub_btn_stop_live"):
                        b_ctrl.stop()
                        st.warning("Bot 24/7 dihentikan.")
                        st.rerun()

                if b_stat["is_running"]:
                    strat_display = "⚡ Pro Sniper (Trader Kakap)" if b_stat.get("strategy_mode") == "pro_sniper" else "🛡️ Institusional Konservatif"
                    tp_val = b_stat.get("take_profit")
                    sl_val = b_stat.get("stop_loss")
                    tp_display = f"${tp_val:,.2f}" if tp_val else "-"
                    sl_display = f"${sl_val:,.2f}" if sl_val else "-"
                    st.success(
                        f"🟢 **BOT AKTIF BERJALAN 24/7 DI SERVER CLOUD**\n\n"
                        f"- Strategi: **`{strat_display}`**\n"
                        f"- Timeframe Aktif: **`{b_stat['timeframe']}`** | Pasangan: **`{b_stat['symbol']}`**\n"
                        f"- Confluence Radar Score: **`{b_stat.get('last_score', 0)}/100`**\n"
                        f"- Target TP: **`{tp_display}`** | Stop Loss: **`{sl_display}`**\n"
                        f"- Trailing Stage: **`{b_stat.get('trailing_stage', 'NONE')}`**\n"
                        f"- Heartbeat Server: `{b_stat['last_heartbeat']}`\n"
                        f"- Lilin Terakhir Dievaluasi: `{b_stat['last_bar']}`\n"
                        f"- Sinyal AI: P(Long) = `{b_stat['last_prob']:.3f}` | Aksi: `{b_stat['last_action']}`\n"
                        f"- Alasan Sinyal: `{b_stat['last_reason']}`\n"
                        f"- Status Loop: `{b_stat['status_msg']}`"
                    )
                    if b_stat["timeframe"] != selected_tf or b_stat.get("strategy_mode") != selected_strat:
                        st.info(f"💡 Pilihan kontrol: **{selected_tf}** ({selected_strat}) | Sedang jalan: **{b_stat['timeframe']}** ({b_stat.get('strategy_mode')}).")
                        if st.button(f"🔄 Beralih Sekarang ke Mode {selected_tf} ({selected_strat})", type="primary", use_container_width=True, key="btn_switch_tf"):
                            b_ctrl.switch_timeframe(new_timeframe=selected_tf, symbol=selected_symbol, strategy_mode=selected_strat)
                            st.success(f"Bot dialihkan ke {selected_tf} ({selected_strat})!")
                            st.rerun()
                else:
                    st.info("⏸️ **STATUS: STANDBY (Mati)** — Klik 'Aktifkan Bot 24/7' untuk mulai trading otomatis.")

    # Top Action Bar: Refresh, Auto-refresh, Reset & Live Status
    c_ref_btn, c_ref_toggle, c_reset_btn, c_stat = st.columns([1, 1.2, 1, 3])
    with c_ref_btn:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    with c_ref_toggle:
        auto_refresh = st.checkbox("⏱️ Auto-Refresh (10s)", value=b_stat["is_running"], help="Otomatis memperbarui metrik setiap 10 detik")
        if auto_refresh:
            if st_autorefresh is not None:
                st_autorefresh(interval=10000, limit=None, key="live_autorefresh")
            else:
                time.sleep(10)
                st.rerun()

    with c_reset_btn:
        if st.button("🗑️ Reset Data", use_container_width=True, help="Hapus riwayat simulasi/order"):
            db.clear_all()
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
    b_ctrl = BotController()
    b_stat = b_ctrl.get_status()
    with c_stat:
        if b_stat["is_running"]:
            strat_short = "⚡ Pro Sniper" if b_stat.get("strategy_mode") == "pro_sniper" else "🛡️ Institusional"
            score_short = f" | Confluence: **`{b_stat.get('last_score', 0)}/100`**" if b_stat.get("strategy_mode") == "pro_sniper" else ""
            st.success(
                f"🟢 **Bot 24/7 Aktif ({b_stat['timeframe']} - {strat_short})**{score_short} | "
                f"P(Long): **`{b_stat['last_prob']:.3f}`** | Aksi: **`{b_stat['last_action']}`**"
            )
        elif signals:
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
            tf_used = raw_dict.get("timeframe", "5m")
            score_val = raw_dict.get("score", None)
            score_str = f" | Score: **`{score_val}/100`**" if score_val is not None else ""
            st.info(
                f"📡 **Sinyal Terakhir ({tf_used})** | `{ts_val} UTC` | "
                f"P(Long): **`{p_val:.3f}`**{score_str} | Aksi: **`{action}`** | Alasan: `{reason}`"
            )
        else:
            st.info("🟢 **Sistem Siap** | Silakan jalankan Simulasi Instan atau Aktifkan Bot 24/7.")

    # Summary balances
    pos_val = current_equity - current_cash
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("Simulated Equity", f"${current_equity:,.2f}", f"{(current_equity/10000.0 - 1.0)*100:.2f}%")
    b_col2.metric("Simulated Cash", f"${current_cash:,.2f}")
    b_col3.metric("Open Position Value", f"${pos_val:,.2f}", f"{(pos_val/current_equity)*100:.1f}% Allocation")
    b_col4.metric("Completed Trades", f"{len(orders)}")

    # -------------------------------------------------------------
    # REAL-TIME AI TELEMETRY HUD COCKPIT
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🛸 Live AI Telemetry & Pro Cockpit Radar")
    hud_c1, hud_c2, hud_c3, hud_c4 = st.columns(4)

    last_prob = b_stat["last_prob"] if b_stat["is_running"] else (signals[0].get("prob_long", 0.50) if signals else 0.50)
    last_act = b_stat["last_action"] if b_stat["is_running"] else (signals[0].get("raw_data", {}).get("action", "FLAT") if signals and isinstance(signals[0].get("raw_data"), dict) else "FLAT")
    active_chart_sym = b_stat["symbol"] if b_stat["is_running"] else selected_symbol
    active_chart_tf = b_stat["timeframe"] if b_stat["is_running"] else selected_tf
    is_gold_active = "XAU" in active_chart_sym.upper() or "GOLD" in active_chart_sym.upper()

    score_val = b_stat.get("last_score", 0) if b_stat["is_running"] else (signals[0].get("raw_data", {}).get("score", 0) if signals and isinstance(signals[0].get("raw_data"), dict) else 0)
    tp_val = b_stat.get("take_profit") if b_stat["is_running"] else None
    sl_val = b_stat.get("stop_loss") if b_stat["is_running"] else None
    trail_stage = b_stat.get("trailing_stage", 0) if b_stat["is_running"] else 0
    strat_current = b_stat.get("strategy_mode", "pro_sniper") if b_stat["is_running"] else selected_strat

    with hud_c1:
        st.markdown("**🎯 Confluence Radar (0-100)**")
        if score_val >= 80:
            score_color = "#00E676"
            score_grade = f"🌟 GRADE A+ ({score_val}/100)"
        elif score_val >= 65:
            score_color = "#00E5FF"
            score_grade = f"⚡ GRADE A ({score_val}/100)"
        elif score_val >= 40:
            score_color = "#FFA726"
            score_grade = f"⚠️ GRADE B ({score_val}/100)"
        else:
            score_color = "#B0BEC5"
            score_grade = f"⚪ CHOP/LOW ({score_val}/100)"

        st.markdown(f"<div style='font-size: 15px; font-weight: 700; color: {score_color}; margin-bottom: 6px;'>{score_grade}</div>", unsafe_allow_html=True)
        st.progress(float(np.clip(score_val / 100.0, 0.0, 1.0)))
        st.caption("Pemicu Entry: Score ≥ 65 (Trend + Momentum + Smart Money + AI Edge).")

    with hud_c2:
        st.markdown("**🧠 AI Conviction Meter**")
        if last_prob >= 0.52:
            badge_color = "#00E676"
            badge_text = f"🟢 BULLISH EDGE ({last_prob:.1%})"
        elif last_prob <= 0.48:
            badge_color = "#FF1744"
            badge_text = f"🔴 BEARISH EDGE ({last_prob:.1%})"
        else:
            badge_color = "#B0BEC5"
            badge_text = f"⚪ NEUTRAL DEADBAND ({last_prob:.1%})"

        st.markdown(f"<div style='font-size: 15px; font-weight: 700; color: {badge_color}; margin-bottom: 6px;'>{badge_text}</div>", unsafe_allow_html=True)
        st.progress(float(np.clip(last_prob, 0.0, 1.0)))
        st.caption("Model B Calibrated LightGBM Prob.")

    with hud_c3:
        st.markdown("**🛡️ Trailing & Risk Brackets**")
        tp_str = f"${tp_val:,.2f}" if tp_val else "Dynamic +1.8 ATR"
        sl_str = f"${sl_val:,.2f}" if sl_val else "Dynamic -1.2 ATR"
        trail_label = "🟢 BREAKEVEN LOCK" if trail_stage >= 1 else "⚪ INITIAL BRACKET"
        st.markdown(f"<div style='font-size: 15px; font-weight: 700; color: #00bcd4; margin-bottom: 6px;'>{trail_label}</div>", unsafe_allow_html=True)
        st.write(f"- TP: **`{tp_str}`**\n- SL: **`{sl_str}`**\n- Trailing Tier: **`{trail_stage}`**")

    with hud_c4:
        st.markdown("**⚡ Live Execution State**")
        act_color = "#00E676" if last_act == "BUY" else ("#FF1744" if last_act == "SELL" else "#FFA726")
        st.markdown(f"<div style='font-size: 15px; font-weight: 700; color: {act_color}; margin-bottom: 6px;'>ACTION: {last_act}</div>", unsafe_allow_html=True)
        st.write(f"- Mode: **`{'Pro Sniper' if strat_current == 'pro_sniper' else 'Institutional'}`**\n- Target WR: **`~70% (Min 7W/3L)`**\n- Asset: **`{active_chart_sym}`**")

    # -------------------------------------------------------------
    # LIVE TRADING CHARTS (PLOTLY OVERLAY + TRADINGVIEW PRO)
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"📈 Live Price Action & Execution Radar ({active_chart_sym})")
    chart_tab1, chart_tab2 = st.tabs(["🎯 Candlestick Bot Executions (AI Radar)", "🌐 TradingView Live Streaming (Pro)"])

    with chart_tab1:
        st.caption("Grafik Candlestick interaktif lengkap dengan garis EMA 12/26, volume, serta titik eksekusi beli (▲ hijau) dan jual (▼ merah) langsung dari bot.")
        try:
            fig_candles = build_live_execution_candlestick_chart(active_chart_sym, active_chart_tf, orders)
            st.plotly_chart(fig_candles, use_container_width=True)
        except Exception as e:
            st.info(f"⏳ Menyiapkan data visualisasi candlestick ({e})...")

    with chart_tab2:
        st.caption("Streaming live candlestick real-time langsung dari bursa pasar dunia via TradingView.")
        try:
            render_tradingview_widget(active_chart_sym, active_chart_tf)
        except Exception as e:
            st.info(f"⏳ Menyiapkan visualisasi TradingView ({e})...")

    st.markdown("---")

    # Tables: Executed Orders & Recent Signals side-by-side
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
                        "Qty": f"{o.get('qty', 0.0):.4f}",
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

                score_display = f"{raw.get('score', '-')}" if "score" in raw else "-"
                sig_rows.append(
                    {
                        "Timestamp": s.get("bar_timestamp", "")[:19],
                        "P(Long)": f"{s.get('prob_long', 0.5):.3f}",
                        "Score": score_display,
                        "Action": raw.get("action", "HOLD"),
                        "Target Wt": f"{raw.get('target_size', 0.0):.2f}",
                        "Decision Reason": raw.get("decision_reason", "N/A"),
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
