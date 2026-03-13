# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Quant Guide Lite",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quant Guide Lite")
st.caption("Minimal and robust quant dashboard for Streamlit Cloud")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", value="QQQ").upper().strip()

period = st.sidebar.selectbox(
    "History Period",
    ["1y", "2y", "3y", "5y", "10y", "max"],
    index=4
)

interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1wk", "1mo"],
    index=0
)

ma_short = st.sidebar.slider("Short MA", 10, 100, 50, 5)
ma_long = st.sidebar.slider("Long MA", 50, 300, 200, 10)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14, 1)
mom_lookback = st.sidebar.slider("Momentum Lookback", 20, 252, 126, 5)

initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)

trading_cost_bps = st.sidebar.slider(
    "Trading Cost (bps)",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)

show_signals = st.sidebar.checkbox("Show Buy/Sell Signals", value=True)

# =========================================================
# Helpers
# =========================================================
def get_annualization_factor(interval_value: str) -> int:
    if interval_value == "1d":
        return 252
    if interval_value == "1wk":
        return 52
    if interval_value == "1mo":
        return 12
    return 252

@st.cache_data(ttl=3600)
def load_data(ticker_value: str, period_value: str, interval_value: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker_value,
        period=period_value,
        interval=interval_value,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.rename(columns=str.title)
    df = df.dropna(how="all")
    return df

def calc_rsi(series: pd.Series, period_value: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period_value, min_periods=period_value).mean()
    avg_loss = loss.rolling(period_value, min_periods=period_value).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    return rsi

def calc_drawdown(series: pd.Series):
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    mdd = drawdown.min()
    return drawdown, mdd

def calc_cagr(series: pd.Series, annual_factor: int) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    years = len(series) / annual_factor
    if years <= 0:
        return np.nan
    start = series.iloc[0]
    end = series.iloc[-1]
    if start <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1

def calc_sharpe(returns: pd.Series, annual_factor: int) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(annual_factor)

def apply_costs(strategy_returns: pd.Series, position: pd.Series, cost_bps: float):
    turnover = position.fillna(0).diff().abs().fillna(position.fillna(0).abs())
    cost_rate = cost_bps / 10000.0
    net_returns = strategy_returns.fillna(0) - turnover * cost_rate
    return net_returns, turnover

def backtest_buy_hold(close: pd.Series, initial_capital_value: float) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    return initial_capital_value * (close / close.iloc[0])

def backtest_trend(close: pd.Series, ma_short_value: int, ma_long_value: int, initial_capital_value: float, cost_bps: float) -> pd.DataFrame:
    close = close.dropna()
    if len(close) < ma_long_value:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["MA_Short"] = close.rolling(ma_short_value).mean()
    bt["MA_Long"] = close.rolling(ma_long_value).mean()
    bt["Signal"] = np.where(bt["MA_Short"] > bt["MA_Long"], 1, 0)
    bt["Position"] = bt["Signal"].shift(1).fillna(0)
    bt["Return"] = bt["Close"].pct_change().fillna(0)
    bt["Strategy_Return"] = bt["Position"] * bt["Return"]
    bt["Signal_Change"] = bt["Signal"].diff()

    net_returns, turnover = apply_costs(bt["Strategy_Return"], bt["Position"], cost_bps)
    bt["Turnover"] = turnover
    bt["Net_Strategy_Return"] = net_returns
    bt["Equity_Net"] = initial_capital_value * (1 + bt["Net_Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital_value * (1 + bt["Return"]).cumprod()
    return bt

def backtest_momentum(close: pd.Series, lookback_value: int, initial_capital_value: float, cost_bps: float) -> pd.DataFrame:
    close = close.dropna()
    if len(close) < lookback_value + 5:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["Momentum"] = close / close.shift(lookback_value) - 1
    bt["Signal"] = np.where(bt["Momentum"] > 0, 1, 0)
    bt["Position"] = bt["Signal"].shift(1).fillna(0)
    bt["Return"] = bt["Close"].pct_change().fillna(0)
    bt["Strategy_Return"] = bt["Position"] * bt["Return"]

    net_returns, turnover = apply_costs(bt["Strategy_Return"], bt["Position"], cost_bps)
    bt["Turnover"] = turnover
    bt["Net_Strategy_Return"] = net_returns
    bt["Equity_Net"] = initial_capital_value * (1 + bt["Net_Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital_value * (1 + bt["Return"]).cumprod()
    return bt

def equity_stats(equity: pd.Series, annual_factor: int):
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan, np.nan, np.nan
    returns = equity.pct_change().dropna()
    cagr_value = calc_cagr(equity, annual_factor)
    sharpe_value = calc_sharpe(returns, annual_factor)
    _, mdd_value = calc_drawdown(equity)
    return cagr_value, sharpe_value, mdd_value

def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2%}"

def fmt_ratio(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}"

# =========================================================
# Load Data
# =========================================================
annual_factor = get_annualization_factor(interval)
df = load_data(ticker, period, interval)

if df.empty:
    st.error("No data loaded. Please check the ticker or try again later.")
    st.stop()

required_cols = ["Open", "High", "Low", "Close", "Volume"]
for c in required_cols:
    if c not in df.columns:
        st.error(f"Missing column: {c}")
        st.stop()

df["MA_Short"] = df["Close"].rolling(ma_short).mean()
df["MA_Long"] = df["Close"].rolling(ma_long).mean()
df["RSI"] = calc_rsi(df["Close"], rsi_period)
df["Momentum"] = df["Close"] / df["Close"].shift(mom_lookback) - 1
df["Return"] = df["Close"].pct_change()

drawdown, mdd = calc_drawdown(df["Close"])
df["Drawdown"] = drawdown
latest = df.iloc[-1]

price_cagr = calc_cagr(df["Close"], annual_factor)
price_sharpe = calc_sharpe(df["Return"], annual_factor)

trend_bt = backtest_trend(df["Close"], ma_short, ma_long, initial_capital, trading_cost_bps)
mom_bt = backtest_momentum(df["Close"], mom_lookback, initial_capital, trading_cost_bps)
buyhold_equity = backtest_buy_hold(df["Close"], initial_capital)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Trend Following",
    "Momentum",
    "Summary"
])

# =========================================================
# Overview
# =========================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Price", f"{latest['Close']:,.2f}")
    c2.metric("RSI", fmt_ratio(latest["RSI"]))
    c3.metric("Drawdown", fmt_pct(latest["Drawdown"]))
    c4.metric("From High", fmt_pct(latest["Close"] / df["Close"].cummax().iloc[-1] - 1))

    d1, d2, d3 = st.columns(3)
    d1.metric("CAGR", fmt_pct(price_cagr))
    d2.metric("Sharpe", fmt_ratio(price_sharpe))
    d3.metric("Max Drawdown", fmt_pct(mdd))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price", "RSI", "Drawdown")
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash")
    fig.add_hline(y=30, row=2, col=1, line_dash="dash")

    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], mode="lines", fill="tozeroy", name="Drawdown"), row=3, col=1)

    fig.update_layout(height=900, title=f"{ticker} Overview")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Drawdown", row=3, col=1, tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Trend Following
# =========================================================
with tab2:
    st.markdown(
        """
**Core idea**  
Follow the dominant market direction using moving averages.

**Simple rule**  
- Long when short MA > long MA  
- Flat when short MA <= long MA
"""
    )

    if trend_bt.empty:
        st.warning("Not enough data for the trend strategy.")
    else:
        trend_cagr, trend_sharpe, trend_mdd = equity_stats(trend_bt["Equity_Net"], annual_factor)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", fmt_pct(trend_cagr))
        c2.metric("Sharpe", fmt_ratio(trend_sharpe))
        c3.metric("Max Drawdown", fmt_pct(trend_mdd))
        c4.metric("Turnover", fmt_pct(trend_bt["Turnover"].mean() * annual_factor))

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=("Price and Signals", "Equity Curve")
        )

        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)

        if show_signals:
            buy_points = trend_bt[trend_bt["Signal_Change"] == 1]
            sell_points = trend_bt[trend_bt["Signal_Change"] == -1]

            fig.add_trace(
                go.Scatter(
                    x=buy_points.index, y=buy_points["Close"],
                    mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=10)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index, y=sell_points["Close"],
                    mode="markers", name="Sell",
                    marker=dict(symbol="triangle-down", size=10)
                ),
                row=1, col=1
            )

        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["BuyHold_Equity"], mode="lines", name="Buy & Hold"), row=2, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Equity_Net"], mode="lines", name="Trend Strategy"), row=2, col=1)

        fig.update_layout(height=850)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Momentum
# =========================================================
with tab3:
    st.markdown(
        """
**Core idea**  
Assets with positive recent momentum may continue to perform well for some time.

**Simple rule**  
- Long when lookback momentum > 0  
- Flat when momentum <= 0
"""
    )

    if mom_bt.empty:
        st.warning("Not enough data for the momentum strategy.")
    else:
        mom_cagr, mom_sharpe, mom_mdd = equity_stats(mom_bt["Equity_Net"], annual_factor)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", fmt_pct(mom_cagr))
        c2.metric("Sharpe", fmt_ratio(mom_sharpe))
        c3.metric("Max Drawdown", fmt_pct(mom_mdd))
        c4.metric("Turnover", fmt_pct(mom_bt["Turnover"].mean() * annual_factor))

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=("Price and Momentum", "Equity Curve")
        )

        fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["Momentum"], mode="lines", name="Momentum"), row=1, col=1)
        fig.add_hline(y=0, row=1, col=1, line_dash="dash")

        fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["BuyHold_Equity"], mode="lines", name="Buy & Hold"), row=2, col=1)
        fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["Equity_Net"], mode="lines", name="Momentum Strategy"), row=2, col=1)

        fig.update_layout(height=820)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Summary
# =========================================================
with tab4:
    summary_rows = []

    if not buyhold_equity.empty:
        bh_cagr, bh_sharpe, bh_mdd = equity_stats(buyhold_equity, annual_factor)
        summary_rows.append({
            "Strategy": "Buy & Hold",
            "Final Value": buyhold_equity.iloc[-1],
            "CAGR": bh_cagr,
            "Sharpe": bh_sharpe,
            "Max Drawdown": bh_mdd,
            "Turnover": 0.0
        })

    if not trend_bt.empty:
        trend_cagr, trend_sharpe, trend_mdd = equity_stats(trend_bt["Equity_Net"], annual_factor)
        summary_rows.append({
            "Strategy": "Trend Following",
            "Final Value": trend_bt["Equity_Net"].iloc[-1],
            "CAGR": trend_cagr,
            "Sharpe": trend_sharpe,
            "Max Drawdown": trend_mdd,
            "Turnover": trend_bt["Turnover"].mean() * annual_factor
        })

    if not mom_bt.empty:
        mom_cagr, mom_sharpe, mom_mdd = equity_stats(mom_bt["Equity_Net"], annual_factor)
        summary_rows.append({
            "Strategy": "Momentum",
            "Final Value": mom_bt["Equity_Net"].iloc[-1],
            "CAGR": mom_cagr,
            "Sharpe": mom_sharpe,
            "Max Drawdown": mom_mdd,
            "Turnover": mom_bt["Turnover"].mean() * annual_factor
        })

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        st.warning("No summary results available.")
    else:
        display_df = summary_df.copy()
        display_df["Final Value"] = display_df["Final Value"].map(lambda x: f"{x:,.0f}")
        display_df["CAGR"] = display_df["CAGR"].map(fmt_pct)
        display_df["Sharpe"] = display_df["Sharpe"].map(fmt_ratio)
        display_df["Max Drawdown"] = display_df["Max Drawdown"].map(fmt_pct)
        display_df["Turnover"] = display_df["Turnover"].map(fmt_pct)
        st.dataframe(display_df, use_container_width=True)

        fig = go.Figure()

        if not buyhold_equity.empty:
            fig.add_trace(go.Scatter(x=buyhold_equity.index, y=buyhold_equity, mode="lines", name="Buy & Hold"))
        if not trend_bt.empty:
            fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Equity_Net"], mode="lines", name="Trend Following"))
        if not mom_bt.empty:
            fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["Equity_Net"], mode="lines", name="Momentum"))

        fig.update_layout(height=520, title="Strategy Equity Comparison")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Footer
# =========================================================
st.caption(f"Last app refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
