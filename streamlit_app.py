# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(
    page_title="QQQ Quant Guide",
    page_icon="📈",
    layout="wide"
)

st.title("📈 QQQ Quant Guide Dashboard")
st.caption("GitHub + Streamlit Community Cloud deploy-ready app")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", value="QQQ").upper()
period = st.sidebar.selectbox(
    "History Period",
    ["1y", "2y", "3y", "5y", "10y", "max"],
    index=3
)
interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1wk", "1mo"],
    index=0
)

ma_short = st.sidebar.slider("Short MA", 10, 100, 50, 5)
ma_long = st.sidebar.slider("Long MA", 50, 300, 200, 10)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14, 1)
vol_window = st.sidebar.slider("Volatility Window", 10, 100, 20, 5)

st.sidebar.markdown("---")
show_buy_sell = st.sidebar.checkbox("Show Buy/Sell Signals", value=True)

# =========================================================
# Helpers
# =========================================================
@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        return df

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.rename(columns=str.title)
    return df

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_max_drawdown(close: pd.Series):
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    mdd = drawdown.min()
    return drawdown, mdd

# =========================================================
# Load
# =========================================================
df = load_data(ticker, period, interval)

if df.empty:
    st.error("No data loaded. Please check ticker or try again later.")
    st.stop()

required_cols = ["Open", "High", "Low", "Close", "Volume"]
for c in required_cols:
    if c not in df.columns:
        st.error(f"Missing column: {c}")
        st.stop()

# =========================================================
# Indicators
# =========================================================
df["MA_Short"] = df["Close"].rolling(ma_short).mean()
df["MA_Long"] = df["Close"].rolling(ma_long).mean()
df["RSI"] = calc_rsi(df["Close"], rsi_period)
df["Return"] = df["Close"].pct_change()
df["Volatility"] = df["Return"].rolling(vol_window).std() * np.sqrt(252)
df["52W_High"] = df["Close"].rolling(252).max()
df["Distance_from_52W_High"] = df["Close"] / df["52W_High"] - 1.0

df["Trend_Signal"] = np.where(df["MA_Short"] > df["MA_Long"], 1, 0)
df["Signal_Change"] = df["Trend_Signal"].diff()

drawdown, mdd = calc_max_drawdown(df["Close"])
df["Drawdown"] = drawdown

latest = df.iloc[-1]
last_price = float(latest["Close"])
last_rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else np.nan
last_vol = float(latest["Volatility"]) if pd.notna(latest["Volatility"]) else np.nan
last_dd = float(latest["Drawdown"]) if pd.notna(latest["Drawdown"]) else np.nan
last_dist_high = float(latest["Distance_from_52W_High"]) if pd.notna(latest["Distance_from_52W_High"]) else np.nan

# =========================================================
# Quant Guide Scoring
# =========================================================
score = 0
reasons = []

# Trend
if pd.notna(latest["MA_Short"]) and pd.notna(latest["MA_Long"]):
    if latest["MA_Short"] > latest["MA_Long"]:
        score += 1
        reasons.append("Trend positive: short MA > long MA")
    else:
        reasons.append("Trend weak: short MA <= long MA")

# RSI
if pd.notna(last_rsi):
    if last_rsi < 30:
        score += 1
        reasons.append("Oversold zone: RSI < 30")
    elif last_rsi > 70:
        reasons.append("Overbought zone: RSI > 70")
    else:
        reasons.append("Neutral RSI")

# Drawdown
if pd.notna(last_dd):
    if last_dd <= -0.15:
        score += 1
        reasons.append("Meaningful pullback: drawdown <= -15%")
    elif last_dd > -0.05:
        reasons.append("Near recent high: drawdown > -5%")
    else:
        reasons.append("Moderate drawdown")

# Volatility
if pd.notna(last_vol):
    if last_vol < 0.25:
        score += 1
        reasons.append("Volatility manageable")
    else:
        reasons.append("Volatility elevated")

if score >= 3:
    guide = "🟢 Favorable zone"
elif score == 2:
    guide = "🟡 Neutral / selective entries"
else:
    guide = "🔴 Cautious zone"

# =========================================================
# Metrics
# =========================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last Price", f"{last_price:,.2f}")
c2.metric("RSI", f"{last_rsi:,.1f}" if pd.notna(last_rsi) else "N/A")
c3.metric("Volatility", f"{last_vol:.2%}" if pd.notna(last_vol) else "N/A")
c4.metric("Current Drawdown", f"{last_dd:.2%}" if pd.notna(last_dd) else "N/A")
c5.metric("From 52W High", f"{last_dist_high:.2%}" if pd.notna(last_dist_high) else "N/A")

st.subheader(f"Quant Guide: {guide}")
for r in reasons:
    st.write(f"- {r}")

# =========================================================
# Buy/Sell markers
# =========================================================
buy_points = df[df["Signal_Change"] == 1]
sell_points = df[df["Signal_Change"] == -1]

# =========================================================
# Main chart
# =========================================================
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.45, 0.18, 0.18, 0.19],
    subplot_titles=(
        f"{ticker} Price / Moving Averages",
        "RSI",
        "Drawdown",
        "Rolling Volatility"
    )
)

# Price
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", mode="lines"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index, y=df["MA_Short"],
        name=f"MA {ma_short}", mode="lines"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index, y=df["MA_Long"],
        name=f"MA {ma_long}", mode="lines"
    ),
    row=1, col=1
)

if show_buy_sell:
    fig.add_trace(
        go.Scatter(
            x=buy_points.index,
            y=buy_points["Close"],
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", size=11)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points.index,
            y=sell_points["Close"],
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", size=11)
        ),
        row=1, col=1
    )

# RSI
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI", mode="lines"
    ),
    row=2, col=1
)
fig.add_hline(y=70, row=2, col=1, line_dash="dash")
fig.add_hline(y=30, row=2, col=1, line_dash="dash")

# Drawdown
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["Drawdown"],
        name="Drawdown", mode="lines", fill="tozeroy"
    ),
    row=3, col=1
)

# Volatility
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["Volatility"],
        name="Volatility", mode="lines"
    ),
    row=4, col=1
)

fig.update_layout(
    height=1100,
    title=f"{ticker} Quant Guide Dashboard",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="Drawdown", row=3, col=1, tickformat=".0%")
fig.update_yaxes(title_text="Vol", row=4, col=1, tickformat=".0%")

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Data table
# =========================================================
st.subheader("Latest Data")
show_cols = [
    "Close", "MA_Short", "MA_Long", "RSI",
    "Drawdown", "Volatility", "Distance_from_52W_High"
]
st.dataframe(df[show_cols].tail(20), use_container_width=True)

# =========================================================
# Strategy note
# =========================================================
st.subheader("How to Read This Dashboard")
st.markdown(
    """
- **Trend**: Short MA above Long MA → medium-term uptrend
- **RSI**: Below 30 can indicate oversold, above 70 can indicate overbought
- **Drawdown**: Helps identify pullback depth from recent peak
- **Volatility**: Higher volatility means higher risk / wider swings
- **Buy/Sell signals**: Simple MA crossover reference only, not a complete trading system
"""
)

st.caption(f"Last app refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
