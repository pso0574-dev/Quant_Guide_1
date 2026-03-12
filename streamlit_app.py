# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="QQQ Quant Guide Pro",
    page_icon="📈",
    layout="wide"
)

st.title("📈 QQQ Quant Guide Pro Dashboard")
st.caption("GitHub + Streamlit Community Cloud deploy-ready app")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Settings")

preset_universe = {
    "QQQ": "QQQ",
    "SPY": "SPY",
    "TLT": "TLT",
    "GLD": "GLD",
    "Custom": None
}

asset_choice = st.sidebar.selectbox(
    "Asset",
    list(preset_universe.keys()),
    index=0
)

if asset_choice == "Custom":
    ticker = st.sidebar.text_input("Custom Ticker", value="QQQ").upper().strip()
else:
    ticker = preset_universe[asset_choice]

comparison_list = st.sidebar.multiselect(
    "Comparison Assets",
    ["QQQ", "SPY", "TLT", "GLD"],
    default=["QQQ", "SPY"]
)

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
vol_window = st.sidebar.slider("Volatility Window", 10, 100, 20, 5)

initial_capital = st.sidebar.number_input(
    "Backtest Initial Capital",
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)

st.sidebar.markdown("---")
show_buy_sell = st.sidebar.checkbox("Show Buy/Sell Signals", value=True)

# =========================================================
# Helper functions
# =========================================================
def get_annualization_factor(interval: str) -> int:
    if interval == "1d":
        return 252
    if interval == "1wk":
        return 52
    if interval == "1mo":
        return 12
    return 252

def get_52w_window(interval: str) -> int:
    if interval == "1d":
        return 252
    if interval == "1wk":
        return 52
    if interval == "1mo":
        return 12
    return 252

@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
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

@st.cache_data(ttl=3600)
def load_close_data(tickers: list, period: str, interval: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        d = yf.download(
            tickers=t,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False
        )
        if d is None or d.empty:
            continue

        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [col[0] for col in d.columns]

        d = d.rename(columns=str.title)
        if "Close" in d.columns:
            frames.append(d[["Close"]].rename(columns={"Close": t}))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).dropna(how="all")
    return out

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)

    return rsi

def calc_drawdown(close: pd.Series):
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    mdd = drawdown.min()
    return drawdown, mdd

def calc_cagr(close: pd.Series, annual_factor: int) -> float:
    if len(close.dropna()) < 2:
        return np.nan
    years = len(close.dropna()) / annual_factor
    if years <= 0:
        return np.nan
    start_val = close.dropna().iloc[0]
    end_val = close.dropna().iloc[-1]
    if start_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1

def calc_sharpe(returns: pd.Series, annual_factor: int) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(annual_factor)

def calc_sortino(returns: pd.Series, annual_factor: int) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if len(returns) < 2 or len(downside) < 2 or downside.std() == 0:
        return np.nan
    return (returns.mean() / downside.std()) * np.sqrt(annual_factor)

def backtest_buy_and_hold(close: pd.Series, initial_capital: float) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    equity = initial_capital * (close / close.iloc[0])
    return equity

def backtest_ma_strategy(close: pd.Series, ma_short: int, ma_long: int, initial_capital: float) -> pd.DataFrame:
    close = close.dropna().copy()
    if len(close) < ma_long:
        return pd.DataFrame()

    df_bt = pd.DataFrame(index=close.index)
    df_bt["Close"] = close
    df_bt["MA_Short"] = close.rolling(ma_short).mean()
    df_bt["MA_Long"] = close.rolling(ma_long).mean()
    df_bt["Signal"] = np.where(df_bt["MA_Short"] > df_bt["MA_Long"], 1, 0)

    # next-bar execution style
    df_bt["Position"] = df_bt["Signal"].shift(1).fillna(0)
    df_bt["Return"] = df_bt["Close"].pct_change().fillna(0)
    df_bt["Strategy_Return"] = df_bt["Position"] * df_bt["Return"]

    df_bt["BuyHold_Equity"] = initial_capital * (1 + df_bt["Return"]).cumprod()
    df_bt["Strategy_Equity"] = initial_capital * (1 + df_bt["Strategy_Return"]).cumprod()
    return df_bt

def summarize_regime(latest_row: pd.Series) -> tuple[str, list]:
    score = 0
    reasons = []

    if pd.notna(latest_row.get("MA_Short")) and pd.notna(latest_row.get("MA_Long")):
        if latest_row["MA_Short"] > latest_row["MA_Long"]:
            score += 1
            reasons.append("Trend positive: short MA > long MA")
        else:
            reasons.append("Trend weak: short MA <= long MA")
    else:
        reasons.append("Trend unavailable")

    rsi_val = latest_row.get("RSI", np.nan)
    if pd.notna(rsi_val):
        if rsi_val < 30:
            score += 1
            reasons.append("Oversold zone: RSI < 30")
        elif rsi_val > 70:
            reasons.append("Overbought zone: RSI > 70")
        else:
            reasons.append("Neutral RSI")
    else:
        reasons.append("RSI unavailable")

    dd_val = latest_row.get("Drawdown", np.nan)
    if pd.notna(dd_val):
        if dd_val <= -0.15:
            score += 1
            reasons.append("Meaningful pullback: drawdown <= -15%")
        elif dd_val > -0.05:
            reasons.append("Near recent high: drawdown > -5%")
        else:
            reasons.append("Moderate drawdown")
    else:
        reasons.append("Drawdown unavailable")

    vol_val = latest_row.get("Volatility", np.nan)
    if pd.notna(vol_val):
        if vol_val < 0.25:
            score += 1
            reasons.append("Volatility manageable")
        else:
            reasons.append("Volatility elevated")
    else:
        reasons.append("Volatility unavailable")

    if score >= 3:
        guide = "🟢 Favorable zone"
    elif score == 2:
        guide = "🟡 Neutral / selective entries"
    else:
        guide = "🔴 Cautious zone"

    return guide, reasons

# =========================================================
# Load main asset data
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

if len(df) < max(ma_long, rsi_period, vol_window):
    st.warning("Not enough historical data for selected settings. Some indicators may be NaN.")

# =========================================================
# Indicators
# =========================================================
annual_factor = get_annualization_factor(interval)
high_window = get_52w_window(interval)

df["MA_Short"] = df["Close"].rolling(ma_short).mean()
df["MA_Long"] = df["Close"].rolling(ma_long).mean()
df["RSI"] = calc_rsi(df["Close"], rsi_period)
df["Return"] = df["Close"].pct_change()
df["Volatility"] = df["Return"].rolling(vol_window).std() * np.sqrt(annual_factor)
df["ATH"] = df["Close"].cummax()
df["Distance_from_ATH"] = df["Close"] / df["ATH"] - 1.0
df["52W_High"] = df["Close"].rolling(high_window).max()
df["Distance_from_52W_High"] = df["Close"] / df["52W_High"] - 1.0
df["Trend_Signal"] = np.where(df["MA_Short"] > df["MA_Long"], 1, 0)
df["Signal_Change"] = df["Trend_Signal"].diff()

drawdown, mdd = calc_drawdown(df["Close"])
df["Drawdown"] = drawdown

latest = df.iloc[-1]

last_price = float(latest["Close"])
last_rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else np.nan
last_vol = float(latest["Volatility"]) if pd.notna(latest["Volatility"]) else np.nan
last_dd = float(latest["Drawdown"]) if pd.notna(latest["Drawdown"]) else np.nan
last_ath_gap = float(latest["Distance_from_ATH"]) if pd.notna(latest["Distance_from_ATH"]) else np.nan
last_52w_gap = float(latest["Distance_from_52W_High"]) if pd.notna(latest["Distance_from_52W_High"]) else np.nan

cagr = calc_cagr(df["Close"], annual_factor)
sharpe = calc_sharpe(df["Return"], annual_factor)
sortino = calc_sortino(df["Return"], annual_factor)

guide, reasons = summarize_regime(latest)

# =========================================================
# Top metrics
# =========================================================
st.subheader(f"{ticker} Snapshot")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Last Price", f"{last_price:,.2f}")
m2.metric("RSI", f"{last_rsi:,.1f}" if pd.notna(last_rsi) else "N/A")
m3.metric("Volatility", f"{last_vol:.2%}" if pd.notna(last_vol) else "N/A")
m4.metric("Drawdown", f"{last_dd:.2%}" if pd.notna(last_dd) else "N/A")
m5.metric("From ATH", f"{last_ath_gap:.2%}" if pd.notna(last_ath_gap) else "N/A")
m6.metric("From 52W High", f"{last_52w_gap:.2%}" if pd.notna(last_52w_gap) else "N/A")

m7, m8, m9, m10 = st.columns(4)
m7.metric("CAGR", f"{cagr:.2%}" if pd.notna(cagr) else "N/A")
m8.metric("Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
m9.metric("Sortino", f"{sortino:.2f}" if pd.notna(sortino) else "N/A")
m10.metric("Max Drawdown", f"{mdd:.2%}" if pd.notna(mdd) else "N/A")

st.subheader(f"Quant Guide: {guide}")
for r in reasons:
    st.write(f"- {r}")

# =========================================================
# Interpretation block
# =========================================================
st.subheader("Action Guide")
if guide.startswith("🟢"):
    st.success(
        "Trend and risk conditions are relatively supportive. "
        "This can be a zone for phased entries rather than all-in buying."
    )
elif guide.startswith("🟡"):
    st.info(
        "Mixed conditions. Prefer selective entries, partial sizing, "
        "or waiting for either trend confirmation or deeper pullback."
    )
else:
    st.warning(
        "Conditions are cautious. Consider smaller size, slower entry pace, "
        "or waiting for stabilization."
    )

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

fig.add_trace(
    go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["MA_Short"], name=f"MA {ma_short}", mode="lines"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["MA_Long"], name=f"MA {ma_long}", mode="lines"),
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

fig.add_trace(
    go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines"),
    row=2, col=1
)
fig.add_hline(y=70, row=2, col=1, line_dash="dash")
fig.add_hline(y=30, row=2, col=1, line_dash="dash")

fig.add_trace(
    go.Scatter(x=df.index, y=df["Drawdown"], name="Drawdown", mode="lines", fill="tozeroy"),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=df.index, y=df["Volatility"], name="Volatility", mode="lines"),
    row=4, col=1
)

fig.update_layout(
    height=1100,
    title=f"{ticker} Quant Guide Pro Dashboard",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="Drawdown", row=3, col=1, tickformat=".0%")
fig.update_yaxes(title_text="Vol", row=4, col=1, tickformat=".0%")

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Comparison chart
# =========================================================
st.subheader("Relative Performance Comparison")

compare_tickers = list(dict.fromkeys(comparison_list))
if ticker not in compare_tickers:
    compare_tickers = [ticker] + compare_tickers

cmp_df = load_close_data(compare_tickers, period, interval)

if not cmp_df.empty:
    normalized = cmp_df.copy()
    for col in normalized.columns:
        first_valid = normalized[col].dropna()
        if not first_valid.empty:
            normalized[col] = 100 * normalized[col] / first_valid.iloc[0]

    fig_cmp = go.Figure()
    for col in normalized.columns:
        fig_cmp.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[col],
                mode="lines",
                name=col
            )
        )

    fig_cmp.update_layout(
        height=450,
        title="Normalized Performance (Start = 100)",
        xaxis_title="Date",
        yaxis_title="Indexed Value"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
else:
    st.warning("Comparison data could not be loaded.")

# =========================================================
# Backtest
# =========================================================
st.subheader("Backtest")

bt_df = backtest_ma_strategy(df["Close"], ma_short, ma_long, initial_capital)
bh_equity = backtest_buy_and_hold(df["Close"], initial_capital)

if not bt_df.empty:
    strategy_equity = bt_df["Strategy_Equity"]
    buyhold_equity = bt_df["BuyHold_Equity"]

    strat_return = strategy_equity.iloc[-1] / initial_capital - 1
    bh_return = buyhold_equity.iloc[-1] / initial_capital - 1

    strat_dd, strat_mdd = calc_drawdown(strategy_equity)
    bh_dd, bh_mdd = calc_drawdown(buyhold_equity)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Buy & Hold Return", f"{bh_return:.2%}")
    b2.metric("MA Strategy Return", f"{strat_return:.2%}")
    b3.metric("Buy & Hold MDD", f"{bh_mdd:.2%}")
    b4.metric("MA Strategy MDD", f"{strat_mdd:.2%}")

    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=buyhold_equity.index,
            y=buyhold_equity.values,
            mode="lines",
            name="Buy & Hold"
        )
    )
    fig_bt.add_trace(
        go.Scatter(
            x=strategy_equity.index,
            y=strategy_equity.values,
            mode="lines",
            name="MA Strategy"
        )
    )

    fig_bt.update_layout(
        height=450,
        title="Backtest Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value"
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    st.caption("Backtest uses a simple next-bar MA regime rule and does not include taxes, slippage, or trading fees.")
else:
    st.warning("Not enough data for backtest with current MA settings.")

# =========================================================
# Latest table
# =========================================================
st.subheader("Latest Data")

show_cols = [
    "Close",
    "MA_Short",
    "MA_Long",
    "RSI",
    "Drawdown",
    "Volatility",
    "Distance_from_ATH",
    "Distance_from_52W_High",
    "Trend_Signal"
]
st.dataframe(df[show_cols].tail(30), use_container_width=True)

# =========================================================
# Summary table
# =========================================================
st.subheader("Summary Table")

summary_rows = []
for t in compare_tickers:
    d = load_data(t, period, interval)
    if d.empty or "Close" not in d.columns:
        continue

    d["Return"] = d["Close"].pct_change()
    _, local_mdd = calc_drawdown(d["Close"])
    local_cagr = calc_cagr(d["Close"], annual_factor)
    local_sharpe = calc_sharpe(d["Return"], annual_factor)

    latest_price = d["Close"].dropna().iloc[-1] if not d["Close"].dropna().empty else np.nan
    ath_gap = (d["Close"].iloc[-1] / d["Close"].cummax().iloc[-1] - 1) if len(d) > 0 else np.nan

    summary_rows.append({
        "Ticker": t,
        "Last Price": latest_price,
        "CAGR": local_cagr,
        "Sharpe": local_sharpe,
        "MDD": local_mdd,
        "From ATH": ath_gap
    })

summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty:
    st.dataframe(
        summary_df.style.format({
            "Last Price": "{:,.2f}",
            "CAGR": "{:.2%}",
            "Sharpe": "{:.2f}",
            "MDD": "{:.2%}",
            "From ATH": "{:.2%}"
        }),
        use_container_width=True
    )

# =========================================================
# How to read
# =========================================================
st.subheader("How to Read This Dashboard")
st.markdown(
    """
- **Trend**: Short MA above Long MA suggests medium-term strength
- **RSI**: Below 30 may indicate oversold, above 70 may indicate overbought
- **Drawdown**: Shows how far price is below its prior peak
- **From ATH / 52W High**: Helps judge whether the asset is extended or in pullback
- **Volatility**: Higher volatility means wider price swings and more risk
- **Backtest**: A simple reference model only, not a production trading system
- **Comparison**: Useful for checking whether QQQ is outperforming or lagging other major assets
"""
)

st.caption(f"Last app refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
