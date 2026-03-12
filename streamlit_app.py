# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =========================================================
# Page
# =========================================================
st.set_page_config(
    page_title="Quant Strategy Learning Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quant Strategy Learning Dashboard")
st.caption("Learn quant strategies with tabs, charts, and simple backtests")

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

asset_choice = st.sidebar.selectbox("Asset", list(preset_universe.keys()), index=0)
if asset_choice == "Custom":
    ticker = st.sidebar.text_input("Custom Ticker", value="QQQ").upper().strip()
else:
    ticker = preset_universe[asset_choice]

comparison_assets = st.sidebar.multiselect(
    "Comparison Assets",
    ["QQQ", "SPY", "TLT", "GLD"],
    default=["QQQ", "SPY", "TLT", "GLD"]
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

mr_z_window = st.sidebar.slider("Mean Reversion Window", 10, 100, 20, 5)
mr_z_entry = st.sidebar.slider("MR Entry Z-Score", 0.5, 3.0, 1.5, 0.1)

initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)

show_signals = st.sidebar.checkbox("Show Signals", value=True)

# =========================================================
# Helpers
# =========================================================
def get_annualization_factor(interval: str) -> int:
    if interval == "1d":
        return 252
    elif interval == "1wk":
        return 52
    elif interval == "1mo":
        return 12
    return 252

def get_52w_window(interval: str) -> int:
    if interval == "1d":
        return 252
    elif interval == "1wk":
        return 52
    elif interval == "1mo":
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
        df = load_data(t, period, interval)
        if not df.empty and "Close" in df.columns:
            frames.append(df[["Close"]].rename(columns={"Close": t}))
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

def calc_drawdown(series: pd.Series):
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    mdd = drawdown.min()
    return drawdown, mdd

def calc_cagr(close: pd.Series, annual_factor: int) -> float:
    close = close.dropna()
    if len(close) < 2:
        return np.nan
    years = len(close) / annual_factor
    if years <= 0:
        return np.nan
    return (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1

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

def equity_stats(equity: pd.Series, annual_factor: int):
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan, np.nan, np.nan
    rets = equity.pct_change().dropna()
    cagr = calc_cagr(equity, annual_factor)
    sharpe = calc_sharpe(rets, annual_factor)
    _, mdd = calc_drawdown(equity)
    return cagr, sharpe, mdd

def backtest_buy_hold(close: pd.Series, initial_capital: float) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    return initial_capital * (close / close.iloc[0])

def backtest_trend(close: pd.Series, ma_short: int, ma_long: int, initial_capital: float) -> pd.DataFrame:
    close = close.dropna().copy()
    if len(close) < ma_long:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["MA_Short"] = close.rolling(ma_short).mean()
    bt["MA_Long"] = close.rolling(ma_long).mean()
    bt["Signal"] = np.where(bt["MA_Short"] > bt["MA_Long"], 1, 0)
    bt["Position"] = bt["Signal"].shift(1).fillna(0)
    bt["Return"] = bt["Close"].pct_change().fillna(0)
    bt["Strategy_Return"] = bt["Position"] * bt["Return"]
    bt["Equity"] = initial_capital * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital * (1 + bt["Return"]).cumprod()
    bt["Signal_Change"] = bt["Signal"].diff()
    return bt

def backtest_mean_reversion(close: pd.Series, window: int, z_entry: float, initial_capital: float) -> pd.DataFrame:
    close = close.dropna().copy()
    if len(close) < window + 5:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["Mean"] = close.rolling(window).mean()
    bt["Std"] = close.rolling(window).std()
    bt["Z"] = (close - bt["Mean"]) / bt["Std"]
    bt["Signal"] = np.where(bt["Z"] < -z_entry, 1, 0)

    # hold until price returns above rolling mean
    signal = []
    in_pos = 0
    for _, row in bt.iterrows():
        z = row["Z"]
        c = row["Close"]
        m = row["Mean"]

        if pd.isna(z) or pd.isna(m):
            signal.append(0)
            in_pos = 0
            continue

        if in_pos == 0 and z < -z_entry:
            in_pos = 1
        elif in_pos == 1 and c >= m:
            in_pos = 0

        signal.append(in_pos)

    bt["Signal"] = signal
    bt["Position"] = bt["Signal"].shift(1).fillna(0)
    bt["Return"] = bt["Close"].pct_change().fillna(0)
    bt["Strategy_Return"] = bt["Position"] * bt["Return"]
    bt["Equity"] = initial_capital * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital * (1 + bt["Return"]).cumprod()
    bt["Signal_Change"] = bt["Signal"].diff()
    return bt

# =========================================================
# Load main data
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
df["Rolling_Mean_MR"] = df["Close"].rolling(mr_z_window).mean()
df["Rolling_Std_MR"] = df["Close"].rolling(mr_z_window).std()
df["ZScore_MR"] = (df["Close"] - df["Rolling_Mean_MR"]) / df["Rolling_Std_MR"]

df["Trend_Signal"] = np.where(df["MA_Short"] > df["MA_Long"], 1, 0)
df["Trend_Signal_Change"] = df["Trend_Signal"].diff()

drawdown, mdd = calc_drawdown(df["Close"])
df["Drawdown"] = drawdown

latest = df.iloc[-1]

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs([
    "Overview",
    "Trend Following",
    "Mean Reversion",
    "Momentum Comparison",
    "Drawdown Guide",
    "Backtest Summary"
])

# =========================================================
# Tab 1: Overview
# =========================================================
with tabs[0]:
    st.subheader(f"{ticker} Overview")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Last Price", f"{latest['Close']:,.2f}")
    c2.metric("RSI", f"{latest['RSI']:.1f}" if pd.notna(latest["RSI"]) else "N/A")
    c3.metric("Volatility", f"{latest['Volatility']:.2%}" if pd.notna(latest["Volatility"]) else "N/A")
    c4.metric("Drawdown", f"{latest['Drawdown']:.2%}" if pd.notna(latest["Drawdown"]) else "N/A")
    c5.metric("From ATH", f"{latest['Distance_from_ATH']:.2%}" if pd.notna(latest["Distance_from_ATH"]) else "N/A")
    c6.metric("From 52W High", f"{latest['Distance_from_52W_High']:.2%}" if pd.notna(latest["Distance_from_52W_High"]) else "N/A")

    cagr = calc_cagr(df["Close"], annual_factor)
    sharpe = calc_sharpe(df["Return"], annual_factor)
    sortino = calc_sortino(df["Return"], annual_factor)

    d1, d2, d3 = st.columns(3)
    d1.metric("CAGR", f"{cagr:.2%}" if pd.notna(cagr) else "N/A")
    d2.metric("Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
    d3.metric("Max Drawdown", f"{mdd:.2%}" if pd.notna(mdd) else "N/A")

    st.markdown(
        """
### What this tab teaches
- **Price trend** shows the current direction of the asset.
- **RSI** helps judge overbought / oversold condition.
- **Drawdown** tells how far the asset is below its previous peak.
- **Volatility** gives a sense of risk and position sizing difficulty.
"""
    )

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} Price", "RSI", "Drawdown")
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash")
    fig.add_hline(y=30, row=2, col=1, line_dash="dash")

    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], mode="lines", fill="tozeroy", name="Drawdown"), row=3, col=1)

    fig.update_layout(height=850, title=f"{ticker} Overview")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="DD", tickformat=".0%", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Tab 2: Trend Following
# =========================================================
with tabs[1]:
    st.subheader("Trend Following Strategy")

    st.markdown(
        """
### Strategy idea
Trend following assumes that **assets that are already going up tend to continue rising** for a period of time.

### Simple rule
- **Buy / hold** when short MA > long MA
- **Stay out** when short MA <= long MA

### Why it matters
- Works well in strong trends
- Can reduce large crashes
- Often suffers from whipsaws in sideways markets
"""
    )

    bt_trend = backtest_trend(df["Close"], ma_short, ma_long, initial_capital)

    if bt_trend.empty:
        st.warning("Not enough data for trend strategy.")
    else:
        buy_points = bt_trend[bt_trend["Signal_Change"] == 1]
        sell_points = bt_trend[bt_trend["Signal_Change"] == -1]

        t1, t2, t3 = st.columns(3)
        strat_cagr, strat_sharpe, strat_mdd = equity_stats(bt_trend["Equity"], annual_factor)
        bh_cagr, bh_sharpe, bh_mdd = equity_stats(bt_trend["BuyHold_Equity"], annual_factor)

        t1.metric("Trend CAGR", f"{strat_cagr:.2%}" if pd.notna(strat_cagr) else "N/A")
        t2.metric("Trend Sharpe", f"{strat_sharpe:.2f}" if pd.notna(strat_sharpe) else "N/A")
        t3.metric("Trend MDD", f"{strat_mdd:.2%}" if pd.notna(strat_mdd) else "N/A")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=("Price with MA Signals", "Equity Curve: Trend vs Buy & Hold")
        )

        fig.add_trace(go.Scatter(x=bt_trend.index, y=bt_trend["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt_trend.index, y=bt_trend["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt_trend.index, y=bt_trend["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)

        if show_signals:
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index, y=buy_points["Close"], mode="markers",
                    name="Buy", marker=dict(symbol="triangle-up", size=11)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index, y=sell_points["Close"], mode="markers",
                    name="Sell", marker=dict(symbol="triangle-down", size=11)
                ),
                row=1, col=1
            )

        fig.add_trace(go.Scatter(x=bt_trend.index, y=bt_trend["BuyHold_Equity"], mode="lines", name="Buy & Hold"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt_trend.index, y=bt_trend["Equity"], mode="lines", name="Trend Strategy"), row=2, col=1)

        fig.update_layout(height=850, title="Trend Following Learning View")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"""
**Interpretation**
- 현재 설정은 **MA {ma_short} / MA {ma_long}** 교차 기준입니다.
- 이 전략은 큰 상승 추세를 따라가려는 목적에 적합합니다.
- 횡보장에서는 신호가 자주 바뀌면서 성과가 약해질 수 있습니다.
- Buy & Hold와 비교해서 **수익률은 조금 낮아도 MDD가 줄어드는지** 보는 것이 중요합니다.
"""
        )

# =========================================================
# Tab 3: Mean Reversion
# =========================================================
with tabs[2]:
    st.subheader("Mean Reversion Strategy")

    st.markdown(
        """
### Strategy idea
Mean reversion assumes that **price often moves back toward its average** after becoming too stretched.

### Simple rule
- Buy when price is far below its rolling mean
- Exit when price recovers back near the average

### Why it matters
- Useful for pullback entries
- Often works better in range-bound or choppy conditions
- Can fail badly in strong downtrends
"""
    )

    bt_mr = backtest_mean_reversion(df["Close"], mr_z_window, mr_z_entry, initial_capital)

    if bt_mr.empty:
        st.warning("Not enough data for mean reversion strategy.")
    else:
        buy_points = bt_mr[bt_mr["Signal_Change"] == 1]
        sell_points = bt_mr[bt_mr["Signal_Change"] == -1]

        m1, m2, m3 = st.columns(3)
        mr_cagr, mr_sharpe, mr_mdd = equity_stats(bt_mr["Equity"], annual_factor)

        m1.metric("MR CAGR", f"{mr_cagr:.2%}" if pd.notna(mr_cagr) else "N/A")
        m2.metric("MR Sharpe", f"{mr_sharpe:.2f}" if pd.notna(mr_sharpe) else "N/A")
        m3.metric("MR MDD", f"{mr_mdd:.2%}" if pd.notna(mr_mdd) else "N/A")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.42, 0.23, 0.35],
            subplot_titles=("Price vs Rolling Mean", "Z-Score", "Equity Curve: Mean Reversion vs Buy & Hold")
        )

        fig.add_trace(go.Scatter(x=bt_mr.index, y=bt_mr["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt_mr.index, y=bt_mr["Mean"], mode="lines", name="Rolling Mean"), row=1, col=1)

        if show_signals:
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index, y=buy_points["Close"], mode="markers",
                    name="Buy", marker=dict(symbol="triangle-up", size=11)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index, y=sell_points["Close"], mode="markers",
                    name="Exit", marker=dict(symbol="triangle-down", size=11)
                ),
                row=1, col=1
            )

        fig.add_trace(go.Scatter(x=bt_mr.index, y=bt_mr["Z"], mode="lines", name="Z-Score"), row=2, col=1)
        fig.add_hline(y=-mr_z_entry, row=2, col=1, line_dash="dash")
        fig.add_hline(y=0, row=2, col=1, line_dash="dot")

        fig.add_trace(go.Scatter(x=bt_mr.index, y=bt_mr["BuyHold_Equity"], mode="lines", name="Buy & Hold"), row=3, col=1)
        fig.add_trace(go.Scatter(x=bt_mr.index, y=bt_mr["Equity"], mode="lines", name="Mean Reversion"), row=3, col=1)

        fig.update_layout(height=950, title="Mean Reversion Learning View")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"""
**Interpretation**
- 현재 전략은 **{mr_z_window}기간 평균 대비 Z-score가 -{mr_z_entry} 아래**로 내려가면 진입합니다.
- 이후 가격이 평균으로 회복하면 청산합니다.
- 이 전략은 **하락 후 반등**을 노리는 구조입니다.
- 강한 하락 추세에서는 평균 회귀가 늦어질 수 있으므로 위험 관리가 중요합니다.
"""
        )

# =========================================================
# Tab 4: Momentum Comparison
# =========================================================
with tabs[3]:
    st.subheader("Momentum Comparison")

    st.markdown(
        """
### Strategy idea
Relative momentum compares multiple assets and asks:

> “Which asset has been strongest recently?”

This is useful for asset rotation or deciding whether growth, bonds, or gold are leading.
"""
    )

    compare_tickers = list(dict.fromkeys(comparison_assets))
    if ticker not in compare_tickers:
        compare_tickers = [ticker] + compare_tickers

    cmp_df = load_close_data(compare_tickers, period, interval)

    if cmp_df.empty:
        st.warning("Comparison data could not be loaded.")
    else:
        normalized = cmp_df.copy()
        for col in normalized.columns:
            series = normalized[col].dropna()
            if not series.empty:
                normalized[col] = 100 * normalized[col] / series.iloc[0]

        momentum_table = []
        lookbacks = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
        if interval == "1wk":
            lookbacks = {"1M": 4, "3M": 13, "6M": 26, "12M": 52}
        elif interval == "1mo":
            lookbacks = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}

        for col in cmp_df.columns:
            row = {"Ticker": col}
            for name, lb in lookbacks.items():
                if len(cmp_df[col].dropna()) > lb:
                    row[name] = cmp_df[col].iloc[-1] / cmp_df[col].shift(lb).iloc[-1] - 1
                else:
                    row[name] = np.nan
            momentum_table.append(row)

        momentum_df = pd.DataFrame(momentum_table)

        fig = go.Figure()
        for col in normalized.columns:
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col))
        fig.update_layout(height=500, title="Normalized Relative Performance (Start = 100)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Momentum Table")
        st.dataframe(
            momentum_df.style.format({
                "1M": "{:.2%}",
                "3M": "{:.2%}",
                "6M": "{:.2%}",
                "12M": "{:.2%}"
            }),
            use_container_width=True
        )

        st.markdown(
            """
**Interpretation**
- 최근 1M, 3M, 6M, 12M 수익률을 비교하면 어떤 자산이 강한지 확인할 수 있습니다.
- 예를 들어 QQQ가 SPY보다 강하면 성장주 주도 흐름일 수 있습니다.
- TLT나 GLD가 상대적으로 강하면 방어적 성격이 부각될 수 있습니다.
"""
        )

# =========================================================
# Tab 5: Drawdown Guide
# =========================================================
with tabs[4]:
    st.subheader("Drawdown Guide")

    st.markdown(
        """
### Strategy idea
Drawdown-based investing asks:

> “How deep is the pullback from the recent peak?”

This is useful for phased entry planning.
"""
    )

    latest_dd = latest["Drawdown"]

    guide_rows = [
        {"Zone": "0% ~ -5%", "Meaning": "Near highs", "Typical Action": "Wait / small entry only"},
        {"Zone": "-5% ~ -10%", "Meaning": "Normal pullback", "Typical Action": "Watchlist / first small entry"},
        {"Zone": "-10% ~ -15%", "Meaning": "Moderate correction", "Typical Action": "Phased buying"},
        {"Zone": "-15% ~ -20%", "Meaning": "Deeper correction", "Typical Action": "More aggressive phased entry"},
        {"Zone": "< -20%", "Meaning": "Severe drawdown", "Typical Action": "Opportunity but high risk"}
    ]
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True)

    if pd.notna(latest_dd):
        if latest_dd > -0.05:
            st.info("Current state: Near recent highs")
        elif latest_dd > -0.10:
            st.info("Current state: Normal pullback")
        elif latest_dd > -0.15:
            st.warning("Current state: Moderate correction")
        elif latest_dd > -0.20:
            st.warning("Current state: Deep correction")
        else:
            st.error("Current state: Severe drawdown")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("Price and ATH", "Drawdown")
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ATH"], mode="lines", name="ATH"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], mode="lines", fill="tozeroy", name="Drawdown"), row=2, col=1)

    fig.add_hline(y=-0.05, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.10, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.15, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.20, row=2, col=1, line_dash="dash")

    fig.update_layout(height=850, title="Drawdown Learning View")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Tab 6: Backtest Summary
# =========================================================
with tabs[5]:
    st.subheader("Backtest Summary")

    bh_equity = backtest_buy_hold(df["Close"], initial_capital)
    bt_trend = backtest_trend(df["Close"], ma_short, ma_long, initial_capital)
    bt_mr = backtest_mean_reversion(df["Close"], mr_z_window, mr_z_entry, initial_capital)

    summary_rows = []

    if not bh_equity.empty:
        cagr_bh, sharpe_bh, mdd_bh = equity_stats(bh_equity, annual_factor)
        summary_rows.append({
            "Strategy": "Buy & Hold",
            "Final Value": bh_equity.iloc[-1],
            "CAGR": cagr_bh,
            "Sharpe": sharpe_bh,
            "MDD": mdd_bh
        })

    if not bt_trend.empty:
        cagr_tf, sharpe_tf, mdd_tf = equity_stats(bt_trend["Equity"], annual_factor)
        summary_rows.append({
            "Strategy": "Trend Following",
            "Final Value": bt_trend["Equity"].iloc[-1],
            "CAGR": cagr_tf,
            "Sharpe": sharpe_tf,
            "MDD": mdd_tf
        })

    if not bt_mr.empty:
        cagr_mr, sharpe_mr, mdd_mr = equity_stats(bt_mr["Equity"], annual_factor)
        summary_rows.append({
            "Strategy": "Mean Reversion",
            "Final Value": bt_mr["Equity"].iloc[-1],
            "CAGR": cagr_mr,
            "Sharpe": sharpe_mr,
            "MDD": mdd_mr
        })

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        st.warning("No backtest results available.")
    else:
        st.dataframe(
            summary_df.style.format({
                "Final Value": "{:,.0f}",
                "CAGR": "{:.2%}",
                "Sharpe": "{:.2f}",
                "MDD": "{:.2%}"
            }),
            use_container_width=True
        )

        fig = go.Figure()

        if not bh_equity.empty:
            fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, mode="lines", name="Buy & Hold"))
        if not bt_trend.empty:
            fig.add_trace(go.Scatter(x=bt_trend.index, y=bt_trend["Equity"], mode="lines", name="Trend Following"))
        if not bt_mr.empty:
            fig.add_trace(go.Scatter(x=bt_mr.index, y=bt_mr["Equity"], mode="lines", name="Mean Reversion"))

        fig.update_layout(height=500, title="Strategy Equity Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
### What this tab teaches
- 같은 자산이라도 전략에 따라 수익률과 MDD가 달라집니다.
- **Buy & Hold**는 가장 단순하지만 낙폭이 클 수 있습니다.
- **Trend Following**은 큰 추세를 따라가면서 낙폭을 줄이는 데 유리할 수 있습니다.
- **Mean Reversion**은 눌림목 매수 관점에서 유용하지만 추세장에서는 불리할 수 있습니다.
"""
        )

# =========================================================
# Footer
# =========================================================
st.caption(f"Last app refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
