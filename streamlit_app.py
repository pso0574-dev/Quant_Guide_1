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
    page_title="Quant Strategy Master Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Quant Strategy Master Dashboard")
st.caption("Learn 10 practical quant methods with charts, rules, and simple backtests")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Settings")

preset_universe = {
    "QQQ": "QQQ",
    "SPY": "SPY",
    "TLT": "TLT",
    "GLD": "GLD",
    "IWM": "IWM",
    "Custom": None
}

asset_choice = st.sidebar.selectbox("Main Asset", list(preset_universe.keys()), index=0)

if asset_choice == "Custom":
    ticker = st.sidebar.text_input("Custom Ticker", value="QQQ").upper().strip()
else:
    ticker = preset_universe[asset_choice]

comparison_assets = st.sidebar.multiselect(
    "Comparison Assets",
    ["QQQ", "SPY", "TLT", "GLD", "IWM"],
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
vol_window = st.sidebar.slider("Vol Window", 10, 100, 20, 5)
mr_window = st.sidebar.slider("Mean Reversion Window", 10, 100, 20, 5)
mr_z = st.sidebar.slider("MR Z Threshold", 0.5, 3.0, 1.5, 0.1)
mom_lookback = st.sidebar.slider("Momentum Lookback", 20, 252, 126, 5)
target_vol = st.sidebar.slider("Target Volatility", 0.05, 0.30, 0.12, 0.01)
initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)
show_signals = st.sidebar.checkbox("Show Buy/Sell Signals", value=True)

# =========================================================
# Helper Functions
# =========================================================
def get_annualization_factor(interval: str) -> int:
    if interval == "1d":
        return 252
    elif interval == "1wk":
        return 52
    elif interval == "1mo":
        return 12
    return 252

def get_high_window(interval: str) -> int:
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
        d = load_data(t, period, interval)
        if not d.empty and "Close" in d.columns:
            frames.append(d[["Close"]].rename(columns={"Close": t}))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).dropna(how="all")
    return out

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

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
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

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
    close = close.dropna()
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
    close = close.dropna()
    if len(close) < window + 5:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["Mean"] = close.rolling(window).mean()
    bt["Std"] = close.rolling(window).std()
    bt["Z"] = (close - bt["Mean"]) / bt["Std"]

    signal = []
    in_pos = 0
    for _, row in bt.iterrows():
        z = row["Z"]
        c = row["Close"]
        m = row["Mean"]

        if pd.isna(z) or pd.isna(m):
            in_pos = 0
            signal.append(0)
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

def backtest_momentum(close: pd.Series, lookback: int, initial_capital: float) -> pd.DataFrame:
    close = close.dropna()
    if len(close) < lookback + 5:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["Momentum"] = close / close.shift(lookback) - 1
    bt["Signal"] = np.where(bt["Momentum"] > 0, 1, 0)
    bt["Position"] = bt["Signal"].shift(1).fillna(0)
    bt["Return"] = bt["Close"].pct_change().fillna(0)
    bt["Strategy_Return"] = bt["Position"] * bt["Return"]
    bt["Equity"] = initial_capital * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital * (1 + bt["Return"]).cumprod()
    return bt

def backtest_vol_target(close: pd.Series, vol_window: int, target_vol: float, annual_factor: int, initial_capital: float) -> pd.DataFrame:
    close = close.dropna()
    if len(close) < vol_window + 5:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["Return"] = close.pct_change().fillna(0)
    realized_vol = bt["Return"].rolling(vol_window).std() * np.sqrt(annual_factor)
    bt["Realized_Vol"] = realized_vol
    bt["Leverage"] = (target_vol / bt["Realized_Vol"]).clip(upper=2.0)
    bt["Leverage"] = bt["Leverage"].replace([np.inf, -np.inf], np.nan).fillna(0)
    bt["Strategy_Return"] = bt["Leverage"].shift(1).fillna(0) * bt["Return"]
    bt["Equity"] = initial_capital * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital * (1 + bt["Return"]).cumprod()
    return bt

def build_risk_parity(close_df: pd.DataFrame, vol_window: int, annual_factor: int, initial_capital: float):
    close_df = close_df.dropna(how="all")
    if close_df.empty or close_df.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()

    returns = close_df.pct_change().fillna(0)
    vol = returns.rolling(vol_window).std() * np.sqrt(annual_factor)
    inv_vol = 1 / vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)

    rp_ret = (weights.shift(1).fillna(0) * returns).sum(axis=1)
    eq_ret = returns.mean(axis=1)

    equity = pd.DataFrame(index=returns.index)
    equity["Risk_Parity"] = initial_capital * (1 + rp_ret).cumprod()
    equity["Equal_Weight"] = initial_capital * (1 + eq_ret).cumprod()
    return weights, equity

def build_rotation_strategy(close_df: pd.DataFrame, lookback: int, initial_capital: float):
    close_df = close_df.dropna(how="all")
    if close_df.empty or close_df.shape[1] < 2:
        return pd.DataFrame(), pd.Series(dtype=float)

    momentum = close_df / close_df.shift(lookback) - 1
    leader = momentum.idxmax(axis=1)
    returns = close_df.pct_change().fillna(0)

    rot_ret = pd.Series(0.0, index=close_df.index)
    for i in range(1, len(close_df)):
        chosen = leader.iloc[i - 1]
        if pd.notna(chosen):
            rot_ret.iloc[i] = returns.iloc[i][chosen]

    equity = initial_capital * (1 + rot_ret).cumprod()
    out = pd.DataFrame(index=close_df.index)
    out["Leader"] = leader
    out["Rotation_Return"] = rot_ret
    out["Rotation_Equity"] = equity
    return momentum, out

def latest_factor_score(df: pd.DataFrame, mom_lookback: int, vol_window: int):
    d = df.copy()
    d["Momentum"] = d["Close"] / d["Close"].shift(mom_lookback) - 1
    d["Volatility"] = d["Close"].pct_change().rolling(vol_window).std()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()
    d["Quality_Proxy"] = d["Close"] / d["Close"].rolling(252).min() - 1

    latest = d.iloc[-1]
    score = 0
    reasons = []

    if pd.notna(latest["Momentum"]) and latest["Momentum"] > 0:
        score += 1
        reasons.append("Positive momentum")
    else:
        reasons.append("Weak momentum")

    if pd.notna(latest["Volatility"]):
        vol_med = d["Volatility"].median()
        if latest["Volatility"] < vol_med:
            score += 1
            reasons.append("Below-median volatility")
        else:
            reasons.append("Above-median volatility")

    if pd.notna(latest["MA50"]) and pd.notna(latest["MA200"]) and latest["MA50"] > latest["MA200"]:
        score += 1
        reasons.append("Trend confirmed")
    else:
        reasons.append("Trend not confirmed")

    if pd.notna(latest["Quality_Proxy"]):
        q_med = d["Quality_Proxy"].median()
        if latest["Quality_Proxy"] > q_med:
            score += 1
            reasons.append("Quality proxy strong")
        else:
            reasons.append("Quality proxy weak")

    return score, reasons, d

# =========================================================
# Data Load
# =========================================================
annual_factor = get_annualization_factor(interval)
high_window = get_high_window(interval)

df = load_data(ticker, period, interval)

if df.empty:
    st.error("No data loaded. Please check ticker or try again later.")
    st.stop()

required_cols = ["Open", "High", "Low", "Close", "Volume"]
for c in required_cols:
    if c not in df.columns:
        st.error(f"Missing column: {c}")
        st.stop()

df["MA_Short"] = df["Close"].rolling(ma_short).mean()
df["MA_Long"] = df["Close"].rolling(ma_long).mean()
df["RSI"] = calc_rsi(df["Close"], rsi_period)
df["Return"] = df["Close"].pct_change()
df["Volatility"] = df["Return"].rolling(vol_window).std() * np.sqrt(annual_factor)
df["ATH"] = df["Close"].cummax()
df["Distance_from_ATH"] = df["Close"] / df["ATH"] - 1
df["52W_High"] = df["Close"].rolling(high_window).max()
df["Distance_from_52W_High"] = df["Close"] / df["52W_High"] - 1
df["Rolling_Mean_MR"] = df["Close"].rolling(mr_window).mean()
df["Rolling_Std_MR"] = df["Close"].rolling(mr_window).std()
df["ZScore_MR"] = (df["Close"] - df["Rolling_Mean_MR"]) / df["Rolling_Std_MR"]
df["Momentum"] = df["Close"] / df["Close"].shift(mom_lookback) - 1

drawdown, mdd = calc_drawdown(df["Close"])
df["Drawdown"] = drawdown
latest = df.iloc[-1]

compare_tickers = list(dict.fromkeys(comparison_assets))
if ticker not in compare_tickers:
    compare_tickers = [ticker] + compare_tickers

cmp_df = load_close_data(compare_tickers, period, interval)

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs([
    "Overview",
    "1. Trend Following",
    "2. Mean Reversion",
    "3. Momentum",
    "4. Relative Momentum",
    "5. Volatility Targeting",
    "6. Risk Parity",
    "7. Drawdown Buying",
    "8. Multi-Factor",
    "9. Strategy Summary"
])

# =========================================================
# 0. Overview
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
**이 탭에서 배우는 것**
- 현재 자산의 추세, 과열/과매도, 낙폭, 변동성
- 투자 판단 전에 가장 먼저 확인해야 할 핵심 상태
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
    fig.update_layout(height=820, title=f"{ticker} Overview")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="DD", tickformat=".0%", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 1. Trend Following
# =========================================================
with tabs[1]:
    st.subheader("Trend Following")

    st.markdown(
        """
**핵심 개념**  
오르는 자산은 한동안 더 오르고, 약한 자산은 더 약해질 수 있다는 가정입니다.

**간단 룰**  
- 진입: Short MA > Long MA  
- 이탈: Short MA <= Long MA
"""
    )

    bt = backtest_trend(df["Close"], ma_short, ma_long, initial_capital)
    if bt.empty:
        st.warning("Not enough data for this strategy.")
    else:
        t_cagr, t_sharpe, t_mdd = equity_stats(bt["Equity"], annual_factor)
        b_cagr, b_sharpe, b_mdd = equity_stats(bt["BuyHold_Equity"], annual_factor)

        a, b, c = st.columns(3)
        a.metric("Trend CAGR", f"{t_cagr:.2%}" if pd.notna(t_cagr) else "N/A")
        b.metric("Trend Sharpe", f"{t_sharpe:.2f}" if pd.notna(t_sharpe) else "N/A")
        c.metric("Trend MDD", f"{t_mdd:.2%}" if pd.notna(t_mdd) else "N/A")

        buy_points = bt[bt["Signal_Change"] == 1]
        sell_points = bt[bt["Signal_Change"] == -1]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=("Price and MA Signals", "Equity Curve")
        )
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)

        if show_signals:
            fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points["Close"], mode="markers", name="Buy",
                                     marker=dict(symbol="triangle-up", size=11)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points["Close"], mode="markers", name="Sell",
                                     marker=dict(symbol="triangle-down", size=11)), row=1, col=1)

        fig.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold_Equity"], mode="lines", name="Buy & Hold"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name="Trend Strategy"), row=2, col=1)
        fig.update_layout(height=840)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2. Mean Reversion
# =========================================================
with tabs[2]:
    st.subheader("Mean Reversion")

    st.markdown(
        """
**핵심 개념**  
가격이 평균에서 너무 멀어지면 다시 평균 쪽으로 되돌아올 가능성을 이용합니다.

**간단 룰**  
- 진입: Z-score < -threshold  
- 청산: 가격이 rolling mean 이상 회복
"""
    )

    bt = backtest_mean_reversion(df["Close"], mr_window, mr_z, initial_capital)
    if bt.empty:
        st.warning("Not enough data for this strategy.")
    else:
        mr_cagr, mr_sharpe, mr_mdd = equity_stats(bt["Equity"], annual_factor)
        a, b, c = st.columns(3)
        a.metric("MR CAGR", f"{mr_cagr:.2%}" if pd.notna(mr_cagr) else "N/A")
        b.metric("MR Sharpe", f"{mr_sharpe:.2f}" if pd.notna(mr_sharpe) else "N/A")
        c.metric("MR MDD", f"{mr_mdd:.2%}" if pd.notna(mr_mdd) else "N/A")

        buy_points = bt[bt["Signal_Change"] == 1]
        sell_points = bt[bt["Signal_Change"] == -1]

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.42, 0.23, 0.35],
            subplot_titles=("Price vs Mean", "Z-Score", "Equity Curve")
        )
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Mean"], mode="lines", name="Rolling Mean"), row=1, col=1)

        if show_signals:
            fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points["Close"], mode="markers", name="Buy",
                                     marker=dict(symbol="triangle-up", size=11)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points["Close"], mode="markers", name="Exit",
                                     marker=dict(symbol="triangle-down", size=11)), row=1, col=1)

        fig.add_trace(go.Scatter(x=bt.index, y=bt["Z"], mode="lines", name="Z-Score"), row=2, col=1)
        fig.add_hline(y=-mr_z, row=2, col=1, line_dash="dash")
        fig.add_hline(y=0, row=2, col=1, line_dash="dot")
        fig.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold_Equity"], mode="lines", name="Buy & Hold"), row=3, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name="Mean Reversion"), row=3, col=1)
        fig.update_layout(height=930)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 3. Momentum
# =========================================================
with tabs[3]:
    st.subheader("Momentum")

    st.markdown(
        """
**핵심 개념**  
최근 강했던 자산이 당분간 더 강할 수 있다는 규칙입니다.

**간단 룰**  
- 진입: lookback momentum > 0  
- 이탈: momentum <= 0
"""
    )

    bt = backtest_momentum(df["Close"], mom_lookback, initial_capital)
    if bt.empty:
        st.warning("Not enough data for this strategy.")
    else:
        m_cagr, m_sharpe, m_mdd = equity_stats(bt["Equity"], annual_factor)
        a, b, c = st.columns(3)
        a.metric("Momentum CAGR", f"{m_cagr:.2%}" if pd.notna(m_cagr) else "N/A")
        b.metric("Momentum Sharpe", f"{m_sharpe:.2f}" if pd.notna(m_sharpe) else "N/A")
        c.metric("Momentum MDD", f"{m_mdd:.2%}" if pd.notna(m_mdd) else "N/A")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=("Price and Momentum", "Equity Curve")
        )
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Momentum"], mode="lines", name="Momentum"), row=1, col=1)
        fig.add_hline(y=0, row=1, col=1, line_dash="dash")
        fig.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold_Equity"], mode="lines", name="Buy & Hold"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name="Momentum Strategy"), row=2, col=1)
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 4. Relative Momentum / Rotation
# =========================================================
with tabs[4]:
    st.subheader("Relative Momentum / Asset Rotation")

    st.markdown(
        """
**핵심 개념**  
여러 자산 중 최근 가장 강한 자산으로 이동하는 방식입니다.  
예: QQQ / SPY / TLT / GLD 중 최근 n기간 수익률 1위만 보유
"""
    )

    if cmp_df.empty or cmp_df.shape[1] < 2:
        st.warning("Need at least 2 assets for relative momentum.")
    else:
        momentum_df, rot = build_rotation_strategy(cmp_df, mom_lookback, initial_capital)

        normalized = cmp_df.copy()
        for col in normalized.columns:
            first_valid = normalized[col].dropna()
            if not first_valid.empty:
                normalized[col] = 100 * normalized[col] / first_valid.iloc[0]

        fig = go.Figure()
        for col in normalized.columns:
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col))
        fig.update_layout(height=450, title="Normalized Performance")
        st.plotly_chart(fig, use_container_width=True)

        if not rot.empty:
            rot_cagr, rot_sharpe, rot_mdd = equity_stats(rot["Rotation_Equity"], annual_factor)
            st.metric("Rotation CAGR", f"{rot_cagr:.2%}" if pd.notna(rot_cagr) else "N/A")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=rot.index, y=rot["Rotation_Equity"], mode="lines", name="Rotation Equity"))
            fig2.update_layout(height=420, title="Rotation Strategy Equity")
            st.plotly_chart(fig2, use_container_width=True)

            last_leaders = rot["Leader"].dropna().tail(20).to_frame(name="Selected Asset")
            st.subheader("Recent Rotation Decisions")
            st.dataframe(last_leaders, use_container_width=True)

# =========================================================
# 5. Volatility Targeting
# =========================================================
with tabs[5]:
    st.subheader("Volatility Targeting")

    st.markdown(
        """
**핵심 개념**  
포트폴리오 변동성을 일정 수준에 맞추려고 포지션 크기를 조절합니다.

**간단 룰**  
- 실현 변동성이 높아지면 비중 감소
- 실현 변동성이 낮아지면 비중 증가
"""
    )

    bt = backtest_vol_target(df["Close"], vol_window, target_vol, annual_factor, initial_capital)
    if bt.empty:
        st.warning("Not enough data for this strategy.")
    else:
        v_cagr, v_sharpe, v_mdd = equity_stats(bt["Equity"], annual_factor)
        a, b, c = st.columns(3)
        a.metric("Vol Target CAGR", f"{v_cagr:.2%}" if pd.notna(v_cagr) else "N/A")
        b.metric("Vol Target Sharpe", f"{v_sharpe:.2f}" if pd.notna(v_sharpe) else "N/A")
        c.metric("Vol Target MDD", f"{v_mdd:.2%}" if pd.notna(v_mdd) else "N/A")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.34, 0.28, 0.38],
            subplot_titles=("Price", "Realized Vol and Leverage", "Equity Curve")
        )
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Realized_Vol"], mode="lines", name="Realized Vol"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Leverage"], mode="lines", name="Leverage"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold_Equity"], mode="lines", name="Buy & Hold"), row=3, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name="Vol Target"), row=3, col=1)
        fig.update_layout(height=900)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 6. Risk Parity
# =========================================================
with tabs[6]:
    st.subheader("Risk Parity")

    st.markdown(
        """
**핵심 개념**  
돈 비중이 아니라 위험 기여도를 비슷하게 맞추는 방식입니다.

예를 들어 변동성이 높은 자산 비중은 줄이고, 낮은 자산 비중은 늘립니다.
"""
    )

    if cmp_df.empty or cmp_df.shape[1] < 2:
        st.warning("Need multiple assets for risk parity.")
    else:
        weights, equity = build_risk_parity(cmp_df, vol_window, annual_factor, initial_capital)

        if equity.empty:
            st.warning("Could not build risk parity result.")
        else:
            rp_cagr, rp_sharpe, rp_mdd = equity_stats(equity["Risk_Parity"], annual_factor)
            ew_cagr, ew_sharpe, ew_mdd = equity_stats(equity["Equal_Weight"], annual_factor)

            a, b, c = st.columns(3)
            a.metric("Risk Parity CAGR", f"{rp_cagr:.2%}" if pd.notna(rp_cagr) else "N/A")
            b.metric("Risk Parity Sharpe", f"{rp_sharpe:.2f}" if pd.notna(rp_sharpe) else "N/A")
            c.metric("Risk Parity MDD", f"{rp_mdd:.2%}" if pd.notna(rp_mdd) else "N/A")

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                row_heights=[0.4, 0.6],
                subplot_titles=("Risk Parity Weights", "Equity Curve")
            )

            for col in weights.columns:
                fig.add_trace(go.Scatter(x=weights.index, y=weights[col], mode="lines", name=f"W_{col}"), row=1, col=1)

            fig.add_trace(go.Scatter(x=equity.index, y=equity["Equal_Weight"], mode="lines", name="Equal Weight"), row=2, col=1)
            fig.add_trace(go.Scatter(x=equity.index, y=equity["Risk_Parity"], mode="lines", name="Risk Parity"), row=2, col=1)
            fig.update_layout(height=860)
            fig.update_yaxes(tickformat=".0%", row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 7. Drawdown Buying
# =========================================================
with tabs[7]:
    st.subheader("Drawdown Buying Guide")

    st.markdown(
        """
**핵심 개념**  
최고점 대비 낙폭 구간별로 분할 진입하는 규칙입니다.

예시
- 0% ~ -5%: 관찰
- -5% ~ -10%: 1차
- -10% ~ -15%: 2차
- -15% ~ -20%: 3차
- -20% 이하: 고위험/고기회
"""
    )

    latest_dd = latest["Drawdown"]

    guide_rows = [
        {"Zone": "0% ~ -5%", "Meaning": "Near highs", "Action": "Watch / small entry"},
        {"Zone": "-5% ~ -10%", "Meaning": "Normal pullback", "Action": "First phased buy"},
        {"Zone": "-10% ~ -15%", "Meaning": "Moderate correction", "Action": "Second phased buy"},
        {"Zone": "-15% ~ -20%", "Meaning": "Deep correction", "Action": "Aggressive phased buy"},
        {"Zone": "< -20%", "Meaning": "Severe drawdown", "Action": "Opportunity with caution"},
    ]
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True)

    if pd.notna(latest_dd):
        if latest_dd > -0.05:
            st.info("Current state: Near highs")
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
        row_heights=[0.5, 0.5],
        subplot_titles=("Price and ATH", "Drawdown Zones")
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ATH"], mode="lines", name="ATH"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], mode="lines", fill="tozeroy", name="Drawdown"), row=2, col=1)
    fig.add_hline(y=-0.05, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.10, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.15, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.20, row=2, col=1, line_dash="dash")
    fig.update_layout(height=840)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 8. Multi-Factor
# =========================================================
with tabs[8]:
    st.subheader("Multi-Factor Score")

    st.markdown(
        """
**핵심 개념**  
한 가지 신호만 보지 않고, 여러 요인을 합쳐 점수화합니다.

이 예시에서는 아래 4개를 사용합니다.
- Momentum
- Low Volatility
- Trend
- Quality Proxy
"""
    )

    score, reasons, fac_df = latest_factor_score(df, mom_lookback, vol_window)

    a, b = st.columns([1, 2])
    with a:
        st.metric("Factor Score", f"{score} / 4")
    with b:
        for r in reasons:
            st.write(f"- {r}")

    fac_df["Momentum_Rank"] = fac_df["Momentum"].rank(pct=True)
    fac_df["LowVol_Rank"] = (1 - fac_df["Volatility"].rank(pct=True))
    fac_df["Trend_Flag"] = np.where(fac_df["MA50"] > fac_df["MA200"], 1.0, 0.0)
    fac_df["Quality_Rank"] = fac_df["Quality_Proxy"].rank(pct=True)

    fac_df["Composite_Score"] = (
        fac_df["Momentum_Rank"].fillna(0) +
        fac_df["LowVol_Rank"].fillna(0) +
        fac_df["Trend_Flag"].fillna(0) +
        fac_df["Quality_Rank"].fillna(0)
    ) / 4.0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.45, 0.55],
        subplot_titles=("Price", "Composite Factor Score")
    )
    fig.add_trace(go.Scatter(x=fac_df.index, y=fac_df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fac_df.index, y=fac_df["Composite_Score"], mode="lines", name="Composite Score"), row=2, col=1)
    fig.update_layout(height=760)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 9. Strategy Summary
# =========================================================
with tabs[9]:
    st.subheader("Strategy Summary")

    summary_rows = []

    bh = backtest_buy_hold(df["Close"], initial_capital)
    if not bh.empty:
        cagr_bh, sharpe_bh, mdd_bh = equity_stats(bh, annual_factor)
        summary_rows.append({
            "Strategy": "Buy & Hold",
            "Final Value": bh.iloc[-1],
            "CAGR": cagr_bh,
            "Sharpe": sharpe_bh,
            "MDD": mdd_bh
        })

    tf = backtest_trend(df["Close"], ma_short, ma_long, initial_capital)
    if not tf.empty:
        cagr_tf, sharpe_tf, mdd_tf = equity_stats(tf["Equity"], annual_factor)
        summary_rows.append({
            "Strategy": "Trend Following",
            "Final Value": tf["Equity"].iloc[-1],
            "CAGR": cagr_tf,
            "Sharpe": sharpe_tf,
            "MDD": mdd_tf
        })

    mr = backtest_mean_reversion(df["Close"], mr_window, mr_z, initial_capital)
    if not mr.empty:
        cagr_mr, sharpe_mr, mdd_mr = equity_stats(mr["Equity"], annual_factor)
        summary_rows.append({
            "Strategy": "Mean Reversion",
            "Final Value": mr["Equity"].iloc[-1],
            "CAGR": cagr_mr,
            "Sharpe": sharpe_mr,
            "MDD": mdd_mr
        })

    mom = backtest_momentum(df["Close"], mom_lookback, initial_capital)
    if not mom.empty:
        cagr_mom, sharpe_mom, mdd_mom = equity_stats(mom["Equity"], annual_factor)
        summary_rows.append({
            "Strategy": "Momentum",
            "Final Value": mom["Equity"].iloc[-1],
            "CAGR": cagr_mom,
            "Sharpe": sharpe_mom,
            "MDD": mdd_mom
        })

    vol_bt = backtest_vol_target(df["Close"], vol_window, target_vol, annual_factor, initial_capital)
    if not vol_bt.empty:
        cagr_v, sharpe_v, mdd_v = equity_stats(vol_bt["Equity"], annual_factor)
        summary_rows.append({
            "Strategy": "Volatility Targeting",
            "Final Value": vol_bt["Equity"].iloc[-1],
            "CAGR": cagr_v,
            "Sharpe": sharpe_v,
            "MDD": mdd_v
        })

    if not cmp_df.empty and cmp_df.shape[1] >= 2:
        _, rot = build_rotation_strategy(cmp_df, mom_lookback, initial_capital)
        if not rot.empty:
            cagr_rot, sharpe_rot, mdd_rot = equity_stats(rot["Rotation_Equity"], annual_factor)
            summary_rows.append({
                "Strategy": "Relative Momentum Rotation",
                "Final Value": rot["Rotation_Equity"].iloc[-1],
                "CAGR": cagr_rot,
                "Sharpe": sharpe_rot,
                "MDD": mdd_rot
            })

        _, rp_eq = build_risk_parity(cmp_df, vol_window, annual_factor, initial_capital)
        if not rp_eq.empty:
            cagr_rp, sharpe_rp, mdd_rp = equity_stats(rp_eq["Risk_Parity"], annual_factor)
            summary_rows.append({
                "Strategy": "Risk Parity",
                "Final Value": rp_eq["Risk_Parity"].iloc[-1],
                "CAGR": cagr_rp,
                "Sharpe": sharpe_rp,
                "MDD": mdd_rp
            })

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        st.warning("No summary results available.")
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

        if not bh.empty:
            fig.add_trace(go.Scatter(x=bh.index, y=bh, mode="lines", name="Buy & Hold"))
        if not tf.empty:
            fig.add_trace(go.Scatter(x=tf.index, y=tf["Equity"], mode="lines", name="Trend"))
        if not mr.empty:
            fig.add_trace(go.Scatter(x=mr.index, y=mr["Equity"], mode="lines", name="Mean Reversion"))
        if not mom.empty:
            fig.add_trace(go.Scatter(x=mom.index, y=mom["Equity"], mode="lines", name="Momentum"))
        if not vol_bt.empty:
            fig.add_trace(go.Scatter(x=vol_bt.index, y=vol_bt["Equity"], mode="lines", name="Vol Target"))
        if not cmp_df.empty and cmp_df.shape[1] >= 2:
            _, rot = build_rotation_strategy(cmp_df, mom_lookback, initial_capital)
            if not rot.empty:
                fig.add_trace(go.Scatter(x=rot.index, y=rot["Rotation_Equity"], mode="lines", name="Rotation"))
            _, rp_eq = build_risk_parity(cmp_df, vol_window, annual_factor, initial_capital)
            if not rp_eq.empty:
                fig.add_trace(go.Scatter(x=rp_eq.index, y=rp_eq["Risk_Parity"], mode="lines", name="Risk Parity"))

        fig.update_layout(height=520, title="Equity Comparison Across Strategies")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
**이 탭에서 배우는 것**
- 같은 자산이라도 전략에 따라 수익률과 낙폭이 다름
- 높은 CAGR만 볼 것이 아니라 MDD와 Sharpe도 함께 봐야 함
- 실전에서는 한 전략보다 조합이 더 중요함
"""
        )

# =========================================================
# Footer
# =========================================================
st.caption(f"Last app refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
