# -*- coding: utf-8 -*-
"""
QQQ / SPY / SCHD Quant Dashboard (Streamlit)
- Trend filter: invest only if price > moving average
- Momentum rotation: select top assets by momentum
- Volatility weighting: inverse volatility weighting
- Monthly rebalance
- Compare with equal-weight buy & hold

Run locally:
    streamlit run streamlit_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Quant ETF Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 QQQ / SPY / SCHD Quant Dashboard")
st.caption("Trend + Momentum + Volatility + Monthly Rebalance")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input("Tickers (comma separated)", "QQQ,SPY,SCHD")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2010-01-01"))
trend_window = st.sidebar.slider("Trend MA window", 50, 300, 200, 10)
momentum_window = st.sidebar.slider("Momentum window (trading days)", 63, 252, 252, 21)
vol_window = st.sidebar.slider("Volatility window (trading days)", 21, 126, 63, 21)
top_n = st.sidebar.slider("Top N momentum assets", 1, 3, 2, 1)
rebalance_freq = st.sidebar.selectbox("Rebalance frequency", ["M", "Q"], index=0)

run_button = st.sidebar.button("Run Strategy")

# =========================================================
# HELPERS
# =========================================================
@st.cache_data(show_spinner=False)
def download_prices(tickers, start_date):
    data = yf.download(
        tickers,
        start=start_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs(tickers[0], axis=1, level=1, drop_level=False)
    else:
        prices = data.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")
    return prices


def calc_cagr(equity_curve):
    if len(equity_curve) < 2:
        return np.nan
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return total_return ** (1 / years) - 1


def calc_mdd(equity_curve):
    if len(equity_curve) < 2:
        return np.nan
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def calc_sharpe(daily_returns, rf=0.0):
    if len(daily_returns) < 2:
        return np.nan
    excess = daily_returns - rf / 252
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return np.sqrt(252) * excess.mean() / vol


def build_strategy(prices, trend_window, momentum_window, vol_window, top_n, rebalance_freq):
    daily_ret = prices.pct_change().fillna(0)

    ma = prices.rolling(trend_window).mean()
    momentum = prices / prices.shift(momentum_window) - 1
    vol = daily_ret.rolling(vol_window).std() * np.sqrt(252)

    rebalance_dates = prices.resample(rebalance_freq).last().index
    rebalance_dates = [d for d in rebalance_dates if d in prices.index]

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current_weights = pd.Series(0.0, index=prices.columns)

    for date in prices.index:
        if date in rebalance_dates:
            candidates = []

            for t in prices.columns:
                p = prices.loc[date, t]
                ma_t = ma.loc[date, t]
                mom_t = momentum.loc[date, t]
                vol_t = vol.loc[date, t]

                if pd.notna(p) and pd.notna(ma_t) and pd.notna(mom_t) and pd.notna(vol_t):
                    if p > ma_t:
                        candidates.append(t)

            if len(candidates) > 0:
                ranked = momentum.loc[date, candidates].sort_values(ascending=False)
                selected = ranked.head(top_n).index.tolist()

                inv_vol = 1 / vol.loc[date, selected]
                inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()

                current_weights = pd.Series(0.0, index=prices.columns)
                if len(inv_vol) > 0:
                    current_weights.loc[inv_vol.index] = inv_vol / inv_vol.sum()
            else:
                current_weights = pd.Series(0.0, index=prices.columns)

        weights.loc[date] = current_weights

    # avoid look-ahead bias
    weights = weights.shift(1).fillna(0)

    strategy_returns = (weights * daily_ret).sum(axis=1)
    strategy_equity = (1 + strategy_returns).cumprod()

    bh_returns = daily_ret.mean(axis=1)
    bh_equity = (1 + bh_returns).cumprod()

    return {
        "weights": weights,
        "strategy_returns": strategy_returns,
        "strategy_equity": strategy_equity,
        "bh_returns": bh_returns,
        "bh_equity": bh_equity,
        "momentum": momentum,
        "vol": vol,
        "ma": ma
    }


def make_metrics_df(result):
    strat_eq = result["strategy_equity"]
    strat_ret = result["strategy_returns"]
    bh_eq = result["bh_equity"]
    bh_ret = result["bh_returns"]

    df = pd.DataFrame({
        "Strategy": [
            calc_cagr(strat_eq),
            calc_mdd(strat_eq),
            calc_sharpe(strat_ret)
        ],
        "Buy & Hold": [
            calc_cagr(bh_eq),
            calc_mdd(bh_eq),
            calc_sharpe(bh_ret)
        ]
    }, index=["CAGR", "MDD", "Sharpe"])

    return df


def format_metric(v, name):
    if pd.isna(v):
        return "-"
    if name in ["CAGR", "MDD"]:
        return f"{v:.2%}"
    return f"{v:.2f}"


# =========================================================
# MAIN
# =========================================================
if run_button:
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
        st.stop()

    with st.spinner("Downloading market data and running strategy..."):
        prices = download_prices(tickers, str(start_date))

        if prices.empty:
            st.error("No price data was downloaded.")
            st.stop()

        # ensure selected order is preserved if possible
        cols = [c for c in tickers if c in prices.columns]
        if len(cols) == 0:
            st.error("Downloaded data does not contain requested tickers.")
            st.stop()

        prices = prices[cols].dropna(how="all")

        result = build_strategy(
            prices=prices,
            trend_window=trend_window,
            momentum_window=momentum_window,
            vol_window=vol_window,
            top_n=top_n,
            rebalance_freq=rebalance_freq
        )

        metrics_df = make_metrics_df(result)

    # =====================================================
    # SUMMARY
    # =====================================================
    st.subheader("Performance Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Strategy CAGR", format_metric(metrics_df.loc["CAGR", "Strategy"], "CAGR"))
    c2.metric("Strategy MDD", format_metric(metrics_df.loc["MDD", "Strategy"], "MDD"))
    c3.metric("Strategy Sharpe", format_metric(metrics_df.loc["Sharpe", "Strategy"], "Sharpe"))

    st.dataframe(
        metrics_df.apply(lambda col: [
            format_metric(col.iloc[0], "CAGR"),
            format_metric(col.iloc[1], "MDD"),
            format_metric(col.iloc[2], "Sharpe")
        ]),
        use_container_width=True
    )

    # =====================================================
    # EQUITY CURVE
    # =====================================================
    st.subheader("Equity Curve")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["strategy_equity"].index,
        y=result["strategy_equity"],
        mode="lines",
        name="Quant Strategy"
    ))
    fig.add_trace(go.Scatter(
        x=result["bh_equity"].index,
        y=result["bh_equity"],
        mode="lines",
        name="Equal Weight Buy & Hold"
    ))
    fig.update_layout(
        height=520,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend_title="Portfolio"
    )
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # DRAWDOWN
    # =====================================================
    st.subheader("Drawdown")

    strat_dd = result["strategy_equity"] / result["strategy_equity"].cummax() - 1
    bh_dd = result["bh_equity"] / result["bh_equity"].cummax() - 1

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=strat_dd.index,
        y=strat_dd,
        mode="lines",
        name="Quant Strategy DD"
    ))
    fig_dd.add_trace(go.Scatter(
        x=bh_dd.index,
        y=bh_dd,
        mode="lines",
        name="Buy & Hold DD"
    ))
    fig_dd.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Drawdown"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # =====================================================
    # CURRENT WEIGHTS
    # =====================================================
    st.subheader("Latest Portfolio Weights")
    latest_weights = result["weights"].iloc[-1]
    latest_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)

    if len(latest_weights) == 0:
        st.info("Current allocation is fully in cash based on the strategy filters.")
    else:
        st.dataframe(
            latest_weights.to_frame("Weight").style.format({"Weight": "{:.2%}"}),
            use_container_width=True
        )

    # =====================================================
    # LATEST INDICATORS
    # =====================================================
    st.subheader("Latest Indicators")

    latest_date = prices.index[-1]
    latest_table = pd.DataFrame({
        "Price": prices.loc[latest_date],
        f"MA({trend_window})": result["ma"].loc[latest_date],
        f"Momentum({momentum_window}d)": result["momentum"].loc[latest_date],
        f"Volatility({vol_window}d)": result["vol"].loc[latest_date]
    })

    st.dataframe(
        latest_table.style.format({
            "Price": "{:.2f}",
            f"MA({trend_window})": "{:.2f}",
            f"Momentum({momentum_window}d)": "{:.2%}",
            f"Volatility({vol_window}d)": "{:.2%}"
        }),
        use_container_width=True
    )

    # =====================================================
    # PRICE CHART
    # =====================================================
    st.subheader("Price Chart")

    selected_ticker = st.selectbox("Select ticker", list(prices.columns))
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=prices.index,
        y=prices[selected_ticker],
        mode="lines",
        name=selected_ticker
    ))
    fig_price.add_trace(go.Scatter(
        x=result["ma"].index,
        y=result["ma"][selected_ticker],
        mode="lines",
        name=f"MA({trend_window})"
    ))
    fig_price.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig_price, use_container_width=True)

else:
    st.info("Set parameters in the sidebar and click 'Run Strategy'.")
