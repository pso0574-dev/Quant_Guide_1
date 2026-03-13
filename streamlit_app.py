# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Nasdaq-100 Quant Screener", layout="wide")

st.title("Nasdaq-100 Quant Screener")
st.caption("Tabs by strategy: momentum, quality, growth, value, defensive, multi-factor")

@st.cache_data(ttl=3600)
def get_nasdaq100_from_wiki():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in cols):
            df = t.copy()
            break
    else:
        raise ValueError("Nasdaq-100 table not found")

    ticker_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "ticker" in lc or "symbol" in lc:
            ticker_col = c
            break

    name_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "company" in lc or "security" in lc or "name" in lc:
            name_col = c
            break

    if ticker_col is None:
        raise ValueError("Ticker column not found")

    out = pd.DataFrame()
    out["Ticker"] = df[ticker_col].astype(str).str.replace(".", "-", regex=False)
    out["Name"] = df[name_col].astype(str) if name_col else out["Ticker"]
    return out.drop_duplicates().reset_index(drop=True)

@st.cache_data(ttl=3600)
def download_price_data(tickers, period="2y"):
    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return data

def compute_metrics(price_data, tickers):
    rows = []

    for t in tickers:
        try:
            df = price_data[t].dropna().copy()
            if len(df) < 220:
                continue

            close = df["Close"]
            vol = df["Volume"]

            price = close.iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]

            ret_1m = close.iloc[-1] / close.iloc[-22] - 1 if len(close) > 22 else np.nan
            ret_3m = close.iloc[-1] / close.iloc[-63] - 1 if len(close) > 63 else np.nan
            ret_6m = close.iloc[-1] / close.iloc[-126] - 1 if len(close) > 126 else np.nan
            ret_12m = close.iloc[-1] / close.iloc[-252] - 1 if len(close) > 252 else np.nan

            high_52w = close.tail(252).max()
            dist_high = price / high_52w - 1

            daily_ret = close.pct_change().dropna()
            vol_1y = daily_ret.tail(252).std() * np.sqrt(252)

            running_max = close.cummax()
            dd = close / running_max - 1
            mdd_1y = dd.tail(252).min()

            avg_vol_dollar = (close * vol).tail(63).mean()

            rows.append({
                "Ticker": t,
                "Price": price,
                "MA50": ma50,
                "MA200": ma200,
                "Ret_1M": ret_1m,
                "Ret_3M": ret_3m,
                "Ret_6M": ret_6m,
                "Ret_12M": ret_12m,
                "High_52W": high_52w,
                "Dist_52W_High": dist_high,
                "Volatility_1Y": vol_1y,
                "MDD_1Y": mdd_1y,
                "Avg_Dollar_Volume": avg_vol_dollar,
                "Trend_OK": int(price > ma50 and ma50 > ma200),
                "Above_MA200": int(price > ma200),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def download_fundamentals(tickers):
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker": t,
                "ROE": info.get("returnOnEquity", np.nan),
                "OperatingMargin": info.get("operatingMargins", np.nan),
                "GrossMargin": info.get("grossMargins", np.nan),
                "DebtToEquity": info.get("debtToEquity", np.nan),
                "RevenueGrowth": info.get("revenueGrowth", np.nan),
                "EpsGrowth": info.get("earningsGrowth", np.nan),
                "ForwardPE": info.get("forwardPE", np.nan),
                "PegRatio": info.get("pegRatio", np.nan),
                "Beta": info.get("beta", np.nan),
            })
        except Exception:
            rows.append({"Ticker": t})
    return pd.DataFrame(rows)

def pct_rank(series, ascending=True):
    s = series.replace([np.inf, -np.inf], np.nan)
    return s.rank(pct=True, ascending=ascending)

def build_scores(df):
    x = df.copy()

    x["MomentumScore"] = (
        0.35 * pct_rank(x["Ret_12M"], ascending=True) +
        0.25 * pct_rank(x["Ret_6M"], ascending=True) +
        0.20 * pct_rank(x["Dist_52W_High"], ascending=True) +
        0.20 * x["Trend_OK"].fillna(0)
    )

    x["QualityScore"] = (
        0.35 * pct_rank(x["ROE"], ascending=True) +
        0.25 * pct_rank(x["OperatingMargin"], ascending=True) +
        0.20 * pct_rank(x["GrossMargin"], ascending=True) +
        0.20 * pct_rank(-x["DebtToEquity"], ascending=True)
    )

    x["GrowthScore"] = (
        0.60 * pct_rank(x["RevenueGrowth"], ascending=True) +
        0.40 * pct_rank(x["EpsGrowth"], ascending=True)
    )

    x["ValueScore"] = (
        0.55 * pct_rank(-x["ForwardPE"], ascending=True) +
        0.45 * pct_rank(-x["PegRatio"], ascending=True)
    )

    x["DefensiveScore"] = (
        0.40 * pct_rank(-x["Volatility_1Y"], ascending=True) +
        0.30 * pct_rank(x["MDD_1Y"], ascending=True) +
        0.15 * pct_rank(-x["Beta"], ascending=True) +
        0.15 * x["Above_MA200"].fillna(0)
    )

    x["TotalScore"] = (
        0.35 * x["MomentumScore"].fillna(0) +
        0.30 * x["QualityScore"].fillna(0) +
        0.20 * x["GrowthScore"].fillna(0) +
        0.15 * x["ValueScore"].fillna(0)
    )

    return x

# -----------------------
# Load data
# -----------------------
universe = get_nasdaq100_from_wiki()
tickers = universe["Ticker"].tolist()

price_data = download_price_data(tickers, period="2y")
tech_df = compute_metrics(price_data, tickers)
fund_df = download_fundamentals(tickers)

df = tech_df.merge(fund_df, on="Ticker", how="left").merge(universe, on="Ticker", how="left")
df = build_scores(df)

st.sidebar.header("Filters")
top_n = st.sidebar.slider("Top N", 5, 30, 15)
min_dollar_volume = st.sidebar.number_input("Min Avg Dollar Volume", value=20_000_000)
filtered = df[df["Avg_Dollar_Volume"].fillna(0) >= min_dollar_volume].copy()

tabs = st.tabs([
    "Overview", "Momentum", "Quality", "Growth",
    "Value", "Defensive", "Multi-Factor"
])

# Overview
with tabs[0]:
    st.subheader("Universe Overview")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            filtered.sort_values("Ret_12M", ascending=False).head(20),
            x="Ticker", y="Ret_12M", title="Top 20 by 12M Return"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            filtered,
            x="Ret_6M",
            y="Ret_12M",
            hover_data=["Ticker", "Name"],
            title="6M vs 12M Return"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        filtered.sort_values("Ret_12M", ascending=False)[[
            "Ticker", "Name", "Price", "Ret_1M", "Ret_3M", "Ret_6M", "Ret_12M", "MA200"
        ]],
        use_container_width=True
    )

# Momentum
with tabs[1]:
    st.subheader("Momentum Strategy")
    st.write("Rule example: Price > MA50 > MA200, positive 6M/12M return, near 52-week high")

    mom = filtered.sort_values("MomentumScore", ascending=False).head(top_n)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(mom, x="Ticker", y="MomentumScore", title="Momentum Score Ranking")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            filtered,
            x="Ret_6M",
            y="Ret_12M",
            size="MomentumScore",
            hover_data=["Ticker"],
            title="Momentum Evidence"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(mom[[
        "Ticker", "Name", "Price", "Ret_6M", "Ret_12M",
        "Dist_52W_High", "Trend_OK", "MomentumScore"
    ]], use_container_width=True)

# Quality
with tabs[2]:
    st.subheader("Quality Strategy")
    st.write("Rule example: high ROE, high margins, controlled debt")

    q = filtered.sort_values("QualityScore", ascending=False).head(top_n)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(q, x="Ticker", y="ROE", title="Top ROE")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            filtered,
            x="ROE",
            y="OperatingMargin",
            size="QualityScore",
            hover_data=["Ticker"],
            title="ROE vs Operating Margin"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(q[[
        "Ticker", "Name", "ROE", "OperatingMargin",
        "GrossMargin", "DebtToEquity", "QualityScore"
    ]], use_container_width=True)

# Growth
with tabs[3]:
    st.subheader("Growth Strategy")
    st.write("Rule example: high revenue growth and EPS growth")

    g = filtered.sort_values("GrowthScore", ascending=False).head(top_n)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(g, x="Ticker", y="RevenueGrowth", title="Revenue Growth Ranking")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            filtered,
            x="RevenueGrowth",
            y="EpsGrowth",
            size="GrowthScore",
            hover_data=["Ticker"],
            title="Revenue Growth vs EPS Growth"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(g[[
        "Ticker", "Name", "RevenueGrowth", "EpsGrowth", "GrowthScore"
    ]], use_container_width=True)

# Value
with tabs[4]:
    st.subheader("Value / Reasonable Growth")
    st.write("Rule example: lower Forward PE and PEG within the Nasdaq-100 universe")

    v = filtered.sort_values("ValueScore", ascending=False).head(top_n)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(v, x="Ticker", y="ForwardPE", title="Forward PE")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            filtered,
            x="ForwardPE",
            y="EpsGrowth",
            size="ValueScore",
            hover_data=["Ticker"],
            title="Forward PE vs EPS Growth"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(v[[
        "Ticker", "Name", "ForwardPE", "PegRatio", "EpsGrowth", "ValueScore"
    ]], use_container_width=True)

# Defensive
with tabs[5]:
    st.subheader("Low Vol / Defensive")
    st.write("Rule example: lower volatility, smaller drawdown, lower beta")

    d = filtered.sort_values("DefensiveScore", ascending=False).head(top_n)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(d, x="Ticker", y="Volatility_1Y", title="1Y Volatility")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            filtered,
            x="Volatility_1Y",
            y="Ret_12M",
            size="DefensiveScore",
            hover_data=["Ticker"],
            title="Return vs Volatility"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(d[[
        "Ticker", "Name", "Volatility_1Y", "MDD_1Y", "Beta", "DefensiveScore"
    ]], use_container_width=True)

# Multi-Factor
with tabs[6]:
    st.subheader("Multi-Factor Ranking")
    st.write("Recommended practical view: Momentum 35%, Quality 30%, Growth 20%, Value 15%")

    mf = filtered.sort_values("TotalScore", ascending=False).head(top_n)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(mf, x="Ticker", y="TotalScore", title="Top Multi-Factor Ranking")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        heat_df = mf[[
            "Ticker", "MomentumScore", "QualityScore", "GrowthScore", "ValueScore", "TotalScore"
        ]].set_index("Ticker")
        fig = px.imshow(heat_df.T, aspect="auto", title="Factor Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(mf[[
        "Ticker", "Name",
        "MomentumScore", "QualityScore", "GrowthScore", "ValueScore", "TotalScore"
    ]], use_container_width=True)
