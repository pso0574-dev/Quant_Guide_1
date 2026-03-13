# -*- coding: utf-8 -*-
"""
Quant Strategy Pro Dashboard
- Multilingual UI: English / 한국어 / English + 한국어
- Market Regime
- Relative Ratio Analysis
- Advanced Backtests with transaction cost
- Rolling metrics
- Monthly return heatmap
- Portfolio construction / risk contribution
- Strategy learning tabs
"""

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
    page_title="Quant Strategy Pro Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# Sidebar - Language
# =========================================================
st.sidebar.header("App Settings / 앱 설정")

language_mode = st.sidebar.selectbox(
    "Language / 언어",
    ["English", "한국어", "English + 한국어"],
    index=2
)

# =========================================================
# Translation Dictionary
# =========================================================
TEXT = {
    "app_title": {
        "en": "📈 Quant Strategy Pro Dashboard",
        "ko": "📈 퀀트 전략 프로 대시보드",
    },
    "app_caption": {
        "en": "Professional research-style quant dashboard with regime, ratios, portfolio, and advanced backtests.",
        "ko": "국면 분석, 비율 분석, 포트폴리오, 고급 백테스트를 포함한 전문가형 퀀트 대시보드입니다.",
    },
    "market_settings": {"en": "Market Settings", "ko": "시장 설정"},
    "backtest_settings": {"en": "Backtest Settings", "ko": "백테스트 설정"},
    "main_asset": {"en": "Main Asset", "ko": "주요 자산"},
    "custom_ticker": {"en": "Custom Ticker", "ko": "사용자 지정 티커"},
    "comparison_assets": {"en": "Comparison Assets", "ko": "비교 자산"},
    "history_period": {"en": "History Period", "ko": "조회 기간"},
    "interval": {"en": "Interval", "ko": "간격"},
    "short_ma": {"en": "Short MA", "ko": "단기 이동평균"},
    "long_ma": {"en": "Long MA", "ko": "장기 이동평균"},
    "rsi_period": {"en": "RSI Period", "ko": "RSI 기간"},
    "vol_window": {"en": "Vol Window", "ko": "변동성 기간"},
    "mr_window": {"en": "Mean Reversion Window", "ko": "평균회귀 기간"},
    "mr_z": {"en": "MR Z Threshold", "ko": "평균회귀 Z 임계값"},
    "mom_lookback": {"en": "Momentum Lookback", "ko": "모멘텀 조회 기간"},
    "target_vol": {"en": "Target Volatility", "ko": "목표 변동성"},
    "initial_capital": {"en": "Initial Capital", "ko": "초기 자본"},
    "trading_cost_bps": {"en": "Trading Cost (bps per turnover)", "ko": "거래 비용 (턴오버당 bps)"},
    "show_signals": {"en": "Show Buy/Sell Signals", "ko": "매수/매도 신호 표시"},
    "show_monthly_heatmap": {"en": "Show Monthly Heatmap", "ko": "월별 히트맵 표시"},
    "no_data": {
        "en": "No data loaded. Please check ticker or try again later.",
        "ko": "데이터를 불러오지 못했습니다. 티커를 확인하거나 나중에 다시 시도해 주세요.",
    },
    "missing_column": {"en": "Missing column", "ko": "누락된 열"},
    "na": {"en": "N/A", "ko": "N/A"},
    "overview_tab": {"en": "Overview", "ko": "개요"},
    "regime_tab": {"en": "Market Regime", "ko": "시장 국면"},
    "ratio_tab": {"en": "Relative Ratios", "ko": "상대 비율"},
    "trend_tab": {"en": "Trend Following", "ko": "추세추종"},
    "meanrev_tab": {"en": "Mean Reversion", "ko": "평균회귀"},
    "momentum_tab": {"en": "Momentum", "ko": "모멘텀"},
    "rotation_tab": {"en": "Asset Rotation", "ko": "자산 로테이션"},
    "voltarget_tab": {"en": "Volatility Targeting", "ko": "변동성 타게팅"},
    "riskparity_tab": {"en": "Risk Parity", "ko": "리스크 패리티"},
    "portfolio_tab": {"en": "Portfolio Construction", "ko": "포트폴리오 구성"},
    "drawdown_tab": {"en": "Drawdown Buying", "ko": "낙폭 기반 매수"},
    "multifactor_tab": {"en": "Multi-Factor", "ko": "멀티팩터"},
    "advanced_bt_tab": {"en": "Advanced Backtest", "ko": "고급 백테스트"},
    "summary_tab": {"en": "Strategy Summary", "ko": "전략 요약"},
    "last_price": {"en": "Last Price", "ko": "현재가"},
    "rsi": {"en": "RSI", "ko": "RSI"},
    "volatility": {"en": "Volatility", "ko": "변동성"},
    "drawdown": {"en": "Drawdown", "ko": "낙폭"},
    "from_ath": {"en": "From ATH", "ko": "ATH 대비"},
    "from_52w_high": {"en": "From 52W High", "ko": "52주 고점 대비"},
    "cagr": {"en": "CAGR", "ko": "연복리수익률"},
    "sharpe": {"en": "Sharpe", "ko": "샤프지수"},
    "sortino": {"en": "Sortino", "ko": "소르티노지수"},
    "calmar": {"en": "Calmar", "ko": "칼마 지수"},
    "max_drawdown": {"en": "Max Drawdown", "ko": "최대낙폭"},
    "rolling_sharpe": {"en": "Rolling Sharpe", "ko": "롤링 샤프"},
    "rolling_cagr": {"en": "Rolling CAGR", "ko": "롤링 CAGR"},
    "overview_title": {"en": "Overview", "ko": "개요"},
    "overview_learn": {
        "en": (
            "**What this tab does**\n"
            "- Summarizes the current state of the asset\n"
            "- Shows price, trend, RSI, and drawdown together\n"
            "- Helps you decide whether the asset is strong, stretched, or correcting"
        ),
        "ko": (
            "**이 탭의 역할**\n"
            "- 자산의 현재 상태를 요약합니다\n"
            "- 가격, 추세, RSI, 낙폭을 함께 보여줍니다\n"
            "- 자산이 강한지, 과열인지, 조정 중인지 판단하는 데 도움을 줍니다"
        ),
    },
    "market_regime_title": {"en": "Market Regime Engine", "ko": "시장 국면 엔진"},
    "regime_desc": {
        "en": (
            "**Core idea**\n"
            "Classify the current market environment before applying any strategy.\n\n"
            "**This panel combines**\n"
            "- Trend status of QQQ / SPY / IWM / TLT / GLD\n"
            "- Leadership ratios like IWM/QQQ and QQQ/SPY\n"
            "- Defensive vs offensive behavior\n"
            "- Cross-asset confirmation"
        ),
        "ko": (
            "**핵심 개념**\n"
            "전략을 적용하기 전에 현재 시장 환경을 먼저 분류합니다.\n\n"
            "**이 패널은 다음을 결합합니다**\n"
            "- QQQ / SPY / IWM / TLT / GLD 추세 상태\n"
            "- IWM/QQQ, QQQ/SPY 같은 리더십 비율\n"
            "- 방어적 자산과 공격적 자산의 상대 강도\n"
            "- 여러 자산의 동시 확인"
        ),
    },
    "regime_state": {"en": "Regime State", "ko": "국면 상태"},
    "broad_risk_on": {"en": "Broad Risk-On", "ko": "광범위 위험선호"},
    "narrow_risk_on": {"en": "Narrow Risk-On", "ko": "협소한 위험선호"},
    "neutral_mixed": {"en": "Neutral / Mixed", "ko": "중립 / 혼조"},
    "defensive": {"en": "Defensive", "ko": "방어적"},
    "risk_off": {"en": "Risk-Off", "ko": "위험회피"},
    "regime_score": {"en": "Regime Score", "ko": "국면 점수"},
    "breadth_score": {"en": "Breadth Score", "ko": "브레드스 점수"},
    "leadership_score": {"en": "Leadership Score", "ko": "리더십 점수"},
    "defense_score": {"en": "Defense Score", "ko": "방어 점수"},
    "asset_state_table": {"en": "Asset State Table", "ko": "자산 상태 표"},
    "asset": {"en": "Asset", "ko": "자산"},
    "close": {"en": "Close", "ko": "종가"},
    "above_ma200": {"en": "Above MA200", "ko": "MA200 상단"},
    "6m_momentum": {"en": "6M Momentum", "ko": "6개월 모멘텀"},
    "drawdown_col": {"en": "Drawdown", "ko": "낙폭"},
    "ratio_title": {"en": "Relative Ratio Analysis", "ko": "상대 비율 분석"},
    "ratio_desc": {
        "en": (
            "**Core idea**\n"
            "Relative ratios show leadership shifts better than absolute price charts.\n\n"
            "**Useful ratios**\n"
            "- QQQ/SPY\n"
            "- IWM/QQQ\n"
            "- IWM/SPY\n"
            "- SPY/TLT\n"
            "- GLD/TLT"
        ),
        "ko": (
            "**핵심 개념**\n"
            "상대 비율 차트는 절대 가격보다 리더십 변화를 더 잘 보여줍니다.\n\n"
            "**유용한 비율**\n"
            "- QQQ/SPY\n"
            "- IWM/QQQ\n"
            "- IWM/SPY\n"
            "- SPY/TLT\n"
            "- GLD/TLT"
        ),
    },
    "trend_desc": {
        "en": (
            "**Core idea**\n"
            "Follow the prevailing market direction using moving averages.\n\n"
            "**Simple rule**\n"
            "- Long when short MA > long MA\n"
            "- Flat when short MA <= long MA"
        ),
        "ko": (
            "**핵심 개념**\n"
            "이동평균으로 지배적인 시장 방향을 따라갑니다.\n\n"
            "**간단 룰**\n"
            "- 단기 MA > 장기 MA 이면 보유\n"
            "- 단기 MA <= 장기 MA 이면 현금"
        ),
    },
    "meanrev_desc": {
        "en": (
            "**Core idea**\n"
            "If price deviates too far below its rolling mean, it may revert upward.\n\n"
            "**Simple rule**\n"
            "- Enter when Z-score < -threshold\n"
            "- Exit when price recovers to the rolling mean"
        ),
        "ko": (
            "**핵심 개념**\n"
            "가격이 이동평균보다 과도하게 아래로 이탈하면 위로 되돌아올 수 있다고 가정합니다.\n\n"
            "**간단 룰**\n"
            "- Z-score < -임계값이면 진입\n"
            "- 가격이 이동평균으로 회복하면 청산"
        ),
    },
    "momentum_desc": {
        "en": (
            "**Core idea**\n"
            "Assets with positive recent returns often continue to outperform for some time.\n\n"
            "**Simple rule**\n"
            "- Long when lookback momentum > 0\n"
            "- Flat when momentum <= 0"
        ),
        "ko": (
            "**핵심 개념**\n"
            "최근 수익률이 좋았던 자산이 일정 기간 계속 outperform할 수 있다고 가정합니다.\n\n"
            "**간단 룰**\n"
            "- 조회 기간 모멘텀 > 0이면 보유\n"
            "- 모멘텀 <= 0이면 현금"
        ),
    },
    "rotation_desc": {
        "en": (
            "**Core idea**\n"
            "Rotate into the strongest asset among several candidates.\n\n"
            "**Simple rule**\n"
            "- Select the asset with the highest lookback momentum\n"
            "- Hold only that leader in the next period"
        ),
        "ko": (
            "**핵심 개념**\n"
            "여러 자산 중 가장 강한 자산으로 이동합니다.\n\n"
            "**간단 룰**\n"
            "- 조회 기간 모멘텀이 가장 높은 자산을 선택\n"
            "- 다음 기간에는 그 자산만 보유"
        ),
    },
    "voltarget_desc": {
        "en": (
            "**Core idea**\n"
            "Scale exposure up or down to target a stable volatility level."
        ),
        "ko": (
            "**핵심 개념**\n"
            "안정적인 목표 변동성 수준을 유지하도록 비중을 조절합니다."
        ),
    },
    "riskparity_desc": {
        "en": (
            "**Core idea**\n"
            "Allocate capital based on risk contribution, not nominal capital."
        ),
        "ko": (
            "**핵심 개념**\n"
            "명목 금액이 아니라 위험 기여도 기준으로 자산을 배분합니다."
        ),
    },
    "portfolio_desc": {
        "en": (
            "**Core idea**\n"
            "Translate signals into weights, risk budgets, and current allocation decisions."
        ),
        "ko": (
            "**핵심 개념**\n"
            "신호를 실제 비중, 위험 예산, 현재 배분 결정으로 연결합니다."
        ),
    },
    "drawdown_desc": {
        "en": (
            "**Core idea**\n"
            "Use drawdown zones to phase entries instead of buying all at once."
        ),
        "ko": (
            "**핵심 개념**\n"
            "한 번에 매수하지 않고 낙폭 구간에 따라 분할 진입합니다."
        ),
    },
    "multifactor_desc": {
        "en": (
            "**Core idea**\n"
            "Combine multiple signals instead of relying on one factor alone."
        ),
        "ko": (
            "**핵심 개념**\n"
            "하나의 요인에만 의존하지 않고 여러 신호를 결합합니다."
        ),
    },
    "advanced_bt_desc": {
        "en": (
            "**Core idea**\n"
            "A professional backtest should include costs, turnover, rolling metrics, and period breakdowns."
        ),
        "ko": (
            "**핵심 개념**\n"
            "전문가형 백테스트는 비용, 턴오버, 롤링 지표, 기간별 분해를 포함해야 합니다."
        ),
    },
    "summary_desc": {
        "en": (
            "**What this tab does**\n"
            "- Compares all strategies on the same asset base\n"
            "- Helps you judge not only return, but also the quality of the return"
        ),
        "ko": (
            "**이 탭의 역할**\n"
            "- 같은 자산 기반에서 여러 전략을 비교합니다\n"
            "- 단순 수익률이 아니라 수익의 질까지 판단하도록 돕습니다"
        ),
    },
    "need_two_assets": {"en": "Need at least 2 assets.", "ko": "최소 2개 이상의 자산이 필요합니다."},
    "not_enough_data": {"en": "Not enough data for this analysis.", "ko": "이 분석을 수행하기에 데이터가 충분하지 않습니다."},
    "buy_hold": {"en": "Buy & Hold", "ko": "매수 후 보유"},
    "trend_strategy": {"en": "Trend Strategy", "ko": "추세 전략"},
    "meanrev_strategy": {"en": "Mean Reversion", "ko": "평균회귀"},
    "momentum_strategy": {"en": "Momentum Strategy", "ko": "모멘텀 전략"},
    "rotation_strategy": {"en": "Rotation Strategy", "ko": "로테이션 전략"},
    "vol_target_strategy": {"en": "Vol Target Strategy", "ko": "변동성 타게팅 전략"},
    "risk_parity_strategy": {"en": "Risk Parity", "ko": "리스크 패리티"},
    "equal_weight_strategy": {"en": "Equal Weight", "ko": "동일가중"},
    "price": {"en": "Price", "ko": "가격"},
    "equity_curve": {"en": "Equity Curve", "ko": "자산곡선"},
    "buy": {"en": "Buy", "ko": "매수"},
    "sell": {"en": "Sell", "ko": "매도"},
    "exit": {"en": "Exit", "ko": "청산"},
    "signal": {"en": "Signal", "ko": "신호"},
    "position": {"en": "Position", "ko": "포지션"},
    "turnover": {"en": "Turnover", "ko": "턴오버"},
    "exposure": {"en": "Exposure", "ko": "노출도"},
    "final_value": {"en": "Final Value", "ko": "최종 자산"},
    "strategy": {"en": "Strategy", "ko": "전략"},
    "recent_rotation_decisions": {"en": "Recent Rotation Decisions", "ko": "최근 로테이션 결정"},
    "selected_asset": {"en": "Selected Asset", "ko": "선택 자산"},
    "factor_score": {"en": "Factor Score", "ko": "팩터 점수"},
    "positive_momentum": {"en": "Positive momentum", "ko": "긍정적 모멘텀"},
    "weak_momentum": {"en": "Weak momentum", "ko": "약한 모멘텀"},
    "below_median_vol": {"en": "Below-median volatility", "ko": "중앙값 이하 변동성"},
    "above_median_vol": {"en": "Above-median volatility", "ko": "중앙값 이상 변동성"},
    "trend_confirmed": {"en": "Trend confirmed", "ko": "추세 확인"},
    "trend_not_confirmed": {"en": "Trend not confirmed", "ko": "추세 미확인"},
    "quality_strong": {"en": "Quality proxy strong", "ko": "퀄리티 대용지표 강함"},
    "quality_weak": {"en": "Quality proxy weak", "ko": "퀄리티 대용지표 약함"},
    "composite_factor_score": {"en": "Composite Factor Score", "ko": "종합 팩터 점수"},
    "current_target_weights": {"en": "Current Target Weights", "ko": "현재 목표 비중"},
    "risk_contribution": {"en": "Risk Contribution", "ko": "위험 기여도"},
    "signal_by_asset": {"en": "Signal by Asset", "ko": "자산별 신호"},
    "monthly_heatmap": {"en": "Monthly Return Heatmap", "ko": "월별 수익률 히트맵"},
    "year": {"en": "Year", "ko": "연도"},
    "underwater_chart": {"en": "Underwater Chart", "ko": "언더워터 차트"},
    "cost_adjusted": {"en": "Cost-Adjusted", "ko": "비용 반영"},
    "raw": {"en": "Raw", "ko": "원시"},
    "last_refresh": {"en": "Last app refresh", "ko": "마지막 앱 새로고침"},
}

# =========================================================
# Translation Helpers
# =========================================================
def tr(key: str) -> str:
    item = TEXT.get(key, {})
    en = item.get("en", key)
    ko = item.get("ko", key)
    if language_mode == "English":
        return en
    elif language_mode == "한국어":
        return ko
    else:
        return f"{en}\n\n{ko}"

def tr_tab(key: str) -> str:
    item = TEXT.get(key, {})
    en = item.get("en", key)
    ko = item.get("ko", key)
    if language_mode == "English":
        return en
    elif language_mode == "한국어":
        return ko
    else:
        return f"{en} / {ko}"

def fmt_metric(value, kind="float"):
    if pd.isna(value):
        return tr("na")
    if kind == "price":
        return f"{value:,.2f}"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "int":
        return f"{value:,.0f}"
    return f"{value}"

# =========================================================
# Title
# =========================================================
st.title(tr("app_title"))
st.caption(tr("app_caption"))

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header(tr("market_settings"))

preset_universe = {
    "QQQ": "QQQ",
    "SPY": "SPY",
    "IWM": "IWM",
    "TLT": "TLT",
    "GLD": "GLD",
    "Custom": None
}

asset_choice = st.sidebar.selectbox(
    tr("main_asset"),
    list(preset_universe.keys()),
    index=0
)

if asset_choice == "Custom":
    ticker = st.sidebar.text_input(tr("custom_ticker"), value="QQQ").upper().strip()
else:
    ticker = preset_universe[asset_choice]

comparison_assets = st.sidebar.multiselect(
    tr("comparison_assets"),
    ["QQQ", "SPY", "IWM", "TLT", "GLD"],
    default=["QQQ", "SPY", "IWM", "TLT", "GLD"]
)

period = st.sidebar.selectbox(
    tr("history_period"),
    ["1y", "2y", "3y", "5y", "10y", "max"],
    index=4
)

interval = st.sidebar.selectbox(
    tr("interval"),
    ["1d", "1wk", "1mo"],
    index=0
)

ma_short = st.sidebar.slider(tr("short_ma"), 10, 100, 50, 5)
ma_long = st.sidebar.slider(tr("long_ma"), 50, 300, 200, 10)
rsi_period = st.sidebar.slider(tr("rsi_period"), 5, 30, 14, 1)
vol_window = st.sidebar.slider(tr("vol_window"), 10, 100, 20, 5)
mr_window = st.sidebar.slider(tr("mr_window"), 10, 100, 20, 5)
mr_z = st.sidebar.slider(tr("mr_z"), 0.5, 3.0, 1.5, 0.1)
mom_lookback = st.sidebar.slider(tr("mom_lookback"), 20, 252, 126, 5)
target_vol = st.sidebar.slider(tr("target_vol"), 0.05, 0.30, 0.12, 0.01)

st.sidebar.markdown("---")
st.sidebar.header(tr("backtest_settings"))

initial_capital = st.sidebar.number_input(
    tr("initial_capital"),
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)

trading_cost_bps = st.sidebar.slider(
    tr("trading_cost_bps"),
    min_value=0,
    max_value=100,
    value=10,
    step=1
)

show_signals = st.sidebar.checkbox(tr("show_signals"), value=True)
show_monthly_heatmap = st.sidebar.checkbox(tr("show_monthly_heatmap"), value=True)

# =========================================================
# Helper Functions
# =========================================================
def get_annualization_factor(interval_value: str) -> int:
    if interval_value == "1d":
        return 252
    elif interval_value == "1wk":
        return 52
    elif interval_value == "1mo":
        return 12
    return 252

def get_high_window(interval_value: str) -> int:
    if interval_value == "1d":
        return 252
    elif interval_value == "1wk":
        return 52
    elif interval_value == "1mo":
        return 12
    return 252

def get_rolling_window_1y(interval_value: str) -> int:
    if interval_value == "1d":
        return 252
    elif interval_value == "1wk":
        return 52
    elif interval_value == "1mo":
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

@st.cache_data(ttl=3600)
def load_close_data(tickers: list, period_value: str, interval_value: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        d = load_data(t, period_value, interval_value)
        if not d.empty and "Close" in d.columns:
            frames.append(d[["Close"]].rename(columns={"Close": t}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all")

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

def calc_sortino(returns: pd.Series, annual_factor: int) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if len(returns) < 2 or len(downside) < 2 or downside.std() == 0:
        return np.nan
    return (returns.mean() / downside.std()) * np.sqrt(annual_factor)

def calc_calmar(cagr_val: float, mdd_val: float) -> float:
    if pd.isna(cagr_val) or pd.isna(mdd_val) or mdd_val == 0:
        return np.nan
    return cagr_val / abs(mdd_val)

def equity_stats(equity: pd.Series, annual_factor: int):
    equity = equity.dropna()
    if len(equity) < 2:
        return {
            "CAGR": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "MDD": np.nan,
            "Exposure": np.nan,
            "Turnover": np.nan,
        }

    rets = equity.pct_change().dropna()
    cagr_val = calc_cagr(equity, annual_factor)
    sharpe_val = calc_sharpe(rets, annual_factor)
    sortino_val = calc_sortino(rets, annual_factor)
    _, mdd_val = calc_drawdown(equity)
    calmar_val = calc_calmar(cagr_val, mdd_val)

    return {
        "CAGR": cagr_val,
        "Sharpe": sharpe_val,
        "Sortino": sortino_val,
        "Calmar": calmar_val,
        "MDD": mdd_val,
    }

def apply_costs(strategy_returns: pd.Series, position: pd.Series, cost_bps: float):
    turnover = position.fillna(0).diff().abs().fillna(position.fillna(0).abs())
    cost_rate = cost_bps / 10000.0
    costs = turnover * cost_rate
    net_returns = strategy_returns.fillna(0) - costs
    return net_returns, turnover

def add_advanced_columns(bt: pd.DataFrame, annual_factor: int, initial_capital_value: float, cost_bps: float) -> pd.DataFrame:
    bt = bt.copy()
    if "Position" not in bt.columns:
        bt["Position"] = 0.0
    if "Strategy_Return" not in bt.columns:
        bt["Strategy_Return"] = 0.0

    net_returns, turnover = apply_costs(bt["Strategy_Return"], bt["Position"], cost_bps)
    bt["Turnover"] = turnover
    bt["Net_Strategy_Return"] = net_returns
    bt["Equity_Net"] = initial_capital_value * (1 + bt["Net_Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital_value * (1 + bt["Return"].fillna(0)).cumprod()
    bt["Exposure"] = bt["Position"].fillna(0)

    roll = get_rolling_window_1y(interval)
    bh_rets = bt["BuyHold_Equity"].pct_change()
    net_rets = bt["Equity_Net"].pct_change()

    bt["Rolling_Sharpe_Net"] = net_rets.rolling(roll).apply(
        lambda x: (np.mean(x) / np.std(x) * np.sqrt(annual_factor)) if np.std(x) != 0 else np.nan,
        raw=True
    )
    bt["Rolling_Sharpe_BH"] = bh_rets.rolling(roll).apply(
        lambda x: (np.mean(x) / np.std(x) * np.sqrt(annual_factor)) if np.std(x) != 0 else np.nan,
        raw=True
    )

    bt["Rolling_CAGR_Net"] = bt["Equity_Net"] / bt["Equity_Net"].shift(roll)
    bt["Rolling_CAGR_Net"] = bt["Rolling_CAGR_Net"] ** (annual_factor / roll) - 1

    bt["Rolling_CAGR_BH"] = bt["BuyHold_Equity"] / bt["BuyHold_Equity"].shift(roll)
    bt["Rolling_CAGR_BH"] = bt["Rolling_CAGR_BH"] ** (annual_factor / roll) - 1

    dd_net, _ = calc_drawdown(bt["Equity_Net"])
    dd_bh, _ = calc_drawdown(bt["BuyHold_Equity"])
    bt["Underwater_Net"] = dd_net
    bt["Underwater_BH"] = dd_bh
    return bt

def backtest_buy_hold(close: pd.Series, initial_capital_value: float) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    return initial_capital_value * (close / close.iloc[0])

def backtest_trend(close: pd.Series, ma_short_value: int, ma_long_value: int, initial_capital_value: float, cost_bps: float, annual_factor: int) -> pd.DataFrame:
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
    bt["Equity_Raw"] = initial_capital_value * (1 + bt["Strategy_Return"]).cumprod()
    bt = add_advanced_columns(bt, annual_factor, initial_capital_value, cost_bps)
    return bt

def backtest_mean_reversion(close: pd.Series, window_value: int, z_entry_value: float, initial_capital_value: float, cost_bps: float, annual_factor: int) -> pd.DataFrame:
    close = close.dropna()
    if len(close) < window_value + 5:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["Mean"] = close.rolling(window_value).mean()
    bt["Std"] = close.rolling(window_value).std()
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

        if in_pos == 0 and z < -z_entry_value:
            in_pos = 1
        elif in_pos == 1 and c >= m:
            in_pos = 0
        signal.append(in_pos)

    bt["Signal"] = signal
    bt["Position"] = bt["Signal"].shift(1).fillna(0)
    bt["Return"] = bt["Close"].pct_change().fillna(0)
    bt["Strategy_Return"] = bt["Position"] * bt["Return"]
    bt["Signal_Change"] = bt["Signal"].diff()
    bt["Equity_Raw"] = initial_capital_value * (1 + bt["Strategy_Return"]).cumprod()
    bt = add_advanced_columns(bt, annual_factor, initial_capital_value, cost_bps)
    return bt

def backtest_momentum(close: pd.Series, lookback_value: int, initial_capital_value: float, cost_bps: float, annual_factor: int) -> pd.DataFrame:
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
    bt["Signal_Change"] = bt["Signal"].diff()
    bt["Equity_Raw"] = initial_capital_value * (1 + bt["Strategy_Return"]).cumprod()
    bt = add_advanced_columns(bt, annual_factor, initial_capital_value, cost_bps)
    return bt

def backtest_vol_target(close: pd.Series, vol_window_value: int, target_vol_value: float, annual_factor: int, initial_capital_value: float, cost_bps: float) -> pd.DataFrame:
    close = close.dropna()
    if len(close) < vol_window_value + 5:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["Return"] = close.pct_change().fillna(0)
    bt["Realized_Vol"] = bt["Return"].rolling(vol_window_value).std() * np.sqrt(annual_factor)
    bt["Signal"] = 1
    bt["Position"] = (target_vol_value / bt["Realized_Vol"]).clip(upper=2.0).replace([np.inf, -np.inf], np.nan).fillna(0)
    bt["Position"] = bt["Position"].shift(1).fillna(0)
    bt["Strategy_Return"] = bt["Position"] * bt["Return"]
    bt["Equity_Raw"] = initial_capital_value * (1 + bt["Strategy_Return"]).cumprod()
    bt = add_advanced_columns(bt, annual_factor, initial_capital_value, cost_bps)
    return bt

def build_risk_parity(close_df: pd.DataFrame, vol_window_value: int, annual_factor: int, initial_capital_value: float, cost_bps: float):
    close_df = close_df.dropna(how="all")
    if close_df.empty or close_df.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    returns = close_df.pct_change().fillna(0)
    vol = returns.rolling(vol_window_value).std() * np.sqrt(annual_factor)
    inv_vol = 1 / vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)

    shifted_weights = weights.shift(1).fillna(0)
    rp_ret = (shifted_weights * returns).sum(axis=1)
    ew_w = pd.DataFrame(1 / returns.shape[1], index=returns.index, columns=returns.columns)
    eq_ret = (ew_w.shift(1).fillna(0) * returns).sum(axis=1)

    turnover_rp = shifted_weights.diff().abs().sum(axis=1).fillna(shifted_weights.abs().sum(axis=1))
    cost_rate = cost_bps / 10000.0

    rp_net = rp_ret - turnover_rp * cost_rate
    eq_net = eq_ret

    equity = pd.DataFrame(index=returns.index)
    equity["Risk_Parity_Raw"] = initial_capital_value * (1 + rp_ret).cumprod()
    equity["Risk_Parity_Net"] = initial_capital_value * (1 + rp_net).cumprod()
    equity["Equal_Weight_Raw"] = initial_capital_value * (1 + eq_ret).cumprod()
    equity["Equal_Weight_Net"] = initial_capital_value * (1 + eq_net).cumprod()
    return weights, equity, turnover_rp

def build_rotation_strategy(close_df: pd.DataFrame, lookback_value: int, initial_capital_value: float, cost_bps: float):
    close_df = close_df.dropna(how="all")
    if close_df.empty or close_df.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()

    momentum = close_df / close_df.shift(lookback_value) - 1
    leader = momentum.idxmax(axis=1)
    returns = close_df.pct_change().fillna(0)

    rot_ret = pd.Series(0.0, index=close_df.index)
    position_matrix = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)

    for i in range(1, len(close_df)):
        chosen = leader.iloc[i - 1]
        if pd.notna(chosen):
            rot_ret.iloc[i] = returns.iloc[i][chosen]
            position_matrix.iloc[i, position_matrix.columns.get_loc(chosen)] = 1.0

    turnover = position_matrix.diff().abs().sum(axis=1).fillna(position_matrix.abs().sum(axis=1))
    cost_rate = cost_bps / 10000.0
    rot_net = rot_ret - turnover * cost_rate

    out = pd.DataFrame(index=close_df.index)
    out["Leader"] = leader
    out["Rotation_Return"] = rot_ret
    out["Rotation_Return_Net"] = rot_net
    out["Turnover"] = turnover
    out["Rotation_Equity_Raw"] = initial_capital_value * (1 + rot_ret).cumprod()
    out["Rotation_Equity_Net"] = initial_capital_value * (1 + rot_net).cumprod()
    return momentum, out

def latest_factor_score(df_in: pd.DataFrame, mom_lookback_value: int, vol_window_value: int):
    d = df_in.copy()
    d["Momentum"] = d["Close"] / d["Close"].shift(mom_lookback_value) - 1
    d["Volatility"] = d["Close"].pct_change().rolling(vol_window_value).std()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()
    d["Quality_Proxy"] = d["Close"] / d["Close"].rolling(252).min() - 1

    latest_row = d.iloc[-1]
    score = 0
    reasons = []

    if pd.notna(latest_row["Momentum"]) and latest_row["Momentum"] > 0:
        score += 1
        reasons.append(tr("positive_momentum"))
    else:
        reasons.append(tr("weak_momentum"))

    if pd.notna(latest_row["Volatility"]):
        vol_med = d["Volatility"].median()
        if latest_row["Volatility"] < vol_med:
            score += 1
            reasons.append(tr("below_median_vol"))
        else:
            reasons.append(tr("above_median_vol"))

    if pd.notna(latest_row["MA50"]) and pd.notna(latest_row["MA200"]) and latest_row["MA50"] > latest_row["MA200"]:
        score += 1
        reasons.append(tr("trend_confirmed"))
    else:
        reasons.append(tr("trend_not_confirmed"))

    if pd.notna(latest_row["Quality_Proxy"]):
        q_med = d["Quality_Proxy"].median()
        if latest_row["Quality_Proxy"] > q_med:
            score += 1
            reasons.append(tr("quality_strong"))
        else:
            reasons.append(tr("quality_weak"))

    return score, reasons, d

def get_monthly_heatmap_from_equity(equity: pd.Series) -> pd.DataFrame:
    equity = equity.dropna()
    if len(equity) < 3:
        return pd.DataFrame()

    monthly = equity.resample("ME").last().pct_change()
    heat = monthly.to_frame("ret")
    heat["Year"] = heat.index.year
    heat["Month"] = heat.index.strftime("%b")
    pivot = heat.pivot(index="Year", columns="Month", values="ret")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=month_order)
    return pivot

def normalize_series(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series * np.nan
    return 100 * series / s.iloc[0]

def create_ratio_series(close_df: pd.DataFrame, num: str, den: str) -> pd.Series:
    if num in close_df.columns and den in close_df.columns:
        return close_df[num] / close_df[den]
    return pd.Series(dtype=float)

def regime_engine(close_df: pd.DataFrame, annual_factor: int):
    needed = ["QQQ", "SPY", "IWM", "TLT", "GLD"]
    available = [x for x in needed if x in close_df.columns]
    if len(available) < 3:
        return {
            "state": tr("na"),
            "regime_score": np.nan,
            "breadth_score": np.nan,
            "leadership_score": np.nan,
            "defense_score": np.nan,
            "table": pd.DataFrame()
        }

    rows = []
    for col in available:
        s = close_df[col].dropna()
        if len(s) < 220:
            continue
        ma200 = s.rolling(200).mean().iloc[-1]
        mom6 = s.iloc[-1] / s.shift(126).iloc[-1] - 1 if len(s) > 126 else np.nan
        dd, _ = calc_drawdown(s)
        rows.append({
            tr("asset"): col,
            tr("close"): s.iloc[-1],
            tr("above_ma200"): bool(s.iloc[-1] > ma200) if pd.notna(ma200) else False,
            tr("6m_momentum"): mom6,
            tr("drawdown_col"): dd.iloc[-1]
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return {
            "state": tr("na"),
            "regime_score": np.nan,
            "breadth_score": np.nan,
            "leadership_score": np.nan,
            "defense_score": np.nan,
            "table": table
        }

    breadth_score = int(table[tr("above_ma200")].sum())
    leadership_score = 0
    defense_score = 0

    ratio_iwm_qqq = create_ratio_series(close_df, "IWM", "QQQ")
    ratio_qqq_spy = create_ratio_series(close_df, "QQQ", "SPY")
    ratio_spy_tlt = create_ratio_series(close_df, "SPY", "TLT")
    ratio_gld_tlt = create_ratio_series(close_df, "GLD", "TLT")

    def ratio_up(s):
        s = s.dropna()
        if len(s) < 60:
            return False
        ma50 = s.rolling(50).mean().iloc[-1]
        return bool(s.iloc[-1] > ma50) if pd.notna(ma50) else False

    if ratio_up(ratio_iwm_qqq):
        leadership_score += 1
    if ratio_up(ratio_qqq_spy):
        leadership_score += 1
    if ratio_up(ratio_spy_tlt):
        leadership_score += 1
    if ratio_up(ratio_gld_tlt):
        defense_score += 1

    regime_score = breadth_score + leadership_score - defense_score

    if breadth_score >= 3 and leadership_score >= 2 and defense_score == 0:
        state = tr("broad_risk_on")
    elif breadth_score >= 2 and leadership_score >= 1:
        state = tr("narrow_risk_on")
    elif breadth_score <= 1 and defense_score >= 1:
        state = tr("risk_off")
    elif defense_score >= 1:
        state = tr("defensive")
    else:
        state = tr("neutral_mixed")

    return {
        "state": state,
        "regime_score": regime_score,
        "breadth_score": breadth_score,
        "leadership_score": leadership_score,
        "defense_score": defense_score,
        "table": table
    }

# =========================================================
# FIXED FUNCTION
# =========================================================
def build_portfolio_suggestion(close_df: pd.DataFrame, lookback_value: int, vol_window_value: int, annual_factor: int):
    if close_df.empty or close_df.shape[1] < 2:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    returns = close_df.pct_change()
    vol = returns.rolling(vol_window_value).std() * np.sqrt(annual_factor)
    latest_vol = vol.iloc[-1]

    latest_mom = close_df.iloc[-1] / close_df.shift(lookback_value).iloc[-1] - 1
    signal = (latest_mom > 0).astype(float)

    inv_vol = 1 / latest_vol.replace(0, np.nan)
    raw_weight = signal * inv_vol
    raw_weight = raw_weight.replace([np.inf, -np.inf], np.nan).fillna(0)

    if raw_weight.sum() > 0:
        weights = raw_weight / raw_weight.sum()
    else:
        weights = pd.Series(0.0, index=close_df.columns)

    ret_clean = returns.dropna(how="all").copy()
    ret_clean = ret_clean.loc[:, close_df.columns]

    if ret_clean.dropna(how="all").empty:
        risk_contrib = pd.Series(0.0, index=close_df.columns)
    else:
        cov = ret_clean.cov() * annual_factor
        cov = cov.reindex(index=close_df.columns, columns=close_df.columns).fillna(0)

        w = weights.reindex(close_df.columns).fillna(0).values.astype(float)

        try:
            port_var = float(w @ cov.values @ w.T)
        except Exception:
            port_var = np.nan

        if pd.isna(port_var) or port_var <= 0:
            risk_contrib = pd.Series(0.0, index=close_df.columns)
        else:
            marginal = cov.values @ w
            contrib = (w * marginal) / port_var
            risk_contrib = pd.Series(contrib, index=close_df.columns)

    signal_df = pd.DataFrame({
        tr("asset"): close_df.columns,
        tr("signal"): signal.reindex(close_df.columns).fillna(0).values,
        tr("6m_momentum"): latest_mom.reindex(close_df.columns).values,
        tr("volatility"): latest_vol.reindex(close_df.columns).values,
        tr("current_target_weights"): weights.reindex(close_df.columns).values,
        tr("risk_contribution"): risk_contrib.reindex(close_df.columns).values
    })

    return signal_df, weights, risk_contrib

# =========================================================
# Data Load
# =========================================================
annual_factor = get_annualization_factor(interval)
high_window = get_high_window(interval)

df = load_data(ticker, period, interval)

if df.empty:
    st.error(tr("no_data"))
    st.stop()

required_cols = ["Open", "High", "Low", "Close", "Volume"]
for c in required_cols:
    if c not in df.columns:
        st.error(f"{tr('missing_column')}: {c}")
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
regime_assets = ["QQQ", "SPY", "IWM", "TLT", "GLD"]
regime_close_df = load_close_data(regime_assets, period, interval)
regime_result = regime_engine(regime_close_df, annual_factor)

trend_bt = backtest_trend(df["Close"], ma_short, ma_long, initial_capital, trading_cost_bps, annual_factor)
mr_bt = backtest_mean_reversion(df["Close"], mr_window, mr_z, initial_capital, trading_cost_bps, annual_factor)
mom_bt = backtest_momentum(df["Close"], mom_lookback, initial_capital, trading_cost_bps, annual_factor)
vol_bt = backtest_vol_target(df["Close"], vol_window, target_vol, annual_factor, initial_capital, trading_cost_bps)
rotation_mom, rotation_bt = build_rotation_strategy(cmp_df, mom_lookback, initial_capital, trading_cost_bps)
rp_weights, rp_equity, rp_turnover = build_risk_parity(cmp_df, vol_window, annual_factor, initial_capital, trading_cost_bps)
signal_df, suggested_weights, risk_contrib = build_portfolio_suggestion(cmp_df, mom_lookback, vol_window, annual_factor)

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs([
    tr_tab("overview_tab"),
    tr_tab("regime_tab"),
    tr_tab("ratio_tab"),
    tr_tab("trend_tab"),
    tr_tab("meanrev_tab"),
    tr_tab("momentum_tab"),
    tr_tab("rotation_tab"),
    tr_tab("voltarget_tab"),
    tr_tab("riskparity_tab"),
    tr_tab("portfolio_tab"),
    tr_tab("drawdown_tab"),
    tr_tab("multifactor_tab"),
    tr_tab("advanced_bt_tab"),
    tr_tab("summary_tab"),
])

# =========================================================
# Overview
# =========================================================
with tabs[0]:
    st.subheader(f"{ticker} - {tr('overview_title')}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(tr("last_price"), fmt_metric(latest["Close"], "price"))
    c2.metric(tr("rsi"), fmt_metric(latest["RSI"], "ratio"))
    c3.metric(tr("volatility"), fmt_metric(latest["Volatility"], "pct"))
    c4.metric(tr("drawdown"), fmt_metric(latest["Drawdown"], "pct"))
    c5.metric(tr("from_ath"), fmt_metric(latest["Distance_from_ATH"], "pct"))
    c6.metric(tr("from_52w_high"), fmt_metric(latest["Distance_from_52W_High"], "pct"))

    cagr_val = calc_cagr(df["Close"], annual_factor)
    sharpe_val = calc_sharpe(df["Return"], annual_factor)
    sortino_val = calc_sortino(df["Return"], annual_factor)
    calmar_val = calc_calmar(cagr_val, mdd)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric(tr("cagr"), fmt_metric(cagr_val, "pct"))
    d2.metric(tr("sharpe"), fmt_metric(sharpe_val, "ratio"))
    d3.metric(tr("sortino"), fmt_metric(sortino_val, "ratio"))
    d4.metric(tr("calmar"), fmt_metric(calmar_val, "ratio"))

    st.markdown(tr("overview_learn"))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} {tr('price')}", tr("rsi"), tr("drawdown"))
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash")
    fig.add_hline(y=30, row=2, col=1, line_dash="dash")
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], mode="lines", fill="tozeroy", name=tr("drawdown")), row=3, col=1)
    fig.update_layout(height=850, title=f"{ticker} {tr('overview_title')}")
    fig.update_yaxes(title_text=tr("price"), row=1, col=1)
    fig.update_yaxes(title_text=tr("rsi"), range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text=tr("drawdown"), tickformat=".0%", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Market Regime
# =========================================================
with tabs[1]:
    st.subheader(tr("market_regime_title"))
    st.markdown(tr("regime_desc"))

    a, b, c, d = st.columns(4)
    a.metric(tr("regime_state"), regime_result["state"])
    b.metric(tr("regime_score"), fmt_metric(regime_result["regime_score"], "ratio"))
    c.metric(tr("breadth_score"), fmt_metric(regime_result["breadth_score"], "ratio"))
    d.metric(tr("defense_score"), fmt_metric(regime_result["defense_score"], "ratio"))

    if not regime_result["table"].empty:
        table_disp = regime_result["table"].copy()
        table_disp[tr("close")] = table_disp[tr("close")].map(lambda x: f"{x:,.2f}")
        table_disp[tr("6m_momentum")] = table_disp[tr("6m_momentum")].map(lambda x: f"{x:.2%}" if pd.notna(x) else tr("na"))
        table_disp[tr("drawdown_col")] = table_disp[tr("drawdown_col")].map(lambda x: f"{x:.2%}" if pd.notna(x) else tr("na"))
        st.subheader(tr("asset_state_table"))
        st.dataframe(table_disp, use_container_width=True)

# =========================================================
# Relative Ratios
# =========================================================
with tabs[2]:
    st.subheader(tr("ratio_title"))
    st.markdown(tr("ratio_desc"))

    if regime_close_df.empty:
        st.warning(tr("not_enough_data"))
    else:
        ratio_map = {
            "QQQ / SPY": create_ratio_series(regime_close_df, "QQQ", "SPY"),
            "IWM / QQQ": create_ratio_series(regime_close_df, "IWM", "QQQ"),
            "IWM / SPY": create_ratio_series(regime_close_df, "IWM", "SPY"),
            "SPY / TLT": create_ratio_series(regime_close_df, "SPY", "TLT"),
            "GLD / TLT": create_ratio_series(regime_close_df, "GLD", "TLT"),
        }

        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            subplot_titles=tuple(ratio_map.keys())
        )

        for i, (name, series) in enumerate(ratio_map.items(), start=1):
            if not series.empty:
                ma50 = series.rolling(50).mean()
                fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name), row=i, col=1)
                fig.add_trace(go.Scatter(x=ma50.index, y=ma50, mode="lines", name=f"{name} MA50"), row=i, col=1)

        fig.update_layout(height=1400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Trend Following
# =========================================================
with tabs[3]:
    st.subheader(tr("trend_tab"))
    st.markdown(tr("trend_desc"))

    if trend_bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        stats = equity_stats(trend_bt["Equity_Net"], annual_factor)
        a, b, c, d = st.columns(4)
        a.metric(tr("cagr"), fmt_metric(stats["CAGR"], "pct"))
        b.metric(tr("sharpe"), fmt_metric(stats["Sharpe"], "ratio"))
        c.metric(tr("max_drawdown"), fmt_metric(stats["MDD"], "pct"))
        d.metric(tr("turnover"), fmt_metric(trend_bt["Turnover"].mean() * annual_factor, "ratio"))

        buy_points = trend_bt[trend_bt["Signal_Change"] == 1]
        sell_points = trend_bt[trend_bt["Signal_Change"] == -1]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=(tr("price"), tr("equity_curve"))
        )
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)

        if show_signals:
            fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points["Close"], mode="markers", name=tr("buy"),
                                     marker=dict(symbol="triangle-up", size=10)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points["Close"], mode="markers", name=tr("sell"),
                                     marker=dict(symbol="triangle-down", size=10)), row=1, col=1)

        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=2, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Equity_Net"], mode="lines", name=tr("cost_adjusted")), row=2, col=1)
        fig.update_layout(height=850)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Mean Reversion
# =========================================================
with tabs[4]:
    st.subheader(tr("meanrev_tab"))
    st.markdown(tr("meanrev_desc"))

    if mr_bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        stats = equity_stats(mr_bt["Equity_Net"], annual_factor)
        a, b, c, d = st.columns(4)
        a.metric(tr("cagr"), fmt_metric(stats["CAGR"], "pct"))
        b.metric(tr("sharpe"), fmt_metric(stats["Sharpe"], "ratio"))
        c.metric(tr("max_drawdown"), fmt_metric(stats["MDD"], "pct"))
        d.metric(tr("turnover"), fmt_metric(mr_bt["Turnover"].mean() * annual_factor, "ratio"))

        buy_points = mr_bt[mr_bt["Signal_Change"] == 1]
        sell_points = mr_bt[mr_bt["Signal_Change"] == -1]

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.42, 0.23, 0.35],
            subplot_titles=(tr("price"), "Z-Score", tr("equity_curve"))
        )
        fig.add_trace(go.Scatter(x=mr_bt.index, y=mr_bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=mr_bt.index, y=mr_bt["Mean"], mode="lines", name="Mean"), row=1, col=1)

        if show_signals:
            fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points["Close"], mode="markers", name=tr("buy"),
                                     marker=dict(symbol="triangle-up", size=10)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points["Close"], mode="markers", name=tr("exit"),
                                     marker=dict(symbol="triangle-down", size=10)), row=1, col=1)

        fig.add_trace(go.Scatter(x=mr_bt.index, y=mr_bt["Z"], mode="lines", name="Z"), row=2, col=1)
        fig.add_hline(y=-mr_z, row=2, col=1, line_dash="dash")
        fig.add_hline(y=0, row=2, col=1, line_dash="dot")
        fig.add_trace(go.Scatter(x=mr_bt.index, y=mr_bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=3, col=1)
        fig.add_trace(go.Scatter(x=mr_bt.index, y=mr_bt["Equity_Net"], mode="lines", name=tr("cost_adjusted")), row=3, col=1)
        fig.update_layout(height=950)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Momentum
# =========================================================
with tabs[5]:
    st.subheader(tr("momentum_tab"))
    st.markdown(tr("momentum_desc"))

    if mom_bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        stats = equity_stats(mom_bt["Equity_Net"], annual_factor)
        a, b, c, d = st.columns(4)
        a.metric(tr("cagr"), fmt_metric(stats["CAGR"], "pct"))
        b.metric(tr("sharpe"), fmt_metric(stats["Sharpe"], "ratio"))
        c.metric(tr("max_drawdown"), fmt_metric(stats["MDD"], "pct"))
        d.metric(tr("turnover"), fmt_metric(mom_bt["Turnover"].mean() * annual_factor, "ratio"))

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=(tr("price"), tr("equity_curve"))
        )
        fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["Momentum"], mode="lines", name="Momentum"), row=1, col=1)
        fig.add_hline(y=0, row=1, col=1, line_dash="dash")
        fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=2, col=1)
        fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["Equity_Net"], mode="lines", name=tr("cost_adjusted")), row=2, col=1)
        fig.update_layout(height=820)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Asset Rotation
# =========================================================
with tabs[6]:
    st.subheader(tr("rotation_tab"))
    st.markdown(tr("rotation_desc"))

    if cmp_df.empty or cmp_df.shape[1] < 2 or rotation_bt.empty:
        st.warning(tr("need_two_assets"))
    else:
        rotation_stats = equity_stats(rotation_bt["Rotation_Equity_Net"], annual_factor)
        a, b, c, d = st.columns(4)
        a.metric(tr("cagr"), fmt_metric(rotation_stats["CAGR"], "pct"))
        b.metric(tr("sharpe"), fmt_metric(rotation_stats["Sharpe"], "ratio"))
        c.metric(tr("max_drawdown"), fmt_metric(rotation_stats["MDD"], "pct"))
        d.metric(tr("turnover"), fmt_metric(rotation_bt["Turnover"].mean() * annual_factor, "ratio"))

        normalized = cmp_df.copy()
        for col in normalized.columns:
            normalized[col] = normalize_series(normalized[col])

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.45, 0.55],
            subplot_titles=("Relative Performance", tr("equity_curve"))
        )
        for col in normalized.columns:
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], mode="lines", name=col), row=1, col=1)
        fig.add_trace(go.Scatter(x=rotation_bt.index, y=rotation_bt["Rotation_Equity_Raw"], mode="lines", name=tr("raw")), row=2, col=1)
        fig.add_trace(go.Scatter(x=rotation_bt.index, y=rotation_bt["Rotation_Equity_Net"], mode="lines", name=tr("cost_adjusted")), row=2, col=1)
        fig.update_layout(height=850)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(tr("recent_rotation_decisions"))
        last_leaders = rotation_bt["Leader"].dropna().tail(20).to_frame(name=tr("selected_asset"))
        st.dataframe(last_leaders, use_container_width=True)

# =========================================================
# Volatility Targeting
# =========================================================
with tabs[7]:
    st.subheader(tr("voltarget_tab"))
    st.markdown(tr("voltarget_desc"))

    if vol_bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        stats = equity_stats(vol_bt["Equity_Net"], annual_factor)
        a, b, c, d = st.columns(4)
        a.metric(tr("cagr"), fmt_metric(stats["CAGR"], "pct"))
        b.metric(tr("sharpe"), fmt_metric(stats["Sharpe"], "ratio"))
        c.metric(tr("max_drawdown"), fmt_metric(stats["MDD"], "pct"))
        d.metric(tr("turnover"), fmt_metric(vol_bt["Turnover"].mean() * annual_factor, "ratio"))

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.34, 0.28, 0.38],
            subplot_titles=(tr("price"), tr("volatility"), tr("equity_curve"))
        )
        fig.add_trace(go.Scatter(x=vol_bt.index, y=vol_bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=vol_bt.index, y=vol_bt["Realized_Vol"], mode="lines", name=tr("volatility")), row=2, col=1)
        fig.add_trace(go.Scatter(x=vol_bt.index, y=vol_bt["Position"], mode="lines", name=tr("position")), row=2, col=1)
        fig.add_trace(go.Scatter(x=vol_bt.index, y=vol_bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=3, col=1)
        fig.add_trace(go.Scatter(x=vol_bt.index, y=vol_bt["Equity_Net"], mode="lines", name=tr("cost_adjusted")), row=3, col=1)
        fig.update_layout(height=930)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Risk Parity
# =========================================================
with tabs[8]:
    st.subheader(tr("riskparity_tab"))
    st.markdown(tr("riskparity_desc"))

    if cmp_df.empty or cmp_df.shape[1] < 2 or rp_equity.empty:
        st.warning(tr("need_two_assets"))
    else:
        rp_stats = equity_stats(rp_equity["Risk_Parity_Net"], annual_factor)
        a, b, c, d = st.columns(4)
        a.metric(tr("cagr"), fmt_metric(rp_stats["CAGR"], "pct"))
        b.metric(tr("sharpe"), fmt_metric(rp_stats["Sharpe"], "ratio"))
        c.metric(tr("max_drawdown"), fmt_metric(rp_stats["MDD"], "pct"))
        d.metric(tr("turnover"), fmt_metric(rp_turnover.mean() * annual_factor, "ratio"))

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.4, 0.6],
            subplot_titles=(tr("current_target_weights"), tr("equity_curve"))
        )

        for col in rp_weights.columns:
            fig.add_trace(go.Scatter(x=rp_weights.index, y=rp_weights[col], mode="lines", name=f"W_{col}"), row=1, col=1)

        fig.add_trace(go.Scatter(x=rp_equity.index, y=rp_equity["Equal_Weight_Net"], mode="lines", name=tr("equal_weight_strategy")), row=2, col=1)
        fig.add_trace(go.Scatter(x=rp_equity.index, y=rp_equity["Risk_Parity_Net"], mode="lines", name=tr("risk_parity_strategy")), row=2, col=1)
        fig.update_layout(height=900)
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Portfolio Construction
# =========================================================
with tabs[9]:
    st.subheader(tr("portfolio_tab"))
    st.markdown(tr("portfolio_desc"))

    if signal_df.empty:
        st.warning(tr("need_two_assets"))
    else:
        disp = signal_df.copy()
        for col in [tr("6m_momentum"), tr("volatility"), tr("current_target_weights"), tr("risk_contribution")]:
            disp[col] = disp[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else tr("na"))
        st.subheader(tr("signal_by_asset"))
        st.dataframe(disp, use_container_width=True)

        pie1, pie2 = st.columns(2)
        with pie1:
            fig_w = go.Figure(data=[go.Pie(labels=suggested_weights.index, values=suggested_weights.fillna(0).values, hole=0.4)])
            fig_w.update_layout(height=420, title=tr("current_target_weights"))
            st.plotly_chart(fig_w, use_container_width=True)

        with pie2:
            rc_vals = np.clip(risk_contrib.fillna(0).values, 0, None)
            if np.nansum(rc_vals) == 0:
                st.info(tr("not_enough_data"))
            else:
                fig_r = go.Figure(data=[go.Pie(labels=risk_contrib.index, values=rc_vals, hole=0.4)])
                fig_r.update_layout(height=420, title=tr("risk_contribution"))
                st.plotly_chart(fig_r, use_container_width=True)

# =========================================================
# Drawdown Buying
# =========================================================
with tabs[10]:
    st.subheader(tr("drawdown_tab"))
    st.markdown(tr("drawdown_desc"))

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
        subplot_titles=("Price and ATH", tr("drawdown"))
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ATH"], mode="lines", name="ATH"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], mode="lines", fill="tozeroy", name=tr("drawdown")), row=2, col=1)
    fig.add_hline(y=-0.05, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.10, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.15, row=2, col=1, line_dash="dash")
    fig.add_hline(y=-0.20, row=2, col=1, line_dash="dash")
    fig.update_layout(height=850)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Multi-Factor
# =========================================================
with tabs[11]:
    st.subheader(tr("multifactor_tab"))
    st.markdown(tr("multifactor_desc"))

    score, reasons, fac_df = latest_factor_score(df, mom_lookback, vol_window)

    a, b = st.columns([1, 2])
    with a:
        st.metric(tr("factor_score"), f"{score} / 4")
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
        subplot_titles=(tr("price"), tr("composite_factor_score"))
    )
    fig.add_trace(go.Scatter(x=fac_df.index, y=fac_df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fac_df.index, y=fac_df["Composite_Score"], mode="lines", name=tr("multifactor_tab")), row=2, col=1)
    fig.update_layout(height=780)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Advanced Backtest
# =========================================================
with tabs[12]:
    st.subheader(tr("advanced_bt_tab"))
    st.markdown(tr("advanced_bt_desc"))

    if trend_bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        adv_stats = equity_stats(trend_bt["Equity_Net"], annual_factor)
        adv_bh_stats = equity_stats(trend_bt["BuyHold_Equity"], annual_factor)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(f"{tr('cost_adjusted')} {tr('cagr')}", fmt_metric(adv_stats["CAGR"], "pct"))
        c2.metric(f"{tr('cost_adjusted')} {tr('sharpe')}", fmt_metric(adv_stats["Sharpe"], "ratio"))
        c3.metric(f"{tr('cost_adjusted')} {tr('max_drawdown')}", fmt_metric(adv_stats["MDD"], "pct"))
        c4.metric(f"{tr('buy_hold')} {tr('cagr')}", fmt_metric(adv_bh_stats["CAGR"], "pct"))
        c5.metric(f"{tr('buy_hold')} {tr('sharpe')}", fmt_metric(adv_bh_stats["Sharpe"], "ratio"))
        c6.metric(f"{tr('buy_hold')} {tr('max_drawdown')}", fmt_metric(adv_bh_stats["MDD"], "pct"))

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.38, 0.31, 0.31],
            subplot_titles=(tr("equity_curve"), tr("rolling_sharpe"), tr("underwater_chart"))
        )
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Equity_Net"], mode="lines", name=tr("cost_adjusted")), row=1, col=1)

        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Rolling_Sharpe_BH"], mode="lines", name=f"{tr('buy_hold')} {tr('rolling_sharpe')}"), row=2, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Rolling_Sharpe_Net"], mode="lines", name=f"{tr('cost_adjusted')} {tr('rolling_sharpe')}"), row=2, col=1)

        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Underwater_BH"], mode="lines", fill="tozeroy", name=f"{tr('buy_hold')} DD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Underwater_Net"], mode="lines", fill="tozeroy", name=f"{tr('cost_adjusted')} DD"), row=3, col=1)

        fig.update_layout(height=1100)
        fig.update_yaxes(tickformat=".0%", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Rolling_CAGR_BH"], mode="lines", name=f"{tr('buy_hold')} {tr('rolling_cagr')}"))
        fig2.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Rolling_CAGR_Net"], mode="lines", name=f"{tr('cost_adjusted')} {tr('rolling_cagr')}"))
        fig2.update_layout(height=420, title=tr("rolling_cagr"))
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

        if show_monthly_heatmap:
            heat = get_monthly_heatmap_from_equity(trend_bt["Equity_Net"])
            if not heat.empty:
                z_vals = heat.values
                text_vals = np.empty_like(z_vals, dtype=object)
                for i in range(z_vals.shape[0]):
                    for j in range(z_vals.shape[1]):
                        if pd.isna(z_vals[i, j]):
                            text_vals[i, j] = ""
                        else:
                            text_vals[i, j] = f"{z_vals[i, j]:.1%}"

                fig3 = go.Figure(
                    data=go.Heatmap(
                        z=z_vals,
                        x=heat.columns,
                        y=heat.index.astype(str),
                        text=text_vals,
                        texttemplate="%{text}",
                        hovertemplate="Year=%{y}<br>Month=%{x}<br>Return=%{z:.2%}<extra></extra>"
                    )
                )
                fig3.update_layout(height=420, title=tr("monthly_heatmap"))
                st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# Strategy Summary
# =========================================================
with tabs[13]:
    st.subheader(tr("summary_tab"))
    st.markdown(tr("summary_desc"))

    summary_rows = []

    bh = backtest_buy_hold(df["Close"], initial_capital)
    if not bh.empty:
        stats = equity_stats(bh, annual_factor)
        summary_rows.append({
            tr("strategy"): tr("buy_hold"),
            tr("final_value"): bh.iloc[-1],
            tr("cagr"): stats["CAGR"],
            tr("sharpe"): stats["Sharpe"],
            tr("calmar"): stats["Calmar"],
            tr("max_drawdown"): stats["MDD"],
            tr("exposure"): 1.0,
            tr("turnover"): 0.0,
        })

    if not trend_bt.empty:
        stats = equity_stats(trend_bt["Equity_Net"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("trend_tab"),
            tr("final_value"): trend_bt["Equity_Net"].iloc[-1],
            tr("cagr"): stats["CAGR"],
            tr("sharpe"): stats["Sharpe"],
            tr("calmar"): stats["Calmar"],
            tr("max_drawdown"): stats["MDD"],
            tr("exposure"): trend_bt["Exposure"].mean(),
            tr("turnover"): trend_bt["Turnover"].mean() * annual_factor,
        })

    if not mr_bt.empty:
        stats = equity_stats(mr_bt["Equity_Net"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("meanrev_tab"),
            tr("final_value"): mr_bt["Equity_Net"].iloc[-1],
            tr("cagr"): stats["CAGR"],
            tr("sharpe"): stats["Sharpe"],
            tr("calmar"): stats["Calmar"],
            tr("max_drawdown"): stats["MDD"],
            tr("exposure"): mr_bt["Exposure"].mean(),
            tr("turnover"): mr_bt["Turnover"].mean() * annual_factor,
        })

    if not mom_bt.empty:
        stats = equity_stats(mom_bt["Equity_Net"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("momentum_tab"),
            tr("final_value"): mom_bt["Equity_Net"].iloc[-1],
            tr("cagr"): stats["CAGR"],
            tr("sharpe"): stats["Sharpe"],
            tr("calmar"): stats["Calmar"],
            tr("max_drawdown"): stats["MDD"],
            tr("exposure"): mom_bt["Exposure"].mean(),
            tr("turnover"): mom_bt["Turnover"].mean() * annual_factor,
        })

    if not vol_bt.empty:
        stats = equity_stats(vol_bt["Equity_Net"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("voltarget_tab"),
            tr("final_value"): vol_bt["Equity_Net"].iloc[-1],
            tr("cagr"): stats["CAGR"],
            tr("sharpe"): stats["Sharpe"],
            tr("calmar"): stats["Calmar"],
            tr("max_drawdown"): stats["MDD"],
            tr("exposure"): vol_bt["Exposure"].mean(),
            tr("turnover"): vol_bt["Turnover"].mean() * annual_factor,
        })

    if not rotation_bt.empty:
        stats = equity_stats(rotation_bt["Rotation_Equity_Net"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("rotation_tab"),
            tr("final_value"): rotation_bt["Rotation_Equity_Net"].iloc[-1],
            tr("cagr"): stats["CAGR"],
            tr("sharpe"): stats["Sharpe"],
            tr("calmar"): stats["Calmar"],
            tr("max_drawdown"): stats["MDD"],
            tr("exposure"): 1.0,
            tr("turnover"): rotation_bt["Turnover"].mean() * annual_factor,
        })

    if not rp_equity.empty:
        stats = equity_stats(rp_equity["Risk_Parity_Net"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("riskparity_tab"),
            tr("final_value"): rp_equity["Risk_Parity_Net"].iloc[-1],
            tr("cagr"): stats["CAGR"],
            tr("sharpe"): stats["Sharpe"],
            tr("calmar"): stats["Calmar"],
            tr("max_drawdown"): stats["MDD"],
            tr("exposure"): 1.0,
            tr("turnover"): rp_turnover.mean() * annual_factor if not rp_turnover.empty else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        st.warning(tr("not_enough_data"))
    else:
        display_df = summary_df.copy()
        display_df[tr("final_value")] = display_df[tr("final_value")].map(lambda x: f"{x:,.0f}" if pd.notna(x) else tr("na"))
        for col in [tr("cagr"), tr("max_drawdown"), tr("exposure"), tr("turnover")]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else tr("na"))
        for col in [tr("sharpe"), tr("calmar")]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else tr("na"))

        st.dataframe(display_df, use_container_width=True)

        fig = go.Figure()
        if not bh.empty:
            fig.add_trace(go.Scatter(x=bh.index, y=bh, mode="lines", name=tr("buy_hold")))
        if not trend_bt.empty:
            fig.add_trace(go.Scatter(x=trend_bt.index, y=trend_bt["Equity_Net"], mode="lines", name=tr("trend_tab")))
        if not mr_bt.empty:
            fig.add_trace(go.Scatter(x=mr_bt.index, y=mr_bt["Equity_Net"], mode="lines", name=tr("meanrev_tab")))
        if not mom_bt.empty:
            fig.add_trace(go.Scatter(x=mom_bt.index, y=mom_bt["Equity_Net"], mode="lines", name=tr("momentum_tab")))
        if not vol_bt.empty:
            fig.add_trace(go.Scatter(x=vol_bt.index, y=vol_bt["Equity_Net"], mode="lines", name=tr("voltarget_tab")))
        if not rotation_bt.empty:
            fig.add_trace(go.Scatter(x=rotation_bt.index, y=rotation_bt["Rotation_Equity_Net"], mode="lines", name=tr("rotation_tab")))
        if not rp_equity.empty:
            fig.add_trace(go.Scatter(x=rp_equity.index, y=rp_equity["Risk_Parity_Net"], mode="lines", name=tr("riskparity_tab")))
        fig.update_layout(height=550, title=tr("equity_curve"))
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Footer
# =========================================================
st.caption(f"{tr('last_refresh')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
