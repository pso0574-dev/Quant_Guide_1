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
        "en": "📈 Quant Strategy Master Dashboard",
        "ko": "📈 퀀트 전략 마스터 대시보드",
    },
    "app_caption": {
        "en": "Learn practical quant methods with charts, rules, and simple backtests.",
        "ko": "차트, 규칙, 간단한 백테스트를 통해 실전 퀀트 방법을 학습합니다.",
    },
    "sidebar_market_settings": {
        "en": "Market Settings",
        "ko": "시장 설정",
    },
    "main_asset": {
        "en": "Main Asset",
        "ko": "주요 자산",
    },
    "custom_ticker": {
        "en": "Custom Ticker",
        "ko": "사용자 지정 티커",
    },
    "comparison_assets": {
        "en": "Comparison Assets",
        "ko": "비교 자산",
    },
    "history_period": {
        "en": "History Period",
        "ko": "조회 기간",
    },
    "interval": {
        "en": "Interval",
        "ko": "간격",
    },
    "short_ma": {
        "en": "Short MA",
        "ko": "단기 이동평균",
    },
    "long_ma": {
        "en": "Long MA",
        "ko": "장기 이동평균",
    },
    "rsi_period": {
        "en": "RSI Period",
        "ko": "RSI 기간",
    },
    "vol_window": {
        "en": "Vol Window",
        "ko": "변동성 기간",
    },
    "mr_window": {
        "en": "Mean Reversion Window",
        "ko": "평균회귀 기간",
    },
    "mr_z": {
        "en": "MR Z Threshold",
        "ko": "평균회귀 Z 임계값",
    },
    "mom_lookback": {
        "en": "Momentum Lookback",
        "ko": "모멘텀 조회 기간",
    },
    "target_vol": {
        "en": "Target Volatility",
        "ko": "목표 변동성",
    },
    "initial_capital": {
        "en": "Initial Capital",
        "ko": "초기 자본",
    },
    "show_signals": {
        "en": "Show Buy/Sell Signals",
        "ko": "매수/매도 신호 표시",
    },
    "no_data": {
        "en": "No data loaded. Please check ticker or try again later.",
        "ko": "데이터를 불러오지 못했습니다. 티커를 확인하거나 나중에 다시 시도해 주세요.",
    },
    "missing_column": {
        "en": "Missing column",
        "ko": "누락된 열",
    },
    "overview_tab": {
        "en": "Overview",
        "ko": "개요",
    },
    "trend_tab": {
        "en": "1. Trend Following",
        "ko": "1. 추세추종",
    },
    "meanrev_tab": {
        "en": "2. Mean Reversion",
        "ko": "2. 평균회귀",
    },
    "momentum_tab": {
        "en": "3. Momentum",
        "ko": "3. 모멘텀",
    },
    "relmom_tab": {
        "en": "4. Relative Momentum",
        "ko": "4. 상대 모멘텀",
    },
    "voltarget_tab": {
        "en": "5. Volatility Targeting",
        "ko": "5. 변동성 타게팅",
    },
    "riskparity_tab": {
        "en": "6. Risk Parity",
        "ko": "6. 리스크 패리티",
    },
    "drawdown_tab": {
        "en": "7. Drawdown Buying",
        "ko": "7. 낙폭 기반 매수",
    },
    "multifactor_tab": {
        "en": "8. Multi-Factor",
        "ko": "8. 멀티팩터",
    },
    "summary_tab": {
        "en": "9. Strategy Summary",
        "ko": "9. 전략 요약",
    },
    "last_price": {
        "en": "Last Price",
        "ko": "현재가",
    },
    "rsi": {
        "en": "RSI",
        "ko": "RSI",
    },
    "volatility": {
        "en": "Volatility",
        "ko": "변동성",
    },
    "drawdown": {
        "en": "Drawdown",
        "ko": "낙폭",
    },
    "from_ath": {
        "en": "From ATH",
        "ko": "ATH 대비",
    },
    "from_52w_high": {
        "en": "From 52W High",
        "ko": "52주 고점 대비",
    },
    "cagr": {
        "en": "CAGR",
        "ko": "연복리수익률",
    },
    "sharpe": {
        "en": "Sharpe",
        "ko": "샤프지수",
    },
    "sortino": {
        "en": "Sortino",
        "ko": "소르티노지수",
    },
    "max_drawdown": {
        "en": "Max Drawdown",
        "ko": "최대낙폭",
    },
    "na": {
        "en": "N/A",
        "ko": "N/A",
    },
    "overview_learn": {
        "en": (
            "**What you learn in this tab**\n"
            "- Trend, overbought/oversold condition, drawdown, and volatility\n"
            "- The core state of the asset before making an investment decision"
        ),
        "ko": (
            "**이 탭에서 배우는 것**\n"
            "- 추세, 과열/과매도 상태, 낙폭, 변동성\n"
            "- 투자 판단 전에 확인해야 할 자산의 핵심 상태"
        ),
    },
    "overview_price": {
        "en": "Price",
        "ko": "가격",
    },
    "overview_dd_short": {
        "en": "DD",
        "ko": "낙폭",
    },
    "trend_title": {
        "en": "Trend Following",
        "ko": "추세추종",
    },
    "trend_desc": {
        "en": (
            "**Core idea**\n"
            "Follow the market direction. A rising asset may continue to rise for some time.\n\n"
            "**Simple rule**\n"
            "- Enter when Short MA > Long MA\n"
            "- Exit when Short MA <= Long MA"
        ),
        "ko": (
            "**핵심 개념**\n"
            "시장 방향을 따라가는 전략입니다. 상승 중인 자산은 일정 기간 더 상승할 수 있다고 가정합니다.\n\n"
            "**간단 룰**\n"
            "- 진입: 단기 MA > 장기 MA\n"
            "- 이탈: 단기 MA <= 장기 MA"
        ),
    },
    "trend_cagr": {
        "en": "Trend CAGR",
        "ko": "추세추종 CAGR",
    },
    "trend_sharpe": {
        "en": "Trend Sharpe",
        "ko": "추세추종 Sharpe",
    },
    "trend_mdd": {
        "en": "Trend MDD",
        "ko": "추세추종 MDD",
    },
    "price_and_ma_signals": {
        "en": "Price and MA Signals",
        "ko": "가격과 이동평균 신호",
    },
    "equity_curve": {
        "en": "Equity Curve",
        "ko": "자산곡선",
    },
    "buy_hold": {
        "en": "Buy & Hold",
        "ko": "매수 후 보유",
    },
    "trend_strategy": {
        "en": "Trend Strategy",
        "ko": "추세추종 전략",
    },
    "buy": {
        "en": "Buy",
        "ko": "매수",
    },
    "sell": {
        "en": "Sell",
        "ko": "매도",
    },
    "not_enough_data": {
        "en": "Not enough data for this strategy.",
        "ko": "이 전략을 실행하기에 데이터가 충분하지 않습니다.",
    },
    "meanrev_title": {
        "en": "Mean Reversion",
        "ko": "평균회귀",
    },
    "meanrev_desc": {
        "en": (
            "**Core idea**\n"
            "If price moves too far below its average, it may revert back toward the mean.\n\n"
            "**Simple rule**\n"
            "- Enter when Z-score < -threshold\n"
            "- Exit when price recovers to the rolling mean"
        ),
        "ko": (
            "**핵심 개념**\n"
            "가격이 평균보다 너무 멀리 아래로 이탈하면 다시 평균 쪽으로 되돌아올 수 있다는 가정입니다.\n\n"
            "**간단 룰**\n"
            "- 진입: Z-score < -임계값\n"
            "- 청산: 가격이 이동평균 수준으로 회복"
        ),
    },
    "mr_cagr": {
        "en": "MR CAGR",
        "ko": "평균회귀 CAGR",
    },
    "mr_sharpe": {
        "en": "MR Sharpe",
        "ko": "평균회귀 Sharpe",
    },
    "mr_mdd": {
        "en": "MR MDD",
        "ko": "평균회귀 MDD",
    },
    "price_vs_mean": {
        "en": "Price vs Mean",
        "ko": "가격 vs 평균",
    },
    "zscore": {
        "en": "Z-Score",
        "ko": "Z-점수",
    },
    "meanrev_strategy": {
        "en": "Mean Reversion",
        "ko": "평균회귀",
    },
    "exit": {
        "en": "Exit",
        "ko": "청산",
    },
    "momentum_title": {
        "en": "Momentum",
        "ko": "모멘텀",
    },
    "momentum_desc": {
        "en": (
            "**Core idea**\n"
            "Assets that have been strong recently may continue to be strong.\n\n"
            "**Simple rule**\n"
            "- Enter when lookback momentum > 0\n"
            "- Exit when momentum <= 0"
        ),
        "ko": (
            "**핵심 개념**\n"
            "최근 강했던 자산이 당분간 계속 강할 수 있다는 가정입니다.\n\n"
            "**간단 룰**\n"
            "- 진입: 조회 기간 모멘텀 > 0\n"
            "- 이탈: 모멘텀 <= 0"
        ),
    },
    "momentum_cagr": {
        "en": "Momentum CAGR",
        "ko": "모멘텀 CAGR",
    },
    "momentum_sharpe": {
        "en": "Momentum Sharpe",
        "ko": "모멘텀 Sharpe",
    },
    "momentum_mdd": {
        "en": "Momentum MDD",
        "ko": "모멘텀 MDD",
    },
    "price_and_momentum": {
        "en": "Price and Momentum",
        "ko": "가격과 모멘텀",
    },
    "momentum_strategy": {
        "en": "Momentum Strategy",
        "ko": "모멘텀 전략",
    },
    "relmom_title": {
        "en": "Relative Momentum / Asset Rotation",
        "ko": "상대 모멘텀 / 자산 로테이션",
    },
    "relmom_desc": {
        "en": (
            "**Core idea**\n"
            "Rotate into the strongest asset among several candidates.\n"
            "Example: hold only the top-performing asset among QQQ / SPY / TLT / GLD."
        ),
        "ko": (
            "**핵심 개념**\n"
            "여러 자산 중 가장 강한 자산으로 이동하는 전략입니다.\n"
            "예: QQQ / SPY / TLT / GLD 중 가장 강한 자산만 보유"
        ),
    },
    "need_two_assets": {
        "en": "Need at least 2 assets for relative momentum.",
        "ko": "상대 모멘텀 전략에는 최소 2개 이상의 자산이 필요합니다.",
    },
    "normalized_performance": {
        "en": "Normalized Performance",
        "ko": "정규화 성과",
    },
    "rotation_cagr": {
        "en": "Rotation CAGR",
        "ko": "로테이션 CAGR",
    },
    "rotation_equity": {
        "en": "Rotation Strategy Equity",
        "ko": "로테이션 전략 자산곡선",
    },
    "recent_rotation_decisions": {
        "en": "Recent Rotation Decisions",
        "ko": "최근 로테이션 결정",
    },
    "selected_asset": {
        "en": "Selected Asset",
        "ko": "선택 자산",
    },
    "voltarget_title": {
        "en": "Volatility Targeting",
        "ko": "변동성 타게팅",
    },
    "voltarget_desc": {
        "en": (
            "**Core idea**\n"
            "Adjust position size to keep portfolio volatility near a target level.\n\n"
            "**Simple rule**\n"
            "- Reduce exposure when realized volatility rises\n"
            "- Increase exposure when realized volatility falls"
        ),
        "ko": (
            "**핵심 개념**\n"
            "포트폴리오 변동성을 목표 수준에 맞추기 위해 비중을 조절하는 전략입니다.\n\n"
            "**간단 룰**\n"
            "- 실현 변동성이 올라가면 비중 축소\n"
            "- 실현 변동성이 내려가면 비중 확대"
        ),
    },
    "voltarget_cagr": {
        "en": "Vol Target CAGR",
        "ko": "변동성 타게팅 CAGR",
    },
    "voltarget_sharpe": {
        "en": "Vol Target Sharpe",
        "ko": "변동성 타게팅 Sharpe",
    },
    "voltarget_mdd": {
        "en": "Vol Target MDD",
        "ko": "변동성 타게팅 MDD",
    },
    "realized_vol_and_leverage": {
        "en": "Realized Vol and Leverage",
        "ko": "실현 변동성과 레버리지",
    },
    "vol_target": {
        "en": "Vol Target",
        "ko": "변동성 타게팅",
    },
    "riskparity_title": {
        "en": "Risk Parity",
        "ko": "리스크 패리티",
    },
    "riskparity_desc": {
        "en": (
            "**Core idea**\n"
            "Allocate based on risk contribution rather than capital amount.\n"
            "Higher-volatility assets get smaller weights, and lower-volatility assets get larger weights."
        ),
        "ko": (
            "**핵심 개념**\n"
            "금액이 아니라 위험 기여도 기준으로 자산을 배분하는 방식입니다.\n"
            "변동성이 큰 자산 비중은 줄이고, 작은 자산 비중은 늘립니다."
        ),
    },
    "need_multiple_assets": {
        "en": "Need multiple assets for risk parity.",
        "ko": "리스크 패리티에는 여러 자산이 필요합니다.",
    },
    "rp_cagr": {
        "en": "Risk Parity CAGR",
        "ko": "리스크 패리티 CAGR",
    },
    "rp_sharpe": {
        "en": "Risk Parity Sharpe",
        "ko": "리스크 패리티 Sharpe",
    },
    "rp_mdd": {
        "en": "Risk Parity MDD",
        "ko": "리스크 패리티 MDD",
    },
    "risk_parity_weights": {
        "en": "Risk Parity Weights",
        "ko": "리스크 패리티 비중",
    },
    "equal_weight": {
        "en": "Equal Weight",
        "ko": "동일가중",
    },
    "risk_parity": {
        "en": "Risk Parity",
        "ko": "리스크 패리티",
    },
    "drawdown_title": {
        "en": "Drawdown Buying Guide",
        "ko": "낙폭 기반 매수 가이드",
    },
    "drawdown_desc": {
        "en": (
            "**Core idea**\n"
            "Buy in phases based on how far price has fallen from its previous high.\n\n"
            "**Example**\n"
            "- 0% ~ -5%: watch\n"
            "- -5% ~ -10%: first buy\n"
            "- -10% ~ -15%: second buy\n"
            "- -15% ~ -20%: third buy\n"
            "- below -20%: high risk / high opportunity"
        ),
        "ko": (
            "**핵심 개념**\n"
            "이전 고점 대비 하락 폭에 따라 분할 매수하는 규칙입니다.\n\n"
            "**예시**\n"
            "- 0% ~ -5%: 관찰\n"
            "- -5% ~ -10%: 1차 매수\n"
            "- -10% ~ -15%: 2차 매수\n"
            "- -15% ~ -20%: 3차 매수\n"
            "- -20% 이하: 고위험 / 고기회"
        ),
    },
    "zone": {
        "en": "Zone",
        "ko": "구간",
    },
    "meaning": {
        "en": "Meaning",
        "ko": "의미",
    },
    "action": {
        "en": "Action",
        "ko": "행동",
    },
    "near_highs": {
        "en": "Near highs",
        "ko": "고점 부근",
    },
    "watch_small_entry": {
        "en": "Watch / small entry",
        "ko": "관찰 / 소액 진입",
    },
    "normal_pullback": {
        "en": "Normal pullback",
        "ko": "일반 조정",
    },
    "first_buy": {
        "en": "First phased buy",
        "ko": "1차 분할매수",
    },
    "moderate_correction": {
        "en": "Moderate correction",
        "ko": "중간 수준 조정",
    },
    "second_buy": {
        "en": "Second phased buy",
        "ko": "2차 분할매수",
    },
    "deep_correction": {
        "en": "Deep correction",
        "ko": "깊은 조정",
    },
    "aggressive_buy": {
        "en": "Aggressive phased buy",
        "ko": "공격적 분할매수",
    },
    "severe_drawdown": {
        "en": "Severe drawdown",
        "ko": "심한 낙폭",
    },
    "opp_with_caution": {
        "en": "Opportunity with caution",
        "ko": "주의가 필요한 기회",
    },
    "current_state_near_highs": {
        "en": "Current state: Near highs",
        "ko": "현재 상태: 고점 부근",
    },
    "current_state_normal_pullback": {
        "en": "Current state: Normal pullback",
        "ko": "현재 상태: 일반 조정",
    },
    "current_state_moderate": {
        "en": "Current state: Moderate correction",
        "ko": "현재 상태: 중간 수준 조정",
    },
    "current_state_deep": {
        "en": "Current state: Deep correction",
        "ko": "현재 상태: 깊은 조정",
    },
    "current_state_severe": {
        "en": "Current state: Severe drawdown",
        "ko": "현재 상태: 심한 낙폭",
    },
    "price_and_ath": {
        "en": "Price and ATH",
        "ko": "가격과 ATH",
    },
    "drawdown_zones": {
        "en": "Drawdown Zones",
        "ko": "낙폭 구간",
    },
    "multifactor_title": {
        "en": "Multi-Factor Score",
        "ko": "멀티팩터 점수",
    },
    "multifactor_desc": {
        "en": (
            "**Core idea**\n"
            "Combine multiple signals instead of relying on only one factor.\n\n"
            "**Factors used here**\n"
            "- Momentum\n"
            "- Low Volatility\n"
            "- Trend\n"
            "- Quality Proxy"
        ),
        "ko": (
            "**핵심 개념**\n"
            "하나의 신호만 보지 않고 여러 요인을 결합해 점수화합니다.\n\n"
            "**여기서 사용하는 요인**\n"
            "- 모멘텀\n"
            "- 저변동성\n"
            "- 추세\n"
            "- 퀄리티 대용지표"
        ),
    },
    "factor_score": {
        "en": "Factor Score",
        "ko": "팩터 점수",
    },
    "positive_momentum": {
        "en": "Positive momentum",
        "ko": "긍정적 모멘텀",
    },
    "weak_momentum": {
        "en": "Weak momentum",
        "ko": "약한 모멘텀",
    },
    "below_median_vol": {
        "en": "Below-median volatility",
        "ko": "중앙값 이하 변동성",
    },
    "above_median_vol": {
        "en": "Above-median volatility",
        "ko": "중앙값 이상 변동성",
    },
    "trend_confirmed": {
        "en": "Trend confirmed",
        "ko": "추세 확인",
    },
    "trend_not_confirmed": {
        "en": "Trend not confirmed",
        "ko": "추세 미확인",
    },
    "quality_strong": {
        "en": "Quality proxy strong",
        "ko": "퀄리티 대용지표 강함",
    },
    "quality_weak": {
        "en": "Quality proxy weak",
        "ko": "퀄리티 대용지표 약함",
    },
    "composite_factor_score": {
        "en": "Composite Factor Score",
        "ko": "종합 팩터 점수",
    },
    "summary_title": {
        "en": "Strategy Summary",
        "ko": "전략 요약",
    },
    "strategy": {
        "en": "Strategy",
        "ko": "전략",
    },
    "final_value": {
        "en": "Final Value",
        "ko": "최종 자산",
    },
    "equity_comparison": {
        "en": "Equity Comparison Across Strategies",
        "ko": "전략별 자산곡선 비교",
    },
    "summary_learn": {
        "en": (
            "**What you learn in this tab**\n"
            "- The same asset can behave very differently under different strategies\n"
            "- Look at MDD and Sharpe, not only CAGR\n"
            "- In practice, combining strategies is often more useful than using just one"
        ),
        "ko": (
            "**이 탭에서 배우는 것**\n"
            "- 같은 자산이라도 전략에 따라 성과가 크게 달라질 수 있음\n"
            "- CAGR뿐 아니라 MDD와 Sharpe도 함께 봐야 함\n"
            "- 실전에서는 한 전략보다 여러 전략의 조합이 더 유용한 경우가 많음"
        ),
    },
    "last_refresh": {
        "en": "Last app refresh",
        "ko": "마지막 앱 새로고침",
    },
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

# =========================================================
# Title
# =========================================================
st.title(tr("app_title"))
st.caption(tr("app_caption"))

# =========================================================
# Sidebar - Market Settings
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header(tr("sidebar_market_settings"))

preset_universe = {
    "QQQ": "QQQ",
    "SPY": "SPY",
    "TLT": "TLT",
    "GLD": "GLD",
    "IWM": "IWM",
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
    ["QQQ", "SPY", "TLT", "GLD", "IWM"],
    default=["QQQ", "SPY", "TLT", "GLD"]
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

initial_capital = st.sidebar.number_input(
    tr("initial_capital"),
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)

show_signals = st.sidebar.checkbox(tr("show_signals"), value=True)

# =========================================================
# Helper Functions
# =========================================================
def fmt_metric(value, kind="float"):
    if pd.isna(value):
        return tr("na")
    if kind == "price":
        return f"{value:,.2f}"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "ratio":
        return f"{value:.2f}"
    return f"{value}"

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
    out = pd.concat(frames, axis=1).dropna(how="all")
    return out

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
    cagr_val = calc_cagr(equity, annual_factor)
    sharpe_val = calc_sharpe(rets, annual_factor)
    _, mdd_val = calc_drawdown(equity)
    return cagr_val, sharpe_val, mdd_val

def backtest_buy_hold(close: pd.Series, initial_capital_value: float) -> pd.Series:
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)
    return initial_capital_value * (close / close.iloc[0])

def backtest_trend(close: pd.Series, ma_short_value: int, ma_long_value: int, initial_capital_value: float) -> pd.DataFrame:
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
    bt["Equity"] = initial_capital_value * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital_value * (1 + bt["Return"]).cumprod()
    bt["Signal_Change"] = bt["Signal"].diff()
    return bt

def backtest_mean_reversion(close: pd.Series, window_value: int, z_entry_value: float, initial_capital_value: float) -> pd.DataFrame:
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
    bt["Equity"] = initial_capital_value * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital_value * (1 + bt["Return"]).cumprod()
    bt["Signal_Change"] = bt["Signal"].diff()
    return bt

def backtest_momentum(close: pd.Series, lookback_value: int, initial_capital_value: float) -> pd.DataFrame:
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
    bt["Equity"] = initial_capital_value * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital_value * (1 + bt["Return"]).cumprod()
    return bt

def backtest_vol_target(close: pd.Series, vol_window_value: int, target_vol_value: float, annual_factor: int, initial_capital_value: float) -> pd.DataFrame:
    close = close.dropna()
    if len(close) < vol_window_value + 5:
        return pd.DataFrame()

    bt = pd.DataFrame(index=close.index)
    bt["Close"] = close
    bt["Return"] = close.pct_change().fillna(0)
    realized_vol = bt["Return"].rolling(vol_window_value).std() * np.sqrt(annual_factor)
    bt["Realized_Vol"] = realized_vol
    bt["Leverage"] = (target_vol_value / bt["Realized_Vol"]).clip(upper=2.0)
    bt["Leverage"] = bt["Leverage"].replace([np.inf, -np.inf], np.nan).fillna(0)
    bt["Strategy_Return"] = bt["Leverage"].shift(1).fillna(0) * bt["Return"]
    bt["Equity"] = initial_capital_value * (1 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = initial_capital_value * (1 + bt["Return"]).cumprod()
    return bt

def build_risk_parity(close_df: pd.DataFrame, vol_window_value: int, annual_factor: int, initial_capital_value: float):
    close_df = close_df.dropna(how="all")
    if close_df.empty or close_df.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()

    returns = close_df.pct_change().fillna(0)
    vol = returns.rolling(vol_window_value).std() * np.sqrt(annual_factor)
    inv_vol = 1 / vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)

    rp_ret = (weights.shift(1).fillna(0) * returns).sum(axis=1)
    eq_ret = returns.mean(axis=1)

    equity = pd.DataFrame(index=returns.index)
    equity["Risk_Parity"] = initial_capital_value * (1 + rp_ret).cumprod()
    equity["Equal_Weight"] = initial_capital_value * (1 + eq_ret).cumprod()
    return weights, equity

def build_rotation_strategy(close_df: pd.DataFrame, lookback_value: int, initial_capital_value: float):
    close_df = close_df.dropna(how="all")
    if close_df.empty or close_df.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()

    momentum = close_df / close_df.shift(lookback_value) - 1
    leader = momentum.idxmax(axis=1)
    returns = close_df.pct_change().fillna(0)

    rot_ret = pd.Series(0.0, index=close_df.index)
    for i in range(1, len(close_df)):
        chosen = leader.iloc[i - 1]
        if pd.notna(chosen):
            rot_ret.iloc[i] = returns.iloc[i][chosen]

    out = pd.DataFrame(index=close_df.index)
    out["Leader"] = leader
    out["Rotation_Return"] = rot_ret
    out["Rotation_Equity"] = initial_capital_value * (1 + rot_ret).cumprod()
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

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs([
    tr_tab("overview_tab"),
    tr_tab("trend_tab"),
    tr_tab("meanrev_tab"),
    tr_tab("momentum_tab"),
    tr_tab("relmom_tab"),
    tr_tab("voltarget_tab"),
    tr_tab("riskparity_tab"),
    tr_tab("drawdown_tab"),
    tr_tab("multifactor_tab"),
    tr_tab("summary_tab"),
])

# =========================================================
# 0. Overview
# =========================================================
with tabs[0]:
    st.subheader(f"{ticker} - {tr('overview_tab')}")

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

    d1, d2, d3 = st.columns(3)
    d1.metric(tr("cagr"), fmt_metric(cagr_val, "pct"))
    d2.metric(tr("sharpe"), fmt_metric(sharpe_val, "ratio"))
    d3.metric(tr("max_drawdown"), fmt_metric(mdd, "pct"))

    st.markdown(tr("overview_learn"))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} {tr('overview_price')}", tr("rsi"), tr("drawdown"))
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash")
    fig.add_hline(y=30, row=2, col=1, line_dash="dash")
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], mode="lines", fill="tozeroy", name="Drawdown"), row=3, col=1)
    fig.update_layout(height=820, title=f"{ticker} {tr('overview_tab')}")
    fig.update_yaxes(title_text=tr("overview_price"), row=1, col=1)
    fig.update_yaxes(title_text=tr("rsi"), range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text=tr("overview_dd_short"), tickformat=".0%", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 1. Trend Following
# =========================================================
with tabs[1]:
    st.subheader(tr("trend_title"))
    st.markdown(tr("trend_desc"))

    bt = backtest_trend(df["Close"], ma_short, ma_long, initial_capital)
    if bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        t_cagr, t_sharpe, t_mdd = equity_stats(bt["Equity"], annual_factor)

        a, b, c = st.columns(3)
        a.metric(tr("trend_cagr"), fmt_metric(t_cagr, "pct"))
        b.metric(tr("trend_sharpe"), fmt_metric(t_sharpe, "ratio"))
        c.metric(tr("trend_mdd"), fmt_metric(t_mdd, "pct"))

        buy_points = bt[bt["Signal_Change"] == 1]
        sell_points = bt[bt["Signal_Change"] == -1]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=(tr("price_and_ma_signals"), tr("equity_curve"))
        )
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["MA_Short"], mode="lines", name=f"MA {ma_short}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["MA_Long"], mode="lines", name=f"MA {ma_long}"), row=1, col=1)

        if show_signals:
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index, y=buy_points["Close"], mode="markers", name=tr("buy"),
                    marker=dict(symbol="triangle-up", size=11)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index, y=sell_points["Close"], mode="markers", name=tr("sell"),
                    marker=dict(symbol="triangle-down", size=11)
                ),
                row=1, col=1
            )

        fig.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name=tr("trend_strategy")), row=2, col=1)
        fig.update_layout(height=840)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2. Mean Reversion
# =========================================================
with tabs[2]:
    st.subheader(tr("meanrev_title"))
    st.markdown(tr("meanrev_desc"))

    bt = backtest_mean_reversion(df["Close"], mr_window, mr_z, initial_capital)
    if bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        mr_cagr_val, mr_sharpe_val, mr_mdd_val = equity_stats(bt["Equity"], annual_factor)
        a, b, c = st.columns(3)
        a.metric(tr("mr_cagr"), fmt_metric(mr_cagr_val, "pct"))
        b.metric(tr("mr_sharpe"), fmt_metric(mr_sharpe_val, "ratio"))
        c.metric(tr("mr_mdd"), fmt_metric(mr_mdd_val, "pct"))

        buy_points = bt[bt["Signal_Change"] == 1]
        sell_points = bt[bt["Signal_Change"] == -1]

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.42, 0.23, 0.35],
            subplot_titles=(tr("price_vs_mean"), tr("zscore"), tr("equity_curve"))
        )
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Mean"], mode="lines", name="Rolling Mean"), row=1, col=1)

        if show_signals:
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index, y=buy_points["Close"], mode="markers", name=tr("buy"),
                    marker=dict(symbol="triangle-up", size=11)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index, y=sell_points["Close"], mode="markers", name=tr("exit"),
                    marker=dict(symbol="triangle-down", size=11)
                ),
                row=1, col=1
            )

        fig.add_trace(go.Scatter(x=bt.index, y=bt["Z"], mode="lines", name=tr("zscore")), row=2, col=1)
        fig.add_hline(y=-mr_z, row=2, col=1, line_dash="dash")
        fig.add_hline(y=0, row=2, col=1, line_dash="dot")
        fig.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=3, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name=tr("meanrev_strategy")), row=3, col=1)
        fig.update_layout(height=930)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 3. Momentum
# =========================================================
with tabs[3]:
    st.subheader(tr("momentum_title"))
    st.markdown(tr("momentum_desc"))

    bt = backtest_momentum(df["Close"], mom_lookback, initial_capital)
    if bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        m_cagr_val, m_sharpe_val, m_mdd_val = equity_stats(bt["Equity"], annual_factor)
        a, b, c = st.columns(3)
        a.metric(tr("momentum_cagr"), fmt_metric(m_cagr_val, "pct"))
        b.metric(tr("momentum_sharpe"), fmt_metric(m_sharpe_val, "ratio"))
        c.metric(tr("momentum_mdd"), fmt_metric(m_mdd_val, "pct"))

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=(tr("price_and_momentum"), tr("equity_curve"))
        )
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Momentum"], mode="lines", name=tr("momentum_title")), row=1, col=1)
        fig.add_hline(y=0, row=1, col=1, line_dash="dash")
        fig.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name=tr("momentum_strategy")), row=2, col=1)
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 4. Relative Momentum / Rotation
# =========================================================
with tabs[4]:
    st.subheader(tr("relmom_title"))
    st.markdown(tr("relmom_desc"))

    if cmp_df.empty or cmp_df.shape[1] < 2:
        st.warning(tr("need_two_assets"))
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
        fig.update_layout(height=450, title=tr("normalized_performance"))
        st.plotly_chart(fig, use_container_width=True)

        if not rot.empty:
            rot_cagr_val, rot_sharpe_val, rot_mdd_val = equity_stats(rot["Rotation_Equity"], annual_factor)
            st.metric(tr("rotation_cagr"), fmt_metric(rot_cagr_val, "pct"))

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=rot.index, y=rot["Rotation_Equity"], mode="lines", name=tr("relmom_title")))
            fig2.update_layout(height=420, title=tr("rotation_equity"))
            st.plotly_chart(fig2, use_container_width=True)

            last_leaders = rot["Leader"].dropna().tail(20).to_frame(name=tr("selected_asset"))
            st.subheader(tr("recent_rotation_decisions"))
            st.dataframe(last_leaders, use_container_width=True)

# =========================================================
# 5. Volatility Targeting
# =========================================================
with tabs[5]:
    st.subheader(tr("voltarget_title"))
    st.markdown(tr("voltarget_desc"))

    bt = backtest_vol_target(df["Close"], vol_window, target_vol, annual_factor, initial_capital)
    if bt.empty:
        st.warning(tr("not_enough_data"))
    else:
        v_cagr_val, v_sharpe_val, v_mdd_val = equity_stats(bt["Equity"], annual_factor)
        a, b, c = st.columns(3)
        a.metric(tr("voltarget_cagr"), fmt_metric(v_cagr_val, "pct"))
        b.metric(tr("voltarget_sharpe"), fmt_metric(v_sharpe_val, "ratio"))
        c.metric(tr("voltarget_mdd"), fmt_metric(v_mdd_val, "pct"))

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.34, 0.28, 0.38],
            subplot_titles=(tr("overview_price"), tr("realized_vol_and_leverage"), tr("equity_curve"))
        )
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Close"], mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Realized_Vol"], mode="lines", name=tr("volatility")), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Leverage"], mode="lines", name="Leverage"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["BuyHold_Equity"], mode="lines", name=tr("buy_hold")), row=3, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name=tr("vol_target")), row=3, col=1)
        fig.update_layout(height=900)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 6. Risk Parity
# =========================================================
with tabs[6]:
    st.subheader(tr("riskparity_title"))
    st.markdown(tr("riskparity_desc"))

    if cmp_df.empty or cmp_df.shape[1] < 2:
        st.warning(tr("need_multiple_assets"))
    else:
        weights, equity = build_risk_parity(cmp_df, vol_window, annual_factor, initial_capital)

        if equity.empty:
            st.warning(tr("not_enough_data"))
        else:
            rp_cagr_val, rp_sharpe_val, rp_mdd_val = equity_stats(equity["Risk_Parity"], annual_factor)

            a, b, c = st.columns(3)
            a.metric(tr("rp_cagr"), fmt_metric(rp_cagr_val, "pct"))
            b.metric(tr("rp_sharpe"), fmt_metric(rp_sharpe_val, "ratio"))
            c.metric(tr("rp_mdd"), fmt_metric(rp_mdd_val, "pct"))

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                row_heights=[0.4, 0.6],
                subplot_titles=(tr("risk_parity_weights"), tr("equity_curve"))
            )

            for col in weights.columns:
                fig.add_trace(go.Scatter(x=weights.index, y=weights[col], mode="lines", name=f"W_{col}"), row=1, col=1)

            fig.add_trace(go.Scatter(x=equity.index, y=equity["Equal_Weight"], mode="lines", name=tr("equal_weight")), row=2, col=1)
            fig.add_trace(go.Scatter(x=equity.index, y=equity["Risk_Parity"], mode="lines", name=tr("risk_parity")), row=2, col=1)
            fig.update_layout(height=860)
            fig.update_yaxes(tickformat=".0%", row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 7. Drawdown Buying
# =========================================================
with tabs[7]:
    st.subheader(tr("drawdown_title"))
    st.markdown(tr("drawdown_desc"))

    latest_dd = latest["Drawdown"]

    guide_rows = [
        {tr("zone"): "0% ~ -5%", tr("meaning"): tr("near_highs"), tr("action"): tr("watch_small_entry")},
        {tr("zone"): "-5% ~ -10%", tr("meaning"): tr("normal_pullback"), tr("action"): tr("first_buy")},
        {tr("zone"): "-10% ~ -15%", tr("meaning"): tr("moderate_correction"), tr("action"): tr("second_buy")},
        {tr("zone"): "-15% ~ -20%", tr("meaning"): tr("deep_correction"), tr("action"): tr("aggressive_buy")},
        {tr("zone"): "< -20%", tr("meaning"): tr("severe_drawdown"), tr("action"): tr("opp_with_caution")},
    ]
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True)

    if pd.notna(latest_dd):
        if latest_dd > -0.05:
            st.info(tr("current_state_near_highs"))
        elif latest_dd > -0.10:
            st.info(tr("current_state_normal_pullback"))
        elif latest_dd > -0.15:
            st.warning(tr("current_state_moderate"))
        elif latest_dd > -0.20:
            st.warning(tr("current_state_deep"))
        else:
            st.error(tr("current_state_severe"))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        subplot_titles=(tr("price_and_ath"), tr("drawdown_zones"))
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ATH"], mode="lines", name="ATH"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"], mode="lines", fill="tozeroy", name=tr("drawdown")), row=2, col=1)
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
    st.subheader(tr("multifactor_title"))
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
        subplot_titles=(tr("overview_price"), tr("composite_factor_score"))
    )
    fig.add_trace(go.Scatter(x=fac_df.index, y=fac_df["Close"], mode="lines", name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fac_df.index, y=fac_df["Composite_Score"], mode="lines", name=tr("multifactor_title")), row=2, col=1)
    fig.update_layout(height=760)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 9. Strategy Summary
# =========================================================
with tabs[9]:
    st.subheader(tr("summary_title"))

    summary_rows = []

    bh = backtest_buy_hold(df["Close"], initial_capital)
    if not bh.empty:
        cagr_bh, sharpe_bh, mdd_bh = equity_stats(bh, annual_factor)
        summary_rows.append({
            tr("strategy"): tr("buy_hold"),
            tr("final_value"): bh.iloc[-1],
            tr("cagr"): cagr_bh,
            tr("sharpe"): sharpe_bh,
            tr("max_drawdown"): mdd_bh
        })

    tf = backtest_trend(df["Close"], ma_short, ma_long, initial_capital)
    if not tf.empty:
        cagr_tf, sharpe_tf, mdd_tf = equity_stats(tf["Equity"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("trend_title"),
            tr("final_value"): tf["Equity"].iloc[-1],
            tr("cagr"): cagr_tf,
            tr("sharpe"): sharpe_tf,
            tr("max_drawdown"): mdd_tf
        })

    mr = backtest_mean_reversion(df["Close"], mr_window, mr_z, initial_capital)
    if not mr.empty:
        cagr_mr, sharpe_mr, mdd_mr = equity_stats(mr["Equity"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("meanrev_title"),
            tr("final_value"): mr["Equity"].iloc[-1],
            tr("cagr"): cagr_mr,
            tr("sharpe"): sharpe_mr,
            tr("max_drawdown"): mdd_mr
        })

    mom = backtest_momentum(df["Close"], mom_lookback, initial_capital)
    if not mom.empty:
        cagr_mom, sharpe_mom, mdd_mom = equity_stats(mom["Equity"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("momentum_title"),
            tr("final_value"): mom["Equity"].iloc[-1],
            tr("cagr"): cagr_mom,
            tr("sharpe"): sharpe_mom,
            tr("max_drawdown"): mdd_mom
        })

    vol_bt = backtest_vol_target(df["Close"], vol_window, target_vol, annual_factor, initial_capital)
    if not vol_bt.empty:
        cagr_v, sharpe_v, mdd_v = equity_stats(vol_bt["Equity"], annual_factor)
        summary_rows.append({
            tr("strategy"): tr("voltarget_title"),
            tr("final_value"): vol_bt["Equity"].iloc[-1],
            tr("cagr"): cagr_v,
            tr("sharpe"): sharpe_v,
            tr("max_drawdown"): mdd_v
        })

    if not cmp_df.empty and cmp_df.shape[1] >= 2:
        _, rot = build_rotation_strategy(cmp_df, mom_lookback, initial_capital)
        if not rot.empty:
            cagr_rot, sharpe_rot, mdd_rot = equity_stats(rot["Rotation_Equity"], annual_factor)
            summary_rows.append({
                tr("strategy"): tr("relmom_title"),
                tr("final_value"): rot["Rotation_Equity"].iloc[-1],
                tr("cagr"): cagr_rot,
                tr("sharpe"): sharpe_rot,
                tr("max_drawdown"): mdd_rot
            })

        _, rp_eq = build_risk_parity(cmp_df, vol_window, annual_factor, initial_capital)
        if not rp_eq.empty:
            cagr_rp, sharpe_rp, mdd_rp = equity_stats(rp_eq["Risk_Parity"], annual_factor)
            summary_rows.append({
                tr("strategy"): tr("riskparity_title"),
                tr("final_value"): rp_eq["Risk_Parity"].iloc[-1],
                tr("cagr"): cagr_rp,
                tr("sharpe"): sharpe_rp,
                tr("max_drawdown"): mdd_rp
            })

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        st.warning(tr("not_enough_data"))
    else:
        display_df = summary_df.copy()
        display_df[tr("final_value")] = display_df[tr("final_value")].map(lambda x: f"{x:,.0f}" if pd.notna(x) else tr("na"))
        display_df[tr("cagr")] = display_df[tr("cagr")].map(lambda x: f"{x:.2%}" if pd.notna(x) else tr("na"))
        display_df[tr("sharpe")] = display_df[tr("sharpe")].map(lambda x: f"{x:.2f}" if pd.notna(x) else tr("na"))
        display_df[tr("max_drawdown")] = display_df[tr("max_drawdown")].map(lambda x: f"{x:.2%}" if pd.notna(x) else tr("na"))
        st.dataframe(display_df, use_container_width=True)

        fig = go.Figure()

        if not bh.empty:
            fig.add_trace(go.Scatter(x=bh.index, y=bh, mode="lines", name=tr("buy_hold")))
        if not tf.empty:
            fig.add_trace(go.Scatter(x=tf.index, y=tf["Equity"], mode="lines", name=tr("trend_title")))
        if not mr.empty:
            fig.add_trace(go.Scatter(x=mr.index, y=mr["Equity"], mode="lines", name=tr("meanrev_title")))
        if not mom.empty:
            fig.add_trace(go.Scatter(x=mom.index, y=mom["Equity"], mode="lines", name=tr("momentum_title")))
        if not vol_bt.empty:
            fig.add_trace(go.Scatter(x=vol_bt.index, y=vol_bt["Equity"], mode="lines", name=tr("voltarget_title")))
        if not cmp_df.empty and cmp_df.shape[1] >= 2:
            _, rot = build_rotation_strategy(cmp_df, mom_lookback, initial_capital)
            if not rot.empty:
                fig.add_trace(go.Scatter(x=rot.index, y=rot["Rotation_Equity"], mode="lines", name=tr("relmom_title")))
            _, rp_eq = build_risk_parity(cmp_df, vol_window, annual_factor, initial_capital)
            if not rp_eq.empty:
                fig.add_trace(go.Scatter(x=rp_eq.index, y=rp_eq["Risk_Parity"], mode="lines", name=tr("riskparity_title")))

        fig.update_layout(height=520, title=tr("equity_comparison"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(tr("summary_learn"))

# =========================================================
# Footer
# =========================================================
st.caption(f"{tr('last_refresh')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
