# streamlit_app.py
# -*- coding: utf-8 -*-
# ==============================================
# Streamlit Dashboard: Temperature & Suicide Rate
# 1) Public Open-Data Dashboard (World Bank + NASA GISTEMP)
# 2) User-Input-Inspired Dashboard (recreate Stanford-style figure with synthetic data)
# ----------------------------------------------
# Data Sources (official):
# - World Bank Suicide mortality rate (per 100,000 population), code: SH.STA.SUIC.P5
#   API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
#   Example JSON: https://api.worldbank.org/v2/country/USA/indicator/SH.STA.SUIC.P5?format=json&per_page=20000
# - NASA GISTEMP v4 Global mean temperature anomaly (annual, °C, base 1951-1980)
#   Direct CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
#   Docs: https://data.giss.nasa.gov/gistemp/
# Notes:
# - If API download fails (no internet / blocked), the app asks for a user CSV upload.
# - If no upload is provided, the app uses a tiny example dataset as a last-resort fallback.
# - Future dates are filtered out.
# ==============================================

import io
import os
import sys
import json
import time
import datetime as dt
from base64 import b64encode

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from scipy import stats

# ------------------------------
# Font injection (optional)
# ------------------------------
def inject_font_css():
    import os
    from base64 import b64encode
    import streamlit as st
    font_path = "/fonts/Pretendard-Bold.ttf"
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            font_data = b64encode(f.read()).decode("utf-8")
        st.markdown(f"""
        <style>
        @font-face {{
            font-family: 'Pretendard';
            src: url(data:font/ttf;base64,{font_data}) format('truetype');
            font-weight: 700; font-style: normal; font-display: swap;
        }}
        html, body, [class*="css"] {{
            font-family: 'Pretendard', sans-serif !important;
        }}
        </style>
        """, unsafe_allow_html=True)

inject_font_css()

st.set_page_config(
    page_title="Temperature & Suicide Rate Dashboard",
    layout="wide",
)

st.title("🌡️ 기온 변화와 자살률: 공개 데이터 & 사용자 입력 기반 대시보드")
st.caption("공개 데이터(우선) → 실패 시 CSV 업로드 → 최후 예시데이터(Fallback). 미래 데이터는 숨깁니다.")

# =========================================================
# Utilities
# =========================================================
TODAY_YEAR = dt.date.today().year

@st.cache_data(ttl=3600)
def fetch_worldbank_suicide(country_code: str) -> pd.DataFrame:
    """Fetch World Bank suicide mortality rate (per 100k), annual.
    URL example:
      https://api.worldbank.org/v2/country/USA/indicator/SH.STA.SUIC.P5?format=json&per_page=20000
    """
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SH.STA.SUIC.P5"
    params = {"format": "json", "per_page": 20000}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    # js[1] holds records
    records = js[1] if isinstance(js, list) and len(js) > 1 else []
    rows = []
    for row in records:
        year = row.get("date")
        value = row.get("value")
        try:
            yi = int(year)
        except:
            continue
        if yi <= TODAY_YEAR:
            rows.append({"year": yi, "suicide_rate": value})
    df = pd.DataFrame(rows).dropna().sort_values("year")
    return df

@st.cache_data(ttl=3600)
def fetch_nasa_gistemp_global() -> pd.DataFrame:
    """Fetch NASA GISTEMP v4 global mean temperature anomaly (annual °C).
    CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    We parse first column (Year) and J-D column (annual mean).
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    content = r.content.decode("utf-8", errors="ignore")
    # Find rows starting with year numbers
    data = []
    for line in content.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 14:
            continue
        # Expect first column = Year, last relevant column "J-D" annual mean
        try:
            year = int(parts[0])
        except:
            continue
        # Find "J-D" column (usually index 13)
        try:
            # Some files have trailing empty col; be safe by searching header first
            pass
        except:
            pass
        # Simple approach: NASA CSV steady format -> J-D is 13th (index 13-1=12) or 14th
    # Re-read using pandas with skiprows until "Year," header
    df = pd.read_csv(io.StringIO(content), skiprows=1)
    # Identify Year + "J-D"
    if "Year" in df.columns and "J-D" in df.columns:
        out = df[["Year", "J-D"]].rename(columns={"Year": "year", "J-D": "temp_anomaly"})
        out = out[pd.to_numeric(out["temp_anomaly"], errors="coerce").notna()]
        out["year"] = out["year"].astype(int)
        out["temp_anomaly"] = out["temp_anomaly"].astype(float) / 100.0  # NASA file in hundredths of °C
        out = out[out["year"] <= TODAY_YEAR].reset_index(drop=True)
        return out
    else:
        # Fallback parse: try first numeric column as year and last numeric as J-D
        df = df.rename(columns={df.columns[0]: "year"})
        jd_col = [c for c in df.columns if "J-D" in c]
        col = jd_col[0] if jd_col else df.columns[-1]
        out = df[["year", col]].rename(columns={col: "temp_anomaly"})
        out = out[pd.to_numeric(out["temp_anomaly"], errors="coerce").notna()]
        out["year"] = out["year"].astype(int)
        out["temp_anomaly"] = out["temp_anomaly"].astype(float) / 100.0
        out = out[out["year"] <= TODAY_YEAR].reset_index(drop=True)
        return out

def tiny_example_df():
    years = np.arange(2000, 2011)
    rng = np.random.default_rng(42)
    temp = np.linspace(0.2, 0.8, len(years)) + rng.normal(0, 0.03, len(years))
    sui  = np.linspace(16, 14, len(years)) + rng.normal(0, 0.2, len(years))
    return pd.DataFrame({"year": years, "temp_anomaly": temp, "suicide_rate": sui})

def ols_fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slope, intercept, r, p, se = stats.linregress(x, y)
    return slope, intercept, r, p, se

# =========================================================
# Sidebar Controls (global)
# =========================================================
st.sidebar.header("⚙️ 옵션")
tab_choice = st.sidebar.radio("대시보드 선택", ["① 공개 데이터 대시보드", "② 사용자 입력(설명/이미지) 대시보드"], index=0)

# =========================================================
# ① Public Open-Data Dashboard
# =========================================================
if tab_choice == "① 공개 데이터 대시보드":
    st.subheader("① 공개 데이터 대시보드 (World Bank + NASA GISTEMP)")
    st.markdown(
        "- **자살률**: World Bank `SH.STA.SUIC.P5` (연도별, 10만명당)\n"
        "- **기온**: NASA GISTEMP v4 글로벌 연평균 온도 이상치(°C, 1951–1980 기준)\n"
        "  - 서로 다른 단위/범위이므로 **동년의 연도** 기준으로 관계를 확인합니다."
    )

    colA, colB = st.columns([1,1])
    with colA:
        country_label = st.selectbox("국가 선택", ["United States (USA)", "Mexico (MEX)", "Korea, Rep. (KOR)", "Japan (JPN)", "Germany (DEU)"], index=0)
    with colB:
        st.write("")

    country_code = country_label.split("(")[-1].replace(")", "").strip()

    # Try fetching
    data_ok = True
    err_msg = None
    try:
        wb = fetch_worldbank_suicide(country_code)
        nasa = fetch_nasa_gistemp_global()
    except Exception as e:
        data_ok = False
        err_msg = str(e)

    df_merge = None
    if data_ok and not wb.empty and not nasa.empty:
        df_merge = pd.merge(wb, nasa, on="year", how="inner")
        df_merge = df_merge[(df_merge["year"] <= TODAY_YEAR)]
        st.success(f"데이터 로드 성공: World Bank {len(wb)}개 연도, NASA {len(nasa)}개 연도 → 교집합 {len(df_merge)}개 연도")
    else:
        st.warning("공개 데이터 다운로드 실패 또는 빈 데이터.")
        if err_msg:
            with st.expander("오류 상세 보기"):
                st.code(err_msg)

    # Offer CSV upload when API fails
    if (df_merge is None or df_merge.empty):
        st.info("🔄 CSV 업로드로 계속할 수 있어요. (열: year, suicide_rate, temp_anomaly)")
        up = st.file_uploader("CSV 업로드 (year, suicide_rate, temp_anomaly)", type=["csv"])
        if up:
            user_df = pd.read_csv(up)
            if set(["year","suicide_rate","temp_anomaly"]).issubset(user_df.columns):
                df_merge = user_df.copy()
                st.success("CSV 로드 성공!")
            else:
                st.error("CSV 컬럼명이 일치하지 않습니다. year, suicide_rate, temp_anomaly 필요.")
        if df_merge is None or df_merge.empty:
            st.info("📦 최후 수단: 예시 데이터 사용")
            df_merge = tiny_example_df()

    # Filter by year
    miny, maxy = int(df_merge["year"].min()), int(df_merge["year"].max())
    y1, y2 = st.slider("연도 범위 선택", min_value=miny, max_value=maxy, value=(max(miny, maxy-30), maxy))
    dm = df_merge[(df_merge["year"] >= y1) & (df_merge["year"] <= y2)].copy()

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.line(dm, x="year", y="suicide_rate", markers=True, title=f"자살률 추이 (World Bank) - {country_code}")
        fig1.update_layout(height=360)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.line(dm, x="year", y="temp_anomaly", markers=True, title="글로벌 기온 이상치 (NASA GISTEMP, °C)")
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter + OLS fit
    if len(dm) >= 3:
        slope, intercept, r, p, se = ols_fit(dm["temp_anomaly"], dm["suicide_rate"])
        xgrid = np.linspace(dm["temp_anomaly"].min(), dm["temp_anomaly"].max(), 100)
        yfit = slope * xgrid + intercept

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=dm["temp_anomaly"], y=dm["suicide_rate"],
                                  mode="markers", name="연도별 관측치"))
        fig3.add_trace(go.Scatter(x=xgrid, y=yfit, mode="lines", name="OLS 추세선"))
        fig3.update_layout(
            title=f"기온 이상치 vs 자살률 ({country_code})  |  r={r:.3f}, p={p:.3g}, slope={slope:.3f}",
            xaxis_title="기온 이상치 (°C, NASA GISTEMP)",
            yaxis_title="자살률 (10만명당, World Bank)",
            height=420
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("주의: 글로벌 평균 기온과 한 국가의 자살률을 단순 연도 매칭하여 본 상관관계는 **인과를 의미하지 않으며** 교란변수(경제, 보건정책 등)를 통제하지 않았습니다.")

    with st.expander("데이터 출처 & 참고 링크"):
        st.markdown(
            "- World Bank API (Suicide mortality rate): `SH.STA.SUIC.P5`\n"
            "  - https://api.worldbank.org/v2/country/USA/indicator/SH.STA.SUIC.P5?format=json&per_page=20000\n"
            "- NASA GISTEMP v4 (Global mean temperature anomaly, annual):\n"
            "  - https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv\n"
            "- 분석 로직: 동년 데이터 병합 → 산점도 & OLS 선형회귀"
        )

# =========================================================
# ② User Input (Image/Description)-based Dashboard
# =========================================================
else:
    st.subheader("② 사용자 입력 기반 대시보드 (이미지/설명 재현)")

    # The prompt described an image: US (left) and Mexico (right), Stanford figure.
    # We'll reconstruct a *similar* synthetic relationship: % change in suicide rate (y)
    # vs monthly average temperature (°C, x), with linear estimate and a confidence band.

    st.markdown("입력 요약: *“기온 상승과 자살률 변화 사이의 상관관계. 왼쪽 미국, 오른쪽 멕시코. 스탠퍼드대학교”*")
    img_path = "/mnt/data/6e24a4be-2147-4211-9f8c-883fb8147347.png"
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption="참고 이미지 (설명 기반 재현용)")

    st.info("아래 그래프는 **설명/이미지에서 추정한 경향**을 토대로 생성한 **예시 데이터**입니다. 실제 연구 수치와 다를 수 있습니다.")

    # Controls for synthetic generation
    col0, col1, col2 = st.columns(3)
    with col0:
        x_min = st.number_input("온도 최소(°C)", value=-20.0)
    with col1:
        x_max = st.number_input("온도 최대(°C)", value=40.0)
    with col2:
        n_pts = st.slider("표본 수", 20, 120, 60)

    def make_synth(country="USA", x_min=-20, x_max=40, n=60, seed=0):
        rng = np.random.default_rng(seed)
        x = np.linspace(x_min, x_max, n)
        # Choose slope/intercept to mimic rising trend; Mexico steeper.
        if country == "USA":
            slope = 0.5     # % change per °C
            intercept = -10
            noise = rng.normal(0, 2.2, size=n)
        else:  # Mexico
            slope = 1.0
            intercept = -15
            noise = rng.normal(0, 3.2, size=n)
        y = intercept + slope * x + noise
        # Build a simple confidence-like band using rolling std
        df = pd.DataFrame({"temp_C": x, "pct_change_suicide": y})
        df["y_hat"] = intercept + slope * df["temp_C"]
        # pseudo CI width increases away from 20°C to echo the visual
        spread = (np.abs(df["temp_C"] - 20) / (x_max - x_min) + 0.1) * (4.0 if country=="MEX" else 3.0)
        df["ci_lo"] = df["y_hat"] - spread * 1.64
        df["ci_hi"] = df["y_hat"] + spread * 1.64
        return df, slope, intercept

    df_us, s_us, b_us = make_synth("USA", x_min, x_max, n_pts, seed=1)
    df_mx, s_mx, b_mx = make_synth("MEX", x_min, x_max, n_pts, seed=2)

    c1, c2 = st.columns(2, gap="large")

    def draw_panel(df, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([df["temp_C"], df["temp_C"][::-1]]),
            y=pd.concat([df["ci_hi"], df["ci_lo"][::-1]]),
            fill="toself", mode="lines", line=dict(width=0), name="Confidence interval", opacity=0.25
        ))
        fig.add_trace(go.Scatter(x=df["temp_C"], y=df["y_hat"], mode="lines", name="Estimate"))
        fig.update_layout(
            title=title,
            xaxis_title="Monthly average temperature (°C)",
            yaxis_title="% change in suicide rate",
            yaxis=dict(range=[-40, 40]),
            height=480,
            showlegend=True
        )
        return fig

    with c1:
        st.plotly_chart(draw_panel(df_us, "United States"), use_container_width=True)
        st.caption(f"가정된 기울기(slope): {s_us:+.2f} %/°C")
    with c2:
        st.plotly_chart(draw_panel(df_mx, "Mexico"), use_container_width=True)
        st.caption(f"가정된 기울기(slope): {s_mx:+.2f} %/°C")

    st.markdown("---")
    st.markdown("**주의**: 위 시각화는 **원문 연구(예: Burke et al., 2018, Stanford 계열 연구)**의 이미지를 모사한 **설명 기반 예시**입니다. 실제 재현을 위해서는 월별 지역 온도와 자살 발생 수(인구구조 통제 포함)가 필요합니다.")

    with st.expander("실제 재현을 위한 데이터 가이드"):
        st.markdown(
            "1) **월별 평균기온**: NOAA CDO 또는 ERA5 월평균 2m 기온 (국가/군집 수준 집계)\n"
            "2) **월별 자살 사망 수**: 각국 통계청/보건부 공개 자료 (인구보정 필요)\n"
            "3) **모형**: 패널 고정효과 + 월·지역 고정효과 + 계절/경향 통제\n"
            "4) **출처 표기**를 코드/보고서에 명확히 남길 것"
        )

st.markdown("---")
st.caption("© Open-data first (World Bank/NASA). If calls fail → CSV 업로드 → 예시 데이터 fallback. 미래 연도 데이터는 제외됩니다.")
