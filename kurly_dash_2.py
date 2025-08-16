import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ================================
# 🎨 Palette (주신 헥사코드)
# ================================
COL_BG = "#F7F5F5"   # 배경
COL_P  = "#5A0F79"   # Primary (포인트/라인)
COL_S1 = "#8B57A0"   # Secondary(진한 보조)
COL_S2 = "#AA73C3"   # Secondary(밝은 보조)
COL_S3 = "#BB7CCE"   # Secondary(파스텔)
COLORWAY = [COL_P, COL_S2, COL_S1, COL_S3]

# ================================
# Streamlit 기본 설정 & 글로벌 스타일
# ================================
st.set_page_config(page_title="월별 매출 대시보드", layout="wide", page_icon="📊")
st.title("📊 월별 매출 대시보드 (Streamlit)")
st.caption("CSV 업로드 후 4가지 시각화가 자동 생성됩니다. 컬럼: 월(YYYY-MM), 매출액, 전년동월, 증감률(%). 미입력 시 증감률은 전년동월로 자동 계산합니다.")

# CSS로 배경/타이틀/메트릭 카드 등 톤 통일
st.markdown(
    f"""
    <style>
    :root {{
      --bg: {COL_BG};
      --primary: {COL_P};
      --s1: {COL_S1};
      --s2: {COL_S2};
      --s3: {COL_S3};
    }}
    .stApp {{ background-color: var(--bg); }}
    h1, h2, h3, h4 {{ color: var(--primary) !important; }}
    /* Metric 카드 톤 정리 */
    div[data-testid="stMetric"] {{
        background: #ffffff;
        border-left: 6px solid var(--primary);
        border-radius: 14px;
        padding: 12px 12px;
        box-shadow: 0 2px 8px rgba(90,15,121,0.06);
    }}
    /* 사이드바 배경/구분선 */
    [data-testid="stSidebar"] {{
        background-color: var(--bg) !important;
        border-right: 1px solid rgba(90,15,121,0.15);
    }}
    /* 버튼/체크박스 포커스 색 */
    .st-emotion-cache-1vt4y43, .st-emotion-cache-7ym5gk, .st-emotion-cache-1vt4y43:focus {{
        accent-color: var(--primary);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# Plotly 공통 테마 함수
# ================================
def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        colorway=COLORWAY,
        font=dict(
            family="Pretendard, Apple SD Gothic Neo, Malgun Gothic, Segoe UI, Arial",
            size=13,
            color="#2C2C2C",
        ),
        paper_bgcolor=COL_BG,
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(gridcolor="#EAE7F2", zeroline=False, linecolor="rgba(90,15,121,0.25)"),
        yaxis=dict(gridcolor="#EAE7F2", zeroline=False, linecolor="rgba(90,15,121,0.25)"),
    )
    return fig

# ================================
# 샘플 데이터
# ================================
SAMPLE_CSV = (
    "월,매출액,전년동월,증감률\n"
    "2024-01,12000000,10500000,14.3\n"
    "2024-02,13500000,11200000,20.5\n"
    "2024-03,11000000,12800000,-14.1\n"
    "2024-04,18000000,15200000,18.4\n"
    "2024-05,21000000,18500000,13.5\n"
    "2024-06,22000000,19000000,15.8\n"
    "2024-07,25000000,20500000,22.0\n"
    "2024-08,28000000,24500000,14.3\n"
    "2024-09,24000000,21000000,14.3\n"
    "2024-10,23000000,20000000,15.0\n"
    "2024-11,19500000,17500000,11.4\n"
    "2024-12,17000000,16500000,3.0\n"
)

@st.cache_data
def read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

@st.cache_data
def parse_sample(sample_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(sample_text))

@st.cache_data
def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 표준화된 컬럼명 가정: 월, 매출액, 전년동월, 증감률
    df["월"] = df["월"].astype(str).str.strip()
    # 날짜 정렬용 컬럼
    df["_date"] = pd.to_datetime(df["월"], format="%Y-%m", errors="coerce")
    df = df.sort_values("_date").reset_index(drop=True)
    # 숫자 캐스팅
    for c in ["매출액", "전년동월"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "증감률" in df.columns:
        df["증감률"] = pd.to_numeric(df.get("증감률"), errors="coerce")
    else:
        df["증감률"] = np.nan
    # 증감률 자동 계산
    missing_mask = df["증감률"].isna()
    df.loc[missing_mask & df["전년동월"].ne(0), "증감률"] = (
        (df.loc[missing_mask, "매출액"] - df.loc[missing_mask, "전년동월"]) / df.loc[missing_mask, "전년동월"] * 100
    )
    df["증감률"] = df["증감률"].fillna(0)
    # 분기 계산
    df["분기"] = df["_date"].dt.quarter
    return df

# Sidebar: 파일 업로드 / 샘플 버튼 / KPI 목표
with st.sidebar:
    st.header("⚙️ 설정")
    uploaded = st.file_uploader("CSV 업로드", type=["csv"], accept_multiple_files=False)
    use_sample = st.checkbox("샘플 데이터 불러오기", value=True if uploaded is None else False)
    target = st.number_input("KPI 목표 매출 (원)", min_value=0, value=20_000_000, step=100_000)

# Load data
if uploaded is not None:
    df_raw = read_csv(uploaded)
elif use_sample:
    df_raw = parse_sample(SAMPLE_CSV)
else:
    st.info("좌측에서 CSV를 업로드하거나 '샘플 데이터 불러오기'를 선택하세요.")
    st.stop()

# Enrich
try:
    df = enrich_df(df_raw)
except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
    st.stop()

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_sales = int(df["매출액"].sum())
    st.metric("총합 매출", f"{total_sales:,.0f}원")
with col2:
    avg_yoy = float(df["증감률"].mean())
    st.metric("평균 증감률", f"{avg_yoy:.1f}%")
with col3:
    max_idx = df["매출액"].idxmax()
    st.metric("최고 매출 (월)", f"{df.loc[max_idx,'월']} · {df.loc[max_idx,'매출액']:,.0f}원")
with col4:
    min_idx = df["매출액"].idxmin()
    st.metric("최저 매출 (월)", f"{df.loc[min_idx,'월']} · {df.loc[min_idx,'매출액']:,.0f}원")

st.divider()

# 1) 월별 매출 추이 (이중선)
with st.container():
    st.subheader("1) 월별 매출 추이 (매출액 vs 전년동월)")
    fig_trend = go.Figure()
    # 매출액(진한 보라)
    fig_trend.add_trace(go.Scatter(
        x=df["월"], y=df["매출액"], mode="lines+markers", name="매출액",
        line=dict(width=3, color=COL_P),
        marker=dict(size=7, color=COL_P)
    ))
    # 전년동월(밝은 보라, 점선)
    fig_trend.add_trace(go.Scatter(
        x=df["월"], y=df["전년동월"], mode="lines+markers", name="전년동월",
        line=dict(width=2, dash="dash", color=COL_S2),
        marker=dict(size=6, color=COL_S2)
    ))
    # 마커(최고/최저)
    fig_trend.add_trace(go.Scatter(
        x=[df.loc[max_idx, "월"]], y=[df.loc[max_idx, "매출액"]],
        mode="markers+text", name="최고",
        text=["최고"], textposition="top center",
        marker=dict(size=12, symbol="star", color=COL_S3, line=dict(width=1, color=COL_P))
    ))
    fig_trend.add_trace(go.Scatter(
        x=[df.loc[min_idx, "월"]], y=[df.loc[min_idx, "매출액"]],
        mode="markers+text", name="최저",
        text=["최저"], textposition="bottom center",
        marker=dict(size=11, symbol="triangle-down", color=COL_S1, line=dict(width=1, color=COL_P))
    ))
    fig_trend.update_layout(yaxis_title="매출액 (원)", xaxis_title="월")
    fig_trend = apply_theme(fig_trend)
    st.plotly_chart(fig_trend, use_container_width=True)

# 2) 전년 대비 증감률 (막대)
with st.container():
    st.subheader("2) 전년 대비 증감률")
    # 양수: 밝은 보라, 음수: 진한 보라 (동일 계열 내 명도 차이로 가독성을 확보)
    bar_colors = [COL_S2 if v >= 0 else COL_S1 for v in df["증감률"]]
    fig_yoy = go.Figure(go.Bar(
        x=df["월"], y=df["증감률"], marker_color=bar_colors, name="증감률",
        hovertemplate="월=%{x}<br>증감률=%{y:.1f}%<extra></extra>"
    ))
    # 0% 기준선
    fig_yoy.add_hline(y=0, line_dash="dash", line_color=COL_P, annotation_text="0% 기준", annotation_position="bottom left")
    fig_yoy.update_layout(yaxis_title="증감률 (%)", xaxis_title="월")
    fig_yoy = apply_theme(fig_yoy)
    st.plotly_chart(fig_yoy, use_container_width=True)

# 3) 분기별 매출 분포 (Boxplot)
with st.container():
    st.subheader("3) 분기별 매출 분포 (Boxplot)")
    fig_box = px.box(
        df, x="분기", y="매출액", points="all",
        color="분기",
        color_discrete_sequence=COLORWAY
    )
    fig_box.update_layout(yaxis_title="매출액 (원)", xaxis_title="분기", showlegend=False)
    fig_box = apply_theme(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)

# 4) 월별 KPI 달성률 (라인 + 목표선)
with st.container():
    st.subheader("4) 월별 KPI 달성률 (목표선 100%)")
    rate = (df["매출액"] / (target if target else 1)) * 100.0
    fig_kpi = go.Figure()
    fig_kpi.add_trace(go.Scatter(
        x=df["월"], y=rate, mode="lines+markers", name="달성률",
        line=dict(width=3, color=COL_P),
        marker=dict(size=7, color=COL_P)
    ))
    fig_kpi.add_hline(y=100, line_dash="dash", line_color=COL_S1,
                      annotation_text="목표 100%", annotation_position="top left")
    fig_kpi.update_layout(yaxis_title="달성률 (%)", xaxis_title="월")
    fig_kpi = apply_theme(fig_kpi)
    st.plotly_chart(fig_kpi, use_container_width=True)

st.divider()
st.subheader("데이터 미리보기")
st.dataframe(df.drop(columns=["_date"]))

st.caption("Tip: 좌측 사이드바에서 KPI 목표를 바꾸면 달성률 차트가 즉시 반영됩니다. 업로드 파일은 동일 스키마를 유지해주세요.")
