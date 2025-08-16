import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="월별 매출 대시보드", layout="wide")

PALETTE = {
    "primary": "#5E0081",
    "secondary": "#61007A",
    "deep": "#610D7C",
    "accent": "#B386C3",
    "surface": "#F0E0F4",
    "on_dark": "#FFFFFF",
    "on_light": "#000000",
}

st.markdown(
    f"""
    <style>
    :root {{
      --primary: {PALETTE["primary"]};
      --secondary: {PALETTE["secondary"]};
      --deep: {PALETTE["deep"]};
      --accent: {PALETTE["accent"]};
      --surface: {PALETTE["surface"]};
    }}
    .stApp {{ background: var(--surface); }}
    .block-container h1, .block-container h2, .block-container h3 {{
      color: var(--primary);
    }}
    .block-container h2 {{
      border-left: 8px solid var(--accent);
      padding-left: .5rem;
    }}
    [data-testid="stSidebar"] > div:first-child {{
      background: #fff;
      border-right: 3px solid var(--accent);
    }}
    [data-testid="stMetric"] {{
      background: #fff;
      border: 1px solid var(--accent);
      border-radius: 12px;
      padding: .75rem;
    }}
    [data-testid="stMetric"] div[data-testid="stMetricLabel"] p {{
      color: var(--secondary) !important;
      font-weight: 600;
    }}
    [data-testid="stMetricValue"] > div {{
      color: var(--primary) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def apply_layout(fig):
    fig.update_layout(
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor="#FFFFFF",
        font=dict(color=PALETTE["deep"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=10, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["surface"], zeroline=False, linecolor=PALETTE["accent"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["surface"], zeroline=False, linecolor=PALETTE["accent"])
    return fig

st.title("📊 월별 매출 대시보드 (Streamlit)")
st.caption("CSV 업로드 후 4가지 시각화가 자동 생성됩니다. 컬럼: 월(YYYY-MM), 매출액, 전년동월, 증감률(%). 미입력 시 증감률은 전년동월로 자동 계산합니다.")

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
    return pd.read_csv(file)

@st.cache_data
def parse_sample(sample_text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(sample_text))

@st.cache_data
def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["월"] = df["월"].astype(str).str.strip()
    df["_date"] = pd.to_datetime(df["월"], format="%Y-%m", errors="coerce")
    df = df.sort_values("_date").reset_index(drop=True)
    for c in ["매출액", "전년동월"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "증감률" in df.columns:
        df["증감률"] = pd.to_numeric(df.get("증감률"), errors="coerce")
    else:
        df["증감률"] = np.nan
    missing_mask = df["증감률"].isna()
    valid_mask = missing_mask & df["전년동월"].ne(0)
    df.loc[valid_mask, "증감률"] = (
        (df.loc[valid_mask, "매출액"] - df.loc[valid_mask, "전년동월"]) / df.loc[valid_mask, "전년동월"] * 100
    )
    df["증감률"] = df["증감률"].fillna(0)
    df["분기"] = df["_date"].dt.quarter
    return df

with st.sidebar:
    st.header("⚙️ 설정")
    uploaded = st.file_uploader("CSV 업로드", type=["csv"], accept_multiple_files=False)
    use_sample = st.checkbox("샘플 데이터 불러오기", value=True if uploaded is None else False)
    target = st.number_input("KPI 목표 매출 (원)", min_value=0, value=20_000_000, step=100_000)

if uploaded is not None:
    df_raw = read_csv(uploaded)
elif use_sample:
    df_raw = parse_sample(SAMPLE_CSV)
else:
    st.info("좌측에서 CSV를 업로드하거나 '샘플 데이터 불러오기'를 선택하세요.")
    st.stop()

try:
    df = enrich_df(df_raw)
except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_sales = int(np.nansum(df["매출액"]))
    st.metric("총합 매출", f"{total_sales:,.0f}원")
with col2:
    avg_yoy = float(df["증감률"].mean())
    st.metric("평균 증감률", f"{avg_yoy:.1f}%")
vals = df["매출액"]
max_idx = vals.idxmax() if vals.notna().any() else None
min_idx = vals.idxmin() if vals.notna().any() else None
with col3:
    if max_idx is not None:
        st.metric("최고 매출 (월)", f"{df.loc[max_idx,'월']} · {df.loc[max_idx,'매출액']:,.0f}원")
    else:
        st.metric("최고 매출 (월)", "-")
with col4:
    if min_idx is not None:
        st.metric("최저 매출 (월)", f"{df.loc[min_idx,'월']} · {df.loc[min_idx,'매출액']:,.0f}원")
    else:
        st.metric("최저 매출 (월)", "-")

st.divider()

with st.container():
    st.subheader("1) 월별 매출 추이 (매출액 vs 전년동월)")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df["월"], y=df["매출액"], mode="lines+markers", name="매출액",
        line=dict(color=PALETTE["primary"], width=3),
        marker=dict(color=PALETTE["primary"])
    ))
    fig_trend.add_trace(go.Scatter(
        x=df["월"], y=df["전년동월"], mode="lines+markers", name="전년동월",
        line=dict(color=PALETTE["accent"], dash="dash", width=3),
        marker=dict(color=PALETTE["accent"])
    ))
    if max_idx is not None:
        fig_trend.add_trace(go.Scatter(
            x=[df.loc[max_idx, "월"]], y=[df.loc[max_idx, "매출액"]], mode="markers+text",
            name="최고", text=["최고"], textposition="top center",
            marker=dict(color=PALETTE["secondary"], size=12, line=dict(width=2, color=PALETTE["accent"]))
        ))
    if min_idx is not None:
        fig_trend.add_trace(go.Scatter(
            x=[df.loc[min_idx, "월"]], y=[df.loc[min_idx, "매출액"]], mode="markers+text",
            name="최저", text=["최저"], textposition="bottom center",
            marker=dict(color=PALETTE["deep"], size=12, line=dict(width=2, color="#FFFFFF"))
        ))
    fig_trend.update_layout(yaxis_title="매출액 (원)", xaxis_title="월")
    st.plotly_chart(apply_layout(fig_trend), use_container_width=True)

with st.container():
    st.subheader("2) 전년 대비 증감률")
    colors = [PALETTE["primary"] if v >= 0 else PALETTE["deep"] for v in df["증감률"]]
    fig_yoy = go.Figure(go.Bar(x=df["월"], y=df["증감률"], marker_color=colors, name="증감률"))
    fig_yoy.add_hline(y=0, line_dash="dot", line_color=PALETTE["accent"])
    fig_yoy.update_layout(yaxis_title="증감률 (%)", xaxis_title="월")
    st.plotly_chart(apply_layout(fig_yoy), use_container_width=True)

with st.container():
    st.subheader("3) 분기별 매출 분포 (Boxplot)")
    fig_box = px.box(df, x="분기", y="매출액", points="all", color_discrete_sequence=[PALETTE["accent"]])
    fig_box.update_traces(marker=dict(color=PALETTE["secondary"], size=6, opacity=0.5))
    fig_box.update_layout(yaxis_title="매출액 (원)", xaxis_title="분기")
    st.plotly_chart(apply_layout(fig_box), use_container_width=True)

with st.container():
    st.subheader("4) 월별 KPI 달성률 (목표선 100%)")
    rate = (df["매출액"] / (target if target else 1)) * 100.0
    fig_kpi = go.Figure()
    fig_kpi.add_trace(go.Scatter(
        x=df["월"], y=rate, mode="lines+markers", name="달성률",
        line=dict(color=PALETTE["primary"], width=3),
        marker=dict(color=PALETTE["primary"])
    ))
    fig_kpi.add_hline(y=100, line_dash="dash", line_color=PALETTE["accent"],
                      annotation_text="목표 100%", annotation_position="top left")
    fig_kpi.update_layout(yaxis_title="달성률 (%)", xaxis_title="월")
    st.plotly_chart(apply_layout(fig_kpi), use_container_width=True)

st.divider()
st.subheader("데이터 미리보기")
st.dataframe(df.drop(columns=["_date"]))

st.caption("Tip: 좌측 사이드바에서 KPI 목표를 바꾸면 달성률 차트가 즉시 반영됩니다. 업로드 파일은 동일 스키마를 유지해주세요.")