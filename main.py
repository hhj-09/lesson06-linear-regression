
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 바탕으로 기온의 장기적인 추세를 살펴보고, "
    "선택한 연도의 예상 평균기온을 확인합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 필요한 열이 있는지 확인
    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV 파일에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )

    # 기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 열 추가
    df["연도"] = df["날짜"].dt.year

    return df


try:
    daily_df = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오는 중 문제가 발생했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# -----------------------------
# 연도별 평균기온과 관측일 계산
# -----------------------------
yearly_df = (
    daily_df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 기준 기간: 2025년까지
# 관측일이 300일 미만인 연도 제외
model_df = yearly_df[
    (yearly_df["연도"] <= 2025)
    & (yearly_df["관측일수"] >= 300)
].copy()

model_df = model_df.sort_values("연도").reset_index(drop=True)

if len(model_df) < 2:
    st.error(
        "회귀 직선을 계산할 수 있는 연도가 부족합니다. "
        "데이터와 관측일 기준을 확인해 주세요."
    )
    st.stop()


# -----------------------------
# 선형 회귀 계산
# -----------------------------
years = model_df["연도"].to_numpy(dtype=float)
temperatures = model_df["연평균기온"].to_numpy(dtype=float)

# 기울기와 절편을 구함
slope, intercept = np.polyfit(years, temperatures, 1)

# 회귀식으로 계산한 각 연도의 기온
model_df["회귀기온"] = slope * years + intercept

# 상관계수 계산
correlation = np.corrcoef(years, temperatures)[0, 1]


# -----------------------------
# 회귀에 사용한 데이터 정보
# -----------------------------
st.subheader("📊 분석에 사용한 데이터")

col1, col2, col3, col4 = st.columns(4)

col1.metric("사용한 연도 수", f"{len(model_df)}년")
col2.metric("시작 연도", f"{int(model_df['연도'].min())}년")
col3.metric("끝 연도", f"{int(model_df['연도'].max())}년")
col4.metric("상관계수", f"{correlation:.3f}")

st.caption(
    "2025년까지의 자료 중 평균기온 관측값이 300일 이상 있는 연도만 사용했습니다. "
    "관측일수는 해당 연도에 유효한 평균기온 자료가 있는 행의 수입니다."
)


# -----------------------------
# 연도 선택 슬라이더
# -----------------------------
st.subheader("🔮 예상 기온 확인")

selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 선택한 연도의 예상 기온
predicted_temperature = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="
        padding: 24px;
        border-radius: 16px;
        background-color: #eef6ff;
        text-align: center;
        margin: 10px 0 20px 0;
    ">
        <div style="font-size: 20px; color: #334155;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 48px;
            font-weight: bold;
            color: #1d4ed8;
            margin-top: 8px;
        ">
            {predicted_temperature:.2f} °C
        </div>
        <div style="font-size: 14px; color: #475569;">
            과거 연평균기온의 선형 회귀식으로 계산한 값
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# 산점도와 회귀 직선
# -----------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=model_df["연도"],
        y=model_df["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=model_df[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 관측 자료의 시작 연도부터 끝 연도까지 회귀 직선 표시
line_years = np.linspace(
    model_df["연도"].min(),
    model_df["연도"].max(),
    300
)
line_temperatures = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도의 회귀 예상값 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=13, symbol="diamond"),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 회귀식과 상관관계 설명
# -----------------------------
st.subheader("🧮 회귀 분석 결과")

st.write(
    f"**회귀식:** 연평균기온 = "
    f"({slope:.5f} × 연도) + ({intercept:.3f})"
)

st.write(f"**상관계수:** {correlation:.4f}")

st.write(
    f"**연도당 회귀 추세 변화량:** {slope:.4f} °C"
)

if correlation > 0:
    st.write(
        "상관계수가 양수이므로, 분석 기간에는 연도가 뒤로 갈수록 "
        "연평균기온이 높아지는 경향이 나타납니다."
    )
elif correlation < 0:
    st.write(
        "상관계수가 음수이므로, 분석 기간에는 연도가 뒤로 갈수록 "
        "연평균기온이 낮아지는 경향이 나타납니다."
    )
else:
    st.write(
        "상관계수가 0에 가까워 연도와 연평균기온 사이의 "
        "선형 관계가 약하게 나타납니다."
    )

st.info(
    "주의: 상관관계가 있다고 해서 연도가 기온 변화를 직접 일으킨다는 뜻은 아닙니다. "
    "또한 1900~2100년의 예상값은 선택한 과거 자료의 선형 추세를 연장한 결과입니다. "
    "미래의 실제 기온은 온실가스 배출, 자연 변동 등 여러 요인에 따라 달라질 수 있습니다."
)


# -----------------------------
# 사용한 연도별 데이터 확인
# -----------------------------
with st.expander("회귀 분석에 사용한 연도별 자료 보기"):
    st.dataframe(
        model_df[
            ["연도", "연평균기온", "관측일수", "회귀기온"]
        ].rename(
            columns={
                "연도": "연도",
                "연평균기온": "실제 연평균기온(°C)",
                "관측일수": "관측일수(일)",
                "회귀기온": "회귀 추정기온(°C)"
            }
        ).round(3),
        use_container_width=True
    )
