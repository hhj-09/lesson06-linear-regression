
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================
# 1. 기본 설정
# =====================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 분석하여 장기적인 기온 변화 추세를 살펴보고, "
    "선택한 연도의 회귀 추정기온을 확인합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =====================================
# 2. 데이터 불러오기
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: " + ", ".join(missing_columns)
        )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )

    # 평균기온을 숫자로 변환
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
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# =====================================
# 3. 연도별 평균기온 계산
# =====================================
yearly_df = (
    daily_df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년까지이며 관측일수가 300일 이상인 연도만 사용
model_df = yearly_df[
    (yearly_df["연도"] <= 2025)
    & (yearly_df["관측일수"] >= 300)
].copy()

model_df = (
    model_df.sort_values("연도")
    .reset_index(drop=True)
)

if len(model_df) < 2:
    st.error(
        "회귀 분석에 사용할 수 있는 연도가 부족합니다. "
        "데이터를 확인해 주세요."
    )
    st.stop()


# =====================================
# 4. 전체 기간 회귀 분석
# =====================================
years = model_df["연도"].to_numpy(dtype=float)
temperatures = model_df["연평균기온"].to_numpy(dtype=float)

# 회귀 직선: y = slope * x + intercept
slope, intercept = np.polyfit(years, temperatures, 1)

# 각 연도의 회귀 추정기온
model_df["회귀기온"] = slope * years + intercept

# 전체 기간 상관계수
correlation = np.corrcoef(years, temperatures)[0, 1]

# 100년당 기온 변화량
rise_100_years = slope * 100


# =====================================
# 5. 최근 20개 유효 연도 회귀 분석
# =====================================
recent_df = model_df.tail(20).copy()

recent_slope = np.nan
recent_intercept = np.nan
recent_rise_100_years = np.nan
recent_correlation = np.nan

if len(recent_df) >= 2:
    recent_years = recent_df["연도"].to_numpy(dtype=float)
    recent_temperatures = (
        recent_df["연평균기온"].to_numpy(dtype=float)
    )

    recent_slope, recent_intercept = np.polyfit(
        recent_years,
        recent_temperatures,
        1
    )

    recent_rise_100_years = recent_slope * 100

    recent_correlation = np.corrcoef(
        recent_years,
        recent_temperatures
    )[0, 1]


# =====================================
# 6. 분석에 사용한 데이터 정보
# =====================================
st.subheader("📊 분석 데이터")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "사용한 연도 수",
    f"{len(model_df)}년"
)

col2.metric(
    "시작 연도",
    f"{int(model_df['연도'].min())}년"
)

col3.metric(
    "끝 연도",
    f"{int(model_df['연도'].max())}년"
)

col4.metric(
    "전체 기간 상관계수",
    f"{correlation:.3f}"
)

st.caption(
    "2025년까지의 자료 중 유효한 평균기온 관측값이 "
    "300일 이상인 연도만 분석했습니다."
)


# =====================================
# 7. 100년당 기온 변화량 비교
# =====================================
st.subheader("🌡️ 100년 동안 기온은 얼마나 변하는가?")

st.write(
    "회귀 직선의 기울기에 100을 곱해 "
    "100년당 기온 변화량으로 환산했습니다."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 전체 기간")

    st.metric(
        label=(
            f"{int(model_df['연도'].min())}~"
            f"{int(model_df['연도'].max())}년"
        ),
        value=f"{rise_100_years:+.2f} °C",
        help="전체 분석 기간의 회귀 기울기를 100년 기준으로 환산"
    )

with col2:
    st.markdown("### 최근 20개 유효 연도")

    if len(recent_df) >= 2:
        st.metric(
            label=(
                f"{int(recent_df['연도'].min())}~"
                f"{int(recent_df['연도'].max())}년"
            ),
            value=f"{recent_rise_100_years:+.2f} °C",
            help="최근 20개 유효 연도의 회귀 기울기를 100년 기준으로 환산"
        )
    else:
        st.warning("최근 기간의 데이터가 부족합니다.")

if len(recent_df) >= 2:
    difference = recent_rise_100_years - rise_100_years

    if difference > 0:
        comparison_text = (
            f"최근 기간의 100년 환산 변화량이 전체 기간보다 "
            f"{difference:.2f} °C 더 큽니다."
        )
    elif difference < 0:
        comparison_text = (
            f"최근 기간의 100년 환산 변화량이 전체 기간보다 "
            f"{abs(difference):.2f} °C 더 작습니다."
        )
    else:
        comparison_text = (
            "두 기간의 100년 환산 변화량이 같습니다."
        )

    st.info(comparison_text)

st.caption(
    "최근 20개 유효 연도란 관측일수 기준을 통과한 자료 중 "
    "연도 순으로 마지막 20개를 의미합니다. "
    "결측 연도가 있으면 정확히 연속된 20년은 아닐 수 있습니다. "
    "100년당 변화량은 회귀 기울기를 환산한 값이며, "
    "실제로 100년 동안 관측된 변화량을 뜻하지 않습니다."
)


# =====================================
# 8. 연도 슬라이더와 예상 기온
# =====================================
st.subheader("🔮 선택한 연도의 예상 평균기온")

selected_year = st.slider(
    "연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 전체 기간 회귀식을 이용한 추정값
predicted_temperature = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="
        padding: 26px;
        border-radius: 16px;
        background-color: #eef6ff;
        text-align: center;
        margin: 10px 0 20px 0;
    ">
        <div style="
            font-size: 20px;
            color: #334155;
        ">
            {selected_year}년 회귀 추정 연평균기온
        </div>
        <div style="
            font-size: 48px;
            font-weight: bold;
            color: #1d4ed8;
            margin-top: 8px;
        ">
            {predicted_temperature:.2f} °C
        </div>
        <div style="
            font-size: 14px;
            color: #475569;
        ">
            전체 분석 기간의 선형 회귀식으로 계산
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================
# 9. 산점도와 전체 기간 회귀 직선
# =====================================
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값
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

# 회귀 직선은 실제 분석 기간에 대해서만 그림
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
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 추정기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도의 추정값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 추정값",
        marker=dict(
            size=14,
            symbol="diamond"
        ),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "추정기온: %{y:.2f} °C"
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

fig.update_xaxes(
    dtick=10,
    tickformat="d"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =====================================
# 10. 회귀 분석 해석
# =====================================
st.subheader("🧮 회귀 분석 결과")

st.write(
    f"**전체 기간 회귀식:** "
    f"연평균기온 = {slope:.5f} × 연도 + ({intercept:.3f})"
)

st.write(
    f"**전체 기간 상관계수:** {correlation:.4f}"
)

st.write(
    f"**전체 기간 100년당 기온 변화량:** "
    f"{rise_100_years:+.2f} °C"
)

if len(recent_df) >= 2:
    st.write(
        f"**최근 기간 회귀식:** "
        f"연평균기온 = {recent_slope:.5f} × 연도 "
        f"+ ({recent_intercept:.3f})"
    )

    st.write(
        f"**최근 기간 상관계수:** {recent_correlation:.4f}"
    )

    st.write(
        f"**최근 기간 100년당 기온 변화량:** "
        f"{recent_rise_100_years:+.2f} °C"
    )

if correlation > 0:
    st.write(
        "전체 기간에는 연도가 뒤로 갈수록 연평균기온이 "
        "높아지는 선형 경향이 나타납니다."
    )
elif correlation < 0:
    st.write(
        "전체 기간에는 연도가 뒤로 갈수록 연평균기온이 "
        "낮아지는 선형 경향이 나타납니다."
    )
else:
    st.write(
        "전체 기간에는 연도와 연평균기온 사이의 "
        "선형 상관관계가 거의 나타나지 않습니다."
    )

st.warning(
    "이 앱은 과거 관측값에 선형 회귀를 적용한 교육용 분석입니다. "
    "슬라이더의 미래 연도 값은 회귀선을 연장해 계산한 추정치이며, "
    "실제 미래 기온을 정확하게 예측하는 기후 모델은 아닙니다."
)


# =====================================
# 11. 분석에 사용한 연도별 자료
# =====================================
with st.expander("📋 회귀 분석에 사용한 연도별 데이터 보기"):
    display_df = model_df[
        ["연도", "연평균기온", "관측일수", "회귀기온"]
    ].rename(
        columns={
            "연평균기온": "실제 연평균기온(°C)",
            "관측일수": "관측일수(일)",
            "회귀기온": "회귀 추정기온(°C)"
        }
    )

    st.dataframe(
        display_df.round(3),
        use_container_width=True,
        hide_index=True
    )
