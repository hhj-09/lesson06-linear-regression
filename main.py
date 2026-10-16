
# -----------------------------
# 선형 회귀 계산
# -----------------------------
years = model_df["연도"].to_numpy(dtype=float)
temperatures = model_df["연평균기온"].to_numpy(dtype=float)

# 전체 기간 회귀 직선
slope, intercept = np.polyfit(years, temperatures, 1)

model_df["회귀기온"] = slope * years + intercept

# 전체 기간 상관계수
correlation = np.corrcoef(years, temperatures)[0, 1]

# 100년 동안의 기온 변화량
rise_100_years = slope * 100

# 최근 20년: 분석에 사용되는 마지막 연도부터 거꾸로 20개 연도
recent_df = model_df.tail(20).copy()

if len(recent_df) >= 2:
    recent_years = recent_df["연도"].to_numpy(dtype=float)
    recent_temperatures = recent_df["연평균기온"].to_numpy(dtype=float)

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
else:
    recent_slope = np.nan
    recent_intercept = np.nan
    recent_rise_100_years = np.nan
    recent_correlation = np.nan
