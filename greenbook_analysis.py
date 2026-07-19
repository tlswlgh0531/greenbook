"""
그린북 시대(1947) vs 현재(2024): 미국 흑인 차별의 지리적 분포 변화 분석
=====================================================================
영화 '그린북' 감상 후 심화활동 — 데이터 분석 프로젝트

데이터 출처:
  1) NYPL(뉴욕공립도서관) 그린북 디지털 아카이브
     https://github.com/NYPL-publicdomain/greenbooks (퍼블릭 도메인)
     - 1947년판 geojson: 흑인 여행자가 이용 가능했던 업소 목록
  2) FBI Crime Data Explorer 증오범죄 통계 (1991~2024)
     https://cde.ucr.cjis.gov (미국 연방정부 퍼블릭 도메인)
     - hate_crime.csv: 사건 단위 증오범죄 기록

실행 방법:
  pip install pandas plotly
  python greenbook_analysis.py

필요 파일 (같은 폴더에 위치):
  - 1947.geojson        (그린북 데이터)
  - hate_crime.csv      (FBI 데이터)
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------------
GREENBOOK_PATH = "1947.geojson"
FBI_PATH = "hate_crime.csv"
OUTPUT_MAP = "greenbook_vs_2024_map.html"
TARGET_YEAR = 2024          # FBI 데이터에서 사용할 연도
TARGET_BIAS = "Anti-Black"  # 그린북과의 비교 논리상 흑인 대상 편견만 필터

# 미국 주 이름 -> 2글자 약자 매핑 (지도를 그릴 때 필요)
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District Of Columbia": "DC",
}


# ---------------------------------------------------------------
# 1. 그린북 데이터 로드 및 전처리 (1947년)
# ---------------------------------------------------------------
def load_greenbook(path: str) -> pd.DataFrame:
    """1947년 그린북 geojson을 읽어 주별 업소 수로 집계한다.

    전처리 내용:
      - state 필드가 비어있는 항목 제거
      - 표기 통일: 대문자(CALIFORNIA) -> Title Case(California)
      - 주 이름 매핑에 실패하는 항목(OCR 오류) 제거
    """
    with open(path, encoding="utf-8") as f:
        geo = json.load(f)

    states = []
    for feature in geo["features"]:
        state = feature["properties"].get("state", "").strip().title()
        if state:
            states.append(state)

    df = pd.Series(states, name="state").value_counts().reset_index()
    df.columns = ["state_name", "greenbook_1947"]

    # 주 약자 매핑 — 매핑 실패 = OCR 노이즈로 판단하고 제거
    df["state_abbr"] = df["state_name"].map(STATE_ABBR)
    noise = df[df["state_abbr"].isna()]
    if len(noise) > 0:
        print(f"[전처리] OCR 노이즈로 제거된 항목 {len(noise)}개:")
        print(noise[["state_name", "greenbook_1947"]].to_string(index=False))
    df = df.dropna(subset=["state_abbr"]).reset_index(drop=True)

    print(f"[그린북 1947] 총 {df['greenbook_1947'].sum()}개 업소, "
          f"{len(df)}개 주 커버")
    return df


# ---------------------------------------------------------------
# 2. FBI 증오범죄 데이터 로드 및 전처리 (2024년, Anti-Black)
# ---------------------------------------------------------------
def load_fbi(path: str, year: int, bias_keyword: str) -> pd.DataFrame:
    """FBI hate_crime.csv에서 특정 연도 + 특정 편견 유형만 필터해
    주별 사건 수로 집계한다.

    참고: 한 사건에 여러 편견이 섞인 경우(multiple bias)도
    bias_desc 문자열에 키워드가 포함되면 카운트한다.
    """
    df = pd.read_csv(path, low_memory=False)

    filtered = df[
        (df["data_year"] == year)
        & (df["bias_desc"].str.contains(bias_keyword, na=False))
    ]
    print(f"[FBI {year}] '{bias_keyword}' 관련 사건 수: {len(filtered)}건")

    out = (
        filtered.groupby(["state_abbr", "state_name"])
        .size()
        .reset_index(name=f"antiblack_{year}")
    )
    print(f"[FBI {year}] {out['state_abbr'].nunique()}개 주 커버")
    return out


# ---------------------------------------------------------------
# 3. 시각화: 나란히 놓인 두 개의 단계구분도
# ---------------------------------------------------------------
def make_maps(gb: pd.DataFrame, hc: pd.DataFrame, output_path: str):
    """왼쪽: 1947 그린북 업소 수 / 오른쪽: 2024 증오범죄 건수"""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
        subplot_titles=(
            "1947년: 그린북 등재 업소 수<br>"
            "<sup>흑인이 안전하게 이용 가능했던 업소 (많을수록 진함)</sup>",
            "2024년: 흑인 대상 증오범죄 건수<br>"
            "<sup>FBI 보고 기준 (많을수록 진함)</sup>",
        ),
    )

    # 왼쪽 지도: 그린북 (초록 계열)
    fig.add_trace(
        go.Choropleth(
            locations=gb["state_abbr"],
            z=gb["greenbook_1947"],
            locationmode="USA-states",
            colorscale="Greens",
            colorbar=dict(title="업소 수", x=0.45, len=0.7),
            hovertext=gb["state_name"],
            hovertemplate="%{hovertext}<br>업소 수: %{z}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 오른쪽 지도: 증오범죄 (빨강 계열)
    fig.add_trace(
        go.Choropleth(
            locations=hc["state_abbr"],
            z=hc[hc.columns[-1]],
            locationmode="USA-states",
            colorscale="Reds",
            colorbar=dict(title="사건 수", x=1.0, len=0.7),
            hovertext=hc["state_name"],
            hovertemplate="%{hovertext}<br>증오범죄: %{z}건<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_geos(scope="usa")
    fig.update_layout(
        title_text="그린북 시대(1947)와 현재(2024): "
                   "미국 흑인 차별의 지리적 분포 변화",
        title_x=0.5,
        height=560, width=1200,
        font=dict(family="sans-serif"),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"[시각화] 지도 저장 완료 -> {output_path}")


# ---------------------------------------------------------------
# 4. 요약 통계 출력 (보고서 작성용)
# ---------------------------------------------------------------
def print_summary(gb: pd.DataFrame, hc: pd.DataFrame):
    col = hc.columns[-1]
    print("\n===== 보고서용 핵심 수치 =====")
    print("\n[1947 그린북] 업소 수 상위 5개 주:")
    print(gb.nlargest(5, "greenbook_1947")
          [["state_name", "greenbook_1947"]].to_string(index=False))
    print("\n[2024 FBI] 흑인 대상 증오범죄 상위 5개 주:")
    print(hc.nlargest(5, col)[["state_name", col]].to_string(index=False))


# ---------------------------------------------------------------
# 메인 실행부
# ---------------------------------------------------------------
if __name__ == "__main__":
    greenbook = load_greenbook(GREENBOOK_PATH)
    fbi = load_fbi(FBI_PATH, TARGET_YEAR, TARGET_BIAS)

    # 중간 결과를 CSV로도 저장 (보고서 표 만들 때 활용)
    greenbook.to_csv("greenbook_1947_by_state.csv", index=False)
    fbi.to_csv(f"antiblack_{TARGET_YEAR}_by_state.csv", index=False)

    make_maps(greenbook, fbi, OUTPUT_MAP)
    print_summary(greenbook, fbi)
