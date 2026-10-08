# -*- coding: utf-8 -*-
"""입력 — 형식이 다른 CSV를 표 하나로 바꾼다.

관광 데이터는 형식이 제각각이다. 격자 데이터는 시간이 가로로 펼쳐져 있고,
API로 받은 데이터는 한 줄에 한 관측이 들어 있다. 그대로 두면 알고리즘마다
데이터 형식을 알아야 하므로, 읽는 단계에서 **하나의 표**로 맞춘다.

    행 = 분석 단위(격자·읍면동·지역),  열 = 시점,  칸 = 값

    group         2014-01|00  2014-01|01  ...
    1000041354          12.3        10.1  ...
    1000041355           4.0         3.2  ...

이 표만 만들어 두면 알고리즘은 '숫자 한 줄'만 받으면 된다.
"""
import glob
import re

import pandas as pd

HOUR_PATTERN = re.compile(r"^(\d{1,2})시")          # "00시-01시 유동인구"


def read_csv(path):
    """한글 CSV 읽기. 인코딩이 파일마다 달라서 차례로 시도한다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("읽을 수 없는 인코딩입니다: %s" % path)


def load_wide(pattern):
    """[형식 1] 가로형 — 시간대가 열로 펼쳐진 격자 데이터.

    원본 한 행 = 격자 하나의 한 달치 24시간 프로파일

        기준년월  격자 ID      00시-01시 유동인구  01시-02시 유동인구  ...
        2014-01  1000041354              12.3              10.1  ...

    24개 시간대 열을 세로로 편 뒤, 파일(달)이 달라도 이어지도록
    '2014-01|00' 형태의 시점 키를 만든다. 그래야 12개월치를 한 계열로 본다.
    """
    frames = []
    for path in sorted(glob.glob(pattern)):
        df = read_csv(path)

        grid_col = [c for c in df.columns if "격자" in c][0]       # 파일마다 이름이 다름
        hour_cols = [c for c in df.columns
                     if HOUR_PATTERN.match(c) and "비율" not in c]  # 비율 열은 제외
        year_month = str(df["기준년월"].iloc[0]).replace("--", "-")[:7]

        melted = df.melt(id_vars=grid_col, value_vars=hour_cols,
                         var_name="hour", value_name="value")
        melted["time"] = year_month + "|" + melted["hour"].str[:2]
        frames.append(melted[[grid_col, "time", "value"]]
                      .rename(columns={grid_col: "group"}))

    table = pd.concat(frames)
    # 격자 ID가 숫자로 읽혀 '1000041354.0'이 되는 경우를 되돌린다
    table["group"] = table["group"].astype(str).str.replace(r"\.0$", "", regex=True)
    return table.pivot_table(index="group", columns="time",
                             values="value").sort_index(axis=1)


def load_long(pattern, group_cols, time_col, value_col):
    """[형식 2] 세로형 — 한 줄에 한 관측이 들어 있는 데이터.

        baseYmd   signguNm  touNum
        20200101  제주시     123456

    group_cols 를 여러 개 주면 '제주시|외지인'처럼 이어 붙여 한 그룹으로 삼는다.
    """
    df = pd.concat([read_csv(p) for p in sorted(glob.glob(pattern))])
    df["group"] = df[list(group_cols)].astype(str).agg("|".join, axis=1)
    df["time"] = df[time_col].astype(str)
    return df.pivot_table(index="group", columns="time",
                          values=value_col).sort_index(axis=1)


def load_single(path, time_col, value_cols):
    """[형식 3] 단일 계열 — 그룹 구분이 없고 지표만 여러 개인 데이터.

        기준일자   내국인방문객수  외국인방문객수  전체방문객수
        20200101         12345          678        13023

    지표 하나가 그룹 하나가 된다(내국인 / 외국인 / 전체).
    """
    df = read_csv(path)
    df["time"] = df[time_col].astype(str)
    table = df.set_index("time")[list(value_cols)].T
    table.index.name = "group"
    return table.sort_index(axis=1)


#: 자주 쓰는 데이터는 이름만으로 읽을 수 있게 미리 적어 둔다.
#:   period = 자연 주기. 시간대 데이터는 하루 24, 일별 데이터는 일주일 7.
PRESETS = {
    "grid300m": dict(
        kind="wide", period=24, group_label="격자", time_label="년월|시간대",
        pattern="제주특별자치도_시간대별_관광객_300그리드단위_유동인구현황"
                "/*/*유동인구_비율포함_20??????.csv"),
    "grid250m": dict(
        kind="wide", period=24, group_label="격자", time_label="년월|시간대",
        pattern="제주특별자치도_시간대별_관광객_250그리드단위_유동인구현황"
                "/*/*유동인구_비율포함_20??????.csv"),
    "tourapi": dict(
        kind="long", period=7, group_label="시군구", time_label="날짜",
        pattern="tourapi_data/total/locgo/*.csv",
        group_cols=["signguNm"], time_col="baseYmd", value_col="touNum"),
    "arrivals": dict(
        kind="single", period=7, group_label="구분", time_label="날짜",
        pattern="제주관광빅데이터_입도 관광객 추이_일별_201301_202608.csv",
        time_col="기준일자(YYYYMMDD)",
        value_cols=["내국인방문객수", "외국인방문객수", "전체방문객수"]),
}


def load(preset, pattern=None):
    """프리셋 이름으로 읽는다.

    반환: (table, info)
        table  그룹 × 시점 DataFrame
        info   {"period": 주기, "group_label": ..., "time_label": ...}
    """
    if preset not in PRESETS:
        raise KeyError("모르는 데이터입니다: %s (가능: %s)"
                       % (preset, ", ".join(PRESETS)))

    spec = dict(PRESETS[preset])
    kind = spec.pop("kind")
    path = pattern or spec.pop("pattern")
    spec.pop("pattern", None)

    if kind == "wide":
        table = load_wide(path)
    elif kind == "long":
        table = load_long(path, spec.pop("group_cols"),
                          spec.pop("time_col"), spec.pop("value_col"))
    else:
        table = load_single(path, spec.pop("time_col"), spec.pop("value_cols"))

    return table, spec
