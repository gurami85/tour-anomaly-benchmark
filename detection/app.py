# -*- coding: utf-8 -*-
"""노트북이 만든 이상탐지 결과 CSV를 들여다보는 스트림릿 앱.

    streamlit run detection/app.py

이 앱은 계산을 하지 않는다. detection/notebooks/anomaly_detection.ipynb 가
detection/notebooks/out/ 에 써 놓은 CSV만 읽는다. 그래서 노트북을 돌린 뒤에 보면 된다.
"""
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parent / "notebooks" / "out"

st.set_page_config(page_title="이상탐지 결과 보기", layout="wide")
st.title("이상탐지 결과 보기")


@st.cache_data(show_spinner=False)
def load(path, mtime):
    # mtime 을 받는 이유: 파일이 다시 쓰이면 캐시가 갱신되도록
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c) for c in frame.columns]
    unit = frame.columns[0]                       # 격자 또는 구분
    frame[unit] = frame[unit].astype(str)
    frame["시점"] = frame["시점"].astype(str)
    return frame, unit


def describe(path):
    # 파일 이름에서 알고리즘과 데이터를 읽는다
    stem = path.stem
    if stem.startswith("example_"):
        return stem[len("example_"):], "예제"
    match = re.match(r"^(.+?)_(iqr_log|iqr|zscore|mzscore|stl)(?:_(.+))?$", stem)
    if match:
        return match.group(2), "%s %s" % (match.group(1), match.group(3) or "")
    return stem, ""


files = sorted(OUT.glob("*.csv"))
if not files:
    st.error("읽을 CSV가 없습니다. 먼저 노트북을 돌려 `detection/notebooks/out/` 에 "
             "결과를 만들어 주십시오.")
    st.stop()

# ---------------------------------------------------------------- 전체 요약
st.subheader("1. 파일 한눈에")
st.caption("요약 파일을 만들지 않기로 했으므로, 이 표가 그 역할을 한다.")

rows = []
for path in files:
    frame, unit = load(str(path), path.stat().st_mtime)
    anomalies = int(frame["이상여부"].sum())
    rows.append({
        "파일": path.name,
        "알고리즘": describe(path)[0],
        "데이터": describe(path)[1],
        "단위": unit,
        "단위 수": frame[unit].nunique(),
        "행": len(frame),
        "이상": anomalies,
        "이상 비율(%)": round(100 * anomalies / len(frame), 2),
        "정상 행 포함": "아니오" if anomalies == len(frame) else "예",
        "크기(MB)": round(path.stat().st_size / 1024 / 1024, 2),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 계열 보기
st.divider()
st.subheader("2. 계열 하나를 펼쳐 보기")

picked = st.selectbox("파일", [p.name for p in files],
                      index=[p.name for p in files].index(files[0].name))
path = OUT / picked
frame, unit = load(str(path), path.stat().st_mtime)
only_anomalies = bool(frame["이상여부"].all())

if only_anomalies:
    st.warning(
        f"이 파일은 **이상 행만** 담고 있습니다 ({len(frame):,}행). "
        "정상값이 없어 계열을 선으로 그릴 수 없습니다 — 이상 지점만 점으로 표시합니다."
    )

with_hits = frame.loc[frame["이상여부"], unit].unique()
left, right = st.columns([3, 1])
with right:
    scope = st.radio(f"{unit} 목록", ["이상이 있는 것만", "전부"], index=0)
    options = sorted(with_hits) if scope == "이상이 있는 것만" \
        else sorted(frame[unit].unique())
    st.caption(f"{len(options):,}개")
    group = st.selectbox(unit, options) if options else None

if group is None:
    st.info("이상으로 잡힌 것이 없습니다.")
    st.stop()

part = frame[frame[unit] == group].sort_values("시점")
with left:
    base = alt.Chart(part).encode(
        x=alt.X("시점:N", sort=None, axis=alt.Axis(labelAngle=-45, title=None)),
        y=alt.Y("값:Q", title="값"),
    )
    layers = []
    if not only_anomalies:
        layers.append(base.mark_line(color="#9aa0a6", strokeWidth=1.5))
    layers.append(
        base.mark_circle(size=90).encode(
            color=alt.Color("이상여부:N",
                            scale=alt.Scale(domain=[False, True],
                                            range=["#9aa0a6", "#d93025"]),
                            legend=alt.Legend(title="이상여부")),
            tooltip=["시점", "값", "이상여부"],
        )
    )
    st.altair_chart(alt.layer(*layers).properties(height=320),
                    use_container_width=True)

    hits = part[part["이상여부"]]
    one, two, three = st.columns(3)
    one.metric("담긴 행", f"{len(part):,}")
    two.metric("이상", f"{len(hits):,}")
    three.metric("이상 비율", f"{100 * len(hits) / len(part):.1f}%")

st.caption("이상으로 잡힌 지점")
st.dataframe(hits, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 알고리즘 비교
st.divider()
st.subheader("3. 같은 단위를 여러 알고리즘이 어떻게 보았나")
st.caption("같은 데이터에서 나온 파일들만 비교된다 — 4종은 300그리드, stl 은 일별 입도 관광객이다.")

same_unit = [p for p in files if load(str(p), p.stat().st_mtime)[1] == unit
             and describe(p)[1] == describe(path)[1]]
if len(same_unit) < 2:
    st.info("비교할 파일이 하나뿐입니다.")
else:
    matrix = None
    for p in same_unit:
        other, _ = load(str(p), p.stat().st_mtime)
        column = (other[other[unit] == group]
                  .set_index("시점")["이상여부"].rename(describe(p)[0]))
        matrix = column.to_frame() if matrix is None else matrix.join(column, how="outer")
    if matrix is not None and not matrix.empty:
        values = part.set_index("시점")["값"]
        matrix.insert(0, "값", values)
        st.dataframe(matrix, use_container_width=True)
        st.caption("알고리즘별 이상 건수 (이 %s 에서)" % unit)
        st.dataframe(matrix.drop(columns="값").sum().rename("건수").to_frame().T,
                     use_container_width=True)

st.divider()
st.caption("읽는 곳: %s" % OUT)
