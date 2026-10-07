# -*- coding: utf-8 -*-
"""실행 — 알고리즘을 돌리고 결과를 CSV로 저장한다.

함수가 둘이다. 하는 일은 같고 범위만 다르다.

    detect_one(계열 하나)   →  그 계열의 이상 지점       [time, value, algorithm, ...]
    detect_all(표 전체)     →  모든 그룹의 이상 지점     [group, time, value, ...]

detect_all 은 표의 행(그룹)마다 detect_one 을 한 번씩 부르고,
결과에 group 열을 붙여 이어 붙인다. 그게 전부다.

알고리즘을 고르지 않는다. 6개를 전부 돌린다.
어떤 알고리즘 결과를 쓸지는 이 결과 CSV를 받는 쪽에서 정한다.

출력 CSV 두 개
    anomalies.csv   이상으로 판정된 지점을 한 줄씩 (알고리즘별)
    summary.csv     알고리즘마다 몇 건을 찾았는지
"""
from pathlib import Path

import pandas as pd

from registry import ALGORITHMS, run

COLUMNS = ["time", "value", "algorithm", "category", "score"]


def detect_one(x, period=None, algorithms=None):
    """계열 **하나**에 모든 알고리즘을 돌린다.

    입력
        x           숫자 한 줄 (pandas Series) — 격자 하나, 지역 하나의 이력
        period      주기 (stl 에 쓰임)
        algorithms  돌릴 알고리즘 이름 목록. 없으면 전부

    출력
        DataFrame [time, value, algorithm, category, score]
        이상으로 판정된 지점만 담는다. 하나도 없으면 빈 표.
    """
    names = algorithms or list(ALGORITHMS)            # 지정한 알고리즘을 실행, 만약 지정한 알고리즘 없으면 모든 알고리즘 실행 
    pieces = []
    for name in names:
        result = run(name, x, period)
        hits = result[result["flag"]]                 # 이상인 지점만 골라낸다
        if hits.empty: 
            continue
        pieces.append(pd.DataFrame({                  # 이상탐지 지점의 시간,관측 값,알고리즘,범주,점수를 추출
            "time": hits.index,
            "value": hits["value"].to_numpy(),
            "algorithm": name,
            "category": ALGORITHMS[name][1],
            "score": hits["score"].to_numpy(),
        }))
    if not pieces:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(pieces, ignore_index=True)      # 모든 알고리즘 이상탐지값을  하나로 합쳐서 반환


def detect_all(table, period=24, algorithms=None, verbose=True):
    """표의 **모든 그룹**에 모든 알고리즘을 돌린다.

    입력
        table       그룹 × 시점 DataFrame (loader 가 만든 것)
        period      주기
        algorithms  돌릴 알고리즘 이름 목록. 없으면 전부

    출력
        DataFrame [group, time, value, algorithm, category, score]
    """
    pieces = []
    step = max(len(table) // 10, 1)                   # 진행 상황을 10번만 찍는다
    for i, (group, series) in enumerate(table.iterrows(), 1):
        found = detect_one(series.dropna(), period, algorithms)   # ← 여기서 detect_one함수 재사용
        if not found.empty:
            found.insert(0, "group", group)           # 어느 그룹인지 표시
            pieces.append(found)
        if verbose and (i % step == 0 or i == len(table)):
            print("  진행 %s / %s" % (format(i, ","), format(len(table), ",")))

    if not pieces:
        return pd.DataFrame(columns=["group"] + COLUMNS)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["group", "time", "algorithm"])     # 최종적으로 [group, time, algorithm] 순으로 정렬하여 반환합니다.


def summarize(found, table):
    """알고리즘마다 몇 건을 찾았는지 정리한다.

    탐지 건수만 보면 안 된다. 전체 칸 대비 비율을 같이 봐야
    "이 알고리즘이 과하게 잡는가"를 알 수 있다.
    """
    total_cells = table.shape[0] * table.shape[1] # 전체 데이터셀 수 를 구함
    rows = []
    for name, (_, category, _, description) in ALGORITHMS.items():
        count = int((found["algorithm"] == name).sum()) if not found.empty else 0 #등록된 알고리즘마다 이상치로 판정한 건수(count)와 전체 대비 비중(percent)을 계산
        rows.append({"algorithm": name, "category": category, "count": count,
                     "percent": round(100 * count / total_cells, 2),
                     "description": description})
    return pd.DataFrame(rows).sort_values("count", ascending=False)  # 요약 데이터프레임을 생성


def save(found, summary, outdir="결과"):
    """CSV 두 개로 저장한다. 엑셀에서 한글이 깨지지 않도록 BOM을 붙인다."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    found.to_csv(outdir / "anomalies.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n저장")
    print("  %s" % (outdir / "anomalies.csv"))
    print("  %s" % (outdir / "summary.csv"))
    return outdir
