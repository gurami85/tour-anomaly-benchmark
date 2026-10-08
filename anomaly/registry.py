# -*- coding: utf-8 -*-
"""알고리즘 목록.

이름 하나로 알고리즘을 부를 수 있게 모아 둔 표다.
알고리즘을 새로 만들면 여기에 한 줄만 추가하면 된다.
"""
import pandas as pd

from . import extreme, shape, trend

#: 이름 → (함수, 범주, 최소 표본 수, 한 줄 설명) , #알고리즘 레지스트리 딕셔너리
ALGORITHMS = {
    # ---------------------------------------------------- 극단치: 값 자체가 튄다
    "iqr":      (extreme.iqr,             "극단치",    4, "사분위 범위(Q1~Q3) 밖"),
    "iqr_log":  (extreme.iqr_log,         "극단치",    4, "로그를 씌운 뒤 사분위 범위 밖"),
    "zscore":   (extreme.zscore,          "극단치",    5, "평균에서 표준편차 3배 밖"),
    "mzscore":  (extreme.modified_zscore, "극단치",    5, "중앙값에서 MAD 3.5배 밖"),

    # ------------------------------------------- 급증·급감: 기대값에서 벗어난다
    "stl":      (trend.stl_residual,      "급증/급감", 48, "추세·주기를 걷어낸 잔차가 큼"),

    # ----------------------------------------- 분포 왜곡: 분포를 쏠리게 만든다
    "skewkurt": (shape.skew_kurtosis,     "분포 왜곡", 20, "왜도·첨도를 만드는 관측치"),
}



#: 주기(period)를 인자로 받는 알고리즘. STL만 주기를 쓴다.
NEEDS_PERIOD = {"stl"}


def run(name, x, period=None, **options):
    """이름으로 알고리즘 하나를 돌린다.

    입력
        name     ALGORITHMS 의 키 (예: "iqr")
        x        숫자 한 줄 (pandas Series)
        period   주기. stl 처럼 필요한 알고리즘에만 쓰인다
        options  임계값 등 알고리즘별 인자 (예: k=2.0)

    출력
        DataFrame [value, score, flag]
    """
    if name not in ALGORITHMS:
        raise KeyError("모르는 알고리즘입니다: %s (가능: %s)"             # 알고리즘 이름이 없는게 들어오면 KeyError 반환
                       % (name, ", ".join(ALGORITHMS)))
    function = ALGORITHMS[name][0]
    if name in NEEDS_PERIOD and period:                                 # 주기가 있는지 검사 하고 주기가 있는 데이터이면 주기 이상탐지 알고리즘도 동작
        options.setdefault("period", period)
    return function(x, **options)                                       # 이상탐지 알고리즘 함수 실행값 반환


def catalog():
    """알고리즘 표를 사람이 보기 좋게 돌려준다."""
    return pd.DataFrame(
        [{"name": name, "category": category, "min_points": min_points,
          "description": description}
         for name, (_, category, min_points, description) in ALGORITHMS.items()])
