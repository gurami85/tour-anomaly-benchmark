# -*- coding: utf-8 -*-
"""알고리즘들이 함께 쓰는 통계 도구.

여기 있는 함수는 '이상이냐 아니냐'를 판단하지 않는다.
판단에 필요한 재료(중앙값, 표준화 점수 등)만 계산한다.
"""
import numpy as np
import pandas as pd


def drop_missing(x):
    """숫자로 바꾸고 빈 값은 버린다. 모든 알고리즘이 맨 앞에서 이걸 부른다.

    입력: 아무 Series
    출력: 숫자만 남은 Series (원래 index 유지)
    """
    x = pd.Series(x).astype(float)
    return x[~x.isna()]


def standardize(x):
    """평균에서 표준편차 몇 배 떨어졌는지. (z = (값 - 평균) / 표준편차)

    평균과 표준편차는 극단값에 끌려간다는 점을 기억해 둘 것.
    값이 하나라도 튀면 평균이 올라가고 표준편차가 커져서,
    정작 그 튄 값의 z가 작아지는 일이 생긴다.
    """
    sd = x.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / sd


def mad(x):
    """MAD(중앙값 절대편차) = 중앙값에서 떨어진 거리들의 중앙값.

    표준편차와 같은 역할(흩어진 정도)을 하지만 극단값에 끌려가지 않는다.
    절반이 넘는 값이 무너져야 MAD가 흔들린다.
    """
    return (x - x.median()).abs().median()


def robust_standardize(x):
    """중앙값·MAD로 만든 표준화 점수. (Modified Z-Score)

        z = 0.6745 * (값 - 중앙값) / MAD

    0.6745를 곱하는 이유: 정규분포에서 MAD는 표준편차의 약 0.6745배다.
    그래서 이 상수를 곱해야 보통의 z 점수와 같은 눈금이 된다.
    """
    center = x.median()
    spread = mad(x)
    if spread == 0 or np.isnan(spread):          # 값이 거의 다 같은 계열
        mean_dev = (x - center).abs().mean()
        if mean_dev == 0 or np.isnan(mean_dev):
            return pd.Series(0.0, index=x.index)
        return (x - center) / (1.2533 * mean_dev)
    return 0.6745 * (x - center) / spread


def cannot_judge(x, min_points):
    """이 계열은 판정하지 말아야 하는가?

    두 가지 경우에 True 다.
        1) 표본이 min_points 보다 적다     — 통계량을 믿을 수 없다
        2) 값이 사실상 하나뿐이다          — 흩어진 정도가 0이라 기준을 못 만든다

    2번을 '값이 전부 같다'가 아니라 '사실상 같다'로 보는 이유:
    계산 과정에서 1e-17 같은 부동소수점 찌꺼기가 남으면, 그 찌꺼기끼리
    비교해 엉뚱한 이상치가 만들어진다.
    """
    if len(x) < min_points:
        return True
    spread = float(x.max() - x.min())
    tolerance = max(abs(float(x.median())), 1.0) * 1e-9
    return spread <= tolerance


def make_result(x, score, flag, thresholds=None):
    """모든 알고리즘이 돌려주는 출력 형식.

    입력
        x           분석한 계열
        score       클수록 이상하다는 뜻의 숫자 (알고리즘마다 의미가 다름)
        flag        이상 여부 True/False
        thresholds  판정에 쓴 경계값 등 (설명용, 없으면 생략)

    출력: index가 시점인 표
        시점        value  score  flag
        2014-05|17  601.8   8.24  True
    """
    result = pd.DataFrame({
        "value": x,
        "score": pd.Series(score, index=x.index).round(4),
        "flag": pd.Series(flag, index=x.index).fillna(False),
    })
    result.attrs["thresholds"] = thresholds or {}
    return result


def empty_result(x):
    """표본이 모자라 판정할 수 없을 때 돌려줄 결과. 전부 '이상 아님'."""
    return make_result(x, np.zeros(len(x)), np.zeros(len(x), dtype=bool),
                       {"note": "표본 부족으로 실행하지 않음"})
