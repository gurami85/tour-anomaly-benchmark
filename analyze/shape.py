# -*- coding: utf-8 -*-
"""[범주 3] 분포 왜곡·편향 — 분포를 한쪽으로 쏠리게 만드는 관측치를 찾는다.

앞의 두 범주는 "이 값이 이상한가"를 묻는다. 이 범주는 질문이 다르다.
**"이 분포를 이상하게 만든 범인이 누구인가"**를 묻는다.

왜도(skewness)와 첨도(kurtosis)가 그 도구다.

    왜도 = 표준화한 값의 세제곱 평균  →  좌우 비대칭 정도
    첨도 = 표준화한 값의 네제곱 평균  →  꼬리의 두꺼움

세제곱·네제곱을 하면 멀리 있는 값일수록 기여가 폭발적으로 커진다.
예를 들어 z=5인 값 하나는 z=1인 값 125개와 맞먹는다(5³=125).
그래서 합에서 각 관측치가 차지하는 몫을 보면 '범인'을 지목할 수 있다.

공통 입출력
    입력  x : pandas Series
    출력  pandas DataFrame [value, score, flag]
"""
import numpy as np
from scipy import stats

from common import (cannot_judge, drop_missing, empty_result, make_result,
                     standardize)


def skew_kurtosis(x, share_limit=0.01):
    """분포 왜곡을 만드는 관측치를 찾는다.

    share_limit=0.01 은 "한 점이 왜도의 1% 넘게 만들었으면 지목"이라는 뜻이다.
    관측치가 n개면 평균 기여도는 1/n 이다. 1000개짜리 계열에서 평균은 0.1%이므로
    1%는 평균의 10배를 뜻한다.

    점수: 기여도 ÷ 임계 (1을 넘으면 이상)
    """
    x = drop_missing(x)
    if cannot_judge(x, 20):                     # 너무 적으면 왜도가 불안정
        return empty_result(x)

    z = standardize(x)

    # 1단계 — 분포 전체의 모양을 잰다 (설명용 숫자)
    skew = float(stats.skew(x))
    excess_kurtosis = float(stats.kurtosis(x))   # 정규분포면 0이 되도록 3을 뺀 값
    normality_p = float(stats.jarque_bera(x).pvalue)   # 0.05 미만이면 정규분포 아님

    # 2단계 — 각 관측치가 왜도/첨도에 얼마나 기여했는지 몫을 구한다
    cubed = z ** 3
    quartic = z ** 4
    skew_share = cubed.abs() / max(cubed.abs().sum(), 1e-12)
    kurtosis_share = quartic / max(quartic.sum(), 1e-12)
    share = np.maximum(skew_share, kurtosis_share)      # 둘 중 큰 쪽으로 판단

    flag = share > share_limit
    return make_result(x, share / share_limit, flag,
                       {"skew": round(skew, 3),
                        "excess_kurtosis": round(excess_kurtosis, 3),
                        "normality_p": round(normality_p, 5),
                        "mean_share": round(1 / len(x), 5),
                        "share_limit": share_limit})
