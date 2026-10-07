# -*- coding: utf-8 -*-
"""[범주 2] 급증·급감 — 기대했던 값에서 갑자기 벗어난 시점을 찾는다.

극단치는 "값이 큰가"를 보고, 이 범주는 "**기대보다** 큰가"를 본다.
새벽 2시의 유동인구 50은 큰 값이 아니지만, 그 격자의 평소 새벽이 5라면 이상이다.

기대값은 STL 계절분해로 만든다. 계열을 추세와 주기로 설명하고,
설명되지 않고 남은 부분(잔차)이 크면 이상으로 본다.

입출력
    입력  x : pandas Series (시간 순서대로 정렬된 숫자 한 줄)
    출력  pandas DataFrame [value, score, flag]
"""
import numpy as np
import pandas as pd

from common import (cannot_judge, drop_missing, empty_result, make_result,
                     robust_standardize)


def stl_residual(x, period=24, k=3.0):
    """STL 계절분해의 잔차로 급증·급감을 찾는다.

    STL은 계열을 세 조각으로 나눈다.

        관측값 = 추세 + 주기 + 잔차

    추세는 길게 보는 흐름, 주기는 반복되는 무늬(24시간·7일),
    잔차는 그 둘로 설명되지 않는 나머지다. 설명되지 않는 부분이 크면 이상이다.

    로그를 먼저 씌우는 이유:
        유동인구는 수준이 커지면 출렁임도 같이 커진다(성수기엔 변동 폭도 큼).
        그대로 분해하면 성수기의 정상적인 큰 출렁임이 전부 잔차에 남는다.
        로그 공간에서 분해하면 잔차가 '기대의 몇 배'라는 비율이 되어
        규모가 다른 시기를 같은 잣대로 볼 수 있다.

    주기를 두 번 이상 관측해야 분해가 된다(예: 주기 24면 최소 48시점).
    """
    x = drop_missing(x)                                                 # 결측치 제거
    if cannot_judge(x, 2 * period):                                     # 주기성이 있는 데이터 최소 2개가 있는지 확인
        return empty_result(x)

    from statsmodels.tsa.seasonal import STL

    # 1단계 — 로그 공간으로 옮긴다 (값이 0이어도 되도록 log1p)
    logged = pd.Series(np.log1p(x.to_numpy()))                          # 관측값을 로그 변환

    # 2단계 — 추세·주기·잔차로 쪼갠다 (robust=True: 이상치에 덜 끌려가게)
    decomposed = STL(logged, period=period, robust=True).fit()
    resid = pd.Series(np.asarray(decomposed.resid), index=x.index)

    # 3단계 — 잔차가 평소 잔차보다 얼마나 큰지 재고 판정
    z = robust_standardize(resid)
    return make_result(x, z.abs(), z.abs() > k,
                       {"period": period, "k": k, "space": "log1p"})   
        ## 원본 시계열 값, 이상 점수($\vert{}Z\vert{}$), 이상 여부 dataframe 형태로 정리하여 반환
