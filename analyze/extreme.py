# -*- coding: utf-8 -*-
"""[범주 1] 극단치 — 값 자체가 유난히 크거나 작은 시점을 찾는다.

네 알고리즘 모두 방식이 같다.

    1. 계열의 '가운데'와 '흩어진 정도'를 잰다
    2. 가운데에서 흩어진 정도의 몇 배까지를 정상으로 볼지 경계를 긋는다
    3. 경계 밖이면 이상

다른 점은 1번에서 무엇을 쓰느냐 하나뿐이다.

    IQR          가운데=사분위수,  흩어짐=사분위 범위(Q3-Q1)
    IQR(로그)    같은 방식을 로그를 씌운 값에 적용
    Z-Score      가운데=평균,      흩어짐=표준편차
    Modified Z   가운데=중앙값,    흩어짐=MAD

공통 입출력
    입력  x : pandas Series (숫자 한 줄)
    출력  pandas DataFrame [value, score, flag]
"""
import numpy as np

from common import (cannot_judge, drop_missing, empty_result, mad,
                     make_result, robust_standardize, standardize)


def iqr(x, k=1.5):
    """사분위 범위(IQR) 기반 극단치.

    Q1(25%)과 Q3(75%) 사이가 '가운데 절반'이다. 그 폭의 k배를 위아래로 더 준다.

        경계 = [Q1 - k*(Q3-Q1),  Q3 + k*(Q3-Q1)]

    k=1.5는 상자그림(box plot)의 수염과 같은 관례값이다.
    분포 모양을 가정하지 않아서 어떤 데이터에나 쓸 수 있다.

    점수: 경계 밖으로 나간 거리 ÷ 사분위 범위 (경계 안이면 0)
    """ 
    x = drop_missing(x)  #결측치 제거
    if cannot_judge(x, 4):                          # 사분위수를 구하는 최소조건을 충족하는지 확인
        return empty_result(x)

    q1, q3 = x.quantile([0.25, 0.75])
    spread = q3 - q1
    low, high = q1 - k * spread, q3 + k * spread

    flag = (x < low) | (x > high)
    scale = spread if spread > 0 else (x.std() or 1.0)   # 폭이 0이면 표준편차로 대체
    score = np.maximum(np.maximum(x - high, low - x), 0) / scale
    return make_result(x, score, flag, {"low": low, "high": high, "k": k})   
        # 이상치 판정 및 점수 계산

def iqr_log(x, k=1.5):
    """로그를 씌운 뒤 IQR.

    유동인구처럼 대부분 작고 일부만 아주 큰 데이터는, 원래 값에서 경계를 그으면
    큰 값 쪽 꼬리가 통째로 이상이 된다. log1p(x) = log(1+x) 로 큰 값을 눌러
    분포를 대칭에 가깝게 만든 뒤 같은 규칙을 적용한다.

    log1p 를 쓰는 이유: 값이 0이어도 계산된다(log(1+0)=0).
    음수가 있으면 로그를 못 쓰므로 그 계열은 건너뛴다.
    """
    x = drop_missing(x)  #결측치 제거
    if cannot_judge(x, 4) or (x < 0).any(): #최소조건 만족하는지 확인 + 음수 없는지 확인
        return empty_result(x)

    logged = np.log1p(x)                            # 여기서만 로그로 값이 바뀜
    result = iqr(logged, k)                         # 로그 변환된 값 입력
    result["value"] = x                             # 보고는 원래 값으로
    return result


def zscore(x, k=3.0):
    """Z-Score 기반 극단치. 정규분포를 전제한다.

        z = (값 - 평균) / 표준편차,   |z| > k 이면 이상

    정규분포라면 |z|>3 인 값은 1000개 중 3개꼴이다. 그래서 k=3이 관례다.
    다만 평균과 표준편차 자체가 극단값에 오염되므로, 분포가 치우친
    데이터에서는 둔감해진다(진짜 이상치를 놓친다).

    점수: |z|
    """
    x = drop_missing(x)   #결측치 제거
    if cannot_judge(x, 5):   # 최소조건 확인, 표본 5개 이상
        return empty_result(x)

    z = standardize(x)
    mean, sd = x.mean(), x.std()
    return make_result(x, z.abs(), z.abs() > k,
                       {"mean": mean, "std": sd,
                        "low": mean - k * sd, "high": mean + k * sd})
                    # 이상치 판정 및 점수 계산

def modified_zscore(x, k=3.5):
    """중앙값·MAD 기반 극단치. Z-Score의 강건한 형태.

        z = 0.6745 * (값 - 중앙값) / MAD,   |z| > k 이면 이상

    평균 대신 중앙값, 표준편차 대신 MAD를 쓴다. 둘 다 극단값에 끌려가지 않아서
    "이상치가 경계를 넓혀 자기를 숨기는" 문제가 생기지 않는다.
    k=3.5는 이 방식에서 널리 쓰는 관례값이다.

    점수: |z|
    """
    x = drop_missing(x)
    if cannot_judge(x, 5):
        return empty_result(x)

    z = robust_standardize(x)
    center, spread = x.median(), mad(x)
    half_width = k * spread / 0.6745 if spread else np.nan   # 경계를 원래 단위로
    return make_result(x, z.abs(), z.abs() > k,
                       {"median": center, "mad": spread,
                        "low": center - half_width, "high": center + half_width})
