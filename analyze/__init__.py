# -*- coding: utf-8 -*-
"""관광 데이터 이상 탐지 — 알고리즘 모음.

이 패키지는 **알고리즘을 고르지 않는다.** 가진 알고리즘 6개를 전부 돌려
결과를 CSV로 내놓는다. 어떤 결과를 쓸지는 이 CSV를 받는 쪽에서 정한다.

구성
    loader.py     형식이 다른 CSV → 그룹 × 시점 표
    extreme.py    극단치 4종      (IQR · IQR로그 · Z-Score · Modified Z)
    trend.py      급증·급감 1종   (STL 분해 잔차)
    shape.py      분포 왜곡 1종   (왜도·첨도)
    common.py     공통 통계 도구
    registry.py   알고리즘 목록
    runner.py     전부 실행 → CSV

알고리즘 하나만 쓰기
    from detection import run
    result = run("iqr", series)          # DataFrame [value, score, flag]

전체 실행
    from detection import load, detect_all
    table, info = load("grid300m")
    found = detect_all(table, period=info["period"])
"""
from loader import PRESETS, load, load_long, load_single, load_wide
from registry import ALGORITHMS, catalog, run
from runner import detect_all, detect_one, save, summarize

__all__ = [
    "load", "load_wide", "load_long", "load_single", "PRESETS",
    "run", "ALGORITHMS", "catalog",
    "detect_one", "detect_all", "summarize", "save",
]
