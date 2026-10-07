# -*- coding: utf-8 -*-
"""명령행 실행.

    python -m detection list                       알고리즘 목록 보기
    python -m detection grid300m                   300m 격자 전체
    python -m detection grid300m --groups 200      앞의 200개만 (빠른 확인)
    python -m detection tourapi --out 결과/방문자수
    python -m detection grid300m --algorithms iqr,stl   일부만 돌리기
"""
import argparse
import time

from loader import PRESETS, load
from registry import catalog
from runner import detect_all, save, summarize


def main():
    parser = argparse.ArgumentParser(
        description="관광 데이터 이상 탐지 — 전체 알고리즘 실행")
    parser.add_argument("dataset", help="프리셋 이름 또는 'list'")
    parser.add_argument("--pattern", help="CSV 경로 (기본: 프리셋에 적힌 경로)")
    parser.add_argument("--out", default="결과", help="저장 폴더 (기본: 결과)")
    parser.add_argument("--groups", type=int, default=0, help="분석할 그룹 수 상한")
    parser.add_argument("--algorithms", help="쉼표로 구분해 일부만 실행")
    args = parser.parse_args()

    if args.dataset == "list":
        print(catalog().to_string(index=False))
        return

    if args.dataset not in PRESETS:
        print("모르는 데이터입니다: %s" % args.dataset)
        print("가능: %s, list" % ", ".join(PRESETS))
        return

    table, info = load(args.dataset, args.pattern)
    if args.groups:
        table = table.head(args.groups)
    print("[데이터] %s %s개 × %s %s개 (주기 %d)"
          % (info["group_label"], format(table.shape[0], ","),
             info["time_label"], format(table.shape[1], ","), info["period"]))

    names = args.algorithms.split(",") if args.algorithms else None
    started = time.time()
    found = detect_all(table, period=info["period"], algorithms=names)
    summary = summarize(found, table)

    print("\n[알고리즘별 탐지]")
    print(summary.to_string(index=False))
    print("\n총 %s건 · %.1f초" % (format(len(found), ","), time.time() - started))
    save(found, summary, args.out)


if __name__ == "__main__":
    main()
