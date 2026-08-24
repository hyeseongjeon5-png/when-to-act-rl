#!/usr/bin/env bash
# 1시간마다 집계 → 그림 → HTML 보고서를 다시 만든다.
# 실험이 도는 동안에도 "지금까지의 λ-성능 지도"가 항상 최신 상태로 존재하게 하려는 것.
# 시드가 모자란 중간 결과여도 신뢰구간이 넓게 나올 뿐 거짓말을 하지는 않는다.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
while true; do
  sleep 3600
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') 중간 집계 ====="
  "$PY" -m src.analysis.aggregate --env all --reps 2000 2>&1 | tail -3
  "$PY" -m src.analysis.plots --env all 2>&1 | tail -3
  "$PY" -m src.report.make_experiment_report 2>&1 | tail -1
done
