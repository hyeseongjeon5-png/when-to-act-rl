#!/usr/bin/env bash
# 스프린트 1단계 — 공정성 파일럿(후보 A·B·C)을 동시에 돌린다. 합계 10워커 = 12코어 중 10개.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
echo "[1단계] 공정성 파일럿 시작 $(date '+%Y-%m-%d %H:%M:%S')"
"$PY" -m src.train.runner --config experiments/configs/fair_pilot_mc_A.yaml --workers 4 > results/logs/runner_fairA.log 2>&1 &
"$PY" -m src.train.runner --config experiments/configs/fair_pilot_mc_B.yaml --workers 3 > results/logs/runner_fairB.log 2>&1 &
"$PY" -m src.train.runner --config experiments/configs/fair_pilot_mc_C.yaml --workers 3 > results/logs/runner_fairC.log 2>&1 &
wait
echo "[1단계] 공정성 파일럿 완료 $(date '+%Y-%m-%d %H:%M:%S')"
