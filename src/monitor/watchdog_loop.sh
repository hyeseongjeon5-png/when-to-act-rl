#!/usr/bin/env bash
# 1시간마다 자가 점검을 돌려 results/watchdog.log에 한 줄씩 남긴다.
# 터미널이 닫혀도 계속 돌도록 nohup으로 띄운다:
#   nohup bash src/monitor/watchdog_loop.sh > results/logs/watchdog_loop.log 2>&1 &
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
while true; do
  "$PY" -m src.monitor.watchdog || true
  sleep 3600
done
