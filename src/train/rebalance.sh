#!/usr/bin/env bash
# LunarLander 본실험이 먼저 끝나면 그쪽 워커 5개가 논다.
# 그 시점에 MountainCar 러너를 워커 10개로 다시 띄워 남은 조건을 두 배 속도로 돌린다.
# 러너를 껐다 켜도 체크포인트와 설정 지문 덕분에 이어서 진행된다 (끝난 조건은 건너뜀).
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
LL=results/progress_main_lunarlander.json
MC=results/progress_main_mountaincar.json

echo "[재분배] LunarLander 본실험 종료를 기다린다"
while true; do
  if [ -f "$LL" ] && "$PY" -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1],encoding='utf-8')).get('finished') else 1)" "$LL"; then
    break
  fi
  sleep 60
done
echo "[재분배] LunarLander 종료 확인 $(date '+%H:%M:%S')"

if [ ! -f "$MC" ]; then echo "[재분배] MountainCar 진행 파일 없음 — 중단"; exit 0; fi
if "$PY" -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1],encoding='utf-8')).get('finished') else 1)" "$MC"; then
  echo "[재분배] MountainCar도 이미 끝남 — 할 일 없음"; exit 0
fi

PID=$("$PY" -c "import json;print(json.load(open('$MC',encoding='utf-8'))['pid'])")
echo "[재분배] MountainCar 러너(pid $PID)를 멈추고 워커 10개로 다시 띄운다"
taskkill //PID "$PID" //T //F > /dev/null 2>&1
sleep 5
exec "$PY" -m src.train.runner --config experiments/configs/main_mountaincar.yaml --workers 10
