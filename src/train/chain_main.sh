#!/usr/bin/env bash
# 파일럿 B 러너가 끝나면 곧바로 본실험 러너를 띄운다.
# 파일럿 B와 본실험은 설정 지문이 같아, 파일럿에서 끝낸 조건은 본실험이 '완료'로 건너뛴다.
# 사용: bash src/train/chain_main.sh <파일럿progress파일> <본실험config>
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
PROG="$1"; CFG="$2"; NAME=$(basename "$CFG" .yaml)
echo "[연결] $PROG 종료를 기다린 뒤 $CFG 발사"
while true; do
  if [ ! -f "$PROG" ]; then break; fi
  if "$PY" -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1],encoding='utf-8')).get('finished') else 1)" "$PROG"; then
    break
  fi
  sleep 30
done
echo "[연결] 파일럿 종료 확인 — 본실험 $NAME 발사 $(date '+%H:%M:%S')"
exec "$PY" -m src.train.runner --config "$CFG"
