#!/usr/bin/env bash
# 작업 대기열 감시자 — 대기열 프로세스가 죽으면 되살린다.
#
# 왜 필요한가: 85시간 자율 운영에서 가장 비싼 사고는 '대기열이 죽어 12코어가 조용히 노는 것'이다.
# 자가 감시(watchdog)는 1시간에 한 번만 보므로 최대 1시간을 잃는다. 이 감시자는 5분마다 본다.
# 대기열은 끝낸 작업을 상태 파일에 남기므로, 되살아나면 하던 다음 작업부터 이어서 한다.
#
# 프로세스 탐지에 wmic를 쓰지 않는다 — 이 환경(Git Bash)에는 wmic가 없어 항상 '없음'으로 읽혀
# 중복 실행을 낳을 뻔했다(2026-08-29 확인). PowerShell의 Get-CimInstance를 쓴다.
# 참고: .venv의 python.exe는 실제 인터프리터를 자식으로 띄우므로 프로세스가 2개로 보인다.
#       개수가 아니라 '1개 이상인가'로만 판단한다.
#
# 실행: nohup bash src/monitor/queue_supervisor.sh > results/logs/queue_supervisor.log 2>&1 &
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

count_queue() {
  powershell -NoProfile -Command \
    "(@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*sprint_queue*' })).Count" \
    2>/dev/null | tr -d '\r' | tail -1
}

pending_jobs() {
  "$PY" - <<'PYEOF' 2>/dev/null
import json
from pathlib import Path
done = set()
q = Path("results/sprint_queue_state.json")
if q.exists():
    try:
        done = set(json.loads(q.read_text(encoding="utf-8")).get("done", []))
    except Exception:
        pass
labels = []
tsv = Path("experiments/sprint_queue.tsv")
if tsv.exists():
    for line in tsv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts = [x for x in line.split("\t") if x.strip()]
            if len(parts) >= 3:
                labels.append(parts[0])
print(len([x for x in labels if x not in done]))
PYEOF
}

N=$(count_queue); [ -z "$N" ] && N="?"
say "대기열 감시자 시작 (5분 주기) — 현재 대기열 프로세스 $N개"
if [ "$N" = "?" ] || [ "$N" = "" ]; then
  say "[경고] 프로세스 탐지가 동작하지 않는다 — 되살리기를 하지 않고 기록만 한다"
  DETECT_OK=0
else
  DETECT_OK=1
fi

while true; do
  sleep 300
  P=$(pending_jobs); [ -z "$P" ] && P=0
  [ "$P" -eq 0 ] && continue
  [ "$DETECT_OK" -eq 0 ] && continue
  N=$(count_queue)
  case "$N" in ''|*[!0-9]*) say "[경고] 프로세스 개수를 읽지 못했다($N) — 이번 점검 건너뜀"; continue;; esac
  [ "$N" -ge 1 ] && continue
  say "대기열 프로세스가 없다 — 남은 작업 ${P}개. 되살린다."
  nohup "$PY" -m src.train.sprint_queue >> results/logs/sprint_queue.log 2>&1 &
  sleep 25
  say "되살림 결과: 대기열 프로세스 $(count_queue)개"
done
