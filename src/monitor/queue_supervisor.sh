#!/usr/bin/env bash
# 작업 대기열 감시자 — 대기열이 죽으면 되살린다. **중복 실행을 절대 만들지 않는 것이 1순위다.**
#
# 2026-08-29 사고 기록:
#   첫 버전은 프로세스 탐지에 wmic를 썼는데 이 환경(Git Bash)에는 wmic가 없어 항상 '없음'으로
#   읽혔다. 그래서 5분마다 대기열을 새로 띄웠고, 같은 실험이 세 벌 돌아 CPU가 3배 과부하가 됐다.
#   게다가 그 감시자를 pkill로 죽였다고 생각했는데 살아남아 계속 띄웠다.
#   결과 오염은 없었지만(확인함) 한 시간을 잃었다.
#
# 그래서 이 버전은 다음을 지킨다:
#   1) **잠금 파일**  — 살아 있는 감시자가 있으면 새 감시자는 즉시 물러난다
#   2) **탐지 검증**  — 시작할 때 탐지가 실제로 동작하는지 확인하고, 안 되면 되살리기를 아예 끈다
#   3) **두 번 확인** — 죽었다고 판단해도 30초 뒤 한 번 더 보고 나서 움직인다
#   4) **재시작 상한** — 최대 3회, 그리고 최소 20분 간격. 계속 죽는다면 사람이 볼 문제다
#   5) **자기 종료**  — 잠금 파일이 사라지면 스스로 끝난다 (kill이 안 먹어도 멈출 수 있는 손잡이)
#
# 실행: nohup bash src/monitor/queue_supervisor.sh > results/logs/queue_supervisor.log 2>&1 &
# 정지: rm results/queue_supervisor.lock   (다음 점검 때 스스로 끝난다)
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
LOCK=results/queue_supervisor.lock
MAX_RESTARTS=3
MIN_GAP=1200          # 재시작 사이 최소 간격(초)
CHECK_EVERY=300

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

count_queue() {
  powershell -NoProfile -Command \
    "(@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*sprint_queue*' })).Count" \
    2>/dev/null | tr -d '\r' | tr -d ' ' | tail -1
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

# ---- 1) 잠금 ----
if [ -f "$LOCK" ]; then
  OLD=$(cat "$LOCK" 2>/dev/null)
  if kill -0 "$OLD" 2>/dev/null; then
    say "이미 감시자(pid $OLD)가 돌고 있다 — 물러난다"
    exit 0
  fi
  say "낡은 잠금 파일(pid $OLD, 죽어 있음) 정리"
fi
echo "$$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
say "감시자 시작 (pid $$)"

# ---- 2) 탐지 검증: 지금 대기열이 도는 것을 아는가 ----
N=$(count_queue)
case "$N" in
  ''|*[!0-9]*) say "[치명] 프로세스 탐지가 숫자를 주지 않는다('$N') — 되살리기를 끄고 기록만 한다"; ARMED=0;;
  0) say "[주의] 시작 시점에 대기열이 없다고 나온다. 정말 없는지 사람이 확인할 것 — 이번 실행은 기록만 한다"; ARMED=0;;
  *) say "탐지 확인: 대기열 프로세스 $N개 인식 — 되살리기 켬"; ARMED=1;;
esac

RESTARTS=0
LAST=0
while true; do
  sleep "$CHECK_EVERY"
  [ -f "$LOCK" ] || { say "잠금 파일이 사라졌다 — 감시자를 끝낸다"; exit 0; }
  P=$(pending_jobs); [ -z "$P" ] && P=0
  [ "$P" -eq 0 ] && continue
  [ "$ARMED" -eq 0 ] && continue

  N=$(count_queue)
  case "$N" in ''|*[!0-9]*) say "[주의] 개수를 읽지 못했다('$N') — 이번 점검 건너뜀"; continue;; esac
  [ "$N" -ge 1 ] && continue

  # ---- 3) 두 번 확인 ----
  sleep 30
  N2=$(count_queue)
  case "$N2" in ''|*[!0-9]*) say "[주의] 재확인 실패 — 건너뜀"; continue;; esac
  [ "$N2" -ge 1 ] && { say "재확인하니 살아 있다 — 건너뜀"; continue; }

  # ---- 4) 상한 ----
  NOW=$(date +%s)
  if [ "$RESTARTS" -ge "$MAX_RESTARTS" ]; then
    say "[정지] 이미 ${RESTARTS}회 되살렸다 — 더는 하지 않는다. 사람이 볼 문제다"
    continue
  fi
  if [ $((NOW - LAST)) -lt "$MIN_GAP" ]; then
    say "[대기] 직전 되살리기로부터 $((NOW - LAST))초밖에 안 지났다 — 건너뜀"
    continue
  fi

  say "대기열이 죽었다(두 번 확인) — 남은 작업 ${P}개. 되살린다 ($((RESTARTS + 1))/${MAX_RESTARTS})"
  nohup "$PY" -m src.train.sprint_queue >> results/logs/sprint_queue.log 2>&1 &
  RESTARTS=$((RESTARTS + 1)); LAST=$NOW
  sleep 25
  say "되살림 결과: 대기열 프로세스 $(count_queue)개"
done
