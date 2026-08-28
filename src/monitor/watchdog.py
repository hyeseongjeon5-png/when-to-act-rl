"""자가 감시 스크립트 — 본실험이 도는 동안 1시간마다 스스로 점검한다.

점검 6가지 (사용자 지시 '상시 자가 감시 루프'):
  ① 러너 프로세스가 살아 있는가
  ② progress.json이 지난 점검 때보다 전진했는가 (멈춤 감지)
  ③ 최신 학습 로그에 NaN/Inf 손실·예외·반복 에러가 없는가
  ④ 디스크 여유가 있고 결과 파일이 정상적으로 쌓이는가
  ⑤ 학습 곡선이 비정상인가 (보상 바닥 고정, 행동 횟수 0 고정 등)
  ⑥ 작업 대기열이 살아 있는가 (러너가 끝났는데 다음이 안 뜨면 12코어가 조용히 논다)

결과는 results/watchdog.log에 한 줄, 자세한 내용은 results/watchdog_last.json에 남긴다.
종료 코드: 0 정상 / 1 경고 / 2 이상(사람 또는 에이전트 개입 필요)

실행: python -m src.monitor.watchdog            (모든 progress_*.json 점검)
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
STATE = RESULTS / "watchdog_state.json"
LOG = RESULTS / "watchdog.log"

ERR_PAT = re.compile(r"(Traceback|Error|Exception|nan|inf)", re.IGNORECASE)


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
        return str(pid) in out
    except Exception:
        return False


def check_logs(minutes: int = 90, fresh_minutes: int = 20) -> dict:
    """최근 수정된 학습 로그에서 예외·NaN을 찾고, '지금 학습이 돌고 있는가'도 함께 본다.

    fresh: 최근 fresh_minutes 안에 갱신된 학습 로그 수. 이것이 0보다 크면
    조건이 아직 하나도 안 끝났어도 '멈춘 것'이 아니다 (조건 1개가 1시간 가까이 걸린다).
    """
    log_dir = RESULTS / "logs" / "train"
    cutoff = time.time() - minutes * 60
    fresh_cut = time.time() - fresh_minutes * 60
    fresh = 0
    bad, nan_hits, scanned = [], [], 0
    if log_dir.exists():
        for f in log_dir.glob("*.log"):
            mt = f.stat().st_mtime
            if mt >= fresh_cut:
                fresh += 1
            if mt < cutoff:
                continue
            scanned += 1
            try:
                tail = f.read_text(encoding="utf-8", errors="replace")[-8000:]
            except Exception:
                continue
            if "Traceback" in tail:
                bad.append(f.name)
            if re.search(r"loss[^\n]*\b(nan|inf)\b", tail, re.IGNORECASE) or "nan" in tail.lower().split("loss")[-1][:200]:
                nan_hits.append(f.name)
    return {"scanned": scanned, "fresh": fresh, "fresh_minutes": fresh_minutes,
            "traceback_logs": bad[:10], "nan_logs": nan_hits[:10]}


def check_curves(max_files: int = 400) -> dict:
    """끝난 조건들의 최종 결과를 훑어 이상 신호를 찾는다."""
    metas = sorted((RESULTS / "raw").rglob("seed*_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]
    bad_nan, all_floor, expected_floor, n = [], [], 0, 0
    for m in metas:
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        f = d.get("final", {})
        n += 1
        for k, v in f.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                bad_nan.append(str(m.relative_to(ROOT)))
                break
        # 고정 규칙(rule_*)은 제외한다. 무행동 규칙은 '행동 0회·성공 0%'가 정상이므로
        # 여기 걸리면 오탐이 된다 (2026-08-24 첫 실행에서 실제로 20건 오탐이 났다).
        agent = str(d.get("agent", ""))
        collapsed = (not agent.startswith("rule_")
                     and f.get("solved_rate", 1) == 0 and f.get("n_actions_mean", 1) == 0)
        if collapsed:
            # λ>0에서 무행동으로 굳는 것은 이 연구가 관찰하려는 현상 자체다(2026-08-25 확인).
            # 그걸 매번 경고하면 진짜 이상 신호가 파묻힌다. 세어만 두고 경고하지 않는다.
            # λ=0은 다르다 — 비용 압력이 아예 없는데 무행동으로 굳는 것은 정상이 아니다.
            if float(d.get("lam", 0)) > 0:
                expected_floor += 1
            else:
                all_floor.append(f"{d.get('env_id')}/{agent}/lam0/seed{d.get('seed')}")
    return {"checked": n, "nan_results": bad_nan[:10],
            "floor_conditions": all_floor[:20], "n_floor": len(all_floor),
            "n_expected_floor": expected_floor}


def main() -> int:
    now = time.time()
    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ts": now, "runs": [], "issues": [], "level": "정상"}
    logs = check_logs()

    for pf in sorted(RESULTS.glob("progress_*.json")):
        try:
            p = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            report["issues"].append(f"{pf.name} 읽기 실패: {e}")
            continue
        key = p["run_name"]
        alive = pid_alive(p.get("pid", -1))
        stale_min = (now - p.get("updated_at", 0)) / 60
        prev_done = prev.get("runs", {}).get(key, {}).get("done")
        # 지난 점검이 너무 최근이면 '전진 없음'을 판정하지 않는다.
        # 조건 1개가 30분 넘게 걸리므로 5분 전과 비교해 놓고 멈췄다고 하면 오탐이다.
        gap_min = (now - prev.get("ts", 0)) / 60 if prev.get("ts") else 0
        min_gap = float(prev.get("min_gap_min", 40))
        advanced = None
        if prev_done is not None and gap_min >= min_gap:
            advanced = p["done"] > prev_done
        r = {"run": key, "done": p["done"], "total": p["total"], "running": p["running"],
             "skipped": p.get("skipped", 0), "pid_alive": alive, "finished": bool(p.get("finished")),
             "stale_min": round(stale_min, 1), "advanced_since_last": advanced,
             "eta_text": p.get("eta_text"), "prev_done": prev_done}
        report["runs"].append(r)

        if not p.get("finished"):
            if not alive:
                report["issues"].append(f"[{key}] ① 러너 프로세스(pid {p.get('pid')})가 죽었다 — 재시작 필요")
                report["level"] = "이상"
            if stale_min > 45:
                report["issues"].append(f"[{key}] ② progress.json이 {stale_min:.0f}분째 갱신 없음 — 멈춤 의심")
                report["level"] = "이상"
            elif advanced is False and p["pending"] > 0 and logs["fresh"] == 0:
                # 최근에 갱신된 학습 로그가 하나도 없을 때만 '멈췄다'고 본다.
                # 조건 1개가 1시간 가까이 걸리는 설정에서는 '완료 수가 안 늘었다'만으로는
                # 멈춤을 판정할 수 없다 (2026-08-29 오탐 3건).
                report["issues"].append(
                    f"[{key}] ② 지난 점검({gap_min:.0f}분 전) 이후 완료 조건이 늘지 않았고 "
                    f"최근 {logs['fresh_minutes']}분간 갱신된 학습 로그도 없음 "
                    f"(완료 {p['done']}/{p['total']})")
                if report["level"] == "정상":
                    report["level"] = "경고"
        if p.get("skipped", 0) > 0:
            report["issues"].append(f"[{key}] 건너뛴 조건 {p['skipped']}개 (3회 실패) — 실험일지 확인")
            if report["level"] == "정상":
                report["level"] = "경고"

    # ---- ⑥ 작업 대기열 점검 ----
    # 러너 하나가 끝나고 다음이 안 떠도 progress.json은 '종료됨'이라 조용하다.
    # 그러면 12코어가 아무 경고 없이 논다 — 85시간 스프린트에서는 이게 가장 비싼 사고다.
    qs = RESULTS / "sprint_queue_state.json"
    if qs.exists():
        try:
            q = json.loads(qs.read_text(encoding="utf-8"))
            qf = ROOT / q.get("queue_file", "experiments/sprint_queue.tsv")
            labels = []
            if qf.exists():
                for line in qf.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = [x for x in line.split("	") if x.strip()]
                        if len(parts) >= 3:
                            labels.append(parts[0])
            pending = [x for x in labels if x not in q.get("done", [])]
            qpid = q.get("pid")
            # PID가 기록되기 전 버전으로 띄운 대기열일 수 있다 → '모름'이지 '죽음'이 아니다
            alive = pid_alive(qpid) if qpid else None
            report["queue"] = {"current": (q.get("current") or {}).get("label"),
                               "done": len(q.get("done", [])), "pending": len(pending),
                               "next": pending[0] if pending else None,
                               "alive": alive, "pid": qpid,
                               "stale_min": round((now - q.get("updated_at", 0)) / 60, 1)}
            # PID를 모를 때(옛 버전으로 띄운 대기열)는 '지금 도는 러너가 있는가'로 대신 판단한다.
            # 대기열이 죽어서 정말 문제가 되는 경우는 '작업 사이에 죽어서 아무도 안 도는' 때다.
            any_running = any(r["running"] > 0 and not r["finished"] for r in report["runs"])
            if alive is None and pending and not any_running and logs["fresh"] == 0:
                report["issues"].append(
                    f"⑥ 대기열 상태를 확인할 수 없는데(PID 미기록) 도는 러너도 없고 남은 작업이 "
                    f"{len(pending)}개다 (다음: {pending[0]}) — 대기열이 살아 있는지 직접 볼 것")
                if report["level"] == "정상":
                    report["level"] = "경고"
            if pending and alive is False:
                report["issues"].append(
                    f"⑥ 작업 대기열 프로세스(pid {q.get('pid')})가 죽었는데 남은 작업이 "
                    f"{len(pending)}개 있다 (다음: {pending[0]}) — 되살릴 것: "
                    f"nohup .venv/Scripts/python.exe -m src.train.sprint_queue &")
                report["level"] = "이상"
        except Exception as e:
            report["issues"].append(f"⑥ 대기열 상태 파일 읽기 실패: {e}")

    curves = check_curves()
    du = shutil.disk_usage(str(ROOT))
    free_gb = du.free / 1e9
    raw_files = sum(1 for _ in (RESULTS / "raw").rglob("*_meta.json"))
    report["logs"], report["curves"] = logs, curves
    report["disk_free_gb"] = round(free_gb, 1)
    report["done_conditions_total"] = raw_files

    if logs["traceback_logs"]:
        report["issues"].append(f"③ 최근 로그에 예외 발생: {', '.join(logs['traceback_logs'])}")
        report["level"] = "이상"
    if curves["nan_results"]:
        report["issues"].append(f"③ 결과에 NaN/Inf: {', '.join(curves['nan_results'])}")
        report["level"] = "이상"
    if free_gb < 5:
        report["issues"].append(f"④ 디스크 여유 {free_gb:.1f}GB — 부족")
        report["level"] = "이상"
    if curves["n_floor"] > 0:
        # λ=0인데 무행동으로 굳었다 — 비용 압력이 없으므로 학습 불안정 또는 버그 신호다.
        report["issues"].append(
            f"⑤ λ=0인데 무행동으로 굳은 조건 {curves['n_floor']}개 (비용이 없는데 안 움직인다 — "
            f"학습 불안정 의심): {', '.join(curves['floor_conditions'][:5])}")
        if report["level"] == "정상":
            report["level"] = "경고"

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "watchdog_last.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    STATE.write_text(json.dumps({"ts": now, "min_gap_min": 40,
                                 "runs": {r["run"]: {"done": r["done"]} for r in report["runs"]}},
                                ensure_ascii=False), encoding="utf-8")

    runs_txt = " ; ".join(f"{r['run']} {r['done']}/{r['total']}"
                          f"{' 진행중' if r['running'] else ''}"
                          f"{' 종료됨' if r['finished'] else ''}"
                          f"{'' if r['pid_alive'] or r['finished'] else ' [프로세스없음]'}"
                          for r in report["runs"]) or "실행 중인 러너 없음"
    issues_txt = " | ".join(report["issues"]) if report["issues"] else "이상 없음"
    q = report.get("queue")
    q_txt = (f" | 대기열 {q['done']}완료/{q['pending']}대기"
             f"{'(' + str(q['current']) + ' 진행중)' if q.get('current') else ''}"
             f"{' [대기열죽음]' if (q['alive'] is False and q['pending']) else ''}"
             f"{' [대기열PID미상]' if q['alive'] is None else ''}") if q else ""
    line = (f"{report['time']} [{report['level']}] {runs_txt}{q_txt} | 디스크 {free_gb:.0f}GB | "
            f"완료조건 누계 {raw_files} | 비용에 의한 무행동 수렴 {curves['n_expected_floor']}건(정상) | {issues_txt}")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return {"정상": 0, "경고": 1, "이상": 2}[report["level"]]


if __name__ == "__main__":
    raise SystemExit(main())
