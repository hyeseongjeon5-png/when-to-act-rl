"""자가 감시 스크립트 — 본실험이 도는 동안 1시간마다 스스로 점검한다.

점검 5가지 (사용자 지시 '상시 자가 감시 루프'):
  ① 러너 프로세스가 살아 있는가
  ② progress.json이 지난 점검 때보다 전진했는가 (멈춤 감지)
  ③ 최신 학습 로그에 NaN/Inf 손실·예외·반복 에러가 없는가
  ④ 디스크 여유가 있고 결과 파일이 정상적으로 쌓이는가
  ⑤ 학습 곡선이 비정상인가 (보상 바닥 고정, 행동 횟수 0 고정 등)

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


def check_logs(minutes: int = 90) -> dict:
    """최근 수정된 학습 로그에서 예외·NaN을 찾는다."""
    log_dir = RESULTS / "logs" / "train"
    cutoff = time.time() - minutes * 60
    bad, nan_hits, scanned = [], [], 0
    if log_dir.exists():
        for f in log_dir.glob("*.log"):
            if f.stat().st_mtime < cutoff:
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
    return {"scanned": scanned, "traceback_logs": bad[:10], "nan_logs": nan_hits[:10]}


def check_curves(max_files: int = 400) -> dict:
    """끝난 조건들의 최종 결과를 훑어 이상 신호를 찾는다."""
    metas = sorted((RESULTS / "raw").rglob("seed*_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]
    bad_nan, all_floor, n = [], [], 0
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
        if (not agent.startswith("rule_")
                and d.get("env_id") == "MountainCar-v0"
                and f.get("solved_rate", 1) == 0 and f.get("n_actions_mean", 1) == 0):
            all_floor.append(f"{agent}/lam{d.get('lam')}/seed{d.get('seed')}")
    return {"checked": n, "nan_results": bad_nan[:10],
            "floor_conditions": all_floor[:20], "n_floor": len(all_floor)}


def main() -> int:
    now = time.time()
    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ts": now, "runs": [], "issues": [], "level": "정상"}

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
            elif advanced is False and p["pending"] > 0:
                report["issues"].append(
                    f"[{key}] ② 지난 점검({gap_min:.0f}분 전) 이후 완료 조건이 늘지 않음 "
                    f"(완료 {p['done']}/{p['total']})")
                if report["level"] == "정상":
                    report["level"] = "경고"
        if p.get("skipped", 0) > 0:
            report["issues"].append(f"[{key}] 건너뛴 조건 {p['skipped']}개 (3회 실패) — 실험일지 확인")
            if report["level"] == "정상":
                report["level"] = "경고"

    logs = check_logs()
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
        # 학습 계열이 '행동 0회 + 성공 0%'로 굳은 것은 무행동 정책으로 무너졌다는 뜻이다.
        # 버그일 수도 있고 비용 λ 때문에 실제로 그렇게 수렴한 것일 수도 있어, 경고로만 남긴다.
        report["issues"].append(
            f"⑤ 학습 계열이 무행동으로 굳은 조건 {curves['n_floor']}개 "
            f"(λ가 크면 정상일 수 있음): {', '.join(curves['floor_conditions'][:5])}")
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
    line = (f"{report['time']} [{report['level']}] {runs_txt} | 디스크 {free_gb:.0f}GB | "
            f"완료조건 누계 {raw_files} | {issues_txt}")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return {"정상": 0, "경고": 1, "이상": 2}[report["level"]]


if __name__ == "__main__":
    raise SystemExit(main())
