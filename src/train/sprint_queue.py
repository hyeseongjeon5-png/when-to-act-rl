"""스프린트 작업 대기열 — 실험을 순서대로 이어 돌려 CPU가 노는 시간을 없앤다.

왜 필요한가: 실험 하나가 끝나고 다음을 사람이 띄울 때까지 12코어가 논다.
85시간 스프린트에서 그 공백은 그대로 손실이다. 이 스크립트는 대기열 파일을
**매번 다시 읽으므로**, 도는 중에 파일 끝에 줄을 덧붙이면 다음 차례로 잡힌다.

대기열 파일 형식 (experiments/sprint_queue.tsv, 탭 구분, #은 주석):
  라벨<탭>종류<탭>내용
    종류 train : 내용 = "설정경로:워커수" 를 쉼표로 나열 → 동시에 띄우고 전부 끝날 때까지 기다림
    종류 rules : 내용 = 설정경로 쉼표 나열 → 고정 규칙 평가를 차례로 실행

상태: results/sprint_queue_state.json (지금 무엇을 하는 중인지, 무엇이 끝났는지)
실행: nohup .venv/Scripts/python.exe -m src.train.sprint_queue > results/logs/sprint_queue.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.utils_atomic import write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable
QUEUE = ROOT / "experiments" / "sprint_queue.tsv"
STATE = ROOT / "results" / "sprint_queue_state.json"
LOGDIR = ROOT / "results" / "logs"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    jobs = []
    for raw in QUEUE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split("\t") if x.strip()]
        if len(parts) < 3:
            log(f"  [무시] 형식이 맞지 않는 줄: {raw!r}")
            continue
        jobs.append({"label": parts[0], "kind": parts[1], "spec": parts[2]})
    return jobs


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": [], "current": None, "history": []}


def save_state(st: dict) -> None:
    st["pid"] = os.getpid()          # 자가 감시가 '대기열이 살아 있는가'를 확인하는 데 쓴다
    st["queue_file"] = str(QUEUE.relative_to(ROOT)).replace(chr(92), "/")
    st["updated_at"] = time.time()
    st["updated_text"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not write_text_atomic(STATE, json.dumps(st, ensure_ascii=False, indent=1)):
        log("  [경고] 대기열 상태 파일을 갱신하지 못했다 (다음 갱신 때 다시 쓴다)")


def train_incomplete(cfg_path: str) -> tuple[int, int, int]:
    """그 설정의 진행 파일을 보고 (완료, 건너뜀, 남은) 조건 수를 돌려준다.

    왜 필요한가: 러너가 중간에 죽으면 대기열의 wait는 그냥 돌아오고, 작업이 '완료'로 표시된 채
    다음으로 넘어간다. 조건이 남아 있어도 아무도 모른다 — 실제로 공정성 파일럿 후보 C가
    이렇게 조건 2개를 잃었다(2026-08-29 사고 2). 그래서 작업이 끝나면 직접 세어 본다.
    """
    import yaml
    try:
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        pf = ROOT / "results" / f"progress_{cfg['name']}.json"
        d = json.loads(pf.read_text(encoding="utf-8"))
        return d["done"], d.get("skipped", 0), d["total"] - d["done"] - d.get("skipped", 0)
    except Exception:
        return -1, -1, -1


def run_job(job: dict) -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    procs = []
    if job["kind"] == "train":
        for item in job["spec"].split(","):
            cfg, _, w = item.partition(":")
            cfg = cfg.strip()
            cmd = [PY, "-m", "src.train.runner", "--config", cfg]
            if w.strip():
                cmd += ["--workers", w.strip()]
            lf = (LOGDIR / f"runner_{job['label']}_{Path(cfg).stem}.log").open("a", encoding="utf-8")
            lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작 =====\n"); lf.flush()
            procs.append((subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
            log(f"  ▶ {cfg} (워커 {w or '설정값'})")
    elif job["kind"] == "rules":
        for cfg in [c.strip() for c in job["spec"].split(",")]:
            lf = (LOGDIR / f"rules_{job['label']}.log").open("a", encoding="utf-8")
            lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {cfg} =====\n"); lf.flush()
            p = subprocess.Popen([PY, "-m", "src.eval.run_rules_sweep", "--config", cfg],
                                 cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env)
            procs.append((p, lf))
            log(f"  ▶ 고정 규칙 {cfg}")
            p.wait()   # 규칙 평가는 1코어라 차례로 돌린다
    else:
        log(f"  [무시] 모르는 종류 {job['kind']}")
        return 0
    codes = []
    for p, lf in procs:
        codes.append(p.wait())
        lf.close()
    code = max(codes) if codes else 0

    # ---- 완주 확인: 러너가 죽어서 조건이 남았으면 최대 2번까지 더 띄운다 ----
    # 러너는 끝난 조건을 건너뛰므로 다시 띄우는 것이 싸고 안전하다(같은 조건을 두 번 계산하지 않는다).
    if job["kind"] == "train":
        for extra in range(1, 3):
            todo = []
            for item in job["spec"].split(","):
                cfg = item.partition(":")[0].strip()
                done, skipped, left = train_incomplete(cfg)
                if left > 0:
                    todo.append((cfg, item, done, skipped, left))
            if not todo:
                break
            log(f"  [완주 확인] 남은 조건이 있다 — 러너를 다시 띄운다 ({extra}/2)")
            again = []
            for cfg, item, done, skipped, left in todo:
                log(f"    {Path(cfg).stem}: 완료 {done} · 건너뜀 {skipped} · 남음 {left}")
                w = item.partition(":")[2].strip()
                cmd = [PY, "-m", "src.train.runner", "--config", cfg] + (["--workers", w] if w else [])
                lf = (LOGDIR / f"runner_{job['label']}_{Path(cfg).stem}.log").open("a", encoding="utf-8")
                lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 완주 확인 재시작 {extra} =====\n")
                lf.flush()
                again.append((subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf,
                                               stderr=subprocess.STDOUT, env=env), lf))
            for pr, lf in again:
                code = max(code, pr.wait())
                lf.close()
        else:
            log("  [주의] 두 번 더 띄웠는데도 조건이 남았다 — 사람이 볼 문제다")
    return code


def main() -> None:
    st = load_state()
    idle_since = None
    log("대기열 감시 시작")
    while True:
        jobs = read_queue()
        pending = [j for j in jobs if j["label"] not in st["done"]]
        if not pending:
            if idle_since is None:
                idle_since = time.time()
                log(f"대기열 비어 있음 — 새 줄이 추가되길 기다린다 ({QUEUE.name})")
            st["current"] = None
            save_state(st)
            time.sleep(60)
            continue
        idle_since = None
        job = pending[0]
        st["current"] = {"label": job["label"], "kind": job["kind"], "spec": job["spec"],
                         "started_at": time.time(), "started_text": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_state(st)
        log(f"=== 작업 시작: {job['label']} ({job['kind']}) ===")
        t0 = time.time()
        code = run_job(job)
        dt = (time.time() - t0) / 3600
        st["done"].append(job["label"])
        st["history"].append({"label": job["label"], "hours": round(dt, 2), "exit": code,
                              "finished_text": time.strftime("%Y-%m-%d %H:%M:%S")})
        st["current"] = None
        save_state(st)
        log(f"=== 작업 완료: {job['label']} | {dt:.2f}시간 | 종료코드 {code} ===")


if __name__ == "__main__":
    main()
