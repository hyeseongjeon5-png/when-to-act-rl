"""본실험 러너 — 조건 여러 개를 CPU 코어에 나눠 돌리고 진행 상황을 파일로 남긴다.

특징 (CLAUDE.md 워크플로 규칙):
  - 이어하기: 이미 끝난 조건(meta.json의 done=True)은 건너뛴다. 그냥 다시 실행하면 된다.
  - progress.json: 완료 조건 수 · 잔여 예상시간 · 조건별 상태를 실시간 기록 (자가 감시용)
  - 실패 조건은 max_attempts 번까지 재시도하고, 그래도 안 되면 '건너뜀'으로 표시하고 계속 간다
    (한 조건 때문에 48시간 전체가 멈추면 안 된다)

실행:
  python -m src.train.runner --config experiments/configs/pilot_mountaincar.yaml
  python -m src.train.runner --config ... --workers 9 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable


def expand(cfg: dict) -> list[dict]:
    """config → 조건 목록. 조건 = 환경 × 계열 × λ × 시드

    **순서가 중요하다.** 시드를 가장 바깥에 두어 (시드0의 전 계열×전 λ) → (시드1의 전 계열×전 λ) …
    순으로 돈다. 이렇게 하면 실험을 중간에 멈춰도 **격자 전체가 채워진 상태**가 되어
    시드 수만 적은 λ-성능 지도를 그릴 수 있다. 계열을 바깥에 두면 중간에 멈췄을 때
    한 계열이 통째로 비어 아무 그림도 못 그린다 (마감이 있는 실험에서 치명적).
    """
    out = []
    lam_by_agent = cfg.get("lambdas_by_agent", {})
    from src.train.train_agent import raw_dir_name
    room = raw_dir_name(cfg)
    for seed in cfg["seeds"]:
        for agent in cfg["agents"]:
            for lam in lam_by_agent.get(agent, cfg["lambdas"]):
                out.append({"env_id": cfg["env_id"], "room": room,
                            "agent": agent, "lam": float(lam), "seed": int(seed)})
    return out


def cond_key(c: dict) -> str:
    return f"{c.get('room', c['env_id'])}/{c['agent']}/lam{c['lam']}/seed{c['seed']}"


def is_done(c: dict, fp: str | None = None) -> bool:
    """이미 끝난 조건인가. 설정 지문이 다르면(하이퍼파라미터가 바뀌었으면) '안 끝난 것'으로 본다."""
    m = ROOT / "results" / "raw" / c.get("room", c["env_id"]) / c["agent"] / f"lam{c['lam']}" / f"seed{c['seed']}_meta.json"
    if not m.exists():
        return False
    try:
        d = json.loads(m.read_text(encoding="utf-8"))
        return bool(d.get("done")) and (fp is None or d.get("fingerprint") == fp)
    except Exception:
        return False


def done_elapsed(c: dict) -> float:
    m = ROOT / "results" / "raw" / c.get("room", c["env_id"]) / c["agent"] / f"lam{c['lam']}" / f"seed{c['seed']}_meta.json"
    try:
        return float(json.loads(m.read_text(encoding="utf-8")).get("elapsed_sec", 0.0))
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress", default=None, help="progress.json 경로 (기본: results/progress_{name}.json)")
    a = ap.parse_args()

    cfg_path = Path(a.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    name = cfg["name"]
    workers = a.workers or int(cfg.get("workers", max(1, (os.cpu_count() or 4) - 3)))
    conds = expand(cfg)
    prog_path = Path(a.progress) if a.progress else ROOT / "results" / f"progress_{name}.json"
    log_dir = ROOT / "results" / "logs" / "train"
    log_dir.mkdir(parents=True, exist_ok=True)

    from src.train.train_agent import fingerprint
    fps = {ag: fingerprint(cfg, ag) for ag in cfg["agents"]}
    state = {cond_key(c): {"status": "완료" if is_done(c, fps.get(c["agent"])) else "대기", "attempts": 0,
                           "elapsed_sec": done_elapsed(c) if is_done(c, fps.get(c["agent"])) else 0.0,
                           "fingerprint": fps.get(c["agent"]), **c} for c in conds}
    started = time.time()
    print(f"[러너] {name} | 총 {len(conds)}조건 | 이미 완료 {sum(1 for v in state.values() if v['status']=='완료')} | 동시 {workers}개", flush=True)
    if a.dry_run:
        for k, v in state.items():
            print(f"  {v['status']}  {k}")
        return

    def write_progress(extra: dict | None = None) -> None:
        vals = list(state.values())
        done = [v for v in vals if v["status"] == "완료"]
        elapsed_list = [v["elapsed_sec"] for v in done if v["elapsed_sec"] > 0]
        avg = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0
        running = [v for v in vals if v["status"] == "진행중"]
        remain = [v for v in vals if v["status"] in ("대기", "진행중")]
        eta = (len(remain) * avg / max(1, workers)) if avg else None
        payload = {
            "run_name": name, "config": str(cfg_path.relative_to(ROOT)).replace("\\", "/") if str(cfg_path).startswith(str(ROOT)) else str(cfg_path),
            "pid": os.getpid(), "workers": workers,
            "started_at": started, "updated_at": time.time(),
            "wall_elapsed_sec": round(time.time() - started, 1),
            "total": len(vals),
            "done": len(done),
            "running": len(running),
            "failed": sum(1 for v in vals if v["status"] == "실패"),
            "skipped": sum(1 for v in vals if v["status"] == "건너뜀"),
            "pending": sum(1 for v in vals if v["status"] == "대기"),
            "avg_sec_per_cond": round(avg, 1),
            "eta_sec": round(eta, 1) if eta else None,
            "eta_text": (f"약 {eta/3600:.1f}시간 남음" if eta else "측정 중"),
            "running_now": [v_k for v_k, v in state.items() if v["status"] == "진행중"],
            "conditions": state,
        }
        if extra:
            payload.update(extra)
        tmp = prog_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(prog_path)

    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    queue = [k for k, v in state.items() if v["status"] != "완료"]
    active: dict[str, dict] = {}
    write_progress()
    last_beat = time.time()

    while queue or active:
        while queue and len(active) < workers:
            k = queue.pop(0)
            c = state[k]
            c["attempts"] += 1
            c["status"] = "진행중"
            logf = (log_dir / (k.replace("/", "__") + ".log")).open("a", encoding="utf-8")
            logf.write(f"\n===== 시도 {c['attempts']} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            logf.flush()
            proc = subprocess.Popen(
                [PY, "-m", "src.train.train_agent", "--config", str(cfg_path),
                 "--agent", c["agent"], "--lam", str(c["lam"]), "--seed", str(c["seed"])],
                cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT, env=env)
            active[k] = {"proc": proc, "log": logf, "t0": time.time()}
            print(f"  ▶ 시작 {k} (시도 {c['attempts']})", flush=True)
            write_progress()

        time.sleep(2.0)
        if time.time() - last_beat > 30:  # 심장박동: 사건이 없어도 30초마다 갱신 (멈춤 감지용)
            write_progress()
            last_beat = time.time()
        for k in list(active):
            proc = active[k]["proc"]
            if proc.poll() is None:
                continue
            dt = time.time() - active[k]["t0"]
            active[k]["log"].close()
            active.pop(k)
            c = state[k]
            if proc.returncode == 0 and is_done(c, c.get("fingerprint")):
                c["status"] = "완료"
                c["elapsed_sec"] = done_elapsed(c)
                print(f"  ✔ 완료 {k} ({dt/60:.1f}분)", flush=True)
            else:
                if c["attempts"] >= a.max_attempts:
                    c["status"] = "건너뜀"
                    c["error"] = f"{a.max_attempts}회 실패 — 건너뜀 (로그: results/logs/train/{k.replace('/','__')}.log)"
                    print(f"  ✖ 건너뜀 {k} — {a.max_attempts}회 실패", flush=True)
                else:
                    c["status"] = "대기"
                    queue.append(k)
                    print(f"  ↻ 재시도 대기 {k} (종료코드 {proc.returncode})", flush=True)
            write_progress()

    write_progress({"finished": True, "finished_at": time.time()})
    d = sum(1 for v in state.values() if v["status"] == "완료")
    print(f"[러너 종료] {name} | 완료 {d}/{len(state)} | 건너뜀 {sum(1 for v in state.values() if v['status']=='건너뜀')} "
          f"| 총 {(time.time()-started)/3600:.2f}시간", flush=True)


if __name__ == "__main__":
    main()
