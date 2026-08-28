"""고정 규칙(라 계열)을 학습 에이전트와 '완전히 같은 형식'으로 평가한다.

왜 따로 만들었나: 학습 에이전트의 최종 평가와 **같은 평가 시드**(500000+seed*1000+i)를 쓰고
같은 파일 구조(seed{n}_final.csv + seed{n}_meta.json)로 남겨야
집계 코드가 규칙과 학습을 구분 없이 똑같이 다룰 수 있다 (공정 비교).

'시드'의 뜻: 고정 규칙은 학습이 없으므로 시드 = 평가 에피소드 묶음 번호다.
시드 10개 × 100 에피소드 = 학습 에이전트와 같은 10×100 구조가 된다.

**λ마다 다시 돌리지 않는다.** 고정 규칙의 정책은 보상을 보지 않고, 환경 동역학도 λ와 무관하다.
따라서 λ가 달라져도 **궤적은 글자 그대로 같다.** 달라지는 것은 비용 회계뿐이므로
    r'(λ) = r - λ × (그 에피소드의 행동 횟수)
로 정확히 계산할 수 있다(근사가 아니라 항등식이다). λ 하나에서 한 번만 굴리고 나머지는 환산한다.
MinAtar처럼 에피소드가 1000스텝인 환경에서는 이것만으로 규칙 평가 시간이 λ 개수만큼 줄어든다.

실행: python -m src.eval.run_rules_sweep --config experiments/configs/main_mountaincar.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import yaml

from src.baselines.fixed_rules import (
    LunarLanderThresholdPolicy,
    MinAtarBreakoutTrackPolicy,
    MinAtarFreewayCautiousPolicy,
    MountainCarPumpPolicy,
    NoOpPolicy,
    PeriodicPolicy,
)
from src.envs.cost_wrapper import NOOP_BY_ENV, make_cost_env
from src.eval.evaluate import evaluate, summarize
from src.train.train_agent import raw_dir_name

ROOT = Path(__file__).resolve().parents[2]

TYPES = {
    "NoOpPolicy": NoOpPolicy,
    "PeriodicPolicy": PeriodicPolicy,
    "MountainCarPumpPolicy": MountainCarPumpPolicy,
    "LunarLanderThresholdPolicy": LunarLanderThresholdPolicy,
    "MinAtarBreakoutTrackPolicy": MinAtarBreakoutTrackPolicy,
    "MinAtarFreewayCautiousPolicy": MinAtarFreewayCautiousPolicy,
}


def build(spec: dict, env_id: str):
    kwargs = {k: v for k, v in spec.items() if k not in ("id", "type")}
    cls = TYPES[spec["type"]]
    if cls in (NoOpPolicy, PeriodicPolicy):
        kwargs["noop_action"] = NOOP_BY_ENV[env_id]
    return cls(**kwargs)


def _already(out: Path, seed, spec: dict, n_final: int) -> bool:
    mp = out / f"seed{seed}_meta.json"
    if not mp.exists() or not (out / f"seed{seed}_final.csv").exists():
        return False
    try:
        prev = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(prev.get("done")) and prev.get("rule_spec") == spec \
        and prev.get("n_eval_episodes_final") == n_final


def _write(out: Path, seed, rows: list[dict], meta: dict) -> dict:
    s = summarize(rows)
    with (out / f"seed{seed}_final.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / f"seed{seed}_meta.json").write_text(
        json.dumps({**meta, "final": s}, ensure_ascii=False, indent=2), encoding="utf-8")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    env_id = cfg["env_id"]
    seeds = cfg["seeds"]
    # 고정 규칙은 학습이 없어 스냅샷 개념이 없다. 학습 계열의 평가 수와 따로 둔다.
    n_final = int(cfg.get("n_eval_episodes_rules", 100))
    env_kwargs = dict(cfg.get("env_kwargs", {}))
    lams = [float(x) for x in cfg["lambdas"]]
    room = raw_dir_name(cfg)

    for spec in cfg["rules"]:
        agent_id = "rule_" + spec["id"]
        dirs = {lam: ROOT / "results" / "raw" / room / agent_id / f"lam{lam}" for lam in lams}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        # λ=0(없으면 가장 작은 λ)에서 한 번만 굴리고, 나머지 λ는 회계만 다시 한다
        base_lam = 0.0 if 0.0 in lams else min(lams)
        env = make_cost_env(env_id, lam=base_lam, **env_kwargs)
        t_run = t_derive = 0.0
        n_run = n_derive = 0
        for seed in seeds:
            todo = [lam for lam in lams if not _already(dirs[lam], seed, spec, n_final)]
            if not todo:
                continue
            eval_seeds = [500_000 + int(seed) * 1000 + i for i in range(n_final)]
            t0 = time.time()
            base_rows = evaluate(env, build(spec, env_id), eval_seeds)
            t_run += time.time() - t0
            n_run += 1
            for lam in todo:
                t1 = time.time()
                # 궤적은 같고 비용 회계만 다르다 — r'(λ) = r − λ × 행동 횟수 (정확한 항등식)
                rows = [dict(r, cost_return=float(r["raw_return"]) - lam * float(r["n_actions"]))
                        for r in base_rows]
                _write(dirs[lam], seed, rows, {
                    "done": True, "env_id": env_id, "agent": agent_id, "lam": lam, "seed": int(seed),
                    "total_steps": 0, "elapsed_sec": round(time.time() - t1, 3),
                    "config_name": cfg.get("name"), "env_kwargs": env_kwargs,
                    "n_eval_episodes_final": n_final, "rule_spec": spec,
                    "derived_from_lam": base_lam,
                    "derivation": "고정 규칙의 궤적은 λ와 무관하므로 λ=" + format(base_lam, "g")
                                  + "에서 굴린 에피소드로부터 r' = r − λ×행동횟수 로 환산했다",
                })
                t_derive += time.time() - t1
                n_derive += 1
        env.close()
        print(f"[규칙 {agent_id:<20}] 굴린 시드 {n_run}개 {t_run:.1f}초 · "
              f"환산 {n_derive}건 {t_derive:.2f}초 · λ {len(lams)}개", flush=True)


if __name__ == "__main__":
    main()
