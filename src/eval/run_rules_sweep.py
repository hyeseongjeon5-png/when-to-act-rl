"""고정 규칙(라 계열)을 학습 에이전트와 '완전히 같은 형식'으로 평가한다.

왜 따로 만들었나: 학습 에이전트의 최종 평가와 **같은 평가 시드**(500000+seed*1000+i)를 쓰고
같은 파일 구조(seed{n}_final.csv + seed{n}_meta.json)로 남겨야
집계 코드가 규칙과 학습을 구분 없이 똑같이 다룰 수 있다 (공정 비교).

'시드'의 뜻: 고정 규칙은 학습이 없으므로 시드 = 평가 에피소드 묶음 번호다.
시드 10개 × 100 에피소드 = 학습 에이전트와 같은 10×100 구조가 된다.

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
    MountainCarPumpPolicy,
    NoOpPolicy,
    PeriodicPolicy,
)
from src.envs.cost_wrapper import NOOP_BY_ENV, make_cost_env
from src.eval.evaluate import evaluate, summarize

ROOT = Path(__file__).resolve().parents[2]

TYPES = {
    "NoOpPolicy": NoOpPolicy,
    "PeriodicPolicy": PeriodicPolicy,
    "MountainCarPumpPolicy": MountainCarPumpPolicy,
    "LunarLanderThresholdPolicy": LunarLanderThresholdPolicy,
}


def build(spec: dict, env_id: str):
    kwargs = {k: v for k, v in spec.items() if k not in ("id", "type")}
    cls = TYPES[spec["type"]]
    if cls in (NoOpPolicy, PeriodicPolicy):
        kwargs["noop_action"] = NOOP_BY_ENV[env_id]
    return cls(**kwargs)


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

    for lam in cfg["lambdas"]:
        for spec in cfg["rules"]:
            agent_id = "rule_" + spec["id"]
            out = ROOT / "results" / "raw" / env_id / agent_id / f"lam{float(lam)}"
            out.mkdir(parents=True, exist_ok=True)
            env = make_cost_env(env_id, lam=float(lam), **env_kwargs)
            for seed in seeds:
                eval_seeds = [500_000 + int(seed) * 1000 + i for i in range(n_final)]
                t0 = time.time()
                rows = evaluate(env, build(spec, env_id), eval_seeds)
                s = summarize(rows)
                with (out / f"seed{seed}_final.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)
                (out / f"seed{seed}_meta.json").write_text(json.dumps({
                    "done": True, "env_id": env_id, "agent": agent_id, "lam": float(lam), "seed": int(seed),
                    "total_steps": 0, "elapsed_sec": round(time.time() - t0, 2),
                    "config_name": cfg.get("name"), "env_kwargs": env_kwargs,
                    "final": s, "n_eval_episodes_final": n_final, "rule_spec": spec,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            env.close()
            print(f"[규칙 {agent_id:<18} λ={lam}] r IQM {s['raw_return_iqm']:8.2f} | "
                  f"r' IQM {s['cost_return_iqm']:8.2f} | 행동 {s['n_actions_mean']:6.1f}회 | "
                  f"성공률 {s['solved_rate']*100:5.1f}% (마지막 시드 기준)", flush=True)


if __name__ == "__main__":
    main()
