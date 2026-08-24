"""고정 규칙 기준선 평가 러너 (계열 라).

config(yaml)를 읽어 MountainCar 등에서 고정 규칙들을 평가하고
에피소드별 결과를 CSV로 남긴다. 학습이 없으므로 환경 시드만 바꾼다.

실행: python -m src.eval.run_fixed_rules --config experiments/configs/smoke_fixed_rules_mountaincar.yaml

기록 항목(docs/02_실험-설계.md §4):
  raw_return   : 비용 빼기 전 원래 보상 합 (r 기준)
  cost_return  : 비용 차감 후 보상 합 (r' 기준)
  n_actions    : 에피소드당 행동(no-op 아닌) 횟수
  steps        : 에피소드 길이
  solved       : 목표 도달 여부(terminated=True, 시간초과는 False)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from src.baselines.fixed_rules import MountainCarPumpPolicy, NoOpPolicy, PeriodicPolicy
from src.envs.cost_wrapper import NOOP_BY_ENV, make_cost_env

ROOT = Path(__file__).resolve().parents[2]

POLICY_TYPES = {
    "NoOpPolicy": NoOpPolicy,
    "PeriodicPolicy": PeriodicPolicy,
    "MountainCarPumpPolicy": MountainCarPumpPolicy,
}


def build_policy(spec: dict, env_id: str):
    """config의 정책 명세 한 줄 -> 정책 객체."""
    kwargs = {k: v for k, v in spec.items() if k not in ("id", "type")}
    cls = POLICY_TYPES[spec["type"]]
    if cls in (NoOpPolicy, PeriodicPolicy):
        kwargs["noop_action"] = NOOP_BY_ENV[env_id]
    return cls(**kwargs)


def run_episode(env, policy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    t, cost_return, terminated, truncated, info = 0, 0.0, False, False, {}
    while not (terminated or truncated):
        action = policy.act(obs, t)
        obs, reward, terminated, truncated, info = env.step(action)
        cost_return += reward
        t += 1
    return {
        "seed": seed,
        "steps": t,
        "raw_return": info["episode_raw_return"],
        "cost_return": cost_return,
        "n_actions": info["episode_actions"],
        "solved": int(bool(terminated)),
    }


def iqm(x) -> float:
    """사분위평균 — 위아래 25%를 버리고 가운데 50%의 평균 (Agarwal et al. 2021)."""
    x = np.sort(np.asarray(x, dtype=float))
    lo, hi = int(np.floor(len(x) * 0.25)), int(np.ceil(len(x) * 0.75))
    return float(np.mean(x[lo:hi]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    env_id = cfg["env_id"]
    n_ep = int(cfg["n_episodes"])
    s0 = int(cfg.get("eval_seeds_start", 0))

    summary = []
    for lam in cfg["lambdas"]:
        for spec in cfg["policies"]:
            out_dir = ROOT / "results" / "raw" / env_id / spec["id"] / f"lam{lam}"
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(cfg_path, out_dir / cfg_path.name)  # 결과 옆에 config 사본 (절대 규칙 3)

            env = make_cost_env(env_id, lam=lam)
            policy = build_policy(spec, env_id)
            t0 = time.time()
            rows = [run_episode(env, policy, s0 + i) for i in range(n_ep)]
            env.close()
            elapsed = time.time() - t0

            with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

            rec = {
                "policy": spec["id"],
                "lam": lam,
                "n_episodes": n_ep,
                "raw_return_iqm": iqm([r["raw_return"] for r in rows]),
                "raw_return_mean": float(np.mean([r["raw_return"] for r in rows])),
                "raw_return_std": float(np.std([r["raw_return"] for r in rows])),
                "cost_return_iqm": iqm([r["cost_return"] for r in rows]),
                "n_actions_mean": float(np.mean([r["n_actions"] for r in rows])),
                "steps_mean": float(np.mean([r["steps"] for r in rows])),
                "solved_rate": float(np.mean([r["solved"] for r in rows])),
                "elapsed_sec": round(elapsed, 2),
                "csv": str((out_dir / "episodes.csv").relative_to(ROOT)).replace("\\", "/"),
            }
            summary.append(rec)
            print(
                f"[{spec['id']:<12} λ={lam}] r IQM {rec['raw_return_iqm']:8.2f} | "
                f"r' IQM {rec['cost_return_iqm']:8.2f} | 행동 {rec['n_actions_mean']:6.1f}회 | "
                f"성공률 {rec['solved_rate']*100:5.1f}% | {rec['elapsed_sec']}초"
            )

    sum_path = ROOT / "results" / f"summary_{cfg['name']}.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n요약 저장: {sum_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
