"""기준선 감사 — "비교 상대인 규칙을 성의 있게 만들었는가".

이 연구는 "학습이 단순 규칙을 이기는가"를 묻는다. 그렇다면 **그 규칙을 얼마나 잘 만들었는지가
결론을 좌우한다.** 대충 만든 규칙을 이기는 것은 쉽다.

그래서 환경마다 기준 규칙의 계수를 격자로 훑어 더 나은 것이 있는지 확인한다.

**튜닝은 평가에 쓰지 않는 에피소드에서 한다** (시드 900000~). 평가 시드(500000 + 시드×1000 + i)와
겹치지 않는다. 평가 집합에서 계수를 고르면 그 규칙의 점수가 부풀려진다 —
CLAUDE.md 절대 규칙 6이 막으라는 실수다. 튜닝셋에서 고른 하나만 평가셋으로 다시 잰다.

결과: results/aggregate/baseline_audit.json

실행: python -m src.analysis.baseline_audit
"""
from __future__ import annotations

import json
from pathlib import Path

from src.baselines.fixed_rules import (
    LunarLanderThresholdPolicy,
    MinAtarFreewayCautiousPolicy,
    MountainCarPumpPolicy,
)
from src.envs.cost_wrapper import make_cost_env
from src.eval.evaluate import evaluate, summarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "aggregate" / "baseline_audit.json"

TUNE_SEEDS = 80    # 시드 900000~ (평가에 쓰지 않는 구간)
EVAL_SEEDS = 100   # 시드 500000~ (본실험 평가와 같은 구간)


def sweep(env_id: str, current: dict, candidates: list[dict], make) -> dict:
    env = make_cost_env(env_id, lam=0.0)
    tune = [900_000 + i for i in range(TUNE_SEEDS)]
    ev = [500_000 + i for i in range(EVAL_SEEDS)]
    rows = []
    for params in candidates:
        st = summarize(evaluate(env, make(**params), tune))
        rows.append({"params": params, "tune_iqm": st["raw_return_iqm"]})
    best = max(rows, key=lambda r: r["tune_iqm"])
    # 튜닝셋에서 고른 하나와 현재 쓰는 것만 평가셋으로 잰다
    out = {}
    for tag, params in (("현재", current), ("튜닝셋 최고", best["params"])):
        se = summarize(evaluate(env, make(**params), ev))
        out[tag] = {"params": params, "eval_iqm": se["raw_return_iqm"],
                    "actions": se["n_actions_mean"], "solved": se["solved_rate"]}
    env.close()
    gap = out["튜닝셋 최고"]["eval_iqm"] - out["현재"]["eval_iqm"]
    out["차이"] = gap
    out["판정"] = ("현재 기준선이 이미 최선이다" if gap <= 1e-6
                 else f"더 나은 계수가 있다 (평가셋에서 {gap:+.1f})")
    out["후보들(튜닝셋)"] = rows
    return out


def main() -> None:
    audit = {
        "설명": ("기준 규칙의 계수를 격자로 훑어 더 나은 것이 있는지 확인한다. "
               "튜닝은 평가에 쓰지 않는 에피소드(시드 900000~)에서 하고, 고른 하나만 "
               "평가용 에피소드(시드 500000~)로 다시 잰다."),
        "왜": ("'학습이 단순 규칙을 이긴다'는 주장은 그 규칙을 얼마나 성의 있게 만들었는지를 "
              "밝혀야 의미가 있다. 대충 만든 규칙을 이기는 것은 쉽다."),
        "results": {},
    }
    specs = [
        ("MountainCar-v0", "pump 규칙",
         {"act_threshold": 0.0},
         [{"act_threshold": t} for t in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05)],
         MountainCarPumpPolicy),
        ("LunarLander-v3", "임계값 규칙",
         {"angle_thresh": 0.10, "vy_thresh": -0.35, "angle_gain": 0.5},
         [{"angle_thresh": a, "vy_thresh": v, "angle_gain": g}
          for a in (0.05, 0.10, 0.20) for v in (-0.25, -0.35, -0.50) for g in (0.5, 1.0)],
         LunarLanderThresholdPolicy),
        ("MinAtar/Freeway-v1", "신중 규칙",
         {"danger": 0},
         [{"danger": d} for d in (0, 1, 2, 3)],
         MinAtarFreewayCautiousPolicy),
    ]
    print(f"{'환경':<22}{'규칙':<12}{'현재 r IQM':>12}{'최고 r IQM':>12}{'차이':>9}  판정")
    for env_id, rule_ko, cur, cands, cls in specs:
        r = sweep(env_id, cur, cands, cls)
        audit["results"][env_id] = {"rule": rule_ko, **r}
        print(f"{env_id:<22}{rule_ko:<12}{r['현재']['eval_iqm']:12.2f}"
              f"{r['튜닝셋 최고']['eval_iqm']:12.2f}{r['차이']:9.1f}  {r['판정']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
