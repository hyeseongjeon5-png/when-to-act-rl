"""비용 부과 방식 민감도 — 고정 규칙의 '전환 횟수'를 재어 λ 눈금이 얼마나 달라지는지 본다.

이 연구는 행동이 실행되는 **매 스텝** 비용을 문다(작동 비용). 대안은 행동이 **바뀔 때만**
무는 것이다(전환 비용). 학습까지 다시 돌리지 않아도, **고정 규칙의 전환 횟수만 재면**
"비용 모형을 바꾸면 λ 눈금이 얼마나 달라지는가"를 정확히 말할 수 있다.

무행동이 최고 규칙을 이기기 시작하는 λ는 다음 한 줄로 정해진다.
    λ_교차 = (규칙 점수 − 무행동 점수) / (규칙이 내는 과금 횟수)
과금 횟수가 방식에 따라 달라지므로 λ 눈금도 달라진다. 규칙은 학습이 없어 결정적이라
이 계산은 근사가 아니다.

결과는 results/aggregate/switch_cost_rules.json 에 저장한다 — 논문이 인용할 수 있게.

실행: python -m src.analysis.switch_cost_rules
"""
from __future__ import annotations

import json
from pathlib import Path

from src.baselines.fixed_rules import (
    LunarLanderThresholdPolicy,
    MinAtarFreewayCautiousPolicy,
    MountainCarPumpPolicy,
    NoOpPolicy,
)
from src.envs.cost_wrapper import make_cost_env
from src.eval.evaluate import evaluate, summarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "aggregate" / "switch_cost_rules.json"
N_EPISODES = 100

SETUPS = [
    ("MountainCar-v0", "pump 규칙", lambda: MountainCarPumpPolicy(), lambda: NoOpPolicy(1)),
    ("LunarLander-v3", "임계값 규칙", lambda: LunarLanderThresholdPolicy(), lambda: NoOpPolicy(0)),
    ("MinAtar/Freeway-v1", "신중 규칙", lambda: MinAtarFreewayCautiousPolicy(danger=0), lambda: NoOpPolicy(0)),
]


def measure(env_id: str, make_pol, mode: str) -> dict:
    env = make_cost_env(env_id, lam=0.0, cost_mode=mode)
    seeds = [500_000 + i for i in range(N_EPISODES)]
    s = summarize(evaluate(env, make_pol(), seeds))
    env.close()
    return {"raw_return_iqm": s["raw_return_iqm"], "charged_mean": s["n_actions_mean"],
            "steps_mean": s["steps_mean"]}


def main() -> None:
    out = {
        "설명": ("고정 규칙의 과금 횟수를 두 비용 방식으로 재고, 무행동이 규칙을 이기기 시작하는 "
               "λ를 계산했다. λ_교차 = (규칙 점수 − 무행동 점수) / 규칙의 과금 횟수. "
               "규칙은 학습이 없어 결정적이므로 이 값은 근사가 아니다."),
        "평가": f"각 조건 {N_EPISODES} 에피소드, 평가 시드 500000~",
        "results": [],
    }
    print(f"{'환경':<22}{'규칙':<12}{'r IQM':>8}{'매스텝 과금':>11}{'전환 과금':>10}"
          f"{'λ교차(매스텝)':>14}{'λ교차(전환)':>13}{'배율':>7}")
    for env_id, rule_ko, mk_rule, mk_noop in SETUPS:
        row = {"env_id": env_id, "rule": rule_ko}
        for mode in ("per_step", "per_switch"):
            r = measure(env_id, mk_rule, mode)
            n = measure(env_id, mk_noop, mode)
            cross = ((r["raw_return_iqm"] - n["raw_return_iqm"]) / r["charged_mean"]
                     if r["charged_mean"] else None)
            row[mode] = {**r, "noop_raw_return_iqm": n["raw_return_iqm"], "lam_cross": cross}
        a, b = row["per_step"], row["per_switch"]
        ratio = (b["lam_cross"] / a["lam_cross"]) if (a["lam_cross"] and b["lam_cross"]) else None
        row["lam_cross_ratio"] = ratio
        out["results"].append(row)
        print(f"{env_id:<22}{rule_ko:<12}{a['raw_return_iqm']:8.2f}{a['charged_mean']:11.1f}"
              f"{b['charged_mean']:10.1f}{(a['lam_cross'] or 0):14.4f}{(b['lam_cross'] or 0):13.4f}"
              f"{(ratio or 0):7.1f}배")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")
    print("읽는 법: '전환 과금'이 '매스텝 과금'보다 훨씬 적은 규칙일수록, 비용 모형을 바꿨을 때"
          " 그 규칙이 유리해지고 λ 눈금이 커진다.")


if __name__ == "__main__":
    main()
