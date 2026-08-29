"""보상 조밀도를 실제로 잰다.

왜 필요한가:
    이 논문의 중심 설명은 "**보상이 조밀한지**가 λ*를 가른다"이다. 그런데 지금까지
    MountainCar를 '희소', LunarLander를 '조밀'이라고 **말로만** 불렀다.
    심사자가 "그 조밀도를 쟀는가"라고 물으면 답할 것이 없다.

무엇을 조밀도로 보나:
    "보상이 0이 아닌 스텝의 비율"은 이 연구에 맞지 않는다. MountainCar는 매 스텝 −1을 주므로
    그 기준으로는 100% 조밀해 보이지만, 그 −1은 **어느 상태에서나 같아** 아무 정보도 주지 않는다.

    그래서 **보상이 상태를 구분해 주는 정도**를 잰다.
      · 서로 다른 보상 값의 개수 — 1이면 보상이 상태를 전혀 구분하지 못한다
      · 최빈값이 아닌 보상을 주는 스텝의 비율 — 신호가 얼마나 자주 '달라지는가'
      · 스텝별 보상의 표준편차

    측정은 **각 환경의 기준 고정 규칙**으로 한다. 무작위 정책은 MountainCar에서 목표에
    한 번도 닿지 못해(도달률 0.0%, 별도 측정) 보상 분포를 대표하지 못하기 때문이다.

실행: python -m src.analysis.reward_density
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from src.baselines.fixed_rules import (LunarLanderThresholdPolicy, MinAtarFreewayCautiousPolicy,
                                       MountainCarPumpPolicy)
from src.envs.cost_wrapper import make_cost_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "aggregate" / "reward_density.json"
N_EPISODES = 60

SETUPS = [
    ("MountainCar-v0", "pump 규칙", lambda: MountainCarPumpPolicy()),
    ("LunarLander-v3", "임계값 규칙(튜닝)",
     lambda: LunarLanderThresholdPolicy(angle_thresh=0.05, vy_thresh=-0.25, angle_gain=1.0)),
    ("MinAtar/Freeway-v1", "신중 규칙", lambda: MinAtarFreewayCautiousPolicy(danger=0)),
]


def measure(env_id: str, make_pol) -> dict:
    env = make_cost_env(env_id, lam=0.0)
    pol = make_pol()
    rewards: list[float] = []
    ep_returns: list[float] = []
    for i in range(N_EPISODES):
        obs, _ = env.reset(seed=500_000 + i)
        if hasattr(pol, "reset"):
            pol.reset()
        done, tot, t = False, 0.0, 0
        while not done:
            a = pol.act(obs, t)
            obs, r, term, trunc, _ = env.step(a)
            rewards.append(float(r))
            tot += float(r)
            t += 1
            done = term or trunc
        ep_returns.append(tot)
    env.close()

    arr = np.asarray(rewards, dtype=float)
    cnt = collections.Counter(np.round(arr, 6).tolist())
    modal, modal_n = cnt.most_common(1)[0]
    return {
        "env_id": env_id,
        "n_episodes": N_EPISODES,
        "n_steps": int(arr.size),
        "distinct_rewards": int(len(cnt)),
        "modal_reward": float(modal),
        "informative_step_rate": float(1.0 - modal_n / arr.size),
        "reward_std_per_step": float(arr.std()),
        "episode_return_mean": float(np.mean(ep_returns)),
    }


def main() -> None:
    rows = []
    print("=" * 96)
    print("보상 조밀도 — 보상이 상태를 얼마나 구분해 주는가 (각 환경의 기준 고정 규칙으로 측정)")
    print("=" * 96)
    print(f"{'환경':<22}{'규칙':<18}{'서로 다른 보상 값':>16}{'신호 있는 스텝':>14}{'스텝 보상 표준편차':>18}")
    for env_id, rule_ko, mk in SETUPS:
        try:
            r = measure(env_id, mk)
        except Exception as e:
            print(f"{env_id:<22}{rule_ko:<18}  측정 실패: {type(e).__name__}: {e}")
            continue
        r["rule"] = rule_ko
        rows.append(r)
        print(f"{env_id:<22}{rule_ko:<18}{r['distinct_rewards']:>16}"
              f"{r['informative_step_rate'] * 100:>13.1f}%{r['reward_std_per_step']:>18.3f}")
    if not rows:
        return
    OUT.write_text(json.dumps({
        "설명": ("보상이 상태를 구분해 주는 정도. '보상이 0이 아닌 비율'은 쓰지 않는다 — "
                "MountainCar는 매 스텝 −1을 주지만 그 −1은 어느 상태에서나 같아 정보가 없다."),
        "측정": f"각 환경의 기준 고정 규칙으로 {N_EPISODES} 에피소드 (평가 시드 500000~)",
        "지표": {
            "distinct_rewards": "서로 다른 보상 값의 개수 (1이면 상태를 전혀 구분 못 한다)",
            "informative_step_rate": "최빈 보상이 아닌 보상을 받은 스텝의 비율",
            "reward_std_per_step": "스텝별 보상의 표준편차",
        },
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 96)
    print(f"저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
