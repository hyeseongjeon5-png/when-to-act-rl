"""무작위 탐험의 목표 도달률 — "이 환경에서 무작위로 움직이면 목표를 볼 수 있는가".

이 숫자가 이 연구의 여러 주장을 떠받친다.
  · MountainCar에서 학습이 규칙에 지는 이유 (탐험이 목표를 못 본다)
  · 무행동 붕괴가 '탐험 실패'라는 해석
  · ε를 크게 감소시키는 흔한 방식이 여기서는 오히려 해로운 이유

그런데 이 값이 결과 파일에 없이 원고에만 적혀 있었다. 숫자는 로그 파일에서만 인용한다는
규칙에 어긋나므로 여기서 다시 재어 저장한다.

'같은 행동 j스텝 유지'를 함께 재는 이유: TempoRL의 주장("행동을 오래 유지하면 탐험이 좋아진다")이
우리 환경에서도 성립하는지 보기 위해서다. 이 표가 Ⅲ장의 환경 설명 근거가 된다.

실행: python -m src.analysis.exploration_reach
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.envs.cost_wrapper import make_cost_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "aggregate" / "exploration_reach.json"

# (환경, 에피소드 수, 유지 길이 후보, terminated가 뜻하는 것)
#
# **중요**: terminated는 환경마다 뜻이 다르다. 같은 이름이라고 나란히 놓으면 안 된다.
#   MountainCar : 목표(깃발)에 닿아서 끝남 → '목표 도달률'로 읽어도 된다
#   LunarLander : 착륙 **또는 추락**으로 끝남 → 성공률이 아니다. 무작위 정책은 거의 항상 추락한다
#   Freeway     : 종료 조건이 없다(1000스텝 제한으로만 끝난다) → 이 지표는 의미가 없다
# 그래서 '탐험이 목표를 보는가'라는 질문에 답하는 것은 MountainCar 하나뿐이다.
SETUPS = [
    ("MountainCar-v0", 500, [1, 2, 4, 8, 16, 32], "목표 도달 (깃발에 닿음)"),
    ("LunarLander-v3", 300, [1, 4, 16], "착륙 또는 추락으로 종료 — 성공률이 아니다"),
    ("MinAtar/Freeway-v1", 300, [1, 4, 16], "종료 조건 없음 — 이 지표는 의미 없다"),
]


def reach_rate(env_id: str, hold: int, n_episodes: int, seed0: int = 900_000) -> dict:
    """무작위 행동을 hold 스텝씩 유지하며 에피소드를 돌고, '종료(terminated)' 비율을 센다.

    terminated는 시간 초과(truncated)가 아닌 환경 자체의 종료 조건으로 끝난 것을 뜻한다.
    **그 조건이 무엇인지는 환경마다 다르다** — SETUPS의 설명을 함께 볼 것.
    MountainCar에서만 '목표 도달'과 같은 뜻이다.
    """
    env = make_cost_env(env_id, lam=0.0)
    n_act = int(env.action_space.n)
    reached = 0
    steps_total = 0
    for i in range(n_episodes):
        rng = np.random.default_rng(seed0 + i)
        env.reset(seed=seed0 + i)
        done = trunc = False
        left, a = 0, 0
        while not (done or trunc):
            if left <= 0:
                a = int(rng.integers(n_act))
                left = hold
            left -= 1
            _, _, done, trunc, _ = env.step(a)
            steps_total += 1
        reached += int(done)
    env.close()
    return {"hold": hold, "episodes": n_episodes, "reached": reached,
            "reach_rate": reached / n_episodes, "steps_mean": steps_total / n_episodes}


def main() -> None:
    out = {
        "설명": ("무작위 정책이 에피소드를 '종료'로 끝내는 비율. '유지 길이'는 같은 무작위 행동을 "
               "몇 스텝 이어서 내는지를 뜻한다. 시간 초과(truncated)는 세지 않는다."),
        "주의": ("terminated의 뜻이 환경마다 다르다. MountainCar만 '목표 도달'로 읽을 수 있고, "
               "LunarLander는 착륙과 추락을 구분하지 않으며, Freeway는 종료 조건이 없어 항상 0이다. "
               "세 값을 나란히 비교하면 안 된다."),
        "평가": "평가 시드 900000~ (학습·최종 평가와 겹치지 않는 구간)",
        "results": [],
    }
    print(f"{'환경':<22}{'유지 길이':>9}{'에피소드':>9}{'종료 비율':>10}{'평균 길이':>10}  terminated의 뜻")
    for env_id, n_ep, holds, meaning in SETUPS:
        for h in holds:
            r = reach_rate(env_id, h, n_ep)
            out["results"].append({"env_id": env_id, "terminated_meaning": meaning, **r})
            print(f"{env_id:<22}{h:>9}{n_ep:>9}{r['reach_rate'] * 100:9.1f}%{r['steps_mean']:10.1f}"
                  f"  {meaning if h == holds[0] else ''}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")
    print(chr(10) + "주의: terminated의 뜻이 환경마다 다르다 — MountainCar만 '목표 도달'로 읽을 수 있다.")
    mc = [r for r in out["results"] if r["env_id"] == "MountainCar-v0"]
    if mc:
        one = next((r for r in mc if r["hold"] == 1), None)
        best = max(mc, key=lambda r: r["reach_rate"])
        if one is not None:
            print(f"읽는 법: MountainCar에서 매 스텝 새 행동을 뽑으면 도달률 {one['reach_rate'] * 100:.1f}%, "
                  f"같은 행동을 {best['hold']}스텝 유지하면 {best['reach_rate'] * 100:.1f}%로 오른다. "
                  f"막히는 것은 시간이 아니라 탐험 방식이다.")


if __name__ == "__main__":
    main()
