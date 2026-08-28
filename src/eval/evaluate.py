"""평가 공통 함수 — 고정 규칙과 학습 에이전트가 '같은 자'로 재어진다.

모든 정책은 .act(obs, t) 인터페이스를 따르고, 있으면 .reset()이 에피소드마다 불린다.
평가는 항상 탐험을 끈 상태(greedy)로 한다 (CLAUDE.md 절대 규칙 2).
"""
from __future__ import annotations

import numpy as np


def run_episode(env, policy, seed: int) -> dict:
    if hasattr(policy, "reset"):
        policy.reset()
    obs, _ = env.reset(seed=int(seed))
    t, cost_return, terminated, truncated, info = 0, 0.0, False, False, {}
    while not (terminated or truncated):
        action = policy.act(obs, t)
        obs, reward, terminated, truncated, info = env.step(action)
        cost_return += reward
        t += 1
    return {
        "seed": int(seed),
        "steps": t,
        "raw_return": info["episode_raw_return"],
        "cost_return": cost_return,
        "n_actions": info["episode_actions"],           # 비용이 부과된 횟수
        # per_switch 방식에서만 위와 달라진다 (전환 횟수 vs 진짜 행동 횟수)
        "n_true_actions": info.get("episode_true_actions", info["episode_actions"]),
        "solved": int(bool(terminated)),
    }


def evaluate(env, policy, seeds) -> list[dict]:
    return [run_episode(env, policy, s) for s in seeds]


def iqm(x) -> float:
    """사분위평균 — 위아래 25%를 버리고 가운데 50%의 평균 (Agarwal et al. 2021)."""
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0:
        return float("nan")
    lo, hi = int(np.floor(len(x) * 0.25)), int(np.ceil(len(x) * 0.75))
    return float(np.mean(x[lo:hi]))


def summarize(rows: list[dict]) -> dict:
    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    return {
        "raw_return_iqm": iqm(g("raw_return")),
        "raw_return_mean": float(g("raw_return").mean()),
        "cost_return_iqm": iqm(g("cost_return")),
        "cost_return_mean": float(g("cost_return").mean()),
        "n_actions_mean": float(g("n_actions").mean()),
        "steps_mean": float(g("steps").mean()),
        "solved_rate": float(g("solved").mean()),
    }
