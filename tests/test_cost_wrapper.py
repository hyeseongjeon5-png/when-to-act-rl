"""비용 래퍼 자동 테스트 — 이슈 #2.

이 연구의 모든 숫자가 이 래퍼 위에 얹혀 있으므로, 여기가 틀리면 전부 틀린다.
실행: python -m pytest tests -q   (또는 python tests/test_cost_wrapper.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 프로젝트 루트를 import 경로에 추가

import gymnasium as gym
import numpy as np

from src.baselines.fixed_rules import MountainCarPumpPolicy, NoOpPolicy, PeriodicPolicy
from src.envs.cost_wrapper import NOOP_BY_ENV, make_cost_env


def _rollout(env, policy, seed=0):
    obs, _ = env.reset(seed=seed)
    t, ret, done, info = 0, 0.0, False, {}
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs, t))
        ret += r
        done = term or trunc
        t += 1
    return ret, info, t


def test_lambda_zero_matches_original():
    """λ=0이면 원래 환경과 보상이 완전히 같아야 한다."""
    plain = gym.make("MountainCar-v0")
    wrapped = make_cost_env("MountainCar-v0", lam=0.0)
    pol = MountainCarPumpPolicy()
    obs, _ = plain.reset(seed=42)
    t, ret_plain, done = 0, 0.0, False
    while not done:
        obs, r, term, trunc, _ = plain.step(pol.act(obs, t))
        ret_plain += r; done = term or trunc; t += 1
    ret_wrapped, _, _ = _rollout(wrapped, MountainCarPumpPolicy(), seed=42)
    assert ret_plain == ret_wrapped, f"λ=0인데 다르다: 원본 {ret_plain} vs 래퍼 {ret_wrapped}"


def test_cost_formula_exact():
    """r' = r − λ × (no-op 아닌 행동 횟수) 가 정확히 성립해야 한다."""
    for lam in (0.05, 0.2, 0.66):
        env = make_cost_env("MountainCar-v0", lam=lam)
        ret, info, _ = _rollout(env, MountainCarPumpPolicy(), seed=7)
        expected = info["episode_raw_return"] - lam * info["episode_actions"]
        assert abs(ret - expected) < 1e-6, f"λ={lam}: {ret} ≠ {expected}"


def test_noop_is_free():
    """no-op에는 비용이 붙지 않아야 한다 — 이 연구의 핵심 가정."""
    env = make_cost_env("MountainCar-v0", lam=1.0)
    ret, info, steps = _rollout(env, NoOpPolicy(NOOP_BY_ENV["MountainCar-v0"]), seed=3)
    assert info["episode_actions"] == 0, "무행동인데 행동 횟수가 0이 아니다"
    assert ret == info["episode_raw_return"], "무행동인데 비용이 붙었다"
    assert steps == 200, "MountainCar 시간제한이 200스텝이 아니다"


def test_action_count_matches_rule():
    """k스텝 주기 규칙의 행동 횟수가 이론값(스텝수/k)과 맞아야 한다."""
    for k in (1, 2, 4, 8):
        env = make_cost_env("MountainCar-v0", lam=0.1)
        _, info, steps = _rollout(env, PeriodicPolicy(2, NOOP_BY_ENV["MountainCar-v0"], k=k), seed=1)
        assert info["episode_actions"] == int(np.ceil(steps / k)), \
            f"k={k}: 행동 {info['episode_actions']}회 ≠ 예상 {int(np.ceil(steps / k))}회"


def test_lunarlander_noop_is_zero_action():
    """LunarLander의 no-op 번호(0)가 맞게 등록돼 있는가."""
    env = make_cost_env("LunarLander-v3", lam=0.5)
    _, info, _ = _rollout(env, NoOpPolicy(0), seed=0)
    assert info["episode_actions"] == 0


def test_lambda_monotonic():
    """같은 정책이면 λ가 커질수록 비용 반영 보상은 작아지거나 같아야 한다."""
    prev = None
    for lam in (0.0, 0.1, 0.3, 0.66):
        env = make_cost_env("MountainCar-v0", lam=lam)
        ret, _, _ = _rollout(env, MountainCarPumpPolicy(), seed=11)
        if prev is not None:
            assert ret <= prev + 1e-9, f"λ={lam}에서 보상이 오히려 늘었다"
        prev = ret


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  통과: {fn.__name__}")
    print(f"\n{len(fns)}개 테스트 모두 통과")
