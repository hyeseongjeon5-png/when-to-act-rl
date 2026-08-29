"""MinAtar 어댑터 자동 테스트.

왜 필요한가: 이 연구는 MinAtar에서 180조건(약 17시간)을 돌린다. 어댑터가 틀리면
그 시간이 전부 헛것이 된다. 특히 **0번 행동이 정말 no-op인가**가 틀리면 비용 회계가
통째로 어긋난다 — 이 논문의 모든 숫자가 "no-op은 공짜"라는 전제 위에 있다.

기존 test_cost_wrapper.py는 MountainCar·LunarLander만 다뤘다 (2026-08-29 확인).

실행: python tests/test_minatar_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.envs.cost_wrapper import NOOP_BY_ENV, make_cost_env
from src.envs import minatar_env  # noqa: F401  (import만 해도 Gymnasium에 등록된다)

ENV_ID = "MinAtar/Freeway-v1"


def test_registered_and_flat_obs():
    """관측이 (10,10,채널)이 아니라 1차원으로 펴져 나온다."""
    env = make_cost_env(ENV_ID, lam=0.0)
    obs, _ = env.reset(seed=0)
    assert np.asarray(obs).ndim == 1, f"관측이 1차원이 아니다: {np.asarray(obs).shape}"
    assert env.observation_space.shape == np.asarray(obs).shape
    env.close()


def test_action_zero_is_noop():
    """0번 행동이 정말 no-op인가 — 비용 회계 전체가 이 전제 위에 있다.

    확인 방법: 0번만 계속 내면 비용이 한 푼도 안 붙어야 한다.
    """
    lam = 1.0
    env = make_cost_env(ENV_ID, lam=lam)
    env.reset(seed=0)
    total_cost = 0.0
    for _ in range(200):
        _, r, term, trunc, info = env.step(0)
        total_cost += float(info.get("action_cost", 0.0))
        if term or trunc:
            break
    assert total_cost == 0.0, f"0번 행동에 비용이 붙었다: {total_cost}"
    assert NOOP_BY_ENV[ENV_ID] == 0
    env.close()


def test_nonzero_action_charged_once():
    """0이 아닌 행동은 1회당 정확히 λ만큼 물린다."""
    lam = 0.25
    env = make_cost_env(ENV_ID, lam=lam)
    env.reset(seed=0)
    n_act, charged = 0, 0.0
    for i in range(120):
        a = 1 if i % 2 else 0          # 번갈아 낸다
        _, r, term, trunc, info = env.step(a)
        if a != 0:
            n_act += 1
        charged += float(info.get("action_cost", 0.0))
        if term or trunc:
            break
    assert abs(charged - n_act * lam) < 1e-9, f"과금이 λ×횟수와 다르다: {charged} vs {n_act * lam}"
    env.close()


def test_sticky_actions_off():
    """끈적임이 켜져 있으면 '낸 행동'과 '실행된 행동'이 어긋나 비용 회계가 흐려진다.

    확인 방법: 같은 시드로 같은 행동 열을 두 번 돌려 관측이 완전히 같아야 한다
    (끈적임은 확률적이므로 켜져 있으면 어긋날 수 있다).
    """
    seq = [0, 1, 1, 0, 2, 1, 0, 0, 1, 2] * 8

    def run():
        env = make_cost_env(ENV_ID, lam=0.0)
        obs, _ = env.reset(seed=123)
        trail = [np.asarray(obs).sum()]
        for a in seq:
            obs, r, term, trunc, _ = env.step(a % env.action_space.n)
            trail.append(float(np.asarray(obs).sum()) + float(r))
            if term or trunc:
                break
        env.close()
        return trail

    assert run() == run(), "같은 시드·같은 행동인데 결과가 달랐다 — 끈적임이 켜져 있을 수 있다"


def test_minimal_action_set():
    """행동 수가 MinAtar 전체(6개)가 아니라 게임에 필요한 만큼만이어야 한다.

    쓸모없는 행동을 남기면 그 행동에도 λ가 붙어 비교가 흐려진다.
    """
    env = make_cost_env(ENV_ID, lam=0.0)
    n = env.action_space.n
    assert 2 <= n <= 6, f"행동 수가 이상하다: {n}"
    assert n < 6, f"최소 행동 집합이 적용되지 않았다 (전체 6개 그대로): {n}"
    env.close()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for t in tests:
        try:
            t()
            print(f"  통과: {t.__name__}")
        except AssertionError as e:
            print(f"  실패: {t.__name__} — {e}")
            bad += 1
        except Exception as e:
            print(f"  오류: {t.__name__} — {type(e).__name__}: {e}")
            bad += 1
    print()
    print(f"{len(tests)}개 중 {len(tests) - bad}개 통과" if bad else f"{len(tests)}개 테스트 모두 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
