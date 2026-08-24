"""고정 규칙 기준선 (라) — 학습하지 않는 비교 대상 3종.

이 연구의 질문이 "학습이 이 단순한 규칙들을 언제 이기는가?"이므로,
학습 에이전트보다 이 파일이 먼저 완성·검증되어야 한다 (CLAUDE.md 절대 규칙 5).

공통 인터페이스: policy.act(obs, t) -> action
  obs : 관측값, t : 에피소드 내 스텝 번호(0부터)
"""
from __future__ import annotations


class NoOpPolicy:
    """무행동 — 항상 no-op. λ가 충분히 크면 이것이 최적이 된다(실험의 한쪽 끝)."""

    def __init__(self, noop_action: int):
        self.noop_action = noop_action

    def act(self, obs, t: int) -> int:
        return self.noop_action


class PeriodicPolicy:
    """k스텝 주기 행동 — k스텝마다 1번 기저 행동을 내고 나머지는 no-op.

    base_action을 정하는 법: λ=0에서 가장 점수가 좋은 단일 행동을 파일럿으로 고른다.
    k=1이면 '매 스텝 행동'이 된다.
    """

    def __init__(self, base_action: int, noop_action: int, k: int = 4):
        assert k >= 1
        self.base_action = base_action
        self.noop_action = noop_action
        self.k = k

    def act(self, obs, t: int) -> int:
        return self.base_action if t % self.k == 0 else self.noop_action


class MountainCarPumpPolicy:
    """MountainCar 임계값 규칙 — '그네 굴리기'.

    속도가 오른쪽(+)이면 오른쪽 가속(2), 왼쪽(−)이면 왼쪽 가속(0).
    속도 절대값이 임계값 이하일 때만 행동하는 변형도 실험 가능(act_threshold).
    이 규칙은 MountainCar를 학습 없이 풀 수 있어서 강력한 기준선이다.
    """

    RIGHT, NOOP, LEFT = 2, 1, 0

    def __init__(self, act_threshold: float = 0.0):
        # act_threshold > 0 이면 |속도|가 그보다 클 때 no-op (관성 활용, 행동 절약)
        self.act_threshold = act_threshold

    def act(self, obs, t: int) -> int:
        velocity = obs[1]
        if self.act_threshold > 0 and abs(velocity) > self.act_threshold:
            return self.NOOP
        return self.RIGHT if velocity >= 0 else self.LEFT


# LunarLander 임계값 규칙은 파일럿에서 정의 확정 후 추가 (docs/02_실험-설계.md §2)
