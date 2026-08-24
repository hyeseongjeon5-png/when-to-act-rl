"""고정 규칙 기준선 (라) — 학습하지 않는 비교 대상 3종.

이 연구의 질문이 "학습이 이 단순한 규칙들을 언제 이기는가?"이므로,
학습 에이전트보다 이 파일이 먼저 완성·검증되어야 한다 (CLAUDE.md 절대 규칙 5).

공통 인터페이스: policy.act(obs, t) -> action
  obs : 관측값, t : 에피소드 내 스텝 번호(0부터)
"""
from __future__ import annotations

import numpy as np


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


class LunarLanderThresholdPolicy:
    """LunarLander 임계값 규칙 — "기울면 바로잡고, 너무 빨리 떨어지면 주엔진".

    docs/02_실험-설계.md §2의 "각도·속도 임계값 초과 시에만 엔진"을 수식으로 고정한 것.
    관측: [x, y, vx, vy, angle, angular_vel, leg_left, leg_right]
    행동: 0 무행동 / 1 왼쪽엔진 / 2 주엔진 / 3 오른쪽엔진

    규칙 (위에서부터 먼저 걸리는 것 하나만 실행):
      1) 두 다리가 모두 땅에 닿았으면 → 무행동 (착륙 완료, 연료 낭비 금지)
      2) |기울기 보정량| 이 임계값을 넘으면 → 자세 보정용 옆엔진
      3) 낙하 속도가 임계값보다 빠르면 → 주엔진
      4) 그 외 → 무행동
    """

    NOOP, LEFT, MAIN, RIGHT = 0, 1, 2, 3

    def __init__(self, angle_thresh: float = 0.10, vy_thresh: float = -0.35, angle_gain: float = 0.5):
        self.angle_thresh = angle_thresh
        self.vy_thresh = vy_thresh
        self.angle_gain = angle_gain

    def act(self, obs, t: int) -> int:
        x, y, vx, vy, ang, ang_v = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4]), float(obs[5])
        legs = float(obs[6]) + float(obs[7])
        if legs >= 2.0:
            return self.NOOP
        # 목표 기울기: 중앙에서 벗어난 만큼 반대로 기울여 되돌아오게 한다
        target_ang = np.clip(0.5 * x + 1.0 * vx, -0.4, 0.4)
        err = (target_ang - ang) * self.angle_gain - 0.5 * ang_v
        if abs(err) > self.angle_thresh:
            return self.RIGHT if err < 0 else self.LEFT
        if vy < self.vy_thresh:
            return self.MAIN
        return self.NOOP
