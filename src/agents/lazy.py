"""(다) Lazy-MDP 방식 — "언제 끼어들지"를 배운다.

Jacq et al., "Lazy-MDPs: Towards Interpretable RL by Learning When to Act" (AAMAS 2022).

원 논문의 뼈대: 행동 집합에 '기본 정책에 맡기기(defer)' 라는 선택지를 하나 더 붙이고,
직접 개입할 때만 벌점 η를 물린다. 배운 결과를 보면 "언제 끼어들었는가"가
곧 "그 상태가 중요한 상태였는가"가 되어 해석이 쉬워진다.

우리 실험에서의 각색:
  - 기본 정책 = 그 환경에서 가장 센 고정 규칙 (MountainCar는 pump, IQM −120.5)
  - 개입 벌점 η는 기본값 0. 우리는 이미 비용 λ를 '실행된 행동'에 물리고 있어
    개입 압력이 λ로 들어오기 때문이다. η는 config로 켤 수 있게만 남긴다.
  - 즉 이 에이전트가 λ를 아끼는 길은 '규칙 대신 no-op을 직접 고르는 것'이다.
"""
from __future__ import annotations

import numpy as np
import torch

from src.agents.common import ReplayBuffer, mlp
from src.agents.dqn import DQNAgent
from src.baselines.fixed_rules import (
    LunarLanderThresholdPolicy,
    MinAtarBreakoutTrackPolicy,
    MinAtarFreewayCautiousPolicy,
    MountainCarPumpPolicy,
    NoOpPolicy,
)
from src.envs.cost_wrapper import NOOP_BY_ENV

DEFAULT_BASE = {
    "MountainCar-v0": "pump",
    "LunarLander-v3": "threshold",
    "LunarLander-v2": "threshold",
    "MinAtar/Freeway-v1": "freeway_cautious",
    "MinAtar/Breakout-v1": "breakout_track",
}


def make_base_policy(name: str, env_id: str):
    if name == "pump":
        return MountainCarPumpPolicy()
    if name == "threshold":
        return LunarLanderThresholdPolicy()
    if name == "threshold_tuned":
        # 평가에 쓰지 않는 에피소드로 계수를 고른 강한 변형 (r IQM 38.3 → 179.8)
        return LunarLanderThresholdPolicy(angle_thresh=0.05, vy_thresh=-0.25, angle_gain=1.0)
    if name == "freeway_cautious":
        # danger=0 — 사전 측정에서 가장 센 설정(r IQM 17.80, 행동 298.4회).
        # 기본 정책은 '그 환경에서 가장 센 고정 규칙'이어야 한다. 기준 규칙과 다른 값을 쓰면
        # Lazy가 위임해도 기준선에 못 미쳐 비교가 어긋난다.
        return MinAtarFreewayCautiousPolicy(danger=0)
    if name == "breakout_track":
        return MinAtarBreakoutTrackPolicy(tol=0)
    if name == "noop":
        return NoOpPolicy(NOOP_BY_ENV[env_id])
    raise KeyError(f"모르는 기본 정책: {name}")


class LazyEvalPolicy:
    def __init__(self, agent):
        self.agent = agent
        self.reset()

    def reset(self):
        if hasattr(self.agent.base_policy, "reset"):
            self.agent.base_policy.reset()

    def act(self, obs, t: int) -> int:
        aug = int(np.argmax(self.agent.q_values(obs)))
        return self.agent.to_env_action(aug, obs, t)


class LazyAgent(DQNAgent):
    """행동 집합 = {0: 기본 정책에 맡기기} ∪ {1..n: 직접 행동 a−1}"""

    name = "lazy"

    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, seed: int, env_id: str = "MountainCar-v0"):
        self.n_env_actions = n_actions
        super().__init__(obs_dim, n_actions + 1, cfg, seed)  # 선택지 1개 추가
        base_name = cfg.get("base_policy", DEFAULT_BASE.get(env_id, "noop"))
        self.base_policy = make_base_policy(base_name, env_id)
        self.base_name = base_name
        self.eta = float(cfg.get("eta", 0.0))  # 개입 벌점 (원 논문의 η). 기본 0
        self.n_defer = 0
        self.n_decisions = 0

    def to_env_action(self, aug_action: int, obs, t: int) -> int:
        """증강 행동 → 실제 환경 행동.

        **기본 정책은 위임하지 않는 스텝에도 항상 호출한다.** 내부 상태를 가진 규칙
        (예: MinAtar Freeway 신중 규칙은 '내가 방금 움직였는가'로 이동 대기시간을 센다)은
        한 스텝이라도 건너뛰면 상태가 어긋나 엉뚱한 행동을 낸다.
        호출만 하고 결과를 버리면 상태는 항상 실제 환경을 따라간다.
        (pump·임계값처럼 상태가 없는 규칙에서는 결과가 달라지지 않는다 — 기존 결과 불변)
        """
        base = int(self.base_policy.act(obs, t))
        return base if aug_action == 0 else int(aug_action - 1)

    def eval_policy(self):
        return LazyEvalPolicy(self)

    def begin_episode(self, obs):
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()

    def interact(self, env, obs, t: int, global_step: int):
        """한 스텝 진행한다. 다만 행동 집합에 **"기본 정책에 맡기기"** 가 하나 더 있다.

        확장된 행동 집합은 [직접 낼 행동 0 … n-1] + [위임] 이고, 마지막 하나를 고르면
        그 스텝의 행동을 사람이 만든 규칙(기본 정책)이 대신 정한다.

        **중요: 기본 정책은 위임하지 않는 스텝에도 매번 호출한다.**
        기본 정책이 상태를 가지면(예: Freeway의 신중 규칙은 이전 프레임을 본다) 건너뛴 스텝만큼
        내부 상태가 어긋나, 정작 위임할 때 엉뚱한 행동이 나온다.
        2026-08-28에 실제로 이 버그로 90스텝 중 26스텝이 틀렸다.

        개입 벌점 η는 이 연구에서 **0으로 둔다**. 원 논문[2]의 η는 해석 가능성을 조절하는
        손잡이지만, 이 연구가 재려는 것은 **환경이 물리는 행동 비용 λ**다. 둘을 함께 켜면
        어느 쪽이 행동을 줄였는지 가릴 수 없다.
        """
        if self.rng.random() < self.eps(global_step):
            aug = int(self.rng.integers(self.n_actions))
        else:
            aug = int(np.argmax(self.q_values(obs)))
        env_action = self.to_env_action(aug, obs, t)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        shaped = float(reward) - (0.0 if aug == 0 else self.eta)
        self.buffer.add(obs, aug, shaped, next_obs, float(terminated))
        self.n_defer += int(aug == 0)
        self.n_decisions += 1
        return next_obs, 1, bool(terminated or truncated), info

    def state_dict(self, with_buffer: bool = True) -> dict:
        sd = super().state_dict(with_buffer)
        sd["n_defer"], sd["n_decisions"] = self.n_defer, self.n_decisions
        return sd

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.n_defer = sd.get("n_defer", 0)
        self.n_decisions = sd.get("n_decisions", 0)
