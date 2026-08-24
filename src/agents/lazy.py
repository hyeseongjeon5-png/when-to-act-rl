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
    MountainCarPumpPolicy,
    NoOpPolicy,
)
from src.envs.cost_wrapper import NOOP_BY_ENV

DEFAULT_BASE = {
    "MountainCar-v0": "pump",
    "LunarLander-v3": "threshold",
    "LunarLander-v2": "threshold",
}


def make_base_policy(name: str, env_id: str):
    if name == "pump":
        return MountainCarPumpPolicy()
    if name == "threshold":
        return LunarLanderThresholdPolicy()
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
        if aug_action == 0:
            return int(self.base_policy.act(obs, t))
        return int(aug_action - 1)

    def eval_policy(self):
        return LazyEvalPolicy(self)

    def begin_episode(self, obs):
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()

    def interact(self, env, obs, t: int, global_step: int):
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
