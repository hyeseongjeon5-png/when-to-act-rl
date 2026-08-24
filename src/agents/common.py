"""학습 에이전트 공통 부품 — 신경망, 재생 버퍼, ε 스케줄.

세 계열(DQN · TempoRL · Lazy-MDP)이 같은 부품을 쓴다.
같은 부품을 써야 "계열 간 차이"가 구현 차이가 아니라 방법 차이가 된다
(CLAUDE.md 절대 규칙 2 — 공정 비교).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden=(128, 128)) -> nn.Sequential:
    """평범한 다층 퍼셉트론. 층 크기는 세 계열이 동일하게 쓴다."""
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class ReplayBuffer:
    """경험 재생 버퍼 — 겪은 일을 저장했다가 무작위로 꺼내 학습한다.

    fields: 추가로 저장할 항목 이름과 모양 (예: TempoRL의 지속 길이 j)
    """

    def __init__(self, capacity: int, obs_dim: int, extra: dict[str, tuple] | None = None):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.extra = {}
        for k, shape in (extra or {}).items():
            self.extra[k] = np.zeros((capacity, *shape) if shape else capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, **kw):
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        for k, v in kw.items():
            self.extra[k][i] = v
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.size, size=batch_size)
        out = {
            "obs": torch.as_tensor(self.obs[idx]),
            "action": torch.as_tensor(self.action[idx]),
            "reward": torch.as_tensor(self.reward[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
        }
        for k, arr in self.extra.items():
            out[k] = torch.as_tensor(arr[idx])
        return out

    def state_dict(self) -> dict:
        return {
            "obs": self.obs, "next_obs": self.next_obs, "action": self.action,
            "reward": self.reward, "done": self.done, "extra": self.extra,
            "ptr": self.ptr, "size": self.size,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.obs, self.next_obs = sd["obs"], sd["next_obs"]
        self.action, self.reward, self.done = sd["action"], sd["reward"], sd["done"]
        self.extra, self.ptr, self.size = sd["extra"], sd["ptr"], sd["size"]


def linear_eps(step: int, total: int, start: float = 1.0, end: float = 0.05, frac: float = 0.3) -> float:
    """ε(무작위로 행동할 확률)를 처음 frac 구간 동안 start→end로 직선 감소."""
    n = max(1, int(total * frac))
    if step >= n:
        return end
    return start + (end - start) * (step / n)
