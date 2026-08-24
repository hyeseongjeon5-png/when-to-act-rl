"""(나) TempoRL 방식 — "무엇을 할지"와 "얼마나 오래 할지"를 함께 배운다.

Biedenkapp et al., "TempoRL: Learning When to Act" (ICML 2021)의 TempoRL-DQN 구조를
automl/TempoRL 공개 코드 기준으로 우리 비용 래퍼에 이식한 것.

구조 (원 논문 그림 2):
  1) 행동망  Q(s, a)        — 어떤 행동을 할지 (평범한 DQN)
  2) 지속망  Q_skip(s, a, j) — 그 행동을 몇 스텝(j=1..J) 유지할지
  갱신식:  Q_skip(s,a,j) ← Σ_{i<j} γ^i r_{t+i} + γ^j max_{a'} Q(s_{t+j}, a')

우리 실험에서 중요한 점: 비용 λ는 '실제로 실행된 no-op 아닌 스텝'마다 붙는다.
따라서 TempoRL이 비용을 아끼는 길은 'no-op을 길게 유지하는 것'(관성으로 미끄러지기)이다.
(결정 시점에만 1번 부과하는 방식은 민감도 분석 항목으로 남긴다 — docs/02_실험-설계.md §1)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.common import ReplayBuffer, linear_eps, mlp
from src.agents.dqn import DQNAgent


class TempoRLEvalPolicy:
    """평가용 — 결정한 행동을 j스텝 동안 유지한다 (고정 규칙과 같은 .act 인터페이스)."""

    def __init__(self, agent):
        self.agent = agent
        self.reset()

    def reset(self):
        self.left = 0
        self.action = None

    def act(self, obs, t: int) -> int:
        if self.left <= 0:
            self.action = int(np.argmax(self.agent.q_values(obs)))
            self.left = int(np.argmax(self.agent.skip_values(obs, self.action))) + 1
        self.left -= 1
        return self.action


class TempoRLAgent(DQNAgent):
    name = "temporl"

    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, seed: int):
        super().__init__(obs_dim, n_actions, cfg, seed)
        self.max_skip = int(cfg.get("max_skip", 10))
        self.skip_augment = bool(cfg.get("skip_augment", True))
        hidden = tuple(cfg.get("hidden", (128, 128)))

        self.q_skip = mlp(obs_dim + n_actions, self.max_skip, hidden)
        self.q_skip_target = mlp(obs_dim + n_actions, self.max_skip, hidden)
        self.q_skip_target.load_state_dict(self.q_skip.state_dict())
        self.opt_skip = torch.optim.Adam(self.q_skip.parameters(), lr=float(cfg.get("lr_skip", cfg.get("lr", 5e-4))))
        # 지속망 버퍼: (s, a, j) → j스텝 누적보상, j스텝 뒤 상태
        self.skip_buffer = ReplayBuffer(int(cfg.get("buffer_size", 100_000)), obs_dim,
                                        extra={"j": (), "behaviour_action": ()})

    # ---------- 지속망 ----------
    def _sa(self, obs, action) -> torch.Tensor:
        onehot = np.zeros(self.n_actions, dtype=np.float32)
        onehot[action] = 1.0
        return torch.as_tensor(np.concatenate([np.asarray(obs, dtype=np.float32).ravel(), onehot]))[None]

    def skip_values(self, obs, action) -> np.ndarray:
        with torch.no_grad():
            return self.q_skip(self._sa(obs, action)).numpy()[0]

    def _sa_batch(self, obs_b: torch.Tensor, act_b: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(act_b.long(), self.n_actions).float()
        return torch.cat([obs_b, onehot], dim=1)

    def eval_policy(self):
        return TempoRLEvalPolicy(self)

    # ---------- 상호작용: 한 번 결정하고 j스텝 실행 ----------
    def interact(self, env, obs, t: int, global_step: int):
        eps = self.eps(global_step)
        if self.rng.random() < eps:
            action = int(self.rng.integers(self.n_actions))
        else:
            action = int(np.argmax(self.q_values(obs)))
        if self.rng.random() < eps:
            j = int(self.rng.integers(self.max_skip)) + 1
        else:
            j = int(np.argmax(self.skip_values(obs, action))) + 1

        s0 = np.asarray(obs, dtype=np.float32).ravel().copy()
        cur = obs
        disc_sum, used, done, terminated, info = 0.0, 0, False, False, {}
        traj = []  # (부분 누적보상, 그 시점의 상태, terminated)
        for i in range(j):
            nxt, reward, term, trunc, info = env.step(action)
            # 행동망은 1스텝 전이로 학습한다 (원 코드와 동일)
            self.buffer.add(cur, action, reward, nxt, float(term))
            disc_sum += (self.gamma ** i) * float(reward)
            used += 1
            cur = nxt
            traj.append((disc_sum, np.asarray(nxt, dtype=np.float32).ravel().copy(), float(term)))
            if term or trunc:
                done, terminated = True, term
                break

        # 지속망 저장: 실제로 실행한 길이 + (선택) 그보다 짧은 길이들도 함께 (skip augmentation)
        idxs = range(len(traj)) if self.skip_augment else [len(traj) - 1]
        for i in idxs:
            r_i, s_i, term_i = traj[i]
            self.skip_buffer.add(s0, i, r_i, s_i, term_i, j=float(i + 1), behaviour_action=float(action))
        return cur, used, done, info

    # ---------- 학습: 환경 스텝 수만큼 두 망을 각각 갱신 ----------
    def update(self, global_step: int, n_updates: int = 1) -> None:
        super().update(global_step, n_updates)
        if self.skip_buffer.size < max(self.learn_start, self.batch_size):
            return
        for _ in range(n_updates):
            b = self.skip_buffer.sample(self.batch_size, self.rng)
            j = b["j"].long()
            with torch.no_grad():
                boot = self.q_target(b["next_obs"]).max(1).values
                target = b["reward"] + (self.gamma ** j.float()) * (1 - b["done"]) * boot
            sa = self._sa_batch(b["obs"], b["behaviour_action"])
            pred = self.q_skip(sa).gather(1, (j - 1)[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(pred, target)
            self.opt_skip.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_skip.parameters(), 10.0)
            self.opt_skip.step()
            if self.n_updates % self.target_update == 0:
                self.q_skip_target.load_state_dict(self.q_skip.state_dict())

    # ---------- 체크포인트 ----------
    def state_dict(self, with_buffer: bool = True) -> dict:
        sd = super().state_dict(with_buffer)
        sd["q_skip"] = self.q_skip.state_dict()
        sd["q_skip_target"] = self.q_skip_target.state_dict()
        sd["opt_skip"] = self.opt_skip.state_dict()
        if with_buffer:
            sd["skip_buffer"] = self.skip_buffer.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.q_skip.load_state_dict(sd["q_skip"])
        self.q_skip_target.load_state_dict(sd["q_skip_target"])
        self.opt_skip.load_state_dict(sd["opt_skip"])
        if "skip_buffer" in sd:
            self.skip_buffer.load_state_dict(sd["skip_buffer"])
