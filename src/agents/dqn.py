"""(가) 표준 DQN — 매 스텝 행동을 고르는 기본 비교 대상.

세 학습 계열의 공통 뼈대이기도 하다. TempoRL·Lazy-MDP는 이 클래스를 물려받아
"한 번의 결정으로 환경을 몇 스텝 진행시키는가"만 바꾼다.

공정 비교를 위한 약속 (CLAUDE.md 절대 규칙 2):
  - 예산은 '환경 스텝 수'로 센다. 한 번의 결정이 여러 스텝을 먹어도 그만큼 예산이 준다.
  - 학습 갱신 횟수도 환경 스텝 수에 비례시킨다.
  - 비용 λ는 래퍼가 '실제로 실행된 no-op 아닌 스텝'마다 부과한다 (모든 계열 동일).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.common import ReplayBuffer, linear_eps, mlp


class GreedyPolicy:
    """평가용 정책 껍데기 — 고정 규칙과 같은 인터페이스(.act(obs,t))로 맞춘다."""

    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        self.agent.reset_eval()

    def act(self, obs, t: int) -> int:
        return self.agent.act_greedy(obs, t)


class DQNAgent:
    name = "dqn"

    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, seed: int):
        self.obs_dim, self.n_actions, self.cfg = obs_dim, n_actions, cfg
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        hidden = tuple(cfg.get("hidden", (128, 128)))
        self.q = mlp(obs_dim, n_actions, hidden)
        self.q_target = mlp(obs_dim, n_actions, hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=float(cfg.get("lr", 5e-4)))

        self.buffer = ReplayBuffer(int(cfg.get("buffer_size", 100_000)), obs_dim)
        self.gamma = float(cfg.get("gamma", 0.99))
        self.batch_size = int(cfg.get("batch_size", 128))
        self.learn_start = int(cfg.get("learn_start", 1_000))
        self.target_update = int(cfg.get("target_update", 500))
        self.total_steps = int(cfg["total_steps"])
        self.eps_end = float(cfg.get("eps_end", 0.05))
        self.eps_frac = float(cfg.get("eps_frac", 0.3))
        self.n_updates = 0
        self.last_loss = float("nan")

    # ---------- 행동 선택 ----------
    def eps(self, step: int) -> float:
        return linear_eps(step, self.total_steps, 1.0, self.eps_end, self.eps_frac)

    def q_values(self, obs) -> np.ndarray:
        with torch.no_grad():
            return self.q(torch.as_tensor(np.asarray(obs, dtype=np.float32))[None]).numpy()[0]

    def act_greedy(self, obs, t: int) -> int:
        return int(np.argmax(self.q_values(obs)))

    def reset_eval(self):
        pass

    def eval_policy(self):
        return GreedyPolicy(self)

    # ---------- 환경과의 상호작용 ----------
    def begin_episode(self, obs):
        pass

    def interact(self, env, obs, t: int, global_step: int):
        """한 번의 '결정'을 수행한다. 반환: (다음 obs, 소비한 환경 스텝 수, done, info)"""
        if self.rng.random() < self.eps(global_step):
            action = int(self.rng.integers(self.n_actions))
        else:
            action = self.act_greedy(obs, t)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # 시간초과(truncated)는 '진짜 끝'이 아니므로 부트스트랩을 끊지 않는다
        self.buffer.add(obs, action, reward, next_obs, float(terminated))
        return next_obs, 1, bool(terminated or truncated), info

    # ---------- 학습 ----------
    def update(self, global_step: int, n_updates: int = 1) -> None:
        if self.buffer.size < max(self.learn_start, self.batch_size):
            return
        for _ in range(n_updates):
            b = self.buffer.sample(self.batch_size, self.rng)
            with torch.no_grad():
                target = b["reward"] + self.gamma * (1 - b["done"]) * self.q_target(b["next_obs"]).max(1).values
            pred = self.q(b["obs"]).gather(1, b["action"][:, None]).squeeze(1)
            loss = F.smooth_l1_loss(pred, target)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()
            self.last_loss = float(loss.item())
            self.n_updates += 1
            if self.n_updates % self.target_update == 0:
                self.q_target.load_state_dict(self.q.state_dict())

    # ---------- 체크포인트 ----------
    def state_dict(self, with_buffer: bool = True) -> dict:
        sd = {
            "q": self.q.state_dict(), "q_target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(), "n_updates": self.n_updates,
            "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        }
        if with_buffer:
            sd["buffer"] = self.buffer.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q_target"])
        self.opt.load_state_dict(sd["opt"])
        self.n_updates = sd["n_updates"]
        self.rng.bit_generator.state = sd["rng"]
        torch.set_rng_state(sd["torch_rng"])
        if "buffer" in sd:
            self.buffer.load_state_dict(sd["buffer"])
