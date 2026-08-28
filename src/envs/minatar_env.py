"""MinAtar 게임을 Gymnasium 환경으로 감싸는 어댑터.

MinAtar 패키지는 Gymnasium에 등록돼 있지 않아 `gym.make("MinAtar/Breakout-v1")`이 바로 안 된다.
이 파일을 한 번 import하면 등록된다.

이 연구에 맞추기 위한 세 가지 결정 (전부 논문 Ⅲ장에 적는다):

1. **관측을 평탄화한다.** MinAtar 관측은 (10, 10, 채널) 3차원 격자다. 이 연구의 세 학습 계열은
   모두 같은 다층 퍼셉트론을 쓰므로 400차원 벡터로 펴서 넣는다. 합성곱망을 쓰면 성능은 오르겠지만
   계열 간 비교가 '방법의 차이'가 아니라 '신경망 구조의 차이'로 오염된다. 대신 은닉층을
   MountainCar/LunarLander보다 크게 잡아 표현력 부족을 보완한다 (세 계열 모두 동일하게).

2. **최소 행동 집합만 쓴다.** MinAtar의 전체 행동은 6개지만 게임마다 실제로 의미 있는 행동은
   일부다(Breakout은 '가만히·왼쪽·오른쪽' 3개). 쓸모없는 행동을 남겨 두면 그 행동에도 비용 λ가
   붙어 비교가 흐려진다. **재배치 후에도 0번은 언제나 no-op('n')이다.**

3. **끈적임(sticky action)을 끈다.** MinAtar 기본값은 10% 확률로 이전 행동을 대신 실행한다.
   그러면 "에이전트가 행동을 냈는가"와 "환경이 무엇을 실행했는가"가 어긋나 비용을 누구에게
   물릴지가 애매해진다. 이 연구는 비용 회계가 정확해야 하므로 0으로 둔다.

실행(자가 점검): python -m src.envs.minatar_env
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

GAMES = ["breakout", "asterix", "freeway", "seaquest", "space_invaders"]
# MinAtar의 action_map은 게임과 무관하게 ['n','l','u','r','d','f'] 이고 0번이 no-op이다.
MINATAR_NOOP_RAW = 0
DEFAULT_MAX_STEPS = 1000


class MinAtarEnv(gym.Env):
    """MinAtar 게임 하나를 Gymnasium 인터페이스로 노출한다."""

    metadata = {"render_modes": []}

    def __init__(self, game: str = "breakout", max_episode_steps: int = DEFAULT_MAX_STEPS,
                 sticky_action_prob: float = 0.0, minimal_actions: bool = True):
        from minatar import Environment
        self.game = game
        self.max_episode_steps = int(max_episode_steps)
        self._env = Environment(game, sticky_action_prob=float(sticky_action_prob))
        h, w, c = self._env.state_shape()
        self.obs_shape = (h, w, c)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(h * w * c,), dtype=np.float32)

        raw = list(self._env.minimal_action_set()) if minimal_actions else list(range(self._env.num_actions()))
        if MINATAR_NOOP_RAW not in raw:
            raw = [MINATAR_NOOP_RAW] + raw          # no-op이 없으면 넣는다 (이 연구에 필수)
        raw.remove(MINATAR_NOOP_RAW)
        self._actions = [MINATAR_NOOP_RAW] + sorted(raw)   # 0번 = no-op 으로 고정
        self.action_space = spaces.Discrete(len(self._actions))
        self._t = 0

    def _obs(self) -> np.ndarray:
        return np.asarray(self._env.state(), dtype=np.float32).reshape(-1)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(int(seed))
        self._env.reset()
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        reward, terminated = self._env.act(self._actions[int(action)])
        self._t += 1
        truncated = (not terminated) and self._t >= self.max_episode_steps
        return self._obs(), float(reward), bool(terminated), bool(truncated), {}

    def close(self):
        pass


def _register() -> list[str]:
    ids = []
    for g in GAMES:
        env_id = "MinAtar/" + "".join(p.capitalize() for p in g.split("_")) + "-v1"
        ids.append(env_id)
        if env_id in gym.registry:
            continue
        gym.register(id=env_id, entry_point="src.envs.minatar_env:MinAtarEnv",
                     kwargs={"game": g}, max_episode_steps=None)
    return ids


REGISTERED = _register()


if __name__ == "__main__":
    from src.envs.cost_wrapper import NOOP_BY_ENV, make_cost_env
    print("등록된 환경:", REGISTERED)
    for env_id in REGISTERED:
        if env_id not in NOOP_BY_ENV:
            print(f"  [주의] {env_id} 는 NOOP_BY_ENV에 아직 없다")
            continue
        env = make_cost_env(env_id, lam=0.5)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape, "관측 모양 불일치"
        # no-op은 공짜, 그 밖의 행동은 λ — 래퍼가 제대로 무는지 확인
        _, _, _, _, i0 = env.step(0)
        _, _, _, _, i1 = env.step(1)
        assert i0["action_cost"] == 0.0 and i0["acted"] is False, "no-op에 비용이 붙었다"
        assert i1["action_cost"] == 0.5 and i1["acted"] is True, "행동에 비용이 안 붙었다"
        print(f"  {env_id:28} 행동 {env.action_space.n}개 · 관측 {obs.shape[0]}차원 · 비용 회계 정상")
        env.close()
    print("자가 점검 통과")
