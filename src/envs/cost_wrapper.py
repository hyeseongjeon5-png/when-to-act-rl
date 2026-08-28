"""행동 비용 래퍼 — 이 연구의 심장.

보상을 r' = r − λ·1[행동 실행] 으로 바꾼다.
"행동 실행"이란 무행동(no-op)이 아닌 행동을 실제로 낸 스텝을 뜻한다.

쉬운 비유: 게임 컨트롤러 버튼을 누를 때마다 동전 λ개를 내는 것.
버튼을 안 누르면(no-op) 공짜다. λ=0이면 원래 게임과 완전히 같다.

환경별 no-op 번호:
  MountainCar-v0  → 1 (가속 안 함)
  LunarLander-v2  → 0 (엔진 안 씀)
  MinAtar         → 0 (원래 action_map ["n","l","u","r","d","f"]의 0번이 no-op)
"""
from __future__ import annotations

import gymnasium as gym


class ActionCostWrapper(gym.Wrapper):
    """행동에 비용 λ를 물리는 Gymnasium 래퍼.

    기록되는 info 키:
      action_cost : 이번 스텝에 낸 비용 (0 또는 λ)
      acted       : 이번 스텝에 행동을 실행했는가 (bool)
      raw_reward  : 비용을 빼기 전 원래 보상 (분석 시 r 기준 성능 계산용)
      episode_actions : 에피소드 종료 스텝에 제공. **비용이 부과된 횟수**다.
                        per_step 방식에서는 곧 행동 횟수와 같고, per_switch 방식에서는 전환 횟수다.
      episode_true_actions : no-op이 아닌 행동을 낸 횟수 (비용 방식과 무관 — 보고용)
      episode_raw_return : 비용을 빼기 전 누적 보상
    """

    COST_MODES = ("per_step", "per_switch")

    def __init__(self, env: gym.Env, lam: float = 0.0, noop_action: int | None = None,
                 cost_mode: str = "per_step"):
        super().__init__(env)
        assert lam >= 0.0, "비용 λ는 0 이상이어야 한다"
        assert cost_mode in self.COST_MODES, f"모르는 비용 방식: {cost_mode}"
        self.lam = float(lam)
        self.noop_action = noop_action  # None이면 모든 행동에 비용 부과
        self.cost_mode = cost_mode
        self._n_actions = 0        # 비용이 부과된 횟수 (per_step이면 곧 행동 횟수)
        self._n_true_actions = 0   # no-op이 아닌 행동을 낸 횟수 (방식과 무관)
        self._raw_return = 0.0
        self._prev_action = None

    def reset(self, **kwargs):
        self._n_actions = 0
        self._n_true_actions = 0
        self._raw_return = 0.0
        self._prev_action = None
        return self.env.reset(**kwargs)

    def _acted(self, action) -> bool:
        """이번 스텝에 '비용을 물어야 할 일'을 했는가.

        per_step  (작동 비용): no-op이 아닌 행동을 실행한 모든 스텝에 부과.
                  버튼을 누르고 있는 동안 계속 전기를 쓰는 기계에 해당한다.
        per_switch(전환 비용): **행동이 직전과 달라진 스텝에만** 부과. 같은 행동을 붙잡고
                  있으면 공짜다. 밸브를 여닫을 때만 부품이 닳는 기계에 해당한다.
                  이 방식은 '행동을 오래 유지하는' TempoRL 계열에 구조적으로 유리하므로
                  본실험의 기본값으로 쓰지 않고 **별도 축**으로만 보고한다 (공정 비교 유지).
        """
        if self.cost_mode == "per_switch":
            changed = self._prev_action is None or int(action) != int(self._prev_action)
            is_noop = self.noop_action is not None and int(action) == int(self.noop_action)
            # 처음부터 no-op이면 아무것도 시작하지 않은 것이므로 전환으로 세지 않는다
            if self._prev_action is None:
                return not is_noop
            return changed
        return (self.noop_action is None) or (int(action) != int(self.noop_action))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        acted = self._acted(action)
        cost = self.lam if acted else 0.0
        self._prev_action = int(action)

        self._n_actions += int(acted)
        self._n_true_actions += int(self.noop_action is None or int(action) != int(self.noop_action))
        self._raw_return += float(reward)

        info = dict(info)
        info["action_cost"] = cost
        info["acted"] = acted
        info["raw_reward"] = float(reward)
        if terminated or truncated:
            info["episode_actions"] = self._n_actions          # 비용이 부과된 횟수
            info["episode_true_actions"] = self._n_true_actions   # no-op 아닌 행동 횟수
            info["episode_raw_return"] = self._raw_return

        return obs, float(reward) - cost, terminated, truncated, info


NOOP_BY_ENV = {
    "MountainCar-v0": 1,
    "LunarLander-v2": 0,
    "LunarLander-v3": 0,
    # MinAtar: src/envs/minatar_env.py 가 행동을 재배치해 0번을 언제나 no-op('n')으로 만든다.
    # (MinAtar 원래 action_map은 게임과 무관하게 ['n','l','u','r','d','f'] 로 0번이 no-op이다.)
    "MinAtar/Breakout-v1": 0,
    "MinAtar/Asterix-v1": 0,
    "MinAtar/Freeway-v1": 0,
    "MinAtar/Seaquest-v1": 0,
    "MinAtar/SpaceInvaders-v1": 0,
}


def make_cost_env(env_id: str, lam: float, cost_mode: str = "per_step", **env_kwargs) -> gym.Env:
    """환경 이름만으로 비용 래퍼가 씌워진 환경을 만든다 (no-op 자동 지정).

    cost_mode: "per_step"(기본, 작동 비용) 또는 "per_switch"(전환 비용 — 민감도 분석용)
    """
    if env_id.startswith("MinAtar/"):
        import src.envs.minatar_env  # noqa: F401  (import만으로 Gymnasium에 등록된다)
    if env_id not in NOOP_BY_ENV:
        raise KeyError(f"{env_id}의 no-op 행동 번호를 NOOP_BY_ENV에 먼저 등록할 것 (확인 필요)")
    env = gym.make(env_id, **env_kwargs)
    return ActionCostWrapper(env, lam=lam, noop_action=NOOP_BY_ENV[env_id], cost_mode=cost_mode)


if __name__ == "__main__":
    # 간단 자가 점검: λ=0이면 원래 보상과 같아야 한다
    env = make_cost_env("MountainCar-v0", lam=0.0)
    obs, _ = env.reset(seed=0)
    _, r, *_ , info = env.step(env.action_space.sample())
    assert r == info["raw_reward"], "λ=0인데 보상이 달라짐 — 버그"
    print("자가 점검 통과: λ=0 → 원래 환경과 동일")
