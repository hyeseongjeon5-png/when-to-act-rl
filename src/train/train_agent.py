"""조건 1개(환경 × 계열 × λ × 시드)를 학습·평가하는 최소 단위 스크립트.

체크포인트 저장과 이어하기를 지원한다 — 48시간 실험에서 중간에 죽어도
마지막 체크포인트부터 다시 시작한다 (CLAUDE.md 워크플로 규칙).

실행 예:
  python -m src.train.train_agent --config experiments/configs/pilot_mountaincar.yaml \
         --agent dqn --lam 0.1 --seed 3
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from src.envs.cost_wrapper import NOOP_BY_ENV, make_cost_env
from src.eval.evaluate import evaluate, summarize

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)  # 조건 여러 개를 동시에 돌리므로 조건당 1스레드


def build_agent(agent_name: str, obs_dim: int, n_actions: int, cfg: dict, seed: int, env_id: str):
    hp = dict(cfg.get("hyperparams", {}).get(agent_name, {}))
    hp["total_steps"] = int(cfg["total_steps"])
    if agent_name == "dqn":
        from src.agents.dqn import DQNAgent
        return DQNAgent(obs_dim, n_actions, hp, seed)
    if agent_name == "temporl":
        from src.agents.temporl import TempoRLAgent
        return TempoRLAgent(obs_dim, n_actions, hp, seed)
    if agent_name == "lazy":
        from src.agents.lazy import LazyAgent
        return LazyAgent(obs_dim, n_actions, hp, seed, env_id=env_id)
    raise KeyError(f"모르는 계열: {agent_name}")


def cond_paths(cfg: dict, agent: str, lam: float, seed: int) -> dict:
    env_id = cfg["env_id"]
    tag = f"{env_id}/{agent}/lam{lam}"
    out = ROOT / "results" / "raw" / tag
    ck = ROOT / "results" / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    return {
        "out": out, "ckpt": ck / f"seed{seed}.pt",
        "curve": out / f"seed{seed}_curve.csv",
        "final": out / f"seed{seed}_final.csv",
        "meta": out / f"seed{seed}_meta.json",
        "tag": f"{tag}/seed{seed}",
    }


def train_one(cfg: dict, agent_name: str, lam: float, seed: int, quiet: bool = False) -> dict:
    env_id = cfg["env_id"]
    total_steps = int(cfg["total_steps"])
    eval_every = int(cfg.get("eval_every", max(1, total_steps // 20)))
    n_eval_train = int(cfg.get("n_eval_episodes_curve", 10))
    n_eval_final = int(cfg.get("n_eval_episodes_final", 100))
    ckpt_every = int(cfg.get("ckpt_every", max(1, total_steps // 4)))
    env_kwargs = dict(cfg.get("env_kwargs", {}))
    p = cond_paths(cfg, agent_name, lam, seed)

    if p["meta"].exists() and json.loads(p["meta"].read_text(encoding="utf-8")).get("done"):
        return {"status": "이미완료", "tag": p["tag"]}

    env = make_cost_env(env_id, lam=lam, **env_kwargs)
    eval_env = make_cost_env(env_id, lam=lam, **env_kwargs)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    agent = build_agent(agent_name, obs_dim, n_actions, cfg, seed, env_id)

    step, curve, t_start = 0, [], time.time()
    elapsed_prev = 0.0
    if p["ckpt"].exists():  # ---- 이어하기 ----
        sd = torch.load(p["ckpt"], weights_only=False)
        agent.load_state_dict(sd["agent"])
        step = sd["step"]
        curve = sd["curve"]
        elapsed_prev = sd.get("elapsed", 0.0)
        if not quiet:
            print(f"[이어하기] {p['tag']} — {step}/{total_steps} 스텝부터")

    eval_seeds_curve = [100_000 + seed * 1000 + i for i in range(n_eval_train)]
    eval_seeds_final = [500_000 + seed * 1000 + i for i in range(n_eval_final)]

    obs, _ = env.reset(seed=seed + 7919)
    agent.begin_episode(obs)
    t_in_ep = 0
    next_eval = ((step // eval_every) + 1) * eval_every
    next_ckpt = ((step // ckpt_every) + 1) * ckpt_every

    while step < total_steps:
        next_obs, used, done, info = agent.interact(env, obs, t_in_ep, step)
        agent.update(step, n_updates=used)
        step += used
        t_in_ep += used
        obs = next_obs
        if done:
            obs, _ = env.reset()
            agent.begin_episode(obs)
            t_in_ep = 0

        if step >= next_eval:
            rows = evaluate(eval_env, agent.eval_policy(), eval_seeds_curve)
            s = summarize(rows)
            s.update({"step": step, "loss": agent.last_loss, "eps": float(agent.eps(step))})
            curve.append(s)
            if not quiet:
                print(f"  {p['tag']} {step:>7}/{total_steps} | r' IQM {s['cost_return_iqm']:8.2f} "
                      f"| 행동 {s['n_actions_mean']:6.1f} | 성공 {s['solved_rate']*100:5.1f}%", flush=True)
            next_eval += eval_every

        if step >= next_ckpt:
            torch.save({"agent": agent.state_dict(), "step": step, "curve": curve,
                        "elapsed": elapsed_prev + time.time() - t_start}, p["ckpt"])
            next_ckpt += ckpt_every

    # ---- 최종 평가 (탐험 끔, 100 에피소드) ----
    rows = evaluate(eval_env, agent.eval_policy(), eval_seeds_final)
    final = summarize(rows)
    elapsed = elapsed_prev + time.time() - t_start

    with p["final"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if curve:
        with p["curve"].open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
            w.writeheader()
            w.writerows(curve)
    meta = {
        "done": True, "env_id": env_id, "agent": agent_name, "lam": lam, "seed": seed,
        "total_steps": total_steps, "elapsed_sec": round(elapsed, 1),
        "config_name": cfg.get("name"), "env_kwargs": env_kwargs,
        "final": final, "n_eval_episodes_final": n_eval_final,
    }
    p["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close(); eval_env.close()
    if p["ckpt"].exists():
        p["ckpt"].unlink()  # 완료된 조건의 체크포인트는 지운다 (디스크 절약)
    if not quiet:
        print(f"[완료] {p['tag']} | r' IQM {final['cost_return_iqm']:.2f} | "
              f"행동 {final['n_actions_mean']:.1f} | 성공 {final['solved_rate']*100:.1f}% | {elapsed:.0f}초")
    return {"status": "완료", "tag": p["tag"], "elapsed_sec": elapsed, **final}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    try:
        train_one(cfg, a.agent, a.lam, a.seed, quiet=a.quiet)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
