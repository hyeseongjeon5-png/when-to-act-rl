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
import hashlib
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


def fingerprint(cfg: dict, agent_name: str) -> str:
    """설정 지문 — 체크포인트가 '지금 이 설정으로 만들어진 것'인지 확인하는 도장.

    왜 필요한가: 하이퍼파라미터를 바꾸고 다시 돌리면 예전 체크포인트가 남아 있다가
    신경망 크기가 안 맞아 터지거나, 더 나쁘게는 조용히 섞인 결과를 만든다.
    지문이 다르면 예전 체크포인트를 무시하고 처음부터 다시 시작한다.
    """
    key = {
        "agent": agent_name,
        "hp": cfg.get("hyperparams", {}).get(agent_name, {}),
        "total_steps": cfg.get("total_steps"),
        "env_id": cfg.get("env_id"),
        "env_kwargs": cfg.get("env_kwargs", {}),
        "n_eval_episodes_final": cfg.get("n_eval_episodes_final"),
        "final_snapshots": cfg.get("final_snapshots", [0.9, 0.95, 1.0]),
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


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
    n_eval_final = int(cfg.get("n_eval_episodes_final", 50))
    # 최종 점수는 '마지막 순간 한 장면'이 아니라 학습 막바지 여러 시점의 평균으로 낸다.
    # 이유: 학습 곡선을 보면 정책이 평가 시점마다 두 상태를 오간다(예: MountainCar lazy가
    # '규칙에 위임(−119)' ↔ '직접 우가속(−200)'을 손실 0.00인 채로 왕복). Q값이 거의 동률이라
    # argmax가 뒤집히는 것인데, 한 시점만 재면 최종 점수가 동전 던지기가 된다.
    # 예산의 90%·95%·100% 시점 정책을 **같은 평가 시드**로 각각 재서 전부 합쳐 평균 낸다.
    snap_fracs = [float(x) for x in cfg.get("final_snapshots", [0.9, 0.95, 1.0])]
    snap_steps = sorted({max(1, int(round(total_steps * f))) for f in snap_fracs})
    ckpt_every = int(cfg.get("ckpt_every", max(1, total_steps // 4)))
    env_kwargs = dict(cfg.get("env_kwargs", {}))
    p = cond_paths(cfg, agent_name, lam, seed)

    fp = fingerprint(cfg, agent_name)
    if p["meta"].exists():
        old = json.loads(p["meta"].read_text(encoding="utf-8"))
        if old.get("done") and old.get("fingerprint") == fp:
            return {"status": "이미완료", "tag": p["tag"]}
        if old.get("done"):
            print(f"[경고] {p['tag']} 의 기존 결과는 다른 설정(지문 {old.get('fingerprint')})으로 만들어진 것 "
                  f"— 현재 지문 {fp}. 다시 계산한다.")

    env = make_cost_env(env_id, lam=lam, **env_kwargs)
    eval_env = make_cost_env(env_id, lam=lam, **env_kwargs)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    agent = build_agent(agent_name, obs_dim, n_actions, cfg, seed, env_id)

    step, curve, t_start = 0, [], time.time()
    final_rows: list[dict] = []
    snap_idx = 0
    elapsed_prev = 0.0
    if p["ckpt"].exists():  # ---- 이어하기 ----
        sd = torch.load(p["ckpt"], weights_only=False)
        if sd.get("fingerprint") != fp:
            # 다른 설정으로 만든 체크포인트 — 이어붙이면 결과가 오염된다. 버리고 처음부터.
            print(f"[체크포인트 무시] {p['tag']} — 설정 지문 불일치 "
                  f"(체크포인트 {sd.get('fingerprint')} ≠ 현재 {fp}). 처음부터 다시 학습한다.")
            p["ckpt"].unlink()
        else:
            agent.load_state_dict(sd["agent"])
            step = sd["step"]
            curve = sd["curve"]
            final_rows = sd.get("final_rows", [])
            snap_idx = sd.get("snap_idx", 0)
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

        # ---- 최종 점수용 스냅샷 평가 (예산의 90%·95%·100% 시점) ----
        while snap_idx < len(snap_steps) and step >= snap_steps[snap_idx]:
            rows = evaluate(eval_env, agent.eval_policy(), eval_seeds_final)
            for r in rows:
                r["snapshot_step"] = snap_steps[snap_idx]
            final_rows.extend(rows)
            if not quiet:
                sm = summarize(rows)
                print(f"  [스냅샷 {snap_steps[snap_idx]}] {p['tag']} r' IQM {sm['cost_return_iqm']:8.2f} "
                      f"| 행동 {sm['n_actions_mean']:6.1f} | 성공 {sm['solved_rate']*100:5.1f}%", flush=True)
            snap_idx += 1

        if step >= next_ckpt:
            torch.save({"agent": agent.state_dict(), "step": step, "curve": curve,
                        "fingerprint": fp, "final_rows": final_rows, "snap_idx": snap_idx,
                        "elapsed": elapsed_prev + time.time() - t_start}, p["ckpt"])
            next_ckpt += ckpt_every

    # ---- 최종 평가: 스냅샷 여러 장을 합쳐서 (탐험 끔) ----
    while snap_idx < len(snap_steps):  # 예산에 딱 맞게 끝난 경우 마지막 스냅샷 보정
        rows = evaluate(eval_env, agent.eval_policy(), eval_seeds_final)
        for r in rows:
            r["snapshot_step"] = snap_steps[snap_idx]
        final_rows.extend(rows)
        snap_idx += 1
    rows = final_rows
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
        "done": True, "fingerprint": fp,
        "hyperparams": cfg.get("hyperparams", {}).get(agent_name, {}),
        "env_id": env_id, "agent": agent_name, "lam": lam, "seed": seed,
        "total_steps": total_steps, "elapsed_sec": round(elapsed, 1),
        "config_name": cfg.get("name"), "env_kwargs": env_kwargs,
        "final": final,
        "n_eval_episodes_final": n_eval_final,
        "final_snapshots": snap_steps,
        "n_final_rows": len(rows),
        "final_by_snapshot": {str(st): summarize([r for r in rows if r["snapshot_step"] == st])
                              for st in snap_steps},
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
