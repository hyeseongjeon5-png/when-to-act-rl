"""실험 설정 파일 사전 점검 — 몇 시간 뒤에 터질 오류를 지금 잡는다.

대기열은 설정을 순서대로 실행한다. 세 번째 설정에 오타가 있으면 앞의 두 실험이 끝난
**몇 시간 뒤에야** 알게 되고, 그 사이 12코어가 논다. 그래서 돌리기 전에 전부 훑는다.

점검하는 것 (실제로 만들어 보는 것까지 한다 — 존재 확인만으로는 부족하다):
  · 필수 항목이 다 있는가 (env_id, total_steps, agents, lambdas, seeds, hyperparams)
  · 환경을 실제로 만들 수 있는가 (no-op 등록 여부 포함)
  · 세 계열의 하이퍼파라미터가 **서로 같은가** (공정 비교 — 계열별 고유 항목은 제외)
  · 고정 규칙을 실제로 만들 수 있는가 (type 이름·인자)
  · Lazy의 기본 정책이 그 환경에서 만들어지는가
  · 에이전트를 실제로 만들 수 있는가 (관측·행동 차원까지 맞춰서)
  · λ·시드·예산 값이 말이 되는가

실행: python -m src.train.check_configs [설정파일...]   (생략하면 experiments/configs/*.yaml 전부)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "experiments" / "configs"

REQUIRED = ["name", "env_id", "total_steps", "agents", "lambdas", "seeds", "hyperparams"]
# 계열마다 고유하게 갖는 항목 — '세 계열이 같은가' 비교에서 뺀다
FAMILY_ONLY = {"max_skip", "skip_augment", "base_policy", "eta", "lr_skip"}


def check(path: Path) -> tuple[list[str], list[str]]:
    errs: list[str] = []
    warns: list[str] = []
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"YAML을 읽지 못함: {e}"], []
    if not isinstance(cfg, dict):
        return ["YAML 최상위가 사전(dict)이 아님"], []

    for k in REQUIRED:
        if k not in cfg:
            errs.append(f"필수 항목 '{k}' 없음")
    if errs:
        return errs, warns

    env_id = cfg["env_id"]

    # ---- 환경을 실제로 만들어 본다 ----
    try:
        from src.envs.cost_wrapper import make_cost_env
        env = make_cost_env(env_id, lam=0.0, cost_mode=cfg.get("cost_mode", "per_step"),
                            **dict(cfg.get("env_kwargs", {})))
        obs, _ = env.reset(seed=0)
        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)
        env.close()
    except Exception as e:
        return errs + [f"환경 생성 실패: {type(e).__name__}: {e}"], warns

    # ---- 세 계열의 공통 하이퍼파라미터가 같은가 (공정 비교) ----
    hps = {a: dict(cfg["hyperparams"].get(a, {})) for a in cfg["agents"]}
    missing = [a for a, h in hps.items() if not h]
    if missing:
        errs.append(f"하이퍼파라미터가 비어 있는 계열: {missing}")
    common = {a: {k: v for k, v in h.items() if k not in FAMILY_ONLY} for a, h in hps.items()}
    keys = sorted({k for h in common.values() for k in h})
    for k in keys:
        vals = {a: h.get(k, "(없음)") for a, h in common.items()}
        if len({repr(v) for v in vals.values()}) > 1:
            errs.append(f"계열마다 '{k}'가 다르다 {vals} — 공정 비교 위반")

    # ---- 고정 규칙을 실제로 만들어 본다 ----
    if "rules" in cfg:
        try:
            from src.eval.run_rules_sweep import build as build_rule
            for spec in cfg["rules"]:
                try:
                    pol = build_rule(spec, env_id)
                    a = pol.act(obs, 0)
                    if not (0 <= int(a) < n_actions):
                        errs.append(f"규칙 '{spec.get('id')}'이 범위 밖 행동 {a}를 냈다 (0~{n_actions - 1})")
                except Exception as e:
                    errs.append(f"규칙 '{spec.get('id')}' 생성/실행 실패: {type(e).__name__}: {e}")
        except Exception as e:
            errs.append(f"규칙 모듈을 불러오지 못함: {e}")
    elif cfg.get("agents"):
        warns.append("rules 항목이 없다 — 이 방에서는 λ* 비교 기준이 없다")

    # ---- 에이전트를 실제로 만들어 본다 ----
    try:
        from src.train.train_agent import build_agent
        for a in cfg["agents"]:
            try:
                ag = build_agent(a, obs_dim, n_actions, cfg, 0, env_id)
                ag.eval_policy().act(obs, 0)
            except Exception as e:
                errs.append(f"계열 '{a}' 생성/평가 실패: {type(e).__name__}: {e}")
    except Exception as e:
        errs.append(f"에이전트 모듈을 불러오지 못함: {e}")

    # ---- 값이 말이 되는가 ----
    lams = [float(x) for x in cfg["lambdas"]]
    if any(l < 0 for l in lams):
        errs.append(f"음수 λ가 있다: {[l for l in lams if l < 0]}")
    if len(set(lams)) != len(lams):
        warns.append(f"λ 격자에 중복이 있다: {lams}")
    seeds = list(cfg["seeds"])
    if len(set(seeds)) != len(seeds):
        errs.append(f"시드에 중복이 있다: {seeds}")
    if len(seeds) < 10 and not cfg.get("variant"):
        warns.append(f"시드가 {len(seeds)}개다 — 본실험 결론에는 10개 이상이 필요하다 "
                     f"(파일럿이면 variant를 지정할 것)")
    ev = int(cfg.get("eval_every", 0) or 0)
    if ev and ev > int(cfg["total_steps"]):
        errs.append(f"eval_every({ev})가 total_steps({cfg['total_steps']})보다 크다 — 곡선이 안 그려진다")
    for f in cfg.get("final_snapshots", [1.0]):
        if not (0 < float(f) <= 1.0):
            errs.append(f"final_snapshots 값이 (0, 1] 범위 밖이다: {f}")
    if float(cfg.get("cost_warmup_frac", 0) or 0) >= 1.0:
        errs.append("cost_warmup_frac가 1 이상이다 — 비용이 끝까지 안 켜진다")

    n_cond = len(cfg["agents"]) * len(lams) * len(seeds)
    warns.append(f"조건 수 {n_cond}개 (계열 {len(cfg['agents'])} × λ {len(lams)} × 시드 {len(seeds)})")
    return errs, warns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", help="점검할 설정 파일 (생략하면 전부)")
    a = ap.parse_args()
    paths = [Path(x) for x in a.configs] if a.configs else sorted(CFG_DIR.glob("*.yaml"))
    paths = [p for p in paths if not p.name.startswith("_")]
    bad = 0
    print("=" * 78)
    print(f"실험 설정 사전 점검 — {len(paths)}개")
    print("=" * 78)
    for p in paths:
        errs, warns = check(p)
        mark = "✖" if errs else "✔"
        print(f"\n{mark} {p.name}")
        for e in errs:
            print(f"    [오류] {e}")
        for w in warns:
            print(f"    [참고] {w}")
        bad += bool(errs)
    print("\n" + "=" * 78)
    print(f"오류 있는 설정 {bad}개 / {len(paths)}개")
    print("=" * 78)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
