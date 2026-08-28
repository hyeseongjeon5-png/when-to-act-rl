"""논문용 표를 집계 파일에서 **자동으로** 만든다.

손으로 옮기지 않는 이유: 표를 손으로 옮기면 (가) 옮기다 틀리고 (나) 실험이 갱신돼도
원고가 낡은 숫자를 그대로 들고 있게 된다. 표 1·2는 항상 이 코드로 다시 만든다.

만드는 표:
  tab1 : 임계 비용 λ* 표 (환경 × 계열, 두 가지 엄격도, 기준 규칙과 최강 규칙 포락선 각각)
  tab2 : 실험 설정 요약 (환경·예산·λ 격자·시드·하이퍼파라미터·평가)
  tab3 : λ=0 공정성 점검 표 (예산을 늘렸을 때 학습이 규칙을 이기는가)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregate"
CFG = ROOT / "experiments" / "configs"

AGENT_KO = {"dqn": "표준 DQN", "temporl": "TempoRL 방식", "lazy": "Lazy-MDP 방식"}
ENV_KO = {"MountainCar-v0": "MountainCar-v0", "LunarLander-v3": "LunarLander-v3",
          "MinAtar_Freeway-v1": "MinAtar Freeway"}
RULE_KO = {"rule_pump": "pump 규칙", "rule_threshold": "임계값 규칙(처음)", "rule_threshold_tuned": "임계값 규칙(튜닝)",
           "rule_cautious": "신중 규칙", "rule_best": "최강 규칙 포락선"}


def _fmt_lam(v) -> str:
    if v is None:
        return "격자 밖"
    return f"{float(v):g}"


def table1_lambda_star(envs=("MountainCar-v0", "LunarLander-v3", "MinAtar_Freeway-v1")) -> dict | None:
    rows = []
    for env in envs:
        p = AGG / f"{env}_lambda_star.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        by_learner = {r["learner"]: r for r in d.get("results", [])}
        by_best = {r["learner"]: r for r in d.get("results_vs_best_rule", [])}
        ref_ko = RULE_KO.get(d.get("rule"), str(d.get("rule")))
        for ag in ("dqn", "temporl", "lazy"):
            r, rb = by_learner.get(ag), by_best.get(ag)
            if not r:
                continue
            rows.append([
                ENV_KO.get(env, env), AGENT_KO.get(ag, ag),
                _fmt_lam(r.get("lam_star_ci")), _fmt_lam(r.get("lam_star_pt")),
                _fmt_lam(rb.get("lam_star_ci")) if rb else "—",
                _fmt_lam(rb.get("lam_star_pt")) if rb else "—",
                str(r.get("min_seeds", "")),
            ])
    if not rows:
        return None
    return {
        "header": ["환경", "학습 계열",
                   f"λ*_CI\n(기준 규칙)", "λ*_점추정\n(기준 규칙)",
                   "λ*_CI\n(최강 규칙)", "λ*_점추정\n(최강 규칙)", "시드"],
        "rows": rows,
        "ko": "임계 비용 λ* — 학습이 고정 규칙을 더 이상 이기지 못하는 지점",
        "en": "Critical action cost λ* at which learning ceases to beat the fixed rule",
        "note": ("λ*_CI는 학습의 95% 신뢰구간 하한이 규칙의 상한 아래로 처음 내려간 λ(엄격), "
                 "λ*_점추정은 두 IQM 곡선이 처음 교차한 λ(느슨)이다. λ*=0은 비용이 없어도 "
                 "규칙을 이기지 못했다는 뜻이며, '격자 밖'은 실험한 최대 λ까지 계속 이겼다는 뜻이다. "
                 "출처: results/aggregate/*_lambda_star.json"),
        "widths": [3.2, 2.6, 2.0, 2.2, 2.0, 2.2, 1.3],
    }


def table2_setup(cfg_names=("main_mountaincar", "main_lunarlander", "main_minatar_freeway")) -> dict | None:
    rows = []
    fields = [
        ("환경", lambda c: c["env_id"]),
        ("학습 예산 (환경 스텝)", lambda c: f"{int(c['total_steps']):,}"),
        ("λ 격자", lambda c: ", ".join(f"{float(x):g}" for x in c["lambdas"])),
        ("시드", lambda c: f"{len(c['seeds'])}개 ({min(c['seeds'])}–{max(c['seeds'])})"),
        ("학습 계열", lambda c: ", ".join(AGENT_KO.get(a, a) for a in c["agents"])),
        ("고정 규칙", lambda c: ", ".join(r["id"] for r in c.get("rules", []))),
        ("최종 평가", lambda c: (f"탐험 끔, 스냅샷 {len(c.get('final_snapshots', [3]))}장 × "
                              f"{c.get('n_eval_episodes_final')}에피소드 "
                              f"(규칙 {c.get('n_eval_episodes_rules')}에피소드)")),
        ("신경망", lambda c: f"은닉 {c['hyperparams']['dqn']['hidden']}, ReLU"),
        ("학습률 / 배치", lambda c: f"{c['hyperparams']['dqn']['lr']} / {c['hyperparams']['dqn']['batch_size']}"),
        ("탐험 ε", lambda c: (f"{c['hyperparams']['dqn'].get('eps_const')} 고정"
                            if c["hyperparams"]["dqn"].get("eps_const") is not None
                            else f"1.0→{c['hyperparams']['dqn'].get('eps_end')} 감소")),
        ("최대 지속 길이 J", lambda c: str(c["hyperparams"]["temporl"].get("max_skip", "—"))),
        ("Lazy 기본 정책", lambda c: str(c["hyperparams"]["lazy"].get("base_policy", "—"))),
    ]
    cfgs = []
    for n in cfg_names:
        p = CFG / f"{n}.yaml"
        if p.exists():
            cfgs.append(yaml.safe_load(p.read_text(encoding="utf-8")))
    if not cfgs:
        return None
    for label, fn in fields:
        row = [label]
        for c in cfgs:
            try:
                row.append(str(fn(c)))
            except Exception:
                row.append("—")
        rows.append(row)
    return {
        "header": ["항목"] + [c["env_id"] for c in cfgs],
        "rows": rows,
        "ko": "실험 설정 요약",
        "en": "Summary of the experimental setup",
        "note": ("하이퍼파라미터는 세 학습 계열이 완전히 동일하며 TempoRL 공개 구현의 기본값을 그대로 썼다. "
                 "출처: experiments/configs/main_*.yaml"),
        "widths": [4.2, 5.6, 5.6],
    }


def table3_fairness(variants: dict[str, str] | None = None) -> dict | None:
    """공정성 점검 — 비용이 없는 λ=0에서 학습이 그 환경의 최고 규칙 수준에 닿는가.

    세 환경을 한 표에 모은다. 환경마다 '예산을 늘리면 달라지는가'를 물었기 때문이다.
    표준 DQN 기준이며, 비교 상대는 그 환경의 기준 규칙이다.
    """
    ENVS = [
        ("MountainCar-v0", "rule_pump", [
            ("MountainCar-v0", "본실험 (30만 스텝, ε=0.2 고정)"),
            ("MountainCar-v0@budget1M_epsconst", "예산 100만 (ε=0.2 고정)"),
            ("MountainCar-v0@budget1M_epsdecay", "예산 100만 + ε 감소"),
            ("MountainCar-v0@budget1M_wide", "예산 100만 + ε 감소 + 용량 확대"),
        ]),
        ("LunarLander-v3", "rule_threshold", [
            ("LunarLander-v3", "본실험 (20만 스텝)"),
            ("LunarLander-v3@budget1M", "예산 100만"),
        ]),
        ("MinAtar_Freeway-v1", "rule_cautious", [
            ("MinAtar_Freeway-v1", "본실험 (30만 스텝)"),
            ("MinAtar_Freeway-v1@budget1M", "예산 100만"),
        ]),
    ]
    rows = []
    for env, rule, settings in ENVS:
        base_p = AGG / f"{env}_iqm.csv"
        if not base_p.exists():
            continue
        ref = pd.read_csv(base_p)
        gr = ref[(ref.agent == rule) & (ref.lam == 0.0)]
        env_rows = []
        for key, label in settings:
            p2 = AGG / f"{key}_iqm.csv"
            if not p2.exists():
                continue
            df = pd.read_csv(p2)
            g = df[(df.agent == "dqn") & (df.lam == 0.0)]
            if g.empty:
                continue
            r = g.iloc[0]
            env_rows.append([env if not env_rows else "", label,
                             f"{int(r.total_steps):,}", str(int(r.n_seeds)),
                             f"{r.raw_return_iqm:.1f}",
                             f"[{r.raw_return_ci_lo:.1f}, {r.raw_return_ci_hi:.1f}]",
                             f"{r.solved_iqm * 100:.0f}%"])
        if not env_rows:
            continue
        if not gr.empty:
            r = gr.iloc[0]
            env_rows.append(["", f"**기준: {RULE_KO.get(rule, rule)} (학습 없음)**", "—",
                             str(int(r.n_seeds)), f"**{r.raw_return_iqm:.1f}**",
                             f"[{r.raw_return_ci_lo:.1f}, {r.raw_return_ci_hi:.1f}]",
                             f"**{r.solved_iqm * 100:.0f}%**"])
        rows += env_rows
    if len(rows) < 2:
        return None
    return {
        "header": ["환경", "설정", "학습 예산", "시드", "r IQM", "95% CI", "성공률"],
        "rows": rows,
        "ko": "공정성 점검 — 비용이 없을 때(λ=0) 학습은 그 환경의 규칙 수준에 닿는가",
        "en": "Fairness check: does learning reach rule-level performance at zero cost (λ=0)",
        "note": ("비용이 없는 조건에서도 학습이 규칙에 지면 '비용 때문에 졌다'고 말할 수 없다. "
                 "그래서 환경마다 학습 예산을 늘려 λ=0 성능을 다시 쟀다. 표준 DQN 기준이며, "
                 "평가는 본실험과 같다(스냅샷 3장 × 최종 에피소드). 시드 수가 10개보다 적은 줄은 "
                 "결정을 위한 파일럿이며 그 자체로 결론이 아니다. "
                 "출처: results/aggregate/*_iqm.csv"),
        "widths": [2.6, 4.4, 1.9, 1.1, 1.6, 2.4, 1.4],
    }


def all_tables(fairness_variants: dict[str, str] | None = None) -> dict:
    out = {}
    for key, t in (("tab1", table1_lambda_star()), ("tab2", table2_setup()),
                   ("tab3", table3_fairness(fairness_variants))):
        if t:
            out[key] = t
    return out


if __name__ == "__main__":
    for k, t in all_tables().items():
        print(f"\n===== {k}: {t['ko']}")
        print(" | ".join(str(h).replace(chr(10), " ") for h in t["header"]))
        for r in t["rows"]:
            print(" | ".join(str(x) for x in r))
