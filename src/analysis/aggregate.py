"""결과 집계 — IQM + 95% 계층 부트스트랩 신뢰구간 (Agarwal et al. 2021, rliable).

왜 평균이 아니라 IQM인가: 시드 몇 개가 유난히 잘/못 되면 평균이 휘둘린다.
IQM(사분위평균)은 위아래 25%를 버리고 가운데 50%만 평균 내므로 이상치에 덜 흔들린다.
신뢰구간은 '이 숫자가 얼마나 믿을 만한가'의 폭이다. 구간이 겹치면 우열을 말하지 않는다.

한 조건 = 환경 × 계열 × λ × 시드. 시드 1개당 점수 = 최종 평가 100 에피소드의 평균.
따라서 계열·λ 하나당 점수 벡터의 길이는 시드 수(본실험 10)다.

실행: python -m src.analysis.aggregate --env MountainCar-v0
      python -m src.analysis.aggregate --env all --reps 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "aggregate"

METRICS = ["cost_return", "raw_return", "n_actions", "solved"]


def load_conditions(env_id: str) -> pd.DataFrame:
    """조건별 seed 점수표를 만든다. 한 줄 = (환경, 계열, λ, 시드) 1개."""
    rows = []
    base = RAW / env_id
    if not base.exists():
        return pd.DataFrame()
    for meta_p in base.rglob("seed*_meta.json"):
        try:
            m = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not m.get("done"):
            continue
        csv_p = meta_p.parent / meta_p.name.replace("_meta.json", "_final.csv")
        if not csv_p.exists():
            continue
        df = pd.read_csv(csv_p)
        rows.append({
            "env_id": m["env_id"], "agent": m["agent"], "lam": float(m["lam"]), "seed": int(m["seed"]),
            "cost_return": float(df["cost_return"].mean()),
            "raw_return": float(df["raw_return"].mean()),
            "n_actions": float(df["n_actions"].mean()),
            "solved": float(df["solved"].mean()),
            "steps": float(df["steps"].mean()),
            "n_eval_episodes": len(df),
            "elapsed_sec": float(m.get("elapsed_sec", 0.0)),
            "total_steps": int(m.get("total_steps", 0)),
            "source_csv": str(csv_p.relative_to(ROOT)).replace("\\", "/"),
        })
    return pd.DataFrame(rows)


def iqm_ci(scores: np.ndarray, reps: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """IQM 점추정과 95% 신뢰구간. rliable이 있으면 그걸 쓰고, 없으면 같은 방식으로 직접 계산."""
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    try:
        from rliable import library as rly
        from rliable import metrics as rl_metrics
        point, cis = rly.get_interval_estimates(
            {"a": x}, lambda s: np.array([rl_metrics.aggregate_iqm(s)]), reps=reps)
        return float(point["a"][0]), float(cis["a"][0][0]), float(cis["a"][1][0])
    except Exception:
        rng = np.random.default_rng(seed)
        def _iqm(v):
            v = np.sort(v)
            lo, hi = int(np.floor(len(v) * 0.25)), int(np.ceil(len(v) * 0.75))
            return float(np.mean(v[lo:hi]))
        boots = [_iqm(rng.choice(x[:, 0], size=len(x), replace=True)) for _ in range(reps)]
        return _iqm(x[:, 0]), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def aggregate(env_id: str, reps: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    cond = load_conditions(env_id)
    if cond.empty:
        return cond, cond
    out = []
    for (agent, lam), g in cond.groupby(["agent", "lam"]):
        rec = {"env_id": env_id, "agent": agent, "lam": lam, "n_seeds": len(g),
               "total_steps": int(g["total_steps"].iloc[0]),
               "elapsed_sec_mean": round(float(g["elapsed_sec"].mean()), 1)}
        for m in METRICS:
            p, lo, hi = iqm_ci(g[m].to_numpy(), reps=reps)
            rec[f"{m}_iqm"] = p
            rec[f"{m}_ci_lo"] = lo
            rec[f"{m}_ci_hi"] = hi
            rec[f"{m}_mean"] = float(g[m].mean())
        out.append(rec)
    agg = pd.DataFrame(out).sort_values(["agent", "lam"]).reset_index(drop=True)
    return cond, agg


def critical_lambda(agg: pd.DataFrame, learner: str, rule: str = "rule_pump",
                    metric: str = "cost_return") -> dict:
    """임계 비용 λ* — 학습이 규칙을 더 이상 이기지 못하게 되는 지점.

    두 가지로 보고한다 (docs/02_실험-설계.md §5의 판정 기준을 그대로 구현):
      lam_star_ci : 학습의 CI 하한이 규칙의 CI 상한 아래로 처음 내려간 λ (엄격 — 통계적으로 우위가 사라짐)
      lam_star_pt : 학습의 점추정(IQM)이 규칙의 점추정 아래로 처음 내려간 λ (느슨 — 곡선 교차점)
    둘 중 어느 것도 없으면(끝까지 이기거나 처음부터 지면) None을 넣고 사유를 적는다.
    """
    L = agg[agg.agent == learner].sort_values("lam")
    R = agg[agg.agent == rule].set_index("lam")
    if L.empty or R.empty:
        return {"learner": learner, "rule": rule, "lam_star_ci": None, "lam_star_pt": None,
                "note": "비교할 데이터 없음"}
    res = {"learner": learner, "rule": rule, "lam_star_ci": None, "lam_star_pt": None, "rows": []}
    for _, r in L.iterrows():
        lam = r["lam"]
        if lam not in R.index:
            continue
        rr = R.loc[lam]
        wins_ci = r[f"{metric}_ci_lo"] > rr[f"{metric}_ci_hi"]     # 통계적으로 확실히 이김
        wins_pt = r[f"{metric}_iqm"] > rr[f"{metric}_iqm"]         # 점추정으로 이김
        res["rows"].append({"lam": lam, "learner_iqm": r[f"{metric}_iqm"], "rule_iqm": rr[f"{metric}_iqm"],
                            "wins_ci": bool(wins_ci), "wins_pt": bool(wins_pt)})
        if res["lam_star_ci"] is None and not wins_ci:
            res["lam_star_ci"] = float(lam)
        if res["lam_star_pt"] is None and not wins_pt:
            res["lam_star_pt"] = float(lam)
    if res["lam_star_pt"] == 0.0:
        res["note"] = "λ=0에서도 규칙을 못 이김 — 비용 때문이 아니라 학습 자체가 규칙보다 약함"
    elif res["lam_star_pt"] is None:
        res["note"] = "실험한 λ 전 구간에서 규칙을 이김 — λ*는 격자 최대값보다 큼"
    else:
        res["note"] = "격자 안에서 교차 지점 발견"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="all")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--rule", default="rule_pump", help="비교 기준이 되는 최고 고정 규칙")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    envs = [p.name for p in RAW.iterdir() if p.is_dir()] if a.env == "all" else [a.env]

    for env_id in envs:
        cond, agg = aggregate(env_id, reps=a.reps)
        if agg.empty:
            print(f"[{env_id}] 완료된 조건 없음 — 건너뜀")
            continue
        cond.to_csv(OUT / f"{env_id}_conditions.csv", index=False, encoding="utf-8-sig")
        agg.to_csv(OUT / f"{env_id}_iqm.csv", index=False, encoding="utf-8-sig")
        rule = a.rule if a.rule in set(agg.agent) else None
        stars = []
        if rule:
            for learner in sorted(set(agg.agent) - {x for x in agg.agent if str(x).startswith("rule_")}):
                stars.append(critical_lambda(agg, learner, rule))
        (OUT / f"{env_id}_lambda_star.json").write_text(
            json.dumps({"env_id": env_id, "rule": rule, "results": stars}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n===== {env_id} =====")
        show = agg[["agent", "lam", "n_seeds", "cost_return_iqm", "cost_return_ci_lo",
                    "cost_return_ci_hi", "n_actions_iqm", "solved_iqm"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
        for s in stars:
            print(f"  λ* [{s['learner']} vs {s['rule']}] CI기준 {s['lam_star_ci']} / 점추정기준 {s['lam_star_pt']} — {s['note']}")
        print(f"  저장: results/aggregate/{env_id}_iqm.csv, {env_id}_lambda_star.json")


if __name__ == "__main__":
    main()
