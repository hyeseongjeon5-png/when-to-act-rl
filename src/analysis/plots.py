"""그림 생성 — λ-성능 지도, 행동 횟수 지도, 학습 곡선.

핵심 그림은 λ-성능 지도다. 읽는 법:
  가로축 = 행동 1번의 비용 λ (오른쪽으로 갈수록 행동이 비싸진다)
  세로축 = 비용까지 반영한 총보상 r' (높을수록 좋다)
  실선 = 각 계열의 IQM, 옅은 띠 = 95% 신뢰구간, 점선 = 최고 고정 규칙(pump)
  두 곡선이 교차하는 지점이 '임계 비용 λ*' — 학습이 규칙을 못 이기게 되는 문턱

실행: python -m src.analysis.plots --env MountainCar-v0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregate"
FIG = ROOT / "results" / "figures"

for f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

LABEL = {"dqn": "표준 DQN", "temporl": "TempoRL 방식", "lazy": "Lazy-MDP 방식",
         "rule_pump": "고정 규칙: pump(임계값)", "rule_noop": "고정 규칙: 무행동",
         "rule_threshold": "고정 규칙: 임계값", "rule_periodic_k1": "고정 규칙: 매 스텝",
         "rule_periodic_k2": "고정 규칙: 2스텝 주기", "rule_periodic_k4": "고정 규칙: 4스텝 주기",
         "rule_periodic_k8": "고정 규칙: 8스텝 주기"}
COLOR = {"dqn": "#1f77b4", "temporl": "#d62728", "lazy": "#2ca02c"}
REF_RULE = {"MountainCar-v0": "rule_pump", "LunarLander-v3": "rule_threshold",
            "LunarLander-v2": "rule_threshold"}


def lam_map(env_id: str, metric: str = "cost_return", ref_rule: str = "rule_pump") -> Path | None:
    p = AGG / f"{env_id}_iqm.csv"
    if not p.exists():
        print(f"[{env_id}] 집계 파일 없음 — 먼저 aggregate를 돌릴 것")
        return None
    agg = pd.read_csv(p)
    learners = [a for a in sorted(agg.agent.unique()) if not str(a).startswith("rule_")]
    rules = [a for a in sorted(agg.agent.unique()) if str(a).startswith("rule_")]

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=140)
    for a in learners:
        g = agg[agg.agent == a].sort_values("lam")
        c = COLOR.get(a)
        ax.plot(g.lam, g[f"{metric}_iqm"], "-o", ms=4, lw=2, color=c, label=LABEL.get(a, a), zorder=3)
        ax.fill_between(g.lam, g[f"{metric}_ci_lo"], g[f"{metric}_ci_hi"], color=c, alpha=0.18, lw=0, zorder=2)
    for a in rules:
        g = agg[agg.agent == a].sort_values("lam")
        style = dict(ls="--", lw=2.0, color="#111111") if a == ref_rule else dict(ls=":", lw=1.2, color="#888888")
        ax.plot(g.lam, g[f"{metric}_iqm"], label=LABEL.get(a, a), zorder=1, **style)
        if a == ref_rule:
            ax.fill_between(g.lam, g[f"{metric}_ci_lo"], g[f"{metric}_ci_hi"],
                            color="#111111", alpha=0.10, lw=0, zorder=1)

    star_p = AGG / f"{env_id}_lambda_star.json"
    if star_p.exists():
        st = json.loads(star_p.read_text(encoding="utf-8"))
        for s in st.get("results", []):
            lam = s.get("lam_star_pt")
            if lam:
                ax.axvline(lam, color=COLOR.get(s["learner"], "#666"), ls="-.", lw=1.0, alpha=0.6)
                ax.annotate(f"λ*({LABEL.get(s['learner'], s['learner'])})={lam:g}", xy=(lam, ax.get_ylim()[0]),
                            xytext=(3, 8), textcoords="offset points", rotation=90,
                            fontsize=8, color=COLOR.get(s["learner"], "#666"))

    ax.set_xlabel("행동 1번의 비용  λ  (오른쪽일수록 행동이 비싸다)")
    ax.set_ylabel("비용 반영 총보상 r'  (높을수록 좋다)" if metric == "cost_return" else metric)
    ax.set_title(f"λ-성능 지도 — {env_id}\n선=IQM, 띠=95% 신뢰구간 (시드 {int(agg.n_seeds.max())}개)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{env_id}_lambda_map_{metric}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  저장: {out.relative_to(ROOT)}")
    return out


def action_map(env_id: str) -> Path | None:
    p = AGG / f"{env_id}_iqm.csv"
    if not p.exists():
        return None
    agg = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=140)
    for a in sorted(agg.agent.unique()):
        g = agg[agg.agent == a].sort_values("lam")
        if str(a).startswith("rule_"):
            ax.plot(g.lam, g.n_actions_iqm, ls="--" if a == "rule_pump" else ":",
                    color="#111111" if a == "rule_pump" else "#999999", lw=1.6, label=LABEL.get(a, a))
        else:
            ax.plot(g.lam, g.n_actions_iqm, "-o", ms=4, color=COLOR.get(a), label=LABEL.get(a, a))
            ax.fill_between(g.lam, g.n_actions_ci_lo, g.n_actions_ci_hi, color=COLOR.get(a), alpha=0.18, lw=0)
    ax.set_xlabel("행동 1번의 비용  λ")
    ax.set_ylabel("에피소드당 행동 횟수 (IQM)")
    ax.set_title(f"비용이 오르면 행동을 얼마나 아끼는가 — {env_id}")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{env_id}_action_map.png"
    fig.savefig(out); plt.close(fig)
    print(f"  저장: {out.relative_to(ROOT)}")
    return out


def learning_curves(env_id: str, lams=None) -> list[Path]:
    """계열별 학습 곡선 (시드 평균). λ마다 한 칸."""
    raw = ROOT / "results" / "raw" / env_id
    if not raw.exists():
        return []
    frames = []
    for c in raw.rglob("seed*_curve.csv"):
        agent = c.parent.parent.name
        lam = float(c.parent.name.replace("lam", ""))
        df = pd.read_csv(c)
        df["agent"], df["lam"], df["seed"] = agent, lam, c.name
        frames.append(df)
    if not frames:
        return []
    all_df = pd.concat(frames)
    lams = lams or sorted(all_df.lam.unique())
    n = len(lams)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), dpi=140, sharey=True, squeeze=False)
    for ax, lam in zip(axes[0], lams):
        sub = all_df[all_df.lam == lam]
        for a in sorted(sub.agent.unique()):
            g = sub[sub.agent == a].groupby("step").cost_return_iqm.agg(["mean", "std"]).reset_index()
            ax.plot(g.step, g["mean"], color=COLOR.get(a), lw=1.8, label=LABEL.get(a, a))
            ax.fill_between(g.step, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=COLOR.get(a), alpha=0.15, lw=0)
        ax.set_title(f"λ = {lam:g}", fontsize=10)
        ax.set_xlabel("환경 스텝")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("평가 총보상 r' (IQM)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"학습 곡선 — {env_id} (선=시드 평균, 띠=±1 표준편차)", y=1.02)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{env_id}_learning_curves.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  저장: {out.relative_to(ROOT)}")
    return [out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="all")
    ap.add_argument("--rule", default=None, help="기준 규칙 (기본: 환경별 자동)")
    a = ap.parse_args()
    envs = ([p.name for p in (ROOT / "results" / "raw").iterdir() if p.is_dir()]
            if a.env == "all" else [a.env])
    for env_id in envs:
        print(f"[{env_id}] 그림 생성")
        ref = a.rule or REF_RULE.get(env_id, "rule_pump")
        lam_map(env_id, "cost_return", ref)
        lam_map(env_id, "raw_return", ref)
        action_map(env_id)
        learning_curves(env_id)


if __name__ == "__main__":
    main()
