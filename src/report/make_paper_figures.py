"""논문용 그림 제작 — 학교 양식(그림 제목은 그림 하단, 국문·영문 병기)에 맞춘 고해상도 출력.

그림 안에는 제목을 넣지 않는다. 제목은 문서(.docx)에서 그림 아래에 붙기 때문이다.
캡션 문구는 results/figures/paper/captions.json에 함께 저장해 문서 조립 때 그대로 쓴다.

실행: python -m src.report.make_paper_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregate"
OUT = ROOT / "results" / "figures" / "paper"
DPI = 300

for f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

LABEL = {"dqn": "표준 DQN", "temporl": "TempoRL 방식", "lazy": "Lazy-MDP 방식",
         "rule_best": "λ마다 가장 센 고정 규칙", "rule_pump": "고정 규칙 (pump)",
         "rule_threshold": "고정 규칙 (임계값)", "rule_noop": "고정 규칙 (무행동)",
         "rule_cautious": "고정 규칙 (신중)", "rule_cautious_d1": "고정 규칙 (신중 d=1)"}
COLOR = {"dqn": "#1f77b4", "temporl": "#d62728", "lazy": "#2ca02c"}
REF = {"MountainCar-v0": "rule_pump", "LunarLander-v3": "rule_threshold",
       "MinAtar_Freeway-v1": "rule_cautious"}
CAPTIONS: dict[str, dict] = {}


def _box(ax, x, y, w, h, text, fc, ec, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018",
                                fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.5, fontweight=weight)


def _arrow(ax, p0, p1, text=None, color="#333333", fs=7.6, rad=0.0, dy=0.016):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.2,
                                 color=color, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, text, ha="center", va="bottom",
                fontsize=fs, color=color, zorder=5)


def fig1_method() -> Path:
    """그림 1 — 비용 래퍼 한 겹과 비교 4계열의 구조."""
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=DPI)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── 왼쪽: 비용 래퍼가 끼어드는 자리 ──────────────────────────────
    ax.text(0.005, 0.965, "(가) 행동 비용 래퍼 — 환경과 학습기 사이에 한 겹",
            fontsize=10.5, fontweight="bold", va="top")
    _box(ax, 0.005, 0.62, 0.125, 0.16, "학습기\n(정책)", "#eef4fb", "#1f77b4", 9.5, "bold")
    _box(ax, 0.205, 0.60, 0.185, 0.20,
         "행동 비용 래퍼\nActionCostWrapper", "#fff6e5", "#e08a00", 9.5, "bold")
    _box(ax, 0.465, 0.62, 0.125, 0.16, "환경\n(Gymnasium)", "#eef7ee", "#2ca02c", 9.5, "bold")

    _arrow(ax, (0.130, 0.750), (0.205, 0.750), None)
    _arrow(ax, (0.390, 0.750), (0.465, 0.750), None)
    _arrow(ax, (0.465, 0.650), (0.390, 0.650), None)
    _arrow(ax, (0.205, 0.650), (0.130, 0.650), None)
    ax.text(0.1675, 0.762, "행동 a", ha="center", fontsize=7.8, color="#333333")
    ax.text(0.4275, 0.762, "행동 a 그대로", ha="center", fontsize=7.8, color="#333333")
    ax.text(0.1675, 0.618, "보상 r\u2032", ha="center", fontsize=7.8, color="#333333")
    ax.text(0.4275, 0.618, "보상 r", ha="center", fontsize=7.8, color="#333333")

    ax.text(0.2975, 0.545,
            "r′ = r - λ · 1[a ≠ a_noop]",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#b35c00",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fffaf0", ec="#e08a00", lw=1.1))
    ax.text(0.2975, 0.475,
            "no-op(아무것도 안 함)은 공짜, 그 밖의 행동은 1회당 λ",
            ha="center", va="center", fontsize=8.6, color="#555555")
    ax.text(0.2975, 0.425,
            "학습·평가 코드는 비용이 붙은 줄 모른다 → 네 계열에 같은 규칙이 적용된다",
            ha="center", va="center", fontsize=8.2, color="#777777", style="italic")

    # ── 오른쪽: 비교 4계열 ──────────────────────────────────────────
    ax.text(0.665, 0.965, "(나) 같은 λ·같은 예산으로 겨루는 네 계열",
            fontsize=10.5, fontweight="bold", va="top")
    series = [
        ("(가) 표준 DQN", "매 스텝 행동을 하나 고른다", "#eef4fb", "#1f77b4"),
        ("(나) TempoRL 방식", "행동 + 유지 길이 j를 함께 배운다", "#fdeeee", "#d62728"),
        ("(다) Lazy-MDP 방식", "직접 할지 / 기본 규칙에 맡길지 고른다", "#eef7ee", "#2ca02c"),
        ("(라) 고정 규칙", "학습 없음 — 무행동 · k스텝 주기 · 임계값", "#f2f2f2", "#777777"),
    ]
    y = 0.745
    for name, desc, fc, ec in series:
        _box(ax, 0.655, y, 0.335, 0.108, "", fc, ec)
        ax.text(0.672, y + 0.070, name, fontsize=9.3, fontweight="bold", va="center", ha="left", zorder=3)
        ax.text(0.672, y + 0.031, desc, fontsize=8.2, color="#444444", va="center", ha="left", zorder=3)
        y -= 0.152

    # ── 아래: 실험 절차 ────────────────────────────────────────────
    ax.text(0.005, 0.335, "(다) 절차 — λ를 키워 가며 같은 자로 재고, 교차점을 λ*로 읽는다",
            fontsize=10.5, fontweight="bold", va="top")
    steps = [
        "λ 격자\n0 → 큰 값",
        "계열 4종\n× 시드 10개",
        "같은 환경 스텝 예산\n탐험 끈 평가",
        "IQM + 95%\n계층 부트스트랩 CI",
        "λ-성능 지도\n임계 비용 λ*",
    ]
    w, gap = 0.168, 0.038
    x = 0.012
    for i, st in enumerate(steps):
        fc = "#eaf0f8" if i < 4 else "#fff2f2"
        ec = "#4a7ebb" if i < 4 else "#d62728"
        _box(ax, x, 0.10, w, 0.165, st, fc, ec, 8.8, "bold" if i == 4 else "normal")
        if i < len(steps) - 1:
            _arrow(ax, (x + w, 0.1825), (x + w + gap, 0.1825))
        x += w + gap

    ax.text(0.5, 0.035,
            "λ* = 학습이 그 λ에서 가장 센 고정 규칙을 더 이상 이기지 못하게 되는 가장 작은 λ",
            ha="center", fontsize=9, color="#b3261e", fontweight="bold")

    fig.tight_layout()
    p = OUT / "fig1_method.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    CAPTIONS["fig1"] = {
        "file": p.name,
        "ko": "행동 비용 래퍼와 비교 4계열의 구조",
        "en": "Structure of the action-cost wrapper and the four compared families",
    }
    print(f"  저장: {p.relative_to(ROOT)}")
    return p


ENV_KO = {"MountainCar-v0": "MountainCar-v0 (보상이 희소한 환경)",
          "LunarLander-v3": "LunarLander-v3 (보상이 조밀한 환경)",
          "MinAtar_Freeway-v1": "MinAtar Freeway (규칙이 쓸 만하지만 이길 여지가 있는 환경)"}


def _load(env_id: str) -> pd.DataFrame | None:
    p = AGG / f"{env_id}_iqm.csv"
    if not p.exists():
        print(f"  [건너뜀] {env_id} 집계 파일 없음")
        return None
    return pd.read_csv(p)


def lambda_map(env_id: str, tag: str, symlog: bool = False) -> Path | None:
    """그림 2·3 — λ-성능 지도. 선=IQM, 띠=95% 신뢰구간, 검은 점선=기준 고정 규칙."""
    agg = _load(env_id)
    if agg is None:
        return None
    ref = REF[env_id]
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    if not learners:
        print(f"  [건너뜀] {env_id} — 학습 계열 결과가 아직 없다 (규칙만으로는 지도를 그리지 않는다)")
        return None

    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=DPI)
    for a in learners:
        g = agg[agg.agent == a].sort_values("lam")
        ax.plot(g.lam, g.cost_return_iqm, "-o", ms=4.2, lw=2.0, color=COLOR[a],
                label=LABEL[a], zorder=4)
        ax.fill_between(g.lam, g.cost_return_ci_lo, g.cost_return_ci_hi,
                        color=COLOR[a], alpha=0.16, lw=0, zorder=2)
    for name, style in ((ref, dict(ls="--", lw=2.0, color="#111111")),
                        ("rule_best", dict(ls="-.", lw=1.6, color="#7b3fb5")),
                        ("rule_noop", dict(ls=":", lw=1.4, color="#888888"))):
        if name not in set(agg.agent):
            continue
        g = agg[agg.agent == name].sort_values("lam")
        ax.plot(g.lam, g.cost_return_iqm, label=LABEL.get(name, name), zorder=3, **style)

    star_p = AGG / f"{env_id}_lambda_star.json"
    lam_stars = []
    if star_p.exists():
        st = json.loads(star_p.read_text(encoding="utf-8"))
        for r in st.get("results", []):
            if r.get("lam_star_pt"):
                lam_stars.append((r["learner"], r["lam_star_pt"]))
    for learner, lam in lam_stars:
        ax.axvline(lam, color=COLOR.get(learner, "#666"), ls="-.", lw=0.9, alpha=0.55, zorder=1)

    n_seeds = int(agg.n_seeds.max())
    if symlog:
        # MountainCar는 흥미로운 변화가 전부 λ<0.05에 몰려 있어 선형 축에서는 벽에 붙어 보이지 않는다.
        # 0 근처를 넓혀 보는 대칭 로그 축을 쓴다 (λ=0도 그대로 표시된다).
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xlabel("행동 1번의 비용  λ   (0 부근을 넓혀 그린 축)", fontsize=10)
    else:
        ax.set_xlabel("행동 1번의 비용  λ   (오른쪽일수록 행동이 비싸다)", fontsize=10)
    ax.set_ylabel("비용 반영 총보상 r′  (높을수록 좋다)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.92)
    ax.tick_params(labelsize=9)
    ax.text(0.99, 0.02, f"시드 {n_seeds}개 · IQM · 95% 계층 부트스트랩 CI",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6, color="#666666")
    fig.tight_layout()
    out = OUT / f"{tag}_lambda_map_{env_id}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    CAPTIONS[tag] = {
        "file": out.name,
        "ko": f"λ-성능 지도 — {ENV_KO.get(env_id, env_id)}",
        "en": f"Performance map over action cost λ on {env_id}",
        "note": (f"선은 IQM(사분위평균), 띠는 95% 계층 부트스트랩 신뢰구간(시드 {n_seeds}개). "
                 "검은 점선은 기준이 되는 최고 고정 규칙, 보라 일점쇄선은 λ마다 가장 센 규칙의 포락선이다."),
        "source": f"results/aggregate/{env_id}_iqm.csv",
    }
    print(f"  저장: {out.relative_to(ROOT)}")
    return out


def collapse_figure(env_id: str = "MountainCar-v0") -> Path | None:
    """그림 4 — 무행동 붕괴. 왼쪽=행동 횟수, 오른쪽=목표 도달률."""
    agg = _load(env_id)
    if agg is None:
        return None
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), dpi=DPI)

    for ax, (col, lo, hi, ylab) in zip(axes, [
            ("n_actions_iqm", "n_actions_ci_lo", "n_actions_ci_hi", "에피소드당 행동 횟수 (IQM)"),
            ("solved_iqm", "solved_ci_lo", "solved_ci_hi", "목표 도달률 (IQM)")]):
        for a in learners:
            g = agg[agg.agent == a].sort_values("lam")
            ax.plot(g.lam, g[col], "-o", ms=4.0, lw=1.9, color=COLOR[a], label=LABEL[a], zorder=3)
            ax.fill_between(g.lam, g[lo], g[hi], color=COLOR[a], alpha=0.15, lw=0, zorder=2)
        ref = REF[env_id]
        if ref in set(agg.agent):
            g = agg[agg.agent == ref].sort_values("lam")
            ax.plot(g.lam, g[col], ls="--", lw=1.7, color="#111111", label=LABEL.get(ref, ref), zorder=3)
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xlabel("행동 1번의 비용  λ  (0 부근을 넓혀 그린 축)", fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8.5)
    axes[0].legend(fontsize=7.8, loc="best", framealpha=0.92)
    n_seeds = int(agg.n_seeds.max())
    axes[1].text(0.99, 0.97, f"시드 {n_seeds}개 · IQM · 95% CI", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=7.4, color="#666666")
    fig.tight_layout()
    out = OUT / f"fig4_collapse_{env_id}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    CAPTIONS["fig4"] = {
        "file": out.name,
        "ko": "무행동 붕괴 — 비용이 조금만 붙어도 행동을 멈춘다 (MountainCar-v0)",
        "en": "Collapse to inaction: action count and success rate versus action cost λ on MountainCar-v0",
        "note": ("가로축은 λ=0 부근을 넓혀 보기 위해 대칭 로그 축(linthresh=0.001)을 썼다. "
                 "왼쪽은 에피소드당 행동 횟수, 오른쪽은 목표 도달률이다."),
        "source": f"results/aggregate/{env_id}_iqm.csv",
    }
    print(f"  저장: {out.relative_to(ROOT)}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_method()
    lambda_map("MountainCar-v0", "fig2", symlog=True)
    lambda_map("LunarLander-v3", "fig3")
    collapse_figure("MountainCar-v0")
    # 세 번째 환경은 학습 결과가 들어온 뒤에만 그린다 (규칙만 있으면 지도가 의미 없다)
    lambda_map("MinAtar_Freeway-v1", "fig5")
    (OUT / "captions.json").write_text(json.dumps(CAPTIONS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  캡션: {(OUT / 'captions.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
