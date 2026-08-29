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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregate"
OUT = ROOT / "results" / "figures" / "paper"
MIN_SEEDS_FOR_FIGURE = 5   # 이보다 적으면 λ 지도를 그리지 않는다
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
         "rule_threshold": "고정 규칙 (임계값·처음)",
         "rule_threshold_tuned": "고정 규칙 (임계값·튜닝)", "rule_noop": "고정 규칙 (무행동)",
         "rule_cautious": "고정 규칙 (신중)", "rule_cautious_d1": "고정 규칙 (신중 d=1)"}
COLOR = {"dqn": "#1f77b4", "temporl": "#d62728", "lazy": "#2ca02c"}
# 세로로 눕혀 쓰는 짧은 이름 — 긴 이름은 그림 밖으로 넘친다
SHORT = {"dqn": "DQN", "temporl": "TempoRL", "lazy": "Lazy-MDP"}
REF = {"MountainCar-v0": "rule_pump", "LunarLander-v3": "rule_threshold_tuned",
       "MinAtar_Freeway-v1": "rule_cautious"}
from matplotlib.ticker import FuncFormatter

PLAIN = FuncFormatter(lambda v, _pos: format(v, "g"))


CAPTIONS: dict[str, dict] = {}


def _box(ax, x, y, w, h, text, fc, ec, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018",
                                fc=fc, ec=ec, lw=1.3, zorder=2, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.5, fontweight=weight, clip_on=False)


def _arrow(ax, p0, p1, text=None, color="#333333", fs=7.6, rad=0.0, dy=0.016):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.2,
                                 color=color, zorder=4, clip_on=False,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, text, ha="center", va="bottom",
                fontsize=fs, color=color, zorder=5, clip_on=False)


def fig1_method() -> Path:
    """그림 1 — 비용 래퍼 한 겹과 비교 4계열의 구조."""
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=DPI)
    ax.set_xlim(-0.015, 1.015); ax.set_ylim(-0.02, 1.02); ax.axis("off")

    # ── 왼쪽: 비용 래퍼가 끼어드는 자리 ──────────────────────────────
    ax.text(0.005, 0.965, "① 행동 비용 래퍼 — 환경과 학습기 사이에 한 겹",
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
    ax.text(0.665, 0.965, "② 같은 λ·같은 예산으로 겨루는 네 계열",
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
    ax.text(0.005, 0.335, "③ 절차 — λ를 키워 가며 같은 자로 재고, 교차점을 λ*로 읽는다",
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
    fig.savefig(p, bbox_inches="tight", pad_inches=0.06, facecolor="white"); plt.close(fig)
    CAPTIONS["fig1"] = {
        "file": p.name,
        "ko": "행동 비용 래퍼와 비교 4계열의 구조",
        "en": "Structure of the action-cost wrapper and the four compared families",
    }
    print(f"  저장: {p.relative_to(ROOT)}")
    return p


ENV_KO = {"MountainCar-v0": "MountainCar-v0 (보상이 희소한 환경)",
          "LunarLander-v3": "LunarLander-v3 (보상이 조밀한 환경)",
          "MinAtar_Freeway-v1": "MinAtar Freeway (규칙 품질은 가운데, 보상은 희소한 환경)"}


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
    n_min = int(agg[agg.agent.isin(learners)].n_seeds.min())
    if n_min < MIN_SEEDS_FOR_FIGURE:
        print(f"  [건너뜀] {env_id} — 시드 {n_min}개뿐이라 지도를 그리지 않는다 "
              f"(최소 {MIN_SEEDS_FOR_FIGURE}개)")
        return None

    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=DPI)
    for a in learners:
        g = agg[agg.agent == a].sort_values("lam")
        ax.plot(g.lam, g.cost_return_iqm, "-o", ms=4.2, lw=2.0, color=COLOR[a],
                label=LABEL[a], zorder=4)
        ax.fill_between(g.lam, g.cost_return_ci_lo, g.cost_return_ci_hi,
                        color=COLOR[a], alpha=0.11, lw=0.7, ec=COLOR[a], zorder=2)
    # 포락선을 두껍고 옅게 **먼저** 깐다. 기준 규칙과 겹치는 구간에서는 검은 점선 둘레로
    # 보라 테가 비쳐 "이 구간은 기준 규칙이 곧 최강 규칙"이라는 사실이 눈에 보인다.
    for name, style in (("rule_best", dict(ls="-", lw=5.0, color="#b98fe0", alpha=0.55)),
                        (ref, dict(ls="--", lw=2.0, color="#111111")),
                        ("rule_noop", dict(ls=":", lw=1.4, color="#888888"))):
        if name not in set(agg.agent):
            continue
        g = agg[agg.agent == name].sort_values("lam")
        ax.plot(g.lam, g.cost_return_iqm, label=LABEL.get(name, name), zorder=3, **style)

    # λ* 표시선은 **최강 규칙 포락선 기준**을 쓴다.
    # 지정 기준 규칙만 보면 격자 끝까지 이기는 환경에서 교차가 없어 선이 안 그려지는데,
    # 비용이 커지면 최강 규칙이 무행동으로 바뀌므로 그쪽이 실질적인 문턱이다.
    star_p = AGG / f"{env_id}_lambda_star.json"
    stars = {}
    if star_p.exists():
        st = json.loads(star_p.read_text(encoding="utf-8"))
        for r in st.get("results_vs_best_rule", st.get("results", [])):
            v = r.get("lam_star_pt")
            if v is not None:                      # 0.0도 뜻이 있다 — 거짓값으로 버리면 안 된다
                stars[r["learner"]] = float(v)
    zero_at = [LABEL[a] for a, v in stars.items() if v == 0.0 and a in learners]
    xt = ax.get_xaxis_transform()          # x는 데이터 좌표, y는 축 비율
    have = [v for a, v in stars.items() if a in learners]

    # 이 그림의 주인공은 λ*다. 모든 계열이 규칙에 지는 구간을 회색으로 깔아
    # "여기서부터는 학습을 쓸 이유가 없다"를 선이 아니라 면적으로 보여 준다.
    # 단, λ*가 전부 0이면 이 음영이 그림 전체를 덮어 아무것도 구분해 주지 못한다.
    # 그때는 칠하지 않는다 — 아래 빨간 상자가 같은 말을 더 분명히 한다.
    if have and len(have) == len(learners) and max(have) > float(agg.lam.min()):
        xr = ax.get_xlim()
        ax.axvspan(max(have), xr[1], color="#9e9e9e", alpha=0.15, lw=0, zorder=0)
        ax.set_xlim(*xr)
        ax.annotate("← 학습이 이기는 구간   |   모든 계열이 규칙에 지는 구간 →",
                    xy=(max(have), 0.012), xycoords=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.4, color="#5f5f5f")

    for a, lam in sorted(stars.items(), key=lambda kv: kv[1]):
        if a not in learners or lam == 0.0:
            continue
        ax.axvline(lam, color=COLOR[a], ls="-.", lw=1.2, alpha=0.7, zorder=1)
        # 가로로, 그림 맨 위에 — 세로 라벨은 데이터 위를 가로질러 읽기 어려웠다
        ax.annotate(f"λ*={lam:g}" + chr(10) + SHORT[a], xy=(lam, 1.005), xycoords=xt,
                    ha="center", va="bottom", fontsize=7.4, color=COLOR[a],
                    fontweight="bold", linespacing=1.1)
    if zero_at:
        ax.annotate("λ*=0 — 비용이 없어도 규칙을 못 이김: " + ", ".join(zero_at),
                    xy=(0.02, 0.035), xycoords="axes fraction", fontsize=7.8,
                    color="#b3261e", fontweight="bold",
                    bbox=dict(fc="white", ec="#b3261e", lw=0.8, alpha=0.9, pad=2.5))

    n_seeds = int(agg.n_seeds.max())
    if symlog:
        # MountainCar는 흥미로운 변화가 전부 λ<0.05에 몰려 있어 선형 축에서는 벽에 붙어 보이지 않는다.
        # 0 근처를 넓혀 보는 대칭 로그 축을 쓴다 (λ=0도 그대로 표시된다).
        ax.set_xscale("symlog", linthresh=0.001)
        ax.xaxis.set_major_formatter(PLAIN)   # 수식 글꼴 눈금($10^{-3}$) 대신 평범한 숫자로
        # 대칭 로그 축은 0을 가운데 두므로 왼쪽 절반이 **음수 λ**가 된다. 비용에 음수는 없다.
        # 그대로 두면 그림 너비의 40%가 빈 채로 낭비되고 데이터가 오른쪽에 몰려 보인다.
        ax.set_xlim(left=-0.00018)
        ax.set_xticks([t for t in ax.get_xticks() if t >= 0])
        ax.set_xlabel("행동 1번의 비용  λ   (0 부근을 넓혀 그린 축 · 음수 구간 없음)", fontsize=10)
    else:
        ax.set_xlabel("행동 1번의 비용  λ   (오른쪽일수록 행동이 비싸다)", fontsize=10)
    ax.set_ylabel("비용 반영 총보상 r′  (높을수록 좋다)", fontsize=10)
    ax.grid(alpha=0.25)
    # 범례를 그림 안에 두면 자리를 어디로 잡아도 선을 가린다. 특히 논문의 비교 기준인
    # 규칙 선을 가리는 것이 가장 나쁘다. 축 아래로 빼면 가리는 것이 없다.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.155),
              ncol=3, frameon=False, handlelength=2.4, columnspacing=1.6)
    ax.tick_params(labelsize=9)
    ax.text(0.99, 0.975, f"시드 {n_seeds}개 · IQM · 95% 계층 부트스트랩 CI",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color="#666666")
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
            ax.fill_between(g.lam, g[lo], g[hi], color=COLOR[a], alpha=0.11, lw=0.7, ec=COLOR[a], zorder=2)
        ref = REF[env_id]
        if ref in set(agg.agent):
            g = agg[agg.agent == ref].sort_values("lam")
            ax.plot(g.lam, g[col], ls="--", lw=1.7, color="#111111", label=LABEL.get(ref, ref), zorder=3)
        ax.set_xscale("symlog", linthresh=0.001)
        ax.xaxis.set_major_formatter(PLAIN)
        ax.set_xlim(left=-0.00018)          # 음수 λ는 없다 — 왼쪽 절반을 비워 두지 않는다
        ax.set_xticks([t for t in ax.get_xticks() if t >= 0])
        ax.set_xlabel("행동 1번의 비용  λ  (0 부근을 넓혀 그린 축)", fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8.5)
    # 이 그림의 발견은 "성능이 먼저 무너지고, 행동은 나중에 줄어든다"이다.
    # 그 사실을 독자가 두 패널을 눈으로 오가며 맞춰 보게 두지 않고, **그 구간을 칠해서**
    # 두 패널 같은 자리에 보여 준다. 기준은 대표 계열(표준 DQN)로 한다.
    #   구간 시작 = 도달률이 λ=0의 절반 아래로 처음 내려가는 λ
    #   구간 끝   = 행동 횟수가 λ=0의 절반 아래로 처음 내려가는 λ
    def _first_below(a: str, col: str, frac: float) -> float | None:
        g0 = agg[(agg.agent == a) & (agg.lam == 0.0)]
        g = agg[(agg.agent == a) & (agg.lam > 0)].sort_values("lam")
        if g0.empty or g.empty:
            return None
        hit = g[g[col] < float(g0.iloc[0][col]) * frac]
        return None if hit.empty else float(hit.iloc[0].lam)

    lead = "dqn" if "dqn" in learners else (learners[0] if learners else None)
    s_half = _first_below(lead, "solved_iqm", 0.5) if lead else None
    a_half = _first_below(lead, "n_actions_iqm", 0.5) if lead else None
    if s_half is not None and a_half is not None and a_half > s_half:
        for ax in axes:
            ax.axvspan(s_half, a_half, color="#f0a500", alpha=0.22, lw=0, zorder=0)
            for x in (s_half, a_half):     # 로그 축에서 이 구간은 매우 좁다 — 경계를 그어 준다
                ax.axvline(x, color="#f0a500", lw=1.1, alpha=0.85, zorder=1)
        # 좁은 띠는 칠하기만 해서는 눈에 띄지 않는다. 빈자리에 설명을 두고 화살표로 가리킨다.
        axes[0].annotate("여기서 성능은 이미 무너졌는데" + chr(10) + "행동 횟수는 아직 λ=0과 비슷하다" + chr(10) + f"(λ {s_half:g} ~ {a_half:g})",
                         xy=(s_half, 0.42), xycoords=axes[0].get_xaxis_transform(),
                         xytext=(0.045, 0.30), textcoords="axes fraction",
                         ha="left", va="center", fontsize=8.0, color="#8a5a00",
                         fontweight="bold", linespacing=1.35,
                         bbox=dict(fc="white", ec="#f0a500", lw=1.0, alpha=0.95, pad=3.5),
                         arrowprops=dict(arrowstyle="->", color="#c98600", lw=1.4,
                                         connectionstyle="arc3,rad=-0.18"))

    for a in learners:
        g0 = agg[(agg.agent == a) & (agg.lam == 0.0)]
        g = agg[(agg.agent == a) & (agg.lam > 0)].sort_values("lam")
        if g0.empty or g.empty:
            continue
        half = g[g.solved_iqm < float(g0.iloc[0].solved_iqm) * 0.5]
        if half.empty:
            continue
        lam_h = float(half.iloc[0].lam)
        for ax in axes:
            ax.axvline(lam_h, color=COLOR[a], ls=":", lw=1.1, alpha=0.55, zorder=1)
    axes[0].annotate("세로 점선 = 계열별로 목표 도달률이 절반이 되는 λ",
                     xy=(0.02, 0.06), xycoords="axes fraction", fontsize=7.4, color="#555555",
                     va="bottom")
    axes[0].legend(fontsize=7.8, loc="best", framealpha=0.92)
    n_seeds = int(agg.n_seeds.max())
    # 시드 수는 그림 안에 적지 않는다 — 오른쪽 패널에서 기준 규칙 점선과 겹친다. 캡션에 넣는다.
    fig.tight_layout()
    out = OUT / f"fig4_collapse_{env_id}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    CAPTIONS["fig4"] = {
        "file": out.name,
        "ko": "무행동 붕괴는 두 단계로 일어난다 — 성능이 먼저 무너지고, 행동은 나중에 줄어든다 (MountainCar-v0)",
        "en": ("Collapse to inaction happens in two stages: success rate falls before the action "
               "count does (MountainCar-v0)"),
        "note": ("가로축은 λ=0 부근을 넓혀 보기 위해 대칭 로그 축을 썼다(0.001 아래는 선형). "
                 "왼쪽은 에피소드당 행동 횟수, 오른쪽은 목표 도달률이며, "
                 f"선은 IQM, 띠는 95% 계층 부트스트랩 신뢰구간이다(시드 {n_seeds}개). "
                 "검은 점선은 기준 규칙(pump)이다. 주황색 띠는 표준 DQN 기준으로 "
                 "**목표 도달률은 이미 절반 아래로 내려갔으나 행동 횟수는 아직 λ=0의 절반 위에 있는** "
                 "구간이다 — 이 구간이 존재한다는 것이 '행동을 아끼다 실패하는 것이 아니라, "
                 "행동을 그대로 하면서 실패한다'는 뜻이다."),
        "source": f"results/aggregate/{env_id}_iqm.csv",
    }
    print(f"  저장: {out.relative_to(ROOT)}")
    return out


# ── 그림 5: 공정성 점검의 학습 곡선 ──────────────────────────────────────────
# 왜 필요한가: 논문의 가장 논쟁적인 주장이 "MountainCar에서는 비용이 없어도(λ=0)
# 학습이 규칙을 못 이긴다"이다. 여기에 대한 첫 반론은 언제나 "학습이 덜 됐다"이다.
# 표의 최종 점수만으로는 그 반론을 못 막는다 — **곡선이 평평해졌는데도 규칙 아래**라는
# 것을 보여야 한다. 강화학습 논문에서 학습 곡선이 빠지면 심사자가 가장 먼저 묻는다.
FAIR_ROOMS = [
    ("MountainCar-v0", "본실험 (30만 스텝)", "#1f77b4", "-"),
    ("MountainCar-v0@budget1M_epsconst", "예산 3.3배 (100만 스텝)", "#d62728", "-"),
    ("MountainCar-v0@budget1M_epsdecay", "예산 3.3배 + ε 감소", "#ff7f0e", "--"),
    ("MountainCar-v0@budget1M_wide", "예산 3.3배 + 신경망 확대", "#2ca02c", "-."),
]


def fairness_curves(env_id: str = "MountainCar-v0", tag: str = "fig_fair") -> Path | None:
    """그림 5 — λ=0에서 예산·탐험·신경망을 바꿔 가며 그린 학습 곡선."""
    import glob
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=DPI)
    drawn = 0
    for room, label, color, ls in FAIR_ROOMS:
        curves = []
        for f in sorted(glob.glob(str(ROOT / "results" / "raw" / room / "dqn" / "lam0.0"
                                       / "seed*_curve.csv"))):
            try:
                c = pd.read_csv(f)
            except Exception:
                continue
            if {"step", "raw_return_iqm"} <= set(c.columns):
                curves.append(c.set_index("step")["raw_return_iqm"])
        if not curves:
            continue
        m = pd.concat(curves, axis=1).sort_index()
        # 시드마다 IQM(사분위평균)을 취한다 — 표·λ 지도와 같은 잣대여야 한다
        q1, q3 = m.quantile(0.25, axis=1), m.quantile(0.75, axis=1)
        iqm = m.apply(lambda r: r[(r >= q1[r.name]) & (r <= q3[r.name])].mean(), axis=1)
        ax.plot(m.index / 1000, iqm, ls, color=color, lw=1.9,
                label=f"{label} · 시드 {m.shape[1]}개", zorder=3)
        drawn += 1
    if not drawn:
        print("  [건너뜀] 공정성 곡선 — λ=0 곡선 자료가 없다")
        plt.close(fig)
        return None

    agg = _load(env_id)
    rule = None
    if agg is not None and REF[env_id] in set(agg.agent):
        g = agg[(agg.agent == REF[env_id]) & (agg.lam == 0.0)]
        if not g.empty:
            rule = float(g.iloc[0].raw_return_iqm)
            ax.axhline(rule, color="#111111", ls="--", lw=2.0, zorder=4,
                       label="고정 규칙 (pump) — 학습 없이 얻는 점수")
            ax.annotate("어떤 조건에서도 이 선 위로 올라가지 못한다",
                        xy=(0.99, rule), xycoords=ax.get_yaxis_transform(),
                        xytext=(-6, 8), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8.2, color="#b3261e",
                        fontweight="bold")

    ax.set_xlabel("학습에 쓴 환경 스텝 (천 스텝)", fontsize=10)
    ax.set_ylabel("총보상 (비용 없음, λ=0)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False, handlelength=2.6, columnspacing=1.6)
    fig.tight_layout()
    out = OUT / f"{tag}_fairness_curves_{env_id}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    CAPTIONS[tag] = {
        "file": out.name,
        "ko": "예산을 늘려도 학습 곡선은 규칙 아래에서 평평해진다 (MountainCar-v0, λ=0)",
        "en": ("Learning curves flatten below the rule even with a larger budget "
               "(MountainCar-v0, λ = 0)"),
        "note": ("비용이 없는 조건(λ=0)에서만 그렸다 — 여기서 지면 비용을 논할 필요가 없기 때문이다. "
                 "선은 시드에 대한 IQM이다. 본실험은 시드 10개, 예산을 늘린 세 조건은 "
                 "**파일럿이라 시드 5개**이므로 이 그림은 경향을 보이는 용도이고, "
                 "우열 판정은 시드 10개로 한 표에서만 말한다. "
                 "곡선의 각 점은 5 에피소드 평가라 출렁인다 — 최종 판정은 "
                 "50 에피소드 평가로 한다. "
                 "검은 점선은 학습 없이 얻는 pump 규칙의 점수다."),
        "source": "results/raw/MountainCar-v0*/dqn/lam0.0/seed*_curve.csv",
    }
    print(f"  저장: {out.relative_to(ROOT)}")
    return out


# ── 인과 실험 그림: 비용을 켜는 시점만 바꾼 대조 ────────────────────────────
# 왜 필요한가: 이 논문의 다섯 기여 중 '무행동 붕괴는 탐험 실패다'만 그림이 없었다.
# 표 8이 숫자를 다 담고 있지만, "같은 비용·같은 예산인데 켜는 시점만 바꿨더니
# 0%가 66%가 되었다"는 것은 **선 두 개의 간격**으로 보여 줄 때 가장 빨리 읽힌다.
def causal_figure(tag: str = "fig_causal") -> Path | None:
    """비용을 처음부터 물릴 때 vs 절반 뒤에 켤 때 — 도달률과 행동 횟수."""
    p = AGG / "causal_warmup.json"
    if not p.exists():
        print("  [건너뜀] 인과 실험 집계가 아직 없다")
        return None
    try:
        res = json.loads(p.read_text(encoding="utf-8")).get("results", [])
    except Exception:
        return None
    if not res:
        return None
    lams = sorted({float(r["lam"]) for r in res})
    ags = [a for a in ("dqn", "temporl", "lazy") if any(r["agent"] == a for r in res)]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=DPI)
    panels = [("solved", "목표 도달률", lambda v: v * 100, "%"),
              ("actions", "에피소드당 행동 횟수", lambda v: v, "회")]
    for ax, (key, ylab, conv, unit) in zip(axes, panels):
        xs, xticks = [], []
        pos = 0
        for lam in lams:
            for ag in ags:
                r = next((x for x in res if float(x["lam"]) == lam and x["agent"] == ag), None)
                if r is None or not r["from_start"].get(key) or not r["warmup"].get(key):
                    pos += 1
                    continue
                a = conv(r["from_start"][key]["iqm"])
                b = conv(r["warmup"][key]["iqm"])
                ax.plot([pos, pos], [a, b], color="#bbbbbb", lw=1.6, zorder=1)
                ax.plot(pos, a, "o", ms=7, color="#ffffff", mec=COLOR[ag], mew=2.0, zorder=3)
                ax.plot(pos, b, "o", ms=7, color=COLOR[ag], mec=COLOR[ag], zorder=3)
                ax.annotate("", xy=(pos, b), xytext=(pos, a), zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=COLOR[ag], lw=1.4,
                                            shrinkA=7, shrinkB=7, alpha=0.75))
                xs.append(pos)
                xticks.append(SHORT[ag])
                pos += 1
            pos += 0.8
        step = len(ags) + 0.8
        ax.set_xticks([i * step + (len(ags) - 1) / 2 for i in range(len(lams))])
        ax.set_xticklabels(["λ=" + format(l, "g") for l in lams], fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylab + (" (%)" if unit == "%" else " (회)"), fontsize=9.5)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(labelsize=9, length=0)
        for i in range(1, len(lams)):        # λ 구간 사이에 옅은 칸막이
            ax.axvline(i * step - 0.9, color="#dddddd", lw=1.0, zorder=0)
    hs = [axes[0].plot([], [], "o", ms=7, color=COLOR[a], mec=COLOR[a], label=LABEL[a])[0]
          for a in ags]
    hs.append(axes[0].plot([], [], "o", ms=7, color="#ffffff", mec="#555555", mew=2.0,
                           label="처음부터 비용")[0])
    hs.append(axes[0].plot([], [], "o", ms=7, color="#555555", mec="#555555",
                           label="절반 뒤 비용")[0])
    fig.legend(handles=hs, fontsize=8.6, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = OUT / (tag + "_warmup_vs_from_start.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)

    n_expl = sum(1 for r in res if str(r["verdict"]).startswith("탐험 실패"))
    CAPTIONS[tag] = {
        "file": out.name,
        "ko": "비용을 켜는 시점만 바꾸면 결과가 달라진다 — 무행동 붕괴는 탐험 실패다 (MountainCar-v0)",
        "en": ("Only the moment the cost is switched on differs: the collapse to inaction is an "
               "exploration failure, not an optimum (MountainCar-v0)"),
        "note": ("빈 점은 처음부터 비용을 문 조건, 채운 점은 예산의 앞 절반을 비용 없이 학습시킨 조건이며 "
                 "화살표는 그 변화다. 두 조건은 환경 스텝 예산·시드·평가 방식이 모두 같고 "
                 "**학습 중 보상 신호만** 다르다. 평가는 양쪽 모두 진짜 λ로 한다 — 성적표는 언제나 "
                 f"비용이 있는 세상에서 매긴다. 비교 {len(res)}건 중 {n_expl}건이 '탐험 실패'로 판정됐고, "
                 "'무행동이 최적'을 지지하는 조건은 하나도 없었다."),
        "source": "results/aggregate/causal_warmup.json",
    }
    print(f"  저장: {out.relative_to(ROOT)}")
    return out


# ── 행동-성능 상충 그림 ─────────────────────────────────────────────────────
# 왜 필요한가: 심사자가 물을 만한데 지금 어떤 그림도 답하지 않는 것이 하나 있다 —
# **"행동을 아낀 만큼 무엇을 얻었는가."** λ 지도는 비용까지 반영한 점수만 보여 주므로
# '행동을 줄여서 이긴 것'과 '그냥 잘해서 이긴 것'이 구분되지 않는다.
#
# 가로를 행동 횟수, 세로를 **비용 빼기 전** 원보상으로 놓으면 그 둘이 분리된다.
# 왼쪽 위로 갈수록 좋다(적게 움직이고 많이 받는다). 그리고 이 축에서는
# 같은 r′를 주는 점들이 기울기 λ인 직선을 이루므로, 독자가 직선을 기울여 보며
# "λ가 이만큼일 때 누가 이기나"를 읽을 수 있다.
TRADEOFF_ENVS = ["MountainCar-v0", "LunarLander-v3", "MinAtar_Freeway-v1"]
RULE_MARK = {"rule_noop": ("무행동", "s"), "rule_pump": ("pump 규칙", "*"),
             "rule_threshold": ("임계값 규칙(처음)", "P"),
             "rule_threshold_tuned": ("임계값 규칙(튜닝)", "*"),
             "rule_cautious": ("신중 규칙", "*")}


# 패널이 여럿인 그림용 짧은 이름. 긴 ENV_KO를 쓰면 제목끼리 겹친다.
ENV_SHORT = {"MountainCar-v0": "MountainCar (희소)",
             "LunarLander-v3": "LunarLander (조밀)",
             "MinAtar_Freeway-v1": "MinAtar Freeway (희소·규칙 중간)"}


def tradeoff_figure(tag: str = "fig_tradeoff") -> Path | None:
    """행동 횟수 대 원보상. 고정 규칙이 지배하는 영역을 함께 칠한다."""
    envs = []
    for e in TRADEOFF_ENVS:
        f = AGG / (e + "_iqm.csv")
        if not f.exists():
            continue
        a = pd.read_csv(f)
        L = a[a.agent.isin(["dqn", "temporl", "lazy"])]
        if L.empty:
            continue
        if int(L.n_seeds.min()) < MIN_SEEDS_FOR_FIGURE:
            print(f"  [건너뜀] 상충 그림에서 {e} 제외 — 시드 {int(L.n_seeds.min())}개뿐")
            continue
        envs.append(e)
    if not envs:
        return None
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6), dpi=DPI)
    # 두 패널일 때를 기준으로, 패널이 늘어난 만큼 글자를 키운다
    fs = 1.0 if n <= 2 else (2 / n) ** -0.85
    axes = [axes] if len(envs) == 1 else list(axes)
    for ax, env in zip(axes, envs):
        agg = _load(env)
        if agg is None:
            continue
        ref = REF[env]
        for a in ("dqn", "temporl", "lazy"):
            if a not in set(agg.agent):
                continue
            d = agg[agg.agent == a].sort_values("lam")
            ax.plot(d.n_actions_iqm, d.raw_return_iqm, "-o", ms=3.4, lw=1.5,
                    color=COLOR[a], alpha=0.9, label=LABEL[a], zorder=3)
            f, l = d.iloc[0], d.iloc[-1]
            ax.annotate("λ=0", xy=(f.n_actions_iqm, f.raw_return_iqm), xytext=(4, 5),
                        textcoords="offset points", fontsize=7.2 * fs, color=COLOR[a])
            ax.annotate(f"λ={l.lam:g}", xy=(l.n_actions_iqm, l.raw_return_iqm), xytext=(4, -10),
                        textcoords="offset points", fontsize=7.2 * fs, color=COLOR[a])
        for rname, (ko, mk) in RULE_MARK.items():
            if rname not in set(agg.agent):
                continue
            d = agg[agg.agent == rname].iloc[0]
            big = rname == ref
            ax.plot(d.n_actions_iqm, d.raw_return_iqm, mk, ms=15 if big else 8,
                    color="#111111" if big else "#777777", zorder=5,
                    label=ko + (" (기준)" if big else ""))
        # 기준 규칙이 지배하는 영역(행동은 더 쓰고 보상은 더 적은 곳)을 칠한다.
        # **데이터를 다 그린 뒤에** 칠해야 한다 — 먼저 칠하면 축 범위가 음영 크기로 잡힌다.
        g = agg[agg.agent == ref]
        if not g.empty:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            rx, ry = float(g.iloc[0].n_actions_iqm), float(g.iloc[0].raw_return_iqm)
            ax.add_patch(Rectangle((rx, y0), max(0.0, x1 - rx), max(0.0, ry - y0),
                                   color="#d62728", alpha=0.07, lw=0, zorder=0))
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xlabel("에피소드당 행동 횟수 (적을수록 아낀다)", fontsize=9.5 * fs)
        ax.set_ylabel("원보상 r  (비용 빼기 전)", fontsize=9.5 * fs)
        ax.set_title(ENV_SHORT.get(env, ENV_KO.get(env, env)), fontsize=10 * fs,
                     fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8.5 * fs)
        ax.legend(fontsize=7.4 * fs, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                  ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.4)
        ax.annotate("← 좋아지는 방향 (적게 움직이고 많이 받는다)", xy=(0.03, 0.965),
                    xycoords="axes fraction", fontsize=7.6 * fs, color="#2e7d32", fontweight="bold")
    fig.tight_layout()
    out = OUT / (tag + "_actions_vs_return.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    CAPTIONS[tag] = {
        "file": out.name,
        "ko": "행동을 아낀 만큼 무엇을 얻었는가 — 행동 횟수와 원보상의 상충",
        "en": "What the saved actions buy: episode actions versus raw return (before cost)",
        "note": ("세로축은 **비용을 빼기 전** 원보상이라 '행동을 줄여서 이긴 것'과 "
                 "'그냥 잘해서 이긴 것'이 구분된다. 선은 λ를 0부터 키우며 이은 것이고 "
                 "양 끝에 λ를 적었다. 왼쪽 위로 갈수록 좋다. 옅은 붉은 영역은 "
                 "**기준 규칙보다 행동은 더 쓰고 보상은 더 적은** 자리다 — 그 안에 있으면 "
                 "어떤 λ에서도 기준 규칙을 이길 수 없다. 이 축에서 같은 비용 반영 점수 r′를 "
                 "주는 점들은 기울기 λ의 직선을 이룬다."),
        "source": "results/aggregate/{MountainCar-v0,LunarLander-v3}_iqm.csv",
    }
    print(f"  저장: {out.relative_to(ROOT)}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_method()
    lambda_map("MountainCar-v0", "fig2", symlog=True)
    lambda_map("LunarLander-v3", "fig3")
    collapse_figure("MountainCar-v0")
    causal_figure("fig_causal")
    tradeoff_figure("fig_tradeoff")
    fairness_curves("MountainCar-v0", "fig_fair")
    # 세 번째 환경은 학습 결과가 들어온 뒤에만 그린다 (규칙만 있으면 지도가 의미 없다)
    lambda_map("MinAtar_Freeway-v1", "fig_minatar")
    (OUT / "captions.json").write_text(json.dumps(CAPTIONS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  캡션: {(OUT / 'captions.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
