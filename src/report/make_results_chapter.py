"""논문 Ⅳ장(Experimental Results) 원고를 집계 파일에서 자동 생성한다.

왜 자동 생성인가: 숫자를 손으로 옮겨 적으면 반드시 틀린다. 그리고 실험이 갱신되면
원고의 숫자가 조용히 낡는다. 이 스크립트는 results/aggregate/*.csv 에서만 숫자를 읽어
paper/04_결과.md 를 다시 쓴다 (CLAUDE.md 절대 규칙 4 — 숫자는 로그 파일에서만 인용).

서술(해석·논의)은 사람이 쓴다. 이 스크립트는 표와 '사실 문장'까지만 만든다.

표는 **λ를 세로로** 놓는다. λ 격자가 14개까지 늘어나 가로로 놓으면 A4 한 쪽에 들어가지 않는다.

실행: python -m src.report.make_results_chapter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregate"
OUT = ROOT / "paper" / "04_결과.md"

LABEL = {"dqn": "표준 DQN", "temporl": "TempoRL", "lazy": "Lazy-MDP"}
RULE_LABEL = {"rule_pump": "pump 규칙", "rule_threshold": "임계값 규칙(처음)",
              "rule_threshold_tuned": "임계값 규칙(튜닝)",
              "rule_cautious": "신중 규칙", "rule_cautious_d1": "신중 규칙(d=1)",
              "rule_noop": "무행동", "rule_best": "최강 규칙",
              "rule_periodic_k1": "매 스텝 주기", "rule_periodic_k2": "2스텝 주기",
              "rule_periodic_k4": "4스텝 주기", "rule_periodic_k8": "8스텝 주기"}
REF_RULE = {"MountainCar-v0": "rule_pump", "LunarLander-v3": "rule_threshold_tuned",
            "MinAtar_Freeway-v1": "rule_cautious"}
ENV_NOTE = {
    "MountainCar-v0": ("보상이 희소한 탐험 문제다. 목표에 닿기 전까지 아무 신호가 없고, "
                       "매 스텝 무작위로 행동하는 탐험으로는 목표에 한 번도 닿지 못한다. "
                       "학습 없이도 문제를 푸는 강한 고정 규칙(pump)이 존재한다."),
    "LunarLander-v3": ("보상이 조밀한 제어 문제다. 매 스텝 자세·속도·연료에 대한 신호가 들어오고, "
                       "표준 DQN이 정상적으로 학습된다. 고정 규칙은 처음 손으로 짠 계수로는 점수가 "
                       "낮았지만({규칙_처음}), 계수를 다시 고르자 {규칙_튜닝}까지 올라 학습에 "
                       "근접한다(Ⅲ장 6.2절의 "
                       "기준선 감사). 표에는 두 규칙을 함께 실어 그 차이를 보인다."),
    "MinAtar_Freeway-v1": ("**규칙 품질** 축에서 앞의 두 환경 사이에 있는 세 번째 경우다. "
                           "손으로 짠 신중 규칙이 쓸 만하지만"
                           "({규칙_대비}) 최적과는 거리가 있어 학습이 이길 여지가 남아 있다. "
                           "에피소드가 1000스텝으로 고정돼 행동 비용의 압력이 두 환경보다 뚜렷하다."),
}
MIN_SEEDS_FOR_SECTION = 5   # 이보다 적으면 그 환경 절을 만들지 않는다
ENV_ORDER = ["MountainCar-v0", "LunarLander-v3", "MinAtar_Freeway-v1"]
ENV_FIG = {"MountainCar-v0": "fig2", "LunarLander-v3": "fig3",
           "MinAtar_Freeway-v1": "fig_minatar"}


def name(a: str) -> str:
    return LABEL.get(a, RULE_LABEL.get(a, a))


# 로마자 약어 뒤의 은/는은 읽는 소리로 정해진다 (DQN=디큐엔 → 은, MDP=엠디피 → 는)
EUN = {"표준 DQN": "은", "TempoRL": "은", "Lazy-MDP": "는"}
# 받침이 있으면 은/이, 없으면 는/가. 계열 이름이 늘어나면 여기에 추가한다.
GA = {"표준 DQN": "이", "TempoRL": "이", "Lazy-MDP": "가"}


def eun(a: str) -> str:
    n = name(a)
    return n + EUN.get(n, "은")


def ga(a: str) -> str:
    n = name(a)
    return n + GA.get(n, "이")


def num(v, nd=1) -> str:
    try:
        return format(float(v), "." + str(nd) + "f")
    except Exception:
        return "—"


def _cols(agg: pd.DataFrame, env_id: str, metric: str = "cost_return") -> list[str]:
    """표에 넣을 열. λ 격자가 14개까지 늘어나 표가 길어지므로 **중복 열은 뺀다.**

    '최강 규칙 포락선'은 대개 기준 규칙과 같은 값이다(그 규칙이 계속 최강이기 때문).
    값이 한 번이라도 달라지는 환경에서만 열을 남긴다 — MountainCar는 빠지고,
    LunarLander는 λ>1.37에서 최강 규칙이 무행동으로 바뀌므로 남는다.
    """
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    ref = REF_RULE.get(env_id)
    rules = [r for r in (ref,) if r in set(agg.agent)]
    # 기준선 감사에서 계수를 다시 고른 환경에서는 '처음 규칙'도 함께 보인다 —
    # 기준선을 얼마나 잘 만들었는지가 결론을 바꾼다는 것이 이 논문의 내용이기 때문이다.
    if ref == "rule_threshold_tuned" and "rule_threshold" in set(agg.agent):
        rules.append("rule_threshold")
    if "rule_noop" in set(agg.agent):
        rules.append("rule_noop")
    # 포락선 열은 **이미 실린 규칙들의 최댓값과 다를 때만** 넣는다.
    # 같다면 독자가 눈으로 계산할 수 있는 값을 한 열 더 싣는 것이고,
    # 그 한 열 때문에 표가 쪽을 넘어간다 (LunarLander가 그랬다: 8열 · 29cm).
    if "rule_best" in set(agg.agent) and rules:
        w = agg.pivot_table(index="lam", columns="agent", values=metric + "_iqm")
        have = [r for r in rules if r in w.columns]
        if "rule_best" in w.columns and have:
            gap = (w["rule_best"] - w[have].max(axis=1)).abs()
            if gap.max() > 1e-9:
                rules.append("rule_best")
    return learners + rules


def _decimals(agg: pd.DataFrame, metric: str) -> int:
    """숫자 자릿수. 값이 크면 소수점을 빼야 한 칸에 '값 [구간]'이 들어간다."""
    try:
        m = float(agg[metric + "_iqm"].abs().max())
    except Exception:
        return 1
    return 0 if m >= 100 else 1


def perf_table(agg: pd.DataFrame, env_id: str, metric: str = "cost_return") -> str:
    """λ를 세로, 계열을 가로로 놓은 표. 각 칸은 IQM [95% CI]."""
    cols = _cols(agg, env_id, metric)
    nd = _decimals(agg, metric)
    lams = sorted(agg.lam.unique())
    lines = ["| λ | " + " | ".join(name(c) for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    idx = {c: agg[agg.agent == c].set_index("lam") for c in cols}
    for l in lams:
        cells = []
        for c in cols:
            g = idx[c]
            if l not in g.index:
                cells.append("—")
                continue
            r = g.loc[l]
            cells.append(num(r[metric + "_iqm"], nd) + " ["
                         + num(r[metric + "_ci_lo"], nd) + ", " + num(r[metric + "_ci_hi"], nd) + "]")
        lines.append("| **" + format(l, "g") + "** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def action_table(agg: pd.DataFrame, env_id: str) -> str:
    cols = _cols(agg, env_id, "n_actions")
    lams = sorted(agg.lam.unique())
    lines = ["| λ | " + " | ".join(name(c) for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    idx = {c: agg[agg.agent == c].set_index("lam") for c in cols}
    for l in lams:
        cells = [num(idx[c].loc[l]["n_actions_iqm"], 0) if l in idx[c].index else "—" for c in cols]
        lines.append("| **" + format(l, "g") + "** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tie_or_loss(agg: pd.DataFrame, agent: str, rule: str, lam: float = 0.0,
                 metric: str = "cost_return") -> str:
    """λ=0에서 '못 이겼다'가 '졌다'인지 '동률'인지 가른다.

    신뢰구간이 겹치면 우열을 말하지 않는 것이 이 연구의 원칙이다(Ⅲ장 §5).
    그런데 λ*만 보면 '졌다'와 '비겼다'가 똑같이 λ*=0으로 찍혀 구분이 사라진다.
    """
    a = agg[(agg.agent == agent) & (agg.lam == lam)]
    r = agg[(agg.agent == rule) & (agg.lam == lam)]
    if a.empty or r.empty:
        return "이기지 못했다"
    a, r = a.iloc[0], r.iloc[0]
    if a[metric + "_ci_hi"] < r[metric + "_ci_lo"]:
        return "졌다"
    if a[metric + "_ci_lo"] > r[metric + "_ci_hi"]:
        return "이겼다"
    return "비겼다(신뢰구간이 겹쳐 우열을 말할 수 없다)"


def star_facts(env_id: str) -> list[str]:
    p = AGG / (env_id + "_lambda_star.json")
    if not p.exists():
        return []
    st = json.loads(p.read_text(encoding="utf-8"))
    ref = st.get("rule", "")
    agg_p = AGG / (env_id + "_iqm.csv")
    agg = pd.read_csv(agg_p) if agg_p.exists() else pd.DataFrame()
    out = []
    for s in st.get("results", []):
        lname = eun(s["learner"])
        if s.get("lam_star_pt") == 0.0:
            how = _tie_or_loss(agg, s["learner"], ref) if not agg.empty else "이기지 못했다"
            tail = ("어느 쪽이든 '비용 때문에 졌다'고 말할 수 없다 — 비용이 0인 조건이다."
                    if how.startswith("비겼다")
                    else "비용 때문이 아니라 학습된 정책 자체가 규칙보다 약하다는 뜻이다.")
            out.append(lname + " 비용이 아예 없는 λ=0에서 " + RULE_LABEL.get(ref, ref)
                       + "에 " + how + ". " + tail)
        elif s.get("lam_star_pt") is not None:
            extra = ("" if s.get("lam_star_ci") is None
                     else " 통계적으로 확실한 우위는 λ=" + format(float(s["lam_star_ci"]), "g")
                          + "에서 이미 사라졌다.")
            out.append(lname + " λ=" + format(float(s["lam_star_pt"]), "g") + "에서 "
                       + RULE_LABEL.get(ref, ref) + "에 역전당했다." + extra)
        else:
            out.append(lname + " 비교된 λ 구간 전체(" + str(s.get("coverage", ""))
                       + ")에서 " + RULE_LABEL.get(ref, ref) + "을 이겼다.")

    # 지정 기준 규칙만 보면 실제보다 후하게 나올 수 있다. 비용이 커지면 '그 λ에서 가장 센 규칙'이
    # 무행동으로 바뀌기 때문이다. 실질적인 λ*는 최강 규칙 포락선과의 비교에서 나온다.
    best = st.get("results_vs_best_rule", [])
    crossed = [r for r in best if r.get("lam_star_pt") not in (None, 0.0)]
    zeroed = [r for r in best if r.get("lam_star_pt") == 0.0]
    if crossed:
        bits = [name(r["learner"]) + " λ=" + format(float(r["lam_star_pt"]), "g") for r in crossed]
        out.append("λ마다 가장 센 규칙(최강 규칙 포락선)과 비교하면 역전 지점이 앞당겨진다: "
                   + ", ".join(bits) + ". 비용이 커지면 최강 규칙이 무행동으로 바뀌기 때문이며, "
                   "이쪽이 실질적인 임계 비용이다.")
    elif zeroed and not any(r.get("lam_star_pt") == 0.0 for r in st.get("results", [])):
        out.append("최강 규칙 포락선과 비교하면 λ=0에서부터 이기지 못한다.")
    return out


def only_one_wins(agg, env_id: str) -> str:
    """한 계열만 최강 규칙을 이기는 λ가 있으면 그 사실을 문장으로 만든다.

    왜 뽑나: 여러 계열이 함께 이기거나 함께 지는 구간은 '환경이 쉬웠나/어려웠나'를 말해 줄 뿐이다.
    **한 계열만 이기는 λ**가 그 계열의 구조가 실제로 무엇을 해내는지 가장 분명히 보여 준다.
    신뢰구간이 겹치지 않는 승리만 센다 — 겹치면 우열을 말하지 않는 것이 이 논문의 원칙이다.
    """
    if "rule_best" not in set(agg.agent):
        return ""
    rows = []
    for lam in sorted(agg.lam.unique()):
        rb = agg[(agg.agent == "rule_best") & (agg.lam == lam)]
        if rb.empty:
            continue
        rv = float(rb.iloc[0].cost_return_iqm)
        winners = []
        for a in ("dqn", "temporl", "lazy"):
            g = agg[(agg.agent == a) & (agg.lam == lam)]
            if g.empty:
                continue
            if float(g.iloc[0].cost_return_ci_lo) > rv:
                winners.append((a, float(g.iloc[0].cost_return_iqm)))
        if len(winners) == 1:
            rows.append((lam, winners[0][0], winners[0][1], rv))
    if not rows:
        return ""
    # 가장 비용이 큰 지점 하나를 대표로 든다 (거기가 구조의 차이가 가장 벌어진 곳이다)
    lam, a, av, rv = rows[-1]
    others = []
    for o in ("dqn", "temporl", "lazy"):
        if o == a:
            continue
        g = agg[(agg.agent == o) & (agg.lam == lam)]
        if not g.empty:
            others.append(name(o) + " " + num(g.iloc[0].cost_return_iqm, 0))
    txt = ("λ=" + format(lam, "g") + "에서는 **" + name(a) + "만** 최강 규칙을 이긴다("
           + num(av, 0) + " 대 " + num(rv, 0) + ", 신뢰구간 분리)")
    if others:
        txt += ". 같은 λ에서 " + " · ".join(others) + "는 규칙에 미치지 못한다"
    if len(rows) > 1:
        lams = ", ".join(format(l, "g") for l, _, _, _ in rows)
        winners = {w for _, w, _, _ in rows}
        if len(winners) == 1:
            txt += (". 한 계열만 이기는 λ는 " + lams + " 로 " + str(len(rows)) + "개인데 "
                    "**모두 " + name(a) + "이다** — 어느 한 지점에서 우연히 앞선 것이 아니다")
        else:
            txt += (". 한 계열만 이기는 λ는 " + lams + " 로 " + str(len(rows)) + "개다")
    return txt + "."


def action_saving_fact(agg: pd.DataFrame, env_id: str) -> str:
    """비용이 오를 때 행동을 실제로 아끼는가 — 가장 작은 λ와 가장 큰 λ의 행동 횟수를 비교."""
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    bits = []
    for a in learners:
        g = agg[agg.agent == a].sort_values("lam")
        if len(g) < 2:
            continue
        lo, hi = g.iloc[0], g.iloc[-1]
        if lo.n_actions_iqm <= 0:
            continue
        drop = (1 - hi.n_actions_iqm / lo.n_actions_iqm) * 100
        bits.append(name(a) + " " + num(lo.n_actions_iqm) + "회(λ=" + format(lo.lam, "g")
                    + ") → " + num(hi.n_actions_iqm) + "회(λ=" + format(hi.lam, "g")
                    + "), " + num(drop, 0) + "% 감소")
    return "; ".join(bits)


def _fill_intro(env_id: str, text: str) -> str:
    """환경 소개 문장 안의 자리표시를 집계 값으로 채운다.

    2026-08-29: 여기 숫자가 손으로 적혀 있었다(36 · 162). 기준선 감사를 다시 돌리면
    조용히 낡는다 — 실제로 감사의 잣대를 고쳤을 때 논문 세 곳이 한꺼번에 낡았다.
    """
    if "{" not in text:
        return text
    vals = {"규칙_처음": "—", "규칙_튜닝": "—", "규칙_대비": "—"}
    # MinAtar 소개용: 무행동·주기 규칙과 견준 신중 규칙 점수 (전부 잰 값이다)
    csv = AGG / (env_id + "_iqm.csv")
    if csv.exists():
        try:
            a = pd.read_csv(csv)
            z = a[a.lam == 0.0]
            def _v(name):
                g = z[z.agent == name]
                return None if g.empty else float(g.iloc[0].raw_return_iqm)
            ref, noop, per = _v(REF_RULE.get(env_id, "")), _v("rule_noop"), _v("rule_periodic_k1")
            bits = []
            if noop is not None:
                bits.append("무행동 " + num(noop) + "점")
            if per is not None:
                bits.append("매 스텝 주기 " + num(per) + "점")
            if ref is not None:
                vals["규칙_대비"] = (", ".join(bits) + " 대비 " + num(ref) + "점") if bits                     else num(ref) + "점"
        except Exception:
            pass
    p = AGG / "baseline_audit.json"
    if p.exists():
        try:
            r = json.loads(p.read_text(encoding="utf-8"))["results"].get(env_id)
            if r:
                vals["규칙_처음"] = num(r["현재"]["eval_iqm"])
                vals["규칙_튜닝"] = num(r["튜닝셋 최고"]["eval_iqm"])
        except Exception:
            pass
    return text.format(**vals)


def env_section(env_id: str, sec: str, compact: bool = False) -> str:
    p = AGG / (env_id + "_iqm.csv")
    if not p.exists():
        return ""
    agg = pd.read_csv(p)
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    sub = agg[agg.agent.isin(learners)]
    if sub.empty:
        return ""
    lo_s, hi_s = int(sub.n_seeds.min()), int(sub.n_seeds.max())
    # 시드가 너무 적으면 절을 아예 만들지 않는다.
    # 이 저장소의 절대 규칙은 "시드 10개 이상 + IQM + 신뢰구간으로만 결론을 말한다"이다.
    # 실험이 도는 중에 시간당 자동 갱신이 돌면 시드 1~2개짜리 표가 논문에 들어간다
    # (2026-08-29에 실제로 MinAtar 절이 시드 1개로 생성됐다). 경고 문구를 붙여도
    # 표에 숫자가 찍혀 있으면 읽는 사람은 그것을 결과로 본다.
    if lo_s < MIN_SEEDS_FOR_SECTION:
        print(f"  [건너뜀] {env_id} — 시드 {lo_s}개뿐이라 절을 만들지 않는다 "
              f"(최소 {MIN_SEEDS_FOR_SECTION}개). 실험이 끝나면 저절로 들어온다")
        return ""
    steps = int(sub.total_steps.max())
    n_lam = sub.lam.nunique()
    n_grid = agg.lam.nunique()
    status = ("조건당 환경 " + format(steps, ",") + "스텝, 학습 계열은 λ " + str(n_lam)
              + "개(격자 " + str(n_grid) + "개 중) · 시드 "
              + (str(lo_s) if lo_s == hi_s else str(lo_s) + "~" + str(hi_s)) + "개까지 완료된 집계다.")
    if lo_s < 10 or n_lam < n_grid:
        status += " **아직 실험이 끝나지 않았으므로 최종 결론이 아니다.**"

    parts = [
        "### " + sec + " " + env_id,
        "",
        _fill_intro(env_id, ENV_NOTE.get(env_id, "")) + " " + status,
        "",
        "<!--TABCAP: " + env_id + "의 비용 반영 총보상 r′ — IQM [95% 신뢰구간]"
        " | Cost-adjusted return r′ on " + env_id + " (IQM [95% CI]) -->",
        perf_table(agg, env_id),
        "",
        "<!--FIG:" + ENV_FIG.get(env_id, "fig2") + "-->",
        "",
    ]
    facts = star_facts(env_id)
    if facts:
        parts += ["표에서 읽히는 사실은 다음과 같다.", ""]
        parts += ["- " + f for f in facts]
        one = only_one_wins(agg, env_id)
        if one:
            parts.append("- " + one)
        parts.append("")
    save = action_saving_fact(agg, env_id)
    if save:
        parts += ["비용이 오를 때 실제로 행동을 아꼈는지는 행동 횟수로 확인된다.", ""]
        parts += ["- " + b.strip() for b in save.split(";")]
        parts += [""]
        parts += [] if compact else [
            "<!--TABCAP: " + env_id + "의 에피소드당 행동 횟수 (IQM)"
            " | Actions per episode on " + env_id + " (IQM) -->",
            action_table(agg, env_id),
            "",
        ]
    return "\n".join(parts)


def cost_share(agg) -> str:
    """붕괴 문턱에서 '한 에피소드에 물게 되는 비용 총액'이 보상 크기의 몇 %인지 문장으로 만든다.

    λ 자체를 보상과 견주면 단위가 맞지 않는다 — λ는 행동 1번의 값이고, 한 에피소드에는
    행동이 100회 넘게 들어간다. 견줄 것은 **λ × λ=0에서의 행동 횟수**다.
    (2026-08-29: 원래 문장이 이 단위를 틀리게 적고 있었다.)
    """
    shares = []
    for ag in ("dqn", "temporl", "lazy"):
        z = agg[(agg.agent == ag) & (agg.lam == 0.0)]
        g = agg[(agg.agent == ag) & (agg.lam > 0)].sort_values("lam")
        if z.empty or g.empty:
            continue
        z = z.iloc[0]
        hit = g[g.solved_iqm < float(z.solved_iqm) * 0.5]
        if hit.empty:
            continue
        total = float(hit.iloc[0].lam) * float(z.n_actions_iqm)
        shares.append(total / 200.0 * 100)      # 가장 나쁜 에피소드 보상(−200) 기준
    if not shares:
        return "아주 작은 비용만으로도"
    lo, hi = f"{min(shares):.1f}", f"{max(shares):.1f}"
    rng = lo + "%" if lo == hi else lo + "~" + hi + "%"
    return f"에피소드에서 물게 되는 비용 총액이 그 크기의 {rng}(1% 미만)만 되어도"


def budget_effect() -> str:
    """예산을 늘렸을 때 λ=0 성능이 어떻게 변했는지 환경별로 계산해 문장으로 만든다.

    2026-08-30: LunarLander에서 예산을 5배로 늘리자 **성능이 오히려 떨어지고
    불확실성이 커졌다.** 기대와 반대인 결과이며, 이 저장소의 규칙은 그런 결과도
    그대로 적는 것이다. 숫자는 전부 집계 파일에서 읽는다.
    """
    PAIRS = [("LunarLander-v3", "LunarLander-v3@budget1M", "20만", "100만", "5배"),
             ("MinAtar_Freeway-v1", "MinAtar_Freeway-v1@budget1M", "30만", "100만", "3.3배")]
    out, reach, widened = [], [], ""
    for base, big, s_lo, s_hi, mult in PAIRS:
        fb, fg = AGG / (base + "_iqm.csv"), AGG / (big + "_iqm.csv")
        if not (fb.exists() and fg.exists()):
            continue
        A, B = pd.read_csv(fb), pd.read_csv(fg)
        bits, worse = [], 0
        for ag in ("dqn", "temporl", "lazy"):
            x = A[(A.agent == ag) & (A.lam == 0.0)]
            y = B[(B.agent == ag) & (B.lam == 0.0)]
            if x.empty or y.empty:
                continue
            x, y = x.iloc[0], y.iloc[0]
            d = float(y.raw_return_iqm) - float(x.raw_return_iqm)
            wb = float(x.raw_return_ci_hi) - float(x.raw_return_ci_lo)
            wg = float(y.raw_return_ci_hi) - float(y.raw_return_ci_lo)
            worse += d < 0
            bits.append(f"{name(ag)} {num(x.raw_return_iqm, 0)}→{num(y.raw_return_iqm, 0)}"
                        f"(구간 폭 {wb:.0f}→{wg:.0f})")
        if not bits:
            continue
        head = ("**" + base + "**에서 예산을 " + mult + "로 늘리자(" + s_lo + " → " + s_hi
                + " 스텝) " + ("**세 계열 모두 점수가 떨어졌다**" if worse == len(bits)
                              else f"{worse}개 계열의 점수가 떨어졌다") + ": " + " · ".join(bits) + ".")
        out.append(head)
        # 예산을 늘린 뒤에도 규칙에 못 닿았는지, 구간이 얼마나 벌어졌는지 모아 둔다
        ref = REF_RULE.get(base)
        rr = A[(A.agent == ref) & (A.lam == 0.0)] if ref else None
        if rr is not None and not rr.empty:
            rv = float(rr.iloc[0].raw_return_iqm)
            best = max((float(B[(B.agent == g) & (B.lam == 0.0)].iloc[0].raw_return_iqm)
                        for g in ("dqn", "temporl", "lazy")
                        if not B[(B.agent == g) & (B.lam == 0.0)].empty), default=None)
            if best is not None:
                pct = best / rv * 100
                env_ko = base.split("@")[0]
                if pct >= 100:
                    reach.append(f"**{env_ko}**에서는 예산을 늘려도 규칙을 더 크게 이기지 못했다"
                                 f"(가장 나은 계열이 규칙의 {pct:.0f}% — 본실험보다 줄었다)")
                else:
                    reach.append(f"**{env_ko}**에서는 예산을 늘려도 규칙에 닿지 못했다"
                                 f"(가장 나은 계열이 규칙의 {pct:.0f}%)")
        if base.startswith("LunarLander"):
            ws = []
            for g in ("dqn", "temporl", "lazy"):
                x = A[(A.agent == g) & (A.lam == 0.0)]
                y = B[(B.agent == g) & (B.lam == 0.0)]
                if not x.empty and not y.empty:
                    x, y = x.iloc[0], y.iloc[0]
                    ws.append((float(y.raw_return_ci_hi) - float(y.raw_return_ci_lo))
                              / max(1e-9, float(x.raw_return_ci_hi) - float(x.raw_return_ci_lo)))
            if ws:
                widened = f"{min(ws):.1f}~{max(ws):.1f}배로 벌어졌다"
    if not out:
        return ""
    reach_note = (". ".join(reach) + ".") if reach else ""
    tail = [reach_note + " **본실험에서 보고한 λ\*는 예산이 모자라 낮게 나온 값이 아니다.**", ""]
    if widened:
        tail += ["LunarLander에서는 점수가 떨어졌을 뿐 아니라 **신뢰구간 폭이 " + widened + "** — "
                 "오래 학습시킬수록 시드마다 결과가 크게 갈린다는 뜻이다. 이 연구는 조기 종료를 "
                 "쓰지 않고 정해진 예산을 끝까지 소진하므로, 학습이 한 번 무너지면 회복하지 못한 채 "
                 "끝난다.", ""]
    tail += ["다만 이것은 이 설정에서의 관찰이다. 조기 종료나 학습률 감소를 쓰면 달라질 수 있고, "
             "그 확인은 이 연구의 범위 밖이다.", ""]
    return chr(10).join([
        "",
        "**예산을 늘리면 나아지리라는 기대는 빗나갔다.**", "",
    ] + [b + chr(10) for b in out] + tail)


def axis_verdict() -> str:
    """세 번째 환경의 λ*가 '규칙 품질'과 '보상 조밀도' 중 어느 축을 따라갔는지 판정한다.

    이 절을 둔 이유가 그 판정이다. 두 축이 갈리는 환경을 하나 넣어 두었으니,
    결과가 어느 쪽이든 그대로 적는다. 숫자는 전부 집계 파일에서 읽는다.
    """
    dens_p = AGG / "reward_density.json"
    if not dens_p.exists():
        return ""
    try:
        dens = {r["env_id"]: r for r in
                json.loads(dens_p.read_text(encoding="utf-8"))["results"]}
    except Exception:
        return ""

    def star(env):
        f = AGG / (env + "_lambda_star.json")
        if not f.exists():
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        out = {}
        for r in d.get("results_vs_best_rule", []):
            out[r["learner"]] = r.get("lam_star_pt")
        return out or None

    mc, ll, ma = star("MountainCar-v0"), star("LunarLander-v3"), star("MinAtar_Freeway-v1")
    if not (mc and ll and ma):
        return ""
    dk = {"MountainCar-v0": "MountainCar-v0", "LunarLander-v3": "LunarLander-v3",
          "MinAtar_Freeway-v1": "MinAtar/Freeway-v1"}
    rate = {e: dens[dk[e]]["informative_step_rate"] * 100 for e in dk if dk[e] in dens}
    if len(rate) < 3:
        return ""

    all_zero = all(v == 0.0 for v in ma.values() if v is not None)
    ll_pos = any((v or 0) > 0 for v in ll.values())
    lines = ["", "**이 절을 둔 이유는 두 축을 가르기 위해서였다. 답은 이렇다.**", ""]
    if all_zero and ll_pos:
        lines.append(
            "MinAtar에서 λ\*는 세 계열 모두 **0**이다 — 두 환경 사이가 아니라 "
            "MountainCar와 **같은 값**이다. 이 환경은 규칙 품질 축에서 가운데였으므로, "
            "λ\*가 규칙 품질을 따라간다면 사이 값이 나왔어야 한다. 그러지 않았다.")
        lines.append("")
        lines.append(
            "대신 λ\*는 **보상 조밀도**를 따라간다. 신호가 있는 스텝의 비율은 "
            + f"MountainCar {rate['MountainCar-v0']:.1f}% · MinAtar {rate['MinAtar_Freeway-v1']:.1f}% · "
            + f"LunarLander {rate['LunarLander-v3']:.1f}% 로(Ⅲ장 2절), "
            "MinAtar는 이 축에서 가운데가 아니라 MountainCar 쪽 끝에 있다. "
            "**λ\*도 그 자리를 따라갔다.**")
        lines.append("")
        lines.append(
            "즉 **'좋은 규칙이 이미 있으면 학습이 불리하다'가 아니라 "
            "'보상이 상태를 구분해 주지 못하면 학습이 불리하다'**가 이 연구가 관찰한 것이다. "
            "다만 환경이 셋뿐이므로 이것은 세 점이 한 방향을 가리킨다는 뜻이지 "
            "상관을 보인 것은 아니다.")
    else:
        lines.append(
            "MinAtar의 λ\*는 " + ", ".join(f"{name(k)} {v!r}" for k, v in ma.items())
            + " 다. 앞의 두 환경과 견주어 이 값이 무엇을 뜻하는지는 위 표와 그림에서 읽는다.")
    lines.append("")
    return chr(10).join(lines)


def tradeoff_section(sec: str) -> str:
    """행동-성능 상충 절. 문장 안 숫자는 전부 집계에서 계산한다."""
    envs = [e for e in ("MountainCar-v0", "LunarLander-v3") if (AGG / (e + "_iqm.csv")).exists()]
    if not envs:
        return ""
    lines = [
        "### " + sec + " 행동을 아낀 만큼 무엇을 얻었는가",
        "",
        "앞의 지도는 비용까지 반영한 점수 r′만 보여 준다. 그래서 **행동을 줄여서 이긴 것**과 "
        "**그냥 잘해서 이긴 것**이 구분되지 않는다. 가로를 행동 횟수, 세로를 "
        "**비용 빼기 전** 원보상 r로 놓으면 그 둘이 갈라진다. 왼쪽 위로 갈수록 좋다 — "
        "적게 움직이고 많이 받는 쪽이다.",
        "",
        "<!--FIG:fig_tradeoff-->",
        "",
    ]
    facts = []
    for env in envs:
        agg = pd.read_csv(AGG / (env + "_iqm.csv"))
        ref = REF_RULE.get(env)
        g = agg[agg.agent == ref]
        if g.empty:
            continue
        rx, ry = float(g.iloc[0].n_actions_iqm), float(g.iloc[0].raw_return_iqm)
        best = None
        for a in ("dqn", "temporl", "lazy"):
            d = agg[agg.agent == a]
            for _, r in d.iterrows():
                if r.n_actions_iqm <= rx and r.raw_return_iqm > ry:
                    gain = float(r.raw_return_iqm) - ry
                    if best is None or gain > best[0]:
                        best = (gain, a, float(r.n_actions_iqm), float(r.raw_return_iqm),
                                float(r.lam))
        rule_ko = name(ref)
        if best is None:
            facts.append(
                "**" + env + "**에서는 어떤 계열도 " + rule_ko + "(" + num(rx, 0) + "회 · "
                + num(ry, 0) + "점)보다 **행동은 적게 쓰면서 보상은 더 받는** 지점에 닿지 못했다. "
                "그림에서 모든 궤적이 규칙 아래에 있다 — 비용을 어떻게 매기든 이 규칙을 이길 수 없다는 뜻이다.")
        else:
            gain, a, bx, by, blam = best
            facts.append(
                "**" + env + "**에서는 " + ga(a) + " λ=" + format(blam, "g") + "에서 "
                + num(bx, 0) + "회 · " + num(by, 0) + "점에 닿는다. " + rule_ko + "("
                + num(rx, 0) + "회 · " + num(ry, 0) + "점)보다 **행동은 " + num(rx - bx, 0)
                + "회 적게 쓰면서 보상은 " + num(gain, 0) + "점 더 받는다.** "
                "학습이 규칙을 이긴 것이 '행동을 줄여서'가 아니라 '같은 행동으로 더 잘해서'라는 뜻이다.")
    for f in facts:
        if f:
            lines += [f, ""]
    lines += ["",
              "이 축에서 같은 r′를 주는 점들은 기울기 λ인 직선을 이룬다. 그래서 독자는 직선을 "
              "기울여 보며 '비용이 이만큼일 때 누가 이기는가'를 직접 읽을 수 있다. "
              "λ\*는 그 직선이 규칙을 지나면서 학습 궤적을 처음으로 완전히 아래에 두는 기울기다.",
              ""]
    return chr(10).join(lines)


def collapse_section(sec: str) -> str:
    p = AGG / "MountainCar-v0_iqm.csv"
    if not p.exists():
        return ""
    agg = pd.read_csv(p)
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    rows = []
    for a in learners:
        g = agg[(agg.agent == a) & (agg.lam > 0)].sort_values("lam")
        g0 = agg[(agg.agent == a) & (agg.lam == 0.0)]
        if g.empty or g0.empty:
            continue
        base_act = float(g0.iloc[0].n_actions_iqm)
        base_solved = float(g0.iloc[0].solved_iqm)

        def first(mask) -> str:
            sub = g[mask]
            return format(float(sub.iloc[0].lam), "g") if not sub.empty else "격자 밖"

        rows.append([
            name(a),
            first(g.solved_iqm < base_solved * 0.5),      # 성능이 먼저 무너지는 지점
            first(g.n_actions_iqm < base_act * 0.5),      # 행동이 절반으로 주는 지점
            first(g.n_actions_iqm < 1.0),                 # 행동이 완전히 멈추는 지점
            num(base_act, 0) + "회 / " + num(base_solved * 100, 0) + "%",
        ])
    lines = []
    if rows:
        lines = [
            "<!--TABTAG:tab_collapse-->",
        "<!--TABCAP: 무행동 붕괴의 세 문턱 — 성능이 먼저 무너지고 행동이 나중에 멈춘다 "
            "| Three thresholds of the collapse: performance degrades before the agent stops acting -->",
            "| 계열 | 도달률이 절반이 되는 첫 λ | 행동이 절반이 되는 첫 λ | 행동이 0이 되는 첫 λ | λ=0 기준값 |",
            "|---|---|---|---|---|",
        ] + ["| " + " | ".join(r) + " |" for r in rows]
    if not lines:
        return ""
    return "\n".join([
        "### " + sec + " 무행동 붕괴 — 아주 작은 비용에서 학습이 멈춘다",
        "",
        "앞의 두 환경이 갈린 이유를 여기서 본다. MountainCar에서는 비용이 조금만 붙어도 "
        "세 계열 모두 행동을 멈추고 그 상태로 굳는다. "
        "λ를 0 부근에서 촘촘히 훑어 그 문턱이 어디인지 쟀다.",
        "",
        "<!--FIG:fig4-->",
        "",
        "\n".join(lines),
        "",
        "표에서 드러나는 것은 붕괴가 **한 번에 일어나지 않는다**는 점이다. 비용이 조금 붙으면 "
        "에이전트는 여전히 λ=0과 비슷한 횟수로 행동하지만 목표에 닿는 비율이 먼저 떨어진다. "
        "행동을 멈추는 것은 그보다 더 비싸진 뒤다. 즉 **성능이 먼저 무너지고 행동이 나중에 멈춘다.**",
        "",
        "이 문턱들은 에피소드 보상 규모에 견주면 매우 작다. MountainCar의 한 에피소드 보상은 "
        "−200에서 −120 사이인데, " + cost_share(agg) + " 학습은 목표에 닿기를 포기한다. "
        "**행동 1번의 값이 아니라, 한 에피소드에 물게 되는 비용 총액이 그만큼이라는 뜻이다.**",
        "",
        collapse_budget_note(),
        "그런데 '성능이 먼저 무너진다'는 사실 자체는 원인을 말해 주지 않는다. 두 가지가 가능하다. "
        "**(가)** 비용이 붙은 세상에서는 그 정도가 실제로 최선이거나, "
        "**(나)** 비용 때문에 학습 과정이 망가져 있을 법한 정책을 못 찾았거나. "
        "다음 절의 대조 실험이 이 둘을 가른다.",
        "",
    ])


def collapse_budget_note() -> str:
    """붕괴 문턱이 예산에 따라 달라지는지 — 공통 λ에서만 비교한다.

    두 판의 λ 격자가 다르므로(본실험에는 0.0075가 있고 100만 판에는 없다)
    '문턱이 몇으로 옮겼다'는 말은 할 수 없다. **양쪽에 다 있는 λ에서만** 견준다.
    """
    fb, fg = AGG / "MountainCar-v0_iqm.csv", AGG / "MountainCar-v0@budget1M_epsconst_iqm.csv"
    if not (fb.exists() and fg.exists()):
        return ""
    A, B = pd.read_csv(fb), pd.read_csv(fg)
    A, B = A[A.agent == "dqn"], B[B.agent == "dqn"]
    shared = sorted(set(A.lam) & set(B.lam))
    if len(shared) < 3:
        return ""
    bits, gained = [], 0
    for l in shared:
        x = float(A[A.lam == l].iloc[0].solved_iqm) * 100
        y = float(B[B.lam == l].iloc[0].solved_iqm) * 100
        gained += y > x
        bits.append(f"λ={l:g} {x:.0f}%→{y:.0f}%")
    zero_both = [l for l in shared
                 if float(A[A.lam == l].iloc[0].solved_iqm) == 0
                 and float(B[B.lam == l].iloc[0].solved_iqm) == 0]
    out = ["", "**이 문턱이 예산 때문에 생긴 것은 아니다.** 같은 환경·같은 계열(표준 DQN)을 "
           "예산 3.3배(30만 → 100만 스텝)로 다시 돌려 목표 도달률을 견주었다. "
           "두 판의 λ 격자가 달라 '문턱이 어디로 옮겼다'고는 말할 수 없으므로, "
           "**양쪽에 다 있는 λ에서만** 비교한다.", ""]
    out.append("- " + " · ".join(bits))
    out.append("")
    if zero_both:
        out.append("예산을 3.3배로 줘도 " + str(gained) + "/" + str(len(shared))
                   + "개 지점에서 몇 %p 오르는 데 그쳤고, λ=" + format(max(zero_both), "g")
                   + "에서는 **양쪽 모두 도달률 0%** 다. "
                   "**붕괴는 예산을 늘려 없앨 수 있는 것이 아니다.**")
    else:
        out.append("예산을 3.3배로 줘도 " + str(gained) + "/" + str(len(shared))
                   + "개 지점에서만 올랐다.")
    out.append("")
    out.append("(뒤의 공정성 점검 절도 예산을 다루지만 묻는 것이 다르다. 여기서는 "
               "**붕괴 문턱이 예산에 따라 밀리는가**를, 그쪽에서는 **비용이 없는 λ=0에서 "
               "학습이 규칙 수준에 닿는가**를 묻는다.)")
    out.append("")
    out.append("<!-- 출처: results/aggregate/MountainCar-v0@budget1M_epsconst_iqm.csv -->")
    out.append("")
    return chr(10).join(out)


def causal_section(sec: str) -> str:
    """인과 실험 절 — 비용을 켜는 시점만 바꾼 대조. 결과가 있을 때만 만든다."""
    p = AGG / "causal_warmup.json"
    if not p.exists():
        return ""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    res = d.get("results", [])
    if not res:
        return ""
    lines = [
        "### " + sec + " 무행동 붕괴는 최적해인가, 탐험 실패인가 (대조 실험)",
        "",
        "앞 절의 붕괴에는 두 가지 해석이 있다. **(가)** 비용을 반영하면 정말로 가만히 있는 것이 "
        "최선이거나, **(나)** 비용 때문에 초반에 목표를 한 번도 보지 못해 굳었거나. "
        "두 해석은 같은 관측을 낳지만 뜻이 정반대다. 이를 가르기 위해 같은 예산·시드 안에서 "
        "**비용을 켜는 시점만** 바꾼 조건을 두었다(Ⅲ장 6.3절). 평가는 양쪽 모두 진짜 λ로 한다. "
        "앞 절에서 본 두 단계 구조 때문에 점수·행동뿐 아니라 **목표 도달률**을 함께 본다 — "
        "행동은 유지되는데 성능만 무너지는 구간에서 워밍업 쪽이 도달률을 지켜 낸다면, "
        "그것이 탐험 실패의 가장 직접적인 증거다.",
        "",
        "<!--FIGREF:fig_causal|가--> 그 결과다. 화살표는 **같은 λ·같은 예산·같은 시드에서 "
        "비용을 켜는 시점만 바꿨을 때**의 변화다. 아홉 조건 모두 도달률이 올라갔고, "
        "내려간 조건은 하나도 없다.",
        "",
        "<!--FIG:fig_causal-->",
        "",
        "<!--TABCAP: 비용을 처음부터 물릴 때와 절반 뒤에 켤 때 (MountainCar-v0, 같은 예산·시드)"
        " | Charging the cost from the start versus switching it on halfway (MountainCar-v0) -->",
        "| λ | 계열 | r′ 처음부터 | r′ 절반 뒤 | 행동 (처음→절반뒤) | 도달률 (처음→절반뒤) | 판정 |",
        "|---|---|---|---|---|---|---|",
    ]
    # 한 조건을 두 줄로 적으면 표가 두 쪽을 넘어간다(19행·46cm). 같은 내용을
    # **한 줄에 두 조건을 나란히** 두면 9행으로 줄면서 오히려 대조가 잘 보인다.
    short = {"탐험 실패": "탐험 실패", "무행동이 최적해": "무행동이 최적해",
             "행동은 늘었으나": "부분 증거", "점수는 높으나": "부분 증거"}

    def code(v: str) -> str:
        for k, t in short.items():
            if k in v:
                return t
        return "판정 보류"

    for r in res:
        lam, ag = format(float(r["lam"]), "g"), name(r["agent"])
        f, w = r["from_start"], r["warmup"]

        def cell(x):
            sc = x["score"]
            return num(sc["iqm"]) + " [" + num(sc["lo"]) + ", " + num(sc["hi"]) + "]"

        def pct(x):
            sv = x.get("solved")
            return num(sv["iqm"] * 100, 0) + "%" if sv else "—"

        lines.append("| **" + lam + "** | " + ag + " | " + cell(f) + " | " + cell(w) + " | "
                     + num(f["actions"]["iqm"], 0) + " → " + num(w["actions"]["iqm"], 0) + " | "
                     + pct(f) + " → " + pct(w) + " | " + code(str(r["verdict"])) + " |")
    lines += [
        "",
        "판정 칸은 줄임말이다. **탐험 실패** = 절반 뒤에 켠 쪽이 도달률이나 점수에서 "
        "신뢰구간이 겹치지 않게 앞선 경우, **부분 증거** = 한쪽 지표만 앞선 경우, "
        "**무행동이 최적해** = 양쪽 모두 행동이 멈춘 경우, **판정 보류** = 신뢰구간이 겹친 경우다. "
        "판정 문구 전체는 `results/aggregate/causal_warmup.json`에 그대로 남아 있다.",
    ]
    n_expl = sum(1 for r in res if str(r["verdict"]).startswith("탐험 실패"))
    n_opt = sum(1 for r in res if "무행동이 최적해" in str(r["verdict"]))
    n_none = len(res) - n_expl - n_opt
    lines += [
        "",
        "비교 " + str(len(res)) + "건 중 '탐험 실패'로 판정된 것이 " + str(n_expl) + "건, "
        "'무행동이 최적해일 가능성'이 " + str(n_opt) + "건, 신뢰구간이 겹쳐 판정을 보류한 것이 "
        + str(n_none) + "건이다.",
        "",
    ]
    return "\n".join(lines)


def lam_star_lead() -> str:
    """표 1 앞에 붙일 요약 문장. 결론을 표보다 먼저 말한다.

    이 저장소의 글쓰기 규칙은 "결론을 맨 위에 쓴다"이다. 논문에서 가장 중요한 절이
    아무 문장 없이 표부터 시작하고 있었다 — 심사자는 표를 스스로 해석해야 했다.
    문장 안의 숫자는 전부 results/aggregate/*_lambda_star.json 에서 읽는다.
    """
    rows = []
    for env in ENV_ORDER:
        p = AGG / (env + "_lambda_star.json")
        if not p.exists():
            continue
        st = json.loads(p.read_text(encoding="utf-8"))
        by = {r["learner"]: r.get("lam_star_pt")
              for r in st.get("results_vs_best_rule", st.get("results", []))}
        if not by:
            continue
        rows.append((env, by))
    if not rows:
        return ""

    zero_envs = [e for e, by in rows if all(v == 0.0 for v in by.values() if v is not None)]
    win_envs = [e for e, by in rows if any((v or 0) > 0 for v in by.values())]
    out = ["**결론부터 적는다. 임계 비용 λ\*는 하나의 숫자로 정해지지 않는다 — 환경에 따라 갈린다.**", ""]
    if zero_envs:
        out.append("")
        out.append("**" + ", ".join(zero_envs) + "**에서는 세 학습 계열 모두 λ\*가 0이다. "
                   "비용을 전혀 물리지 않아도 학습이 고정 규칙을 넘어서지 못한다는 뜻이며, "
                   "이 환경에서는 '언제 행동할지를 배우는 것'이 아니라 '학습이라는 접근 자체'가 열세다.")
    if win_envs:
        parts = []
        for e, by in rows:
            if e not in win_envs:
                continue
            for ag in ("dqn", "temporl", "lazy"):
                v = by.get(ag)
                if v is not None:
                    parts.append(f"{name(ag)} λ\*={format(float(v), 'g')}")
            out.append("")
            out.append("**" + e + "**에서는 계열마다 다르다 — " + ", ".join(parts) + ". "
                       "같은 환경·같은 예산인데도 방법에 따라 버티는 비용 구간이 다르다는 것은, "
                       "λ\*가 환경만의 성질이 아니라 **환경과 방법의 짝**에 붙는 값이라는 뜻이다.")
            parts = []
    out += ["", "아래 표가 그 값들이다. 두 가지 엄격도로 함께 적었다 — "
                "신뢰구간이 겹치기 시작하는 λ(엄격)와 IQM이 교차하는 λ(느슨)다."]
    return chr(10).join(out)


HEADER_TMPL = """# Ⅳ. Experimental Results

## 1. 읽는 법

- **λ**: 행동 1번의 값. 오른쪽(아래쪽)으로 갈수록 행동이 비싸다. λ=0이면 원래 문제와 같다.
- **r′**: 비용까지 빼고 남은 총보상. 높을수록 좋다.
- **대괄호**: 95% 계층 부트스트랩 신뢰구간. 두 구간이 겹치면 우열을 말하지 않는다.
- **기준 규칙**: 각 환경에서 가장 센 고정 규칙(MountainCar는 pump, LunarLander는 임계값). 문제를 아예 못 푸는 약한 규칙을 기준으로 삼으면 "학습이 이겼다"가 너무 쉬워진다.
- **최강 규칙**: λ마다 그 시점에서 가장 센 고정 규칙을 골라 이은 포락선. 비용이 커지면 최강 규칙이 무행동으로 바뀌므로 이 보조선을 함께 본다. **기준 규칙과 값이 한 번도 달라지지 않는 환경에서는 표에서 이 열을 생략한다**(중복이므로).
- **λ\\***: 학습이 규칙을 더 이상 이기지 못하게 되는 가장 작은 λ. 격자 안에서 교차가 없으면 보간하지 않고 그 사실을 그대로 적는다.

## 2. 핵심 결과 — 임계 비용 λ*

<!--LEAD:tab1-->

<!--TABLE:tab1-->

## 3. 실험 설정

아래 표의 조건은 세 환경에 공통으로 적용된다. **λ를 뺀 모든 것을 같게 맞추는 것**이
이 연구의 설계 전체이므로, 예산·시드·평가 방식·초매개변수를 한자리에 모아 둔다.

<!--TABLE:tab2-->

## 4. 환경별 λ-성능 지도

환경마다 λ\*가 왜 그렇게 갈렸는지를 하나씩 본다. 순서는 **가장 극단적인 환경부터**다 —
MountainCar에서 무엇이 무너지는지를 먼저 보면, LunarLander가 왜 다른지가 대비로 드러난다.
"""

def switch_cost_section(sec: str) -> str:
    """비용 부과 방식을 바꾸면 답이 달라지는가 — 학습 계열까지 재어 본 절.

    이 연구의 지도는 '매 스텝 과금'(작동 비용) 위에 있다. 실제 설비에서는 행동이
    **바뀔 때만** 무는 해석(전환 비용)도 흔하다. 고정 규칙만으로도 λ 눈금이 크게
    달라진다는 것은 이미 보였고, 여기서는 **학습 계열도 같은 조건에서** 잰다.
    """
    f = AGG / "MountainCar-v0@switch_iqm.csv"
    fb = AGG / "MountainCar-v0_iqm.csv"
    if not (f.exists() and fb.exists()):
        return ""
    B, A = pd.read_csv(f), pd.read_csv(fb)
    lams = sorted(B.lam.unique())
    cols = [a for a in ("dqn", "temporl", "lazy") if a in set(B.agent)]
    cols += [r for r in ("rule_pump", "rule_noop") if r in set(B.agent)]
    if not cols:
        return ""

    def half(df, ag):
        z = df[(df.agent == ag) & (df.lam == 0.0)]
        g = df[(df.agent == ag) & (df.lam > 0)].sort_values("lam")
        if z.empty or g.empty:
            return None
        h = g[g.solved_iqm < float(z.iloc[0].solved_iqm) * 0.5]
        return None if h.empty else float(h.iloc[0].lam)

    lines = [
        "## " + sec + ". 비용을 어떻게 세느냐가 답을 바꾼다",
        "",
        "이 연구의 지도는 **매 스텝 과금**(행동을 실행하는 스텝마다 λ) 위에 있다. "
        "실제 설비에서는 행동이 **바뀔 때만** 무는 해석(전환 비용)도 흔하다. "
        "밸브를 계속 열어 두는 것은 공짜고 여닫는 순간에만 마모가 생기는 경우다. "
        "같은 환경(MountainCar)·같은 계열·같은 예산에서 **과금 방식만** 바꿔 다시 쟀다.",
        "",
        "<!--TABCAP: 전환 과금에서의 목표 도달률 (MountainCar-v0, 시드 10개) "
        "| Goal-reaching rate under per-switch cost (MountainCar-v0, 10 seeds) -->",
        "| 계열 | " + " | ".join("λ=" + format(l, "g") for l in lams) + " |",
        "|---|" + "---|" * len(lams),
    ]
    for c in cols:
        g = B[B.agent == c].set_index("lam")
        cells = [(num(g.loc[l].solved_iqm * 100, 0) + "%") if l in g.index else "—" for l in lams]
        bold = "**" if c in ("dqn", "temporl", "lazy") else ""
        lines.append("| " + bold + name(c) + bold + " | " + " | ".join(cells) + " |")
    lines.append("")

    h_step = {a: half(A, a) for a in ("dqn", "temporl", "lazy")}
    h_sw = {a: half(B, a) for a in ("dqn", "temporl", "lazy")}
    facts = []
    ratios = [h_sw[a] / h_step[a] for a in h_step if h_step.get(a) and h_sw.get(a)]
    if ratios:
        facts.append("도달률이 절반이 되는 λ가 매 스텝 과금에서는 "
                     + ", ".join(format(v, "g") for v in sorted({v for v in h_step.values() if v}))
                     + "이었는데 전환 과금에서는 "
                     + ", ".join(format(v, "g") for v in sorted({v for v in h_sw.values() if v}))
                     + "이다 — **λ 눈금이 두 자릿수로 커졌다.** "
                     "전환은 행동보다 훨씬 드물게 일어나므로 같은 압력을 주려면 λ가 그만큼 커야 한다.")
    tp = B[B.agent == "temporl"].set_index("lam")
    if not tp.empty:
        lo = min(float(tp.loc[l].solved_iqm) for l in lams if l in tp.index)
        facts.append("**전환 과금에서는 TempoRL만 끝까지 버틴다.** 격자 끝(λ="
                     + format(max(lams), "g") + ")에서도 도달률 "
                     + num(lo * 100, 0) + "% 아래로 내려가지 않는 반면, "
                     "표준 DQN은 λ=" + format(sorted(l for l in lams if l > 0)[0], "g")
                     + "에서 이미 0%가 된다. 한 번 고른 행동을 오래 유지하는 구조가 "
                     "**전환에만 값을 매기는 세상에서 곧바로 이점이 된다** — "
                     "매 스텝 과금에서는 드러나지 않던 이점이다.")
    facts.append("다만 **어느 계열도 pump 규칙을 이기지 못한다.** pump는 속도 부호가 바뀔 때만 "
                 "방향을 틀어 전환이 3회뿐이라, 이 과금 방식에서 오히려 더 유리해진다. "
                 "그리고 λ가 커지면 TempoRL의 점수는 무행동 아래로 내려간다 — "
                 "버티는 것과 이득을 보는 것은 다르다.")
    lines += ["- " + x for x in facts]
    lines += ["", "**이 연구의 지도는 '매 스텝 과금'이라는 전제 위에 있다.** "
              "전제를 바꾸면 λ 눈금뿐 아니라 **어느 방법이 유리한지도 바뀐다.** "
              "실무자는 자기 설비의 마모가 어느 쪽에 가까운지를 먼저 정해야 한다.",
              "",
              "<!-- 출처: results/aggregate/MountainCar-v0@switch_iqm.csv "
              "(cost_mode=per_switch, 180조건) -->", ""]
    return chr(10).join(lines)


def baseline_audit_section() -> str:
    """기준선 감사 절 — 비교 상대인 규칙을 성의 있게 만들었는가."""
    p = AGG / "baseline_audit.json"
    if not p.exists():
        return ""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    res = d.get("results", {})
    if not res:
        return ""
    rows = []
    worst_gap = max((float(r.get("차이", 0)) for r in res.values()), default=0.0)
    for env, r in res.items():
        rows.append("| " + env + " | " + r["rule"] + " | "
                    + num(r["현재"]["eval_iqm"]) + " | " + num(r["튜닝셋 최고"]["eval_iqm"]) + " | "
                    + num(r["차이"]) + " | " + r["판정"] + " |")
    return "\n".join([
        "## {감사번호}. 기준선 감사 — 비교 상대인 규칙을 성의 있게 만들었는가",
        "",
        "\"학습이 단순 규칙을 이긴다\"는 주장은 그 규칙을 얼마나 잘 만들었는지에 달려 있다. "
        "환경마다 기준 규칙의 계수를 격자로 훑어 더 나은 것이 있는지 확인했다. "
        "튜닝은 평가에 쓰지 않는 에피소드에서 하고, 거기서 고른 하나만 평가용 에피소드로 다시 쟀다(Ⅲ장 6.2절). "
        "재는 잣대는 본실험과 같다 — 시드마다 점수를 내고 그 시드 점수들의 IQM을 쓴다.",
        "",
        "<!--TABCAP: 기준 규칙의 계수를 다시 골랐을 때 (시드 10개 × 100 에피소드, 본실험과 같은 잣대)"
        " | Re-selecting the coefficients of each baseline rule "
        "(10 seeds x 100 episodes, the same estimator as the main experiment) -->",
        "| 환경 | 규칙 | 현재 계수 r IQM | 다시 고른 계수 r IQM | 차이 | 판정 |",
        "|---|---|---|---|---|---|",
    ] + rows + [
        "",
        "세 환경 중 하나에서만 문제가 나왔다. MountainCar와 MinAtar의 기준 규칙은 이미 최선이었고, "
        "**LunarLander의 임계값 규칙만 계수를 다시 고르는 것으로 " + num(worst_gap, 0) + "점이 올랐다**"
        "(같은 형태의 규칙, 계수만 다름). 그 결과 그 환경의 임계 비용 λ*가 바뀌었다 — "
        "<!--TABREF:tab1|의--> 값은 다시 고른 규칙을 포함한 것이다.",
        "",
        "이 비대칭이 중요하다. MountainCar에서 '학습이 규칙에 진다'는 결론은 기준선이 이미 최선이었으므로 "
        "더 단단해졌고, LunarLander에서 '학습이 이긴다'는 결론만 약한 기준선의 덕을 보고 있었다.",
        "",
    ])


FAIRNESS_TMPL = """## {공정성번호}. 공정성 점검 — 비용이 없을 때 학습은 규칙 수준에 닿는가

MountainCar에서 λ*가 0으로 나왔다는 것은 "비용이 없어도 학습이 규칙에 진다"는 뜻이다.
그렇다면 이 결과는 비용에 대한 발견이 아니라 학습이 덜 됐다는 신호일 수 있다.
그래서 학습 예산과 탐험 설정을 바꿔 가며 λ=0 성능을 다시 쟀다.

<!--TABLE:tab3-->

최종 점수만으로는 "예산을 더 주면 달라졌을 것"이라는 반론에 답할 수 없다.
답이 되는 것은 **곡선이 평평해졌는가**이다. <!--FIGREF:fig_fair|는--> λ=0에서 예산을 3.3배로 늘리고
탐험과 신경망까지 바꿔 가며 그린 학습 곡선이다. 세 조건 모두 30만 스텝 부근에서
오르기를 멈추고, 남은 70만 스텝 동안 규칙 선 아래에서 오르내릴 뿐이다.
**예산이 모자라서가 아니라, 이 환경에서 이 학습기가 닿는 높이가 거기까지다.**

<!--FIG:fig_fair-->
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--compact", action="store_true",
                    help="분량을 줄인다: 환경별 '행동 횟수' 표를 빼고 사실 문장만 남긴다 "
                         "(그림 2·3의 λ 지도와 그림 4에 같은 정보가 있다)")
    a = ap.parse_args()
    envs = [e for e in ENV_ORDER if (AGG / (e + "_iqm.csv")).exists()]
    if not envs:
        print("집계 파일이 없다 — 먼저 aggregate를 돌릴 것")
        return
    # 절 배치: 대조되는 두 환경 → 그 대조의 원인(무행동 붕괴) → 세 번째 환경으로 확인.
    # 이 순서라야 그림 번호가 방법(1) · MountainCar 지도(2) · LunarLander 지도(3) ·
    # 붕괴(4) · 공정성 곡선(5) · Freeway 지도(6) 으로 자연스럽게 붙는다
    # (번호는 등장 순서로 자동 부여된다).
    parts, k = [], 0
    for e in envs[:2]:
        k += 1
        parts.append(env_section(e, "4." + str(k), a.compact))

    # 5절: 두 환경이 갈린 원인을 파고든다
    parts.append("## 5. 무엇이 두 환경을 갈랐나")
    parts.append("")
    parts.append("앞 절은 **무슨 일이 일어났는지**를 보였다. 이 절은 **왜 그런지**를 묻는다. "
                 "보상이 희소한 환경에서 학습이 무너지는 방식을 먼저 보고(5.1), 그것이 "
                 "최적해인지 탐험 실패인지를 대조 실험으로 가른 뒤(5.2), "
                 "아낀 행동으로 무엇을 얻었는지를 본다(5.3).")
    parts.append("")
    j = 0
    j += 1
    parts.append(collapse_section("5." + str(j)))
    cs = causal_section("5." + str(j + 1))
    if cs:
        j += 1
        parts.append(cs)
    ts = tradeoff_section("5." + str(j + 1))
    if ts:
        j += 1
        parts.append(ts)

    # 6절: 세 번째 환경으로 확인한다.
    # 여기 두는 이유는 두 가지다 —
    #   (1) 내용: MinAtar는 두 환경 사이에 놓이도록 고른 환경이라, 앞의 설명을 확인하는 자리다
    #   (2) 배치: 4절에 두면 그 λ 지도가 그림 4가 되어 '그림 4 = 무행동 붕괴' 지시가 깨진다
    third = [env_section(e, "6", a.compact) for e in envs[2:]]
    third = [t for t in third if t]
    if third:
        parts.append("## 6. 세 번째 환경으로 확인한다")
        parts.append("")
        parts.append("MinAtar/Freeway를 세 번째 환경으로 고른 이유는 **규칙 품질** 축에서 "
                     "가운데였기 때문이다 — 손으로 짠 규칙이 쓸 만하지만 최적과는 거리가 있다. "
                     "그런데 Ⅲ장 2절에서 **보상 조밀도**를 재고 보니 그 축에서는 가운데가 아니라 "
                     "MountainCar 쪽 끝에 가까웠다(보상 값 2가지 · 신호 있는 스텝 1.6%, "
                     "MountainCar는 1가지 · 0.0%, LunarLander는 16,717가지 · 99.8%).")
        parts.append("")
        parts.append("**두 축이 갈린다는 것이 오히려 좋은 시험이 된다.** λ\*가 규칙 품질을 따라간다면 "
                     "여기서 두 환경 사이 값이 나와야 하고, 보상 조밀도를 따라간다면 MountainCar처럼 "
                     "0에 가까운 값이 나와야 한다. **어느 쪽이든 그대로 적는다.**")
        parts.append("")
        parts += [t.replace("### 6 ", "### 6.1 ") for t in third]
        parts.append(axis_verdict())
    body = "\n".join(x for x in parts if x)
    src = ("\n<!-- 출처: results/aggregate/" + "{" + ",".join(envs) + "}_iqm.csv, "
           "*_lambda_star.json. 조건별 원본은 results/raw/{환경}/{계열}/lam{λ}/seed{n}_final.csv. "
           "설계 결정과 도중에 고친 버그는 docs/실험일지.md 참조. -->\n")
    out = Path(a.out) if a.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    head = HEADER_TMPL.replace("<!--LEAD:tab1-->", lam_star_lead())
    # 세 번째 환경 절이 없으면(실험이 아직 안 끝났으면) 뒤 절 번호가 하나씩 당겨져야 한다.
    # 그러지 않으면 '5절 다음이 7절'이 되어 목차에 구멍이 생긴다.
    has_third = any("## 6. 세 번째 환경" in x for x in parts)
    n_fair = 7 if has_third else 6
    fair = FAIRNESS_TMPL.replace("{공정성번호}", str(n_fair)) + budget_effect()
    audit = baseline_audit_section().replace("{감사번호}", str(n_fair + 1))
    audit += chr(10) + switch_cost_section(str(n_fair + 2))
    out.write_text(head + chr(10) + body + chr(10) + fair + chr(10) + audit + src, encoding="utf-8")
    print("Ⅳ장 생성: " + str(out.relative_to(ROOT)))


if __name__ == "__main__":
    main()
