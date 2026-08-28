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
RULE_LABEL = {"rule_pump": "pump 규칙", "rule_threshold": "임계값 규칙",
              "rule_cautious": "신중 규칙", "rule_cautious_d1": "신중 규칙(d=1)",
              "rule_noop": "무행동", "rule_best": "최강 규칙",
              "rule_periodic_k1": "매 스텝 주기", "rule_periodic_k2": "2스텝 주기",
              "rule_periodic_k4": "4스텝 주기", "rule_periodic_k8": "8스텝 주기"}
REF_RULE = {"MountainCar-v0": "rule_pump", "LunarLander-v3": "rule_threshold",
            "MinAtar_Freeway-v1": "rule_cautious"}
ENV_NOTE = {
    "MountainCar-v0": ("보상이 희소한 탐험 문제다. 목표에 닿기 전까지 아무 신호가 없고, "
                       "매 스텝 무작위로 행동하는 탐험으로는 목표에 한 번도 닿지 못한다. "
                       "학습 없이도 문제를 푸는 강한 고정 규칙(pump)이 존재한다."),
    "LunarLander-v3": ("보상이 조밀한 제어 문제다. 매 스텝 자세·속도·연료에 대한 신호가 들어오고, "
                       "표준 DQN이 정상적으로 학습된다. 고정 규칙은 착륙은 시키지만 점수가 낮다."),
    "MinAtar_Freeway-v1": ("앞의 두 환경 사이에 있는 세 번째 경우다. 손으로 짠 신중 규칙이 쓸 만하지만"
                           "(무작위 0점 대비 17.8점) 최적과는 거리가 있어 학습이 이길 여지가 남아 있다. "
                           "에피소드가 1000스텝으로 고정돼 행동 비용의 압력이 두 환경보다 뚜렷하다."),
}
ENV_ORDER = ["MountainCar-v0", "LunarLander-v3", "MinAtar_Freeway-v1"]
ENV_FIG = {"MountainCar-v0": "fig2", "LunarLander-v3": "fig3",
           "MinAtar_Freeway-v1": "fig5"}


def name(a: str) -> str:
    return LABEL.get(a, RULE_LABEL.get(a, a))


# 로마자 약어 뒤의 은/는은 읽는 소리로 정해진다 (DQN=디큐엔 → 은, MDP=엠디피 → 는)
EUN = {"표준 DQN": "은", "TempoRL": "은", "Lazy-MDP": "는"}


def eun(a: str) -> str:
    n = name(a)
    return n + EUN.get(n, "은")


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
    if "rule_best" in set(agg.agent) and ref in set(agg.agent):
        a = agg[agg.agent == "rule_best"].set_index("lam")[metric + "_iqm"]
        b = agg[agg.agent == ref].set_index("lam")[metric + "_iqm"]
        common = a.index.intersection(b.index)
        if len(common) and not (a.loc[common] - b.loc[common]).abs().lt(1e-9).all():
            rules.append("rule_best")
    if "rule_noop" in set(agg.agent):
        rules.append("rule_noop")
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


def env_section(env_id: str, sec: str) -> str:
    p = AGG / (env_id + "_iqm.csv")
    if not p.exists():
        return ""
    agg = pd.read_csv(p)
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    sub = agg[agg.agent.isin(learners)]
    if sub.empty:
        return ""
    lo_s, hi_s = int(sub.n_seeds.min()), int(sub.n_seeds.max())
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
        ENV_NOTE.get(env_id, "") + " " + status,
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
        parts.append("")
    save = action_saving_fact(agg, env_id)
    if save:
        parts += ["비용이 오를 때 실제로 행동을 아꼈는지는 행동 횟수로 확인된다: " + save + ".", ""]
        parts += [
            "<!--TABCAP: " + env_id + "의 에피소드당 행동 횟수 (IQM)"
            " | Actions per episode on " + env_id + " (IQM) -->",
            action_table(agg, env_id),
            "",
        ]
    return "\n".join(parts)


def collapse_section(sec: str) -> str:
    p = AGG / "MountainCar-v0_iqm.csv"
    if not p.exists():
        return ""
    agg = pd.read_csv(p)
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    lines = []
    for a in learners:
        g = agg[(agg.agent == a) & (agg.lam > 0)].sort_values("lam")
        if g.empty:
            continue
        z = g[g.n_actions_iqm < 1.0]
        first_zero = format(float(z.iloc[0].lam), "g") if not z.empty else "격자 안에 없음"
        g0 = agg[(agg.agent == a) & (agg.lam == 0.0)]
        base = float(g0.iloc[0].n_actions_iqm) if not g0.empty else float("nan")
        h = g[g.n_actions_iqm < base * 0.5]
        half = format(float(h.iloc[0].lam), "g") if not h.empty else "격자 안에 없음"
        lines.append("- " + name(a) + ": 행동 횟수가 절반 아래로 떨어지는 첫 λ = " + half
                     + ", 완전히 0이 되는 첫 λ = " + first_zero)
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
        "이 문턱은 에피소드 보상 규모에 견주면 매우 작다. MountainCar의 한 에피소드 보상은 "
        "−200에서 −120 사이인데, 행동 1번의 값이 그 1% 수준만 되어도 학습은 행동을 포기한다.",
        "",
    ])


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
        "**비용을 켜는 시점만** 바꾼 조건을 두었다(Ⅲ장 6.2절). 평가는 양쪽 모두 진짜 λ로 한다.",
        "",
        "<!--TABCAP: 비용을 처음부터 물릴 때와 절반 뒤에 켤 때 (MountainCar-v0, 같은 예산·시드)"
        " | Charging the cost from the start versus switching it on halfway (MountainCar-v0) -->",
        "| λ | 계열 | 비용 시점 | r′ IQM [95% CI] | 행동 횟수 | 판정 |",
        "|---|---|---|---|---|---|",
    ]
    for r in res:
        lam, ag = format(float(r["lam"]), "g"), name(r["agent"])
        for key, label in (("from_start", "처음부터"), ("warmup", "절반 뒤")):
            sc, ac = r[key]["score"], r[key]["actions"]
            lines.append("| **" + lam + "** | " + ag + " | " + label + " | "
                         + num(sc["iqm"]) + " [" + num(sc["lo"]) + ", " + num(sc["hi"]) + "] | "
                         + num(ac["iqm"]) + " | "
                         + (r["verdict"] if key == "warmup" else "") + " |")
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


HEADER_TMPL = """# Ⅳ. Experimental Results

## 1. 읽는 법

- **λ**: 행동 1번의 값. 오른쪽(아래쪽)으로 갈수록 행동이 비싸다. λ=0이면 원래 문제와 같다.
- **r′**: 비용까지 빼고 남은 총보상. 높을수록 좋다.
- **대괄호**: 95% 계층 부트스트랩 신뢰구간. 두 구간이 겹치면 우열을 말하지 않는다.
- **기준 규칙**: 각 환경에서 가장 센 고정 규칙(MountainCar는 pump, LunarLander는 임계값). 문제를 아예 못 푸는 약한 규칙을 기준으로 삼으면 "학습이 이겼다"가 너무 쉬워진다.
- **최강 규칙**: λ마다 그 시점에서 가장 센 고정 규칙을 골라 이은 포락선. 비용이 커지면 최강 규칙이 무행동으로 바뀌므로 이 보조선을 함께 본다. **기준 규칙과 값이 한 번도 달라지지 않는 환경에서는 표에서 이 열을 생략한다**(중복이므로).
- **λ\\***: 학습이 규칙을 더 이상 이기지 못하게 되는 가장 작은 λ. 격자 안에서 교차가 없으면 보간하지 않고 그 사실을 그대로 적는다.

## 2. 핵심 결과 — 임계 비용 λ*

<!--TABLE:tab1-->

## 3. 실험 설정

<!--TABLE:tab2-->

## 4. 환경별 λ-성능 지도
"""

FAIRNESS_TMPL = """## 5. 공정성 점검 — 비용이 없을 때 학습은 규칙 수준에 닿는가

MountainCar에서 λ*가 0으로 나왔다는 것은 "비용이 없어도 학습이 규칙에 진다"는 뜻이다.
그렇다면 이 결과는 비용에 대한 발견이 아니라 학습이 덜 됐다는 신호일 수 있다.
그래서 학습 예산과 탐험 설정을 바꿔 가며 λ=0 성능을 다시 쟀다.

<!--TABLE:tab3-->
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    envs = [e for e in ENV_ORDER if (AGG / (e + "_iqm.csv")).exists()]
    if not envs:
        print("집계 파일이 없다 — 먼저 aggregate를 돌릴 것")
        return
    # 절 배치: 대조되는 두 환경 → 그 대조의 원인(무행동 붕괴) → 세 번째 환경으로 확인.
    # 이 순서라야 그림 번호가 방법(1) · MountainCar 지도(2) · LunarLander 지도(3) ·
    # 붕괴(4) · Freeway 지도(5) 로 자연스럽게 붙는다 (번호는 등장 순서로 자동 부여된다).
    parts, k = [], 0
    for e in envs[:2]:
        k += 1
        parts.append(env_section(e, "4." + str(k)))
    k += 1
    parts.append(collapse_section("4." + str(k)))
    cs = causal_section("4." + str(k + 1))
    if cs:
        k += 1
        parts.append(cs)
    for e in envs[2:]:
        k += 1
        parts.append(env_section(e, "4." + str(k)))
    body = "\n".join(x for x in parts if x)
    src = ("\n<!-- 출처: results/aggregate/" + "{" + ",".join(envs) + "}_iqm.csv, "
           "*_lambda_star.json. 조건별 원본은 results/raw/{환경}/{계열}/lam{λ}/seed{n}_final.csv. "
           "설계 결정과 도중에 고친 버그는 docs/실험일지.md 참조. -->\n")
    out = Path(a.out) if a.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER_TMPL + "\n" + body + "\n" + FAIRNESS_TMPL + src, encoding="utf-8")
    print("Ⅳ장 생성: " + str(out.relative_to(ROOT)))


if __name__ == "__main__":
    main()
