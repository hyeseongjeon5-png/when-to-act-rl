"""논문 4장(결과) 원고를 집계 파일에서 자동 생성한다.

왜 자동 생성인가: 숫자를 손으로 옮겨 적으면 반드시 틀린다. 그리고 실험이 갱신되면
원고의 숫자가 조용히 낡는다. 이 스크립트는 results/aggregate/*.csv 에서만 숫자를 읽어
paper/04_결과.md 를 다시 쓴다 (CLAUDE.md 절대 규칙 4 — 숫자는 로그 파일에서만 인용).

서술(해석·논의)은 사람이 쓴다. 이 스크립트는 표와 사실 문장까지만 만든다.

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

LABEL = {"dqn": "표준 DQN", "temporl": "TempoRL 방식", "lazy": "Lazy-MDP 방식"}
RULE_LABEL = {"rule_pump": "pump(임계값) 규칙", "rule_threshold": "임계값 규칙",
              "rule_noop": "무행동", "rule_periodic_k1": "매 스텝 주기",
              "rule_periodic_k2": "2스텝 주기", "rule_periodic_k4": "4스텝 주기",
              "rule_periodic_k8": "8스텝 주기"}
REF_RULE = {"MountainCar-v0": "rule_pump", "LunarLander-v3": "rule_threshold"}
ENV_NOTE = {
    "MountainCar-v0": "보상이 희소한 탐험 문제. 표준 ε-greedy 탐험으로는 목표에 닿지 못한다.",
    "LunarLander-v3": "보상이 조밀한 제어 문제. 표준 DQN이 정상적으로 학습된다.",
}


def name(a: str) -> str:
    return LABEL.get(a, RULE_LABEL.get(a, a))


def num(v, nd=1) -> str:
    try:
        return format(float(v), "." + str(nd) + "f")
    except Exception:
        return "—"


def perf_table(agg: pd.DataFrame, env_id: str, metric: str = "cost_return") -> str:
    lams = sorted(agg.lam.unique())
    learners = [a for a in sorted(agg.agent.unique()) if not str(a).startswith("rule_")]
    rules = [a for a in sorted(agg.agent.unique()) if str(a).startswith("rule_")]
    head = "| 계열 / 규칙 | " + " | ".join("λ=" + format(l, "g") for l in lams) + " |"
    sep = "|---|" + "---|" * len(lams)
    lines = [head, sep]
    for a in learners + rules:
        g = agg[agg.agent == a].set_index("lam")
        cells = []
        for l in lams:
            if l not in g.index:
                cells.append("—")
                continue
            r = g.loc[l]
            cells.append(num(r[metric + "_iqm"]) + " <sub>[" + num(r[metric + "_ci_lo"])
                         + ", " + num(r[metric + "_ci_hi"]) + "]</sub>")
        bold = "**" if a in learners else ""
        lines.append("| " + bold + name(a) + bold + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def action_table(agg: pd.DataFrame) -> str:
    lams = sorted(agg.lam.unique())
    learners = [a for a in sorted(agg.agent.unique()) if not str(a).startswith("rule_")]
    ref = [a for a in agg.agent.unique() if a in REF_RULE.values()]
    head = "| 계열 | " + " | ".join("λ=" + format(l, "g") for l in lams) + " |"
    lines = [head, "|---|" + "---|" * len(lams)]
    for a in learners + sorted(ref):
        g = agg[agg.agent == a].set_index("lam")
        cells = [num(g.loc[l]["n_actions_iqm"]) if l in g.index else "—" for l in lams]
        lines.append("| " + name(a) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def star_table(env_id: str) -> tuple[str, list]:
    p = AGG / (env_id + "_lambda_star.json")
    if not p.exists():
        return "*(λ\\* 계산 결과 없음)*", []
    st = json.loads(p.read_text(encoding="utf-8"))
    rows = ["| 학습 계열 | λ\\*<sub>CI</sub> (엄격) | λ\\*<sub>점추정</sub> (느슨) | 상태 |", "|---|---|---|---|"]
    for s in st.get("results", []):
        ci = s["lam_star_ci"] if s["lam_star_ci"] is not None else "격자 안에 없음"
        pt = s["lam_star_pt"] if s["lam_star_pt"] is not None else "격자 안에 없음"
        rows.append("| " + name(s["learner"]) + " | " + str(ci) + " | " + str(pt) + " | " + s.get("note", "") + " |")
    return "\n".join(rows), st.get("results", [])


def env_chapter(env_id: str) -> str:
    p = AGG / (env_id + "_iqm.csv")
    if not p.exists():
        return "### " + env_id + "\n\n*(집계 결과 없음)*\n"
    agg = pd.read_csv(p)
    ref = REF_RULE.get(env_id, "rule_pump")
    learners = [a for a in sorted(agg.agent.unique()) if not str(a).startswith("rule_")]
    seeds_learner = agg[agg.agent.isin(learners)]["n_seeds"]
    lo_s = int(seeds_learner.min()) if len(seeds_learner) else 0
    hi_s = int(seeds_learner.max()) if len(seeds_learner) else 0
    n_lam_learner = agg[agg.agent.isin(learners)].lam.nunique()
    n_lam_grid = agg.lam.nunique()
    steps = int(agg[agg.agent.isin(learners)]["total_steps"].max()) if len(seeds_learner) else 0

    status = ("학습 계열은 λ 격자 " + str(n_lam_grid) + "개 중 " + str(n_lam_lam(n_lam_learner))
              + "개, 시드 " + (str(lo_s) if lo_s == hi_s else str(lo_s) + "~" + str(hi_s))
              + "개까지 완료된 시점의 집계다.")
    if lo_s < 10 or n_lam_learner < n_lam_grid:
        status += " **아직 실험이 끝나지 않았으므로 최종 결론이 아니다.**"

    st_tbl, st_rows = star_table(env_id)
    facts = []
    for s in st_rows:
        lname = name(s["learner"])
        if s.get("lam_star_pt") == 0.0:
            facts.append("- " + lname + "은 λ=0에서도 " + RULE_LABEL.get(ref, ref)
                         + "을 이기지 못했다. 비용 때문이 아니라 학습된 정책 자체가 규칙보다 약하다는 뜻이다.")
        elif s.get("lam_star_pt") is not None:
            facts.append("- " + lname + "은 λ=" + str(s["lam_star_pt"])
                         + "에서 " + RULE_LABEL.get(ref, ref) + "에 역전당했다"
                         + (" (통계적으로 확실한 우위는 λ=" + str(s["lam_star_ci"]) + "에서 이미 사라졌다)"
                            if s.get("lam_star_ci") is not None else "") + ".")
        else:
            facts.append("- " + lname + "은 비교된 λ 구간 전체에서 " + RULE_LABEL.get(ref, ref)
                         + "을 이겼다 (" + str(s.get("coverage", "")) + " 비교됨).")

    return ("### " + env_id + "\n\n"
            + ENV_NOTE.get(env_id, "") + " 조건당 환경 " + format(steps, ",") + "스텝, "
            + "평가는 탐험을 끈 상태로 100 에피소드. " + status + "\n\n"
            + "#### 비용 반영 총보상 r' — IQM [95% 신뢰구간]\n\n"
            + perf_table(agg, env_id) + "\n\n"
            + "#### 에피소드당 행동 횟수 (IQM)\n\n"
            + action_table(agg) + "\n\n"
            + "#### 임계 비용 λ\\*\n\n" + st_tbl + "\n\n"
            + "\n".join(facts) + "\n\n"
            + "![λ-성능 지도](../results/figures/" + env_id + "_lambda_map_cost_return.png)\n\n"
            + "![행동 횟수 지도](../results/figures/" + env_id + "_action_map.png)\n\n"
            + "![학습 곡선](../results/figures/" + env_id + "_learning_curves.png)\n")


def n_lam_lam(n: int) -> int:
    return n


HEADER = """# 4장 · 결과

> **이 문서는 자동 생성된다.** `python -m src.report.make_results_chapter` 를 돌리면
> `results/aggregate/*.csv` 에서 숫자를 다시 읽어 이 파일을 덮어쓴다.
> 숫자를 손으로 고치지 말 것 — 다음 생성 때 사라지고, 그 사이 원고가 실험과 어긋난다.
> 해석과 논의는 5장(`05_논의.md`)에 사람이 쓴다.

## 4.1 읽는 법

- **λ**: 행동 1번의 값. 오른쪽으로 갈수록 행동이 비싸다. λ=0이면 원래 문제와 같다.
- **r'**: 비용까지 빼고 남은 총보상. 높을수록 좋다.
- **대괄호**: 95% 계층 부트스트랩 신뢰구간. 두 구간이 겹치면 우열을 말하지 않는다.
- **기준선**: 각 환경에서 가장 센 고정 규칙 (MountainCar는 pump 규칙, LunarLander는 임계값 규칙).
  문제를 아예 못 푸는 약한 규칙(무행동·주기)을 기준으로 삼으면 "학습이 이겼다"가 너무 쉬워진다.
- **λ\\***: 학습이 그 기준 규칙을 더 이상 이기지 못하게 되는 가장 작은 λ.
  격자 안에서 교차가 없으면 보간하지 않고 "격자 안에 없음"이라고 적는다.

## 4.2 환경별 결과
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    envs = sorted({p.name.replace("_iqm.csv", "") for p in AGG.glob("*_iqm.csv")})
    if not envs:
        print("집계 파일이 없다 — 먼저 aggregate를 돌릴 것")
        return
    body = "\n".join(env_chapter(e) for e in envs)
    src = ("\n---\n\n*출처: `results/aggregate/{" + ",".join(envs) + "}_iqm.csv`, "
           "`*_lambda_star.json`. 조건별 원본은 `results/raw/{환경}/{계열}/lam{λ}/seed{n}_final.csv`. "
           "설계 결정과 도중에 고친 버그는 `docs/실험일지.md` 참조.*\n")
    out = Path(a.out) if a.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER + "\n" + body + src, encoding="utf-8")
    print("결과 장 생성: " + str(out.relative_to(ROOT)))


if __name__ == "__main__":
    main()
