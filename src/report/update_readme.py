"""README의 '결과' 절을 집계 파일에서 다시 쓴다.

왜: README는 채용 담당자가 보는 첫 화면인데, 손으로 옮겨 적은 숫자는 실험이 갱신되면
조용히 낡는다. 실제로 이 저장소의 README도 λ 격자가 9개에서 14개로 늘어난 뒤 낡아 있었다.
표시 구간(<!--AUTO:결과--> … <!--/AUTO:결과-->) 안쪽만 기계가 다시 쓰고, 나머지는 손대지 않는다.

실행: python -m src.report.update_readme
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregate"
README = ROOT / "README.md"

BEGIN, END = "<!--AUTO:결과-->", "<!--/AUTO:결과-->"
ENVS = [("MountainCar-v0", "MountainCar-v0 — 보상이 희소하고 좋은 규칙이 있다", "rule_pump"),
        ("LunarLander-v3", "LunarLander-v3 — 보상이 조밀하고 규칙이 약하다", "rule_threshold"),
        ("MinAtar_Freeway-v1", "MinAtar Freeway — 규칙이 쓸 만하지만 이길 여지가 있다", "rule_cautious")]
AG = {"dqn": "표준 DQN", "temporl": "TempoRL", "lazy": "Lazy-MDP"}


def num(v, nd=1) -> str:
    try:
        return format(float(v), "." + str(nd) + "f")
    except Exception:
        return "—"


def lam_star_table() -> list[str]:
    rows = ["| 환경 | 표준 DQN | TempoRL | Lazy-MDP | 시드 | 해석 |", "|---|---|---|---|---|---|"]
    any_row = False
    for env, label, _ in ENVS:
        p = AGG / f"{env}_lambda_star.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        by = {r["learner"]: r for r in d.get("results_vs_best_rule", [])}
        if not by:
            continue
        cells = []
        for ag in ("dqn", "temporl", "lazy"):
            v = by.get(ag, {}).get("lam_star_pt")
            cells.append("**격자 밖**" if v is None else f"**{float(v):g}**")
        seeds = min((r.get("min_seeds", 0) for r in d.get("results", [])), default=0)
        note = ("λ=0에서도 규칙을 넘어서지 못한다" if all(c == "**0**" for c in cells)
                else "넓은 비용 구간에서 학습이 이긴다")
        rows.append(f"| {env} | " + " | ".join(cells) + f" | {seeds} | {note} |")
        any_row = True
    return rows if any_row else []


def env_block(env: str, label: str, rule: str) -> list[str]:
    p = AGG / f"{env}_iqm.csv"
    if not p.exists():
        return []
    agg = pd.read_csv(p)
    learners = [a for a in ("dqn", "temporl", "lazy") if a in set(agg.agent)]
    if not learners:
        return []
    lams = sorted(agg.lam.unique())
    show = lams if len(lams) <= 8 else [lams[0]] + lams[1:-1:max(1, (len(lams) - 2) // 5)] + [lams[-1]]
    cols = learners + [r for r in (rule, "rule_best", "rule_noop") if r in set(agg.agent)]
    names = {**AG, rule: f"{rule.replace('rule_', '')} 규칙(기준)",
             "rule_best": "최강 규칙", "rule_noop": "무행동"}
    out = [f"### {label}", "",
           "| 계열 | " + " | ".join("λ=" + format(l, "g") for l in show) + " |",
           "|---|" + "---|" * len(show)]
    for c in cols:
        g = agg[agg.agent == c].set_index("lam")
        cells = [num(g.loc[l]["cost_return_iqm"], 0) if l in g.index else "—" for l in show]
        bold = "**" if c in learners else ""
        out.append(f"| {bold}{names.get(c, c)}{bold} | " + " | ".join(cells) + " |")
    n = int(agg[agg.agent.isin(learners)].n_seeds.max())
    steps = int(agg[agg.agent.isin(learners)].total_steps.max())
    out += ["", f"조건당 환경 {steps:,}스텝 · 시드 {n}개 · 값은 비용 반영 총보상 r′의 IQM "
                f"(λ 격자 {len(lams)}개 중 일부만 표시)", "",
            f"![{env} λ-성능 지도](results/figures/{env}_lambda_map_cost_return.png)", ""]
    return out


def build() -> str:
    lines = ["", "시드 10개, IQM + 95% 계층 부트스트랩 신뢰구간 (Agarwal et al. 2021).",
             "λ\\* = 학습이 **그 λ에서 가장 센 고정 규칙**을 더 이상 이기지 못하게 되는 가장 작은 행동 비용.",
             "**이 표는 `results/aggregate/`에서 자동 생성된다** — 손으로 고치면 다음 실행에서 사라진다.",
             ""]
    t = lam_star_table()
    if t:
        lines += t + [""]
    for env, label, rule in ENVS:
        lines += env_block(env, label, rule)
    lines += ["> 자세한 표·그림·해석은 `paper/04_결과.md`(자동 생성)와 "
              "`results/reports/`의 HTML 보고서에 있다.", ""]
    return "\n".join(lines)


def main() -> None:
    text = README.read_text(encoding="utf-8")
    body = build()
    if BEGIN in text and END in text:
        new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                     BEGIN + "\n" + body + END, text, flags=re.S)
    else:
        # 표시 구간이 없으면 '## 결과' 절을 통째로 갈아 끼운다 (한 번만 일어난다)
        m = re.search(r"^## 결과.*?(?=^## )", text, flags=re.S | re.M)
        if not m:
            print("README에서 '## 결과' 절을 찾지 못했다 — 표시 구간을 직접 넣을 것")
            return
        new = text[:m.start()] + "## 결과 — 임계 비용 λ\\*\n\n" + BEGIN + "\n" + body + END + "\n\n" + text[m.end():]
    README.write_text(new, encoding="utf-8")
    print(f"README 결과 절 갱신 ({len(body)}자)")


if __name__ == "__main__":
    main()
