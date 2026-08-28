"""공정성 파일럿 판정 — λ=0에서 학습이 pump 규칙 수준에 닿는가.

이 판정 하나가 스프린트의 20시간짜리 재실험(1-B)을 돌릴지 말지를 정한다.
그래서 '눈으로 곡선을 보고' 정하지 않고, 본실험과 **같은 자**로 잰다:
최종 평가(스냅샷 3장 × 50 에피소드) · 시드별 점수 · IQM · 95% 계층 부트스트랩 신뢰구간.

판정 규칙 (docs/02_실험-설계.md §5의 λ* 판정과 같은 기준):
  이김   : 학습의 CI 하한 > 규칙의 CI 상한      → 재격자를 돌린다
  비김   : 두 CI가 겹친다                        → 재격자를 돌린다(개선 신호는 있다)
  짐     : 학습의 CI 상한 < 규칙의 CI 하한       → 재격자 불필요, 결론 확정

실행: python -m src.analysis.fairness_verdict
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from src.analysis.aggregate import iqm_ci

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"

CANDIDATES = {
    "MountainCar-v0": "1차 본실험 (300k, ε=0.2 고정)",
    "MountainCar-v0@budget1M_epsconst": "후보 A — 예산 1M (ε=0.2 고정)",
    "MountainCar-v0@budget1M_epsdecay": "후보 B — 예산 1M + ε 감소",
    "MountainCar-v0@budget1M_wide": "후보 C — 예산 1M + ε 감소 + 용량 확대",
}
REF_ROOM, REF_RULE = "MountainCar-v0", "rule_pump"


def seed_scores(room: str, agent: str, lam: float = 0.0, metric: str = "raw_return") -> list[float]:
    """조건별 시드 점수 = 그 시드의 최종 평가 에피소드 평균 (본실험 집계와 동일한 방식)."""
    import pandas as pd
    out = []
    for m in sorted(glob.glob(str(RAW / room / agent / f"lam{lam}" / "seed*_meta.json"))):
        d = json.loads(Path(m).read_text(encoding="utf-8"))
        if not d.get("done"):
            continue
        csv = Path(m).parent / Path(m).name.replace("_meta.json", "_final.csv")
        if not csv.exists():
            continue
        out.append(float(pd.read_csv(csv)[metric].mean()))
    return out


def summarize(scores: list[float], reps: int = 10000) -> dict | None:
    if len(scores) < 2:
        return None
    p, lo, hi = iqm_ci(np.array(scores), reps=reps)
    return {"n": len(scores), "iqm": p, "lo": lo, "hi": hi}


def main() -> None:
    ref = summarize(seed_scores(REF_ROOM, REF_RULE))
    if ref is None:
        print("기준 규칙 결과가 없다 — 먼저 규칙 평가를 돌릴 것")
        return
    print(f"기준선  {'pump 규칙 (학습 없음)':<34} 시드 {ref['n']:>2} | "
          f"r IQM {ref['iqm']:8.1f}  95%CI [{ref['lo']:.1f}, {ref['hi']:.1f}]")
    print("-" * 100)

    # 본실험 방에서는 세 계열을 모두 본다 — λ*=0이 'DQN만의 문제'인지 확인해야 한다
    print("1차 본실험(300k)에서 세 계열이 각각 규칙에 얼마나 못 미쳤나")
    for ag, ko in (("dqn", "표준 DQN"), ("temporl", "TempoRL"), ("lazy", "Lazy-MDP")):
        s = summarize(seed_scores(REF_ROOM, ag))
        if not s:
            continue
        v = "이김" if s["lo"] > ref["hi"] else ("짐" if s["hi"] < ref["lo"] else "비김")
        print(f"  [{v}] {ko:<10} 시드 {s['n']:>2} | r IQM {s['iqm']:8.1f} "
              f"95%CI [{s['lo']:.1f}, {s['hi']:.1f}] | 규칙 대비 {s['iqm'] - ref['iqm']:+.1f}")
    print("-" * 100)
    print("공정성 파일럿 — 예산·탐험을 바꾼 표준 DQN이 규칙 수준에 닿는가")

    verdicts = {}
    for room, label in CANDIDATES.items():
        sc = seed_scores(room, "dqn")
        s = summarize(sc)
        if s is None:
            print(f"        {label:<34} 시드 {len(sc):>2} | (조건이 2개 미만 — 판정 불가)")
            continue
        if s["lo"] > ref["hi"]:
            v = "이김"
        elif s["hi"] < ref["lo"]:
            v = "짐"
        else:
            v = "비김"
        verdicts[room] = v
        gap = s["iqm"] - ref["iqm"]
        print(f"[{v}]   {label:<34} 시드 {s['n']:>2} | "
              f"r IQM {s['iqm']:8.1f}  95%CI [{s['lo']:.1f}, {s['hi']:.1f}] | "
              f"규칙 대비 {gap:+.1f}")
        print(f"        시드별: {', '.join(f'{x:.0f}' for x in sorted(sc))}")

    print("-" * 100)
    improved = [r for r, v in verdicts.items() if v in ("이김", "비김") and r != REF_ROOM]
    base_v = verdicts.get(REF_ROOM)
    if improved:
        print(f"판정: 개선 신호 있음 → λ 격자 재실험(1-B)을 돌린다. 후보: {', '.join(improved)}")
    else:
        print("판정: 예산·탐험을 바꿔도 pump 규칙을 이기지 못했다 "
              f"(1차 본실험도 '{base_v}').")
        print("      → λ 격자 재실험(1-B)은 불필요하다. 결론을 확정한다:")
        print("        '좋은 고정 규칙이 있는 희소 보상 환경에서는 비용과 무관하게 학습 자체가 열세다.'")
        print("      이 판정은 λ=0(비용 없음) 조건에서 났으므로 비용 탓으로 돌릴 수 없다.")


if __name__ == "__main__":
    main()
