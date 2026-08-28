"""인과 실험 분석 — 무행동 붕괴는 최적해인가, 탐험 실패인가.

같은 λ·같은 예산·같은 시드에서 **비용을 켜는 시점만** 다른 두 조건을 비교한다.

  처음부터 비용   : results/raw/MountainCar-v0/…              (본실험)
  절반 뒤에 비용  : results/raw/MountainCar-v0@warmup50/…     (인과 실험)

읽는 법:
  워밍업 쪽이 **목표에 더 자주 닿으면** → 탐험 실패다. 비용이 있는 세상에서도 목표에 닿는
    정책이 존재하는데, 처음부터 비용을 물리면 그것을 못 찾는 것이다.
  워밍업 쪽도 **똑같이 무행동으로 굳으면** → 비용을 반영한 최적해가 정말 무행동이다.

  실험 2에서 붕괴가 두 단계로 일어난다는 것이 밝혀졌다 — 행동은 유지되는데 성능이 먼저
  무너지는 구간이 있다. 그래서 점수·행동뿐 아니라 **목표 도달률**을 함께 본다.
  그 구간에서 워밍업 쪽이 도달률을 지켜 낸다면 그것이 가장 직접적인 증거다.

두 조건은 환경 스텝 예산·시드·평가 방식이 모두 같고, 다른 것은 학습 중 보상 신호뿐이다.
평가는 양쪽 모두 **진짜 λ**로 한다 — 성적표는 언제나 비용이 있는 세상에서 매긴다.

실행: python -m src.analysis.causal_compare
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.aggregate import iqm_ci

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "aggregate" / "causal_warmup.json"

BASE_ROOM = "MountainCar-v0"
WARM_ROOM = "MountainCar-v0@warmup50"
AGENT_KO = {"dqn": "표준 DQN", "temporl": "TempoRL", "lazy": "Lazy-MDP"}


def seed_values(room: str, agent: str, lam: float, metric: str) -> list[float]:
    out = []
    for m in sorted(glob.glob(str(RAW / room / agent / f"lam{lam}" / "seed*_meta.json"))):
        try:
            d = json.loads(Path(m).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d.get("done"):
            continue
        csv = Path(m).parent / Path(m).name.replace("_meta.json", "_final.csv")
        if not csv.exists():
            continue
        out.append(float(pd.read_csv(csv)[metric].mean()))
    return out


def stat(vals: list[float], reps: int = 10000) -> dict | None:
    if len(vals) < 2:
        return None
    p, lo, hi = iqm_ci(np.array(vals), reps=reps)
    return {"n": len(vals), "iqm": p, "lo": lo, "hi": hi}


def verdict(base: dict, warm: dict, base_act: dict, warm_act: dict,
            base_solved: dict | None = None, warm_solved: dict | None = None) -> str:
    """두 조건의 신뢰구간으로 판정한다. 겹치면 우열을 말하지 않는다.

    실험 2에서 붕괴가 두 단계로 일어난다는 것이 밝혀졌다 — 행동은 유지되는데 성능이 먼저
    무너지는 구간이 있다. 그래서 **목표 도달률**을 함께 본다. 그 구간에서 워밍업 쪽이
    도달률을 지켜 낸다면, 그것이 '탐험 실패'의 가장 직접적인 증거다.
    """
    acts_up = warm_act["lo"] > base_act["hi"]
    score_up = warm["lo"] > base["hi"]
    solved_up = bool(base_solved and warm_solved and warm_solved["lo"] > base_solved["hi"])
    if solved_up and score_up:
        return "탐험 실패 (워밍업 쪽이 도달률도 점수도 높다)"
    if solved_up:
        return "탐험 실패 (워밍업 쪽이 목표에 더 자주 닿는다)"
    if score_up and acts_up:
        return "탐험 실패 (워밍업 쪽이 행동도 점수도 높다)"
    if acts_up and not score_up:
        return "행동은 늘었으나 점수 차이는 불확실"
    if score_up and not acts_up:
        return "점수는 높으나 행동 차이는 불확실"
    if warm_act["hi"] < 1.0 and base_act["hi"] < 1.0:
        return "양쪽 모두 무행동으로 굳음 → 무행동이 최적해일 가능성"
    return "차이 불확실 (신뢰구간이 겹친다)"


def main() -> None:
    if not (RAW / WARM_ROOM).exists():
        print(f"{WARM_ROOM} 결과가 아직 없다 — 인과 실험(실험 3)이 끝난 뒤에 돌릴 것")
        return
    lams = sorted({float(p.name[3:]) for p in (RAW / WARM_ROOM).glob("*/lam*") if p.is_dir()})
    rows, payload = [], []
    print("=" * 104)
    print("인과 실험 — 비용을 처음부터 물릴 때 vs 절반 뒤에 켤 때 (MountainCar-v0, 같은 예산·시드)")
    print("=" * 104)
    print(f"{'λ':>7} {'계열':<10} {'조건':<12} {'시드':>4} {'r′ IQM':>9} {'95% CI':>20} "
          f"{'행동':>9} {'도달률':>6} {'판정':<38}")
    for lam in lams:
        for ag in ("dqn", "temporl", "lazy"):
            b = stat(seed_values(BASE_ROOM, ag, lam, "cost_return"))
            w = stat(seed_values(WARM_ROOM, ag, lam, "cost_return"))
            ba = stat(seed_values(BASE_ROOM, ag, lam, "n_actions"))
            wa = stat(seed_values(WARM_ROOM, ag, lam, "n_actions"))
            bs = stat(seed_values(BASE_ROOM, ag, lam, "solved"))
            ws = stat(seed_values(WARM_ROOM, ag, lam, "solved"))
            if not (b and w and ba and wa):
                continue
            v = verdict(b, w, ba, wa, bs, ws)
            for label, sc, a, sv in (("처음부터 비용", b, ba, bs), ("절반 뒤 비용", w, wa, ws)):
                sv_txt = f"{sv['iqm']*100:5.0f}%" if sv else "    —"
                print(f"{lam:>7g} {AGENT_KO[ag]:<10} {label:<12} {sc['n']:>4} {sc['iqm']:>9.1f} "
                      f"{'[' + format(sc['lo'], '.1f') + ', ' + format(sc['hi'], '.1f') + ']':>20} "
                      f"{a['iqm']:>9.1f} {sv_txt} {v if label == '절반 뒤 비용' else '':<38}")
            payload.append({"lam": lam, "agent": ag, "verdict": v,
                            "from_start": {"score": b, "actions": ba, "solved": bs},
                            "warmup": {"score": w, "actions": wa, "solved": ws}})
            rows.append((lam, ag, v))
    if not payload:
        print("비교할 수 있는 조건이 아직 없다 (양쪽 방에 같은 λ·계열이 모두 있어야 한다)")
        return
    OUT.write_text(json.dumps({
        "설명": "같은 λ·예산·시드에서 비용을 켜는 시점만 바꾼 대조. 평가는 양쪽 모두 진짜 λ로 한다.",
        "base_room": BASE_ROOM, "warmup_room": WARM_ROOM, "results": payload,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 104)
    n_expl = sum(1 for _, _, v in rows if v.startswith("탐험 실패"))
    n_opt = sum(1 for _, _, v in rows if "무행동이 최적해" in v)
    print(f"요약: 비교 {len(rows)}건 중 '탐험 실패' {n_expl}건 · '무행동이 최적해' {n_opt}건 · "
          f"나머지 {len(rows) - n_expl - n_opt}건은 판정 보류(신뢰구간 겹침)")
    print(f"저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
