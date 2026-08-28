"""원고 일관성 점검 — 실험이 갱신되면서 낡아버린 문장을 찾는다.

논문의 가장 흔한 사고는 '숫자를 고쳤는데 그 숫자를 언급한 다른 문장을 안 고친 것'이다.
자동 생성되는 Ⅳ장과 달리 사람이 쓴 장(Ⅰ·Ⅱ·Ⅲ·Ⅴ·초록)은 손으로 고쳐야 하므로
여기서 위험한 표현을 기계적으로 훑는다. **자동으로 고치지 않는다** — 사람이 보고 판단할 목록만 만든다.

실행: python -m src.report.check_manuscript
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
AGG = ROOT / "results" / "aggregate"

# (이름, 정규식, 왜 위험한가)
RISKS = [
    ("환경 수 표현", r"두 환경|환경이 둘|양쪽 환경",
     "환경이 셋이 되면 낡는다 — 개수를 세는 표현은 확인할 것"),
    ("낡은 pump 값", r"120\.5|119\.8",
     "pump 규칙의 현재 값은 results/aggregate에서 확인할 것"),
    ("λ* 하드코딩", r"λ\?\*\s*(=|는)\s*\d",
     "λ*는 실험이 갱신되면 바뀐다 — 본문 대신 표를 가리킬 것"),
    ("기여 개수", r"얻은 것은 (두|세|네|다섯) 가지|기여는 (두|세|네) 가지",
     "항목을 늘리면 개수 표현이 낡는다"),
    ("확정 어투", r"확인했다|입증했다|증명했다",
     "실제로 확인한 범위를 넘지 않는지 볼 것"),
    ("결과 대기 표시", r"\[실험 대기 — 결과 나오면 확정\]",
     "그 실험이 끝나면 이 문장을 결과로 채우고 표시를 지울 것"),
    ("본인 확인 표시", r"\[본인 확인 필요[^\]]*\]",
     "제출 전 본인이 채워야 하는 자리"),
]


def numbers_in_use() -> dict:
    """집계 파일의 현재 기준 수치 — 원고와 대조할 때 쓴다."""
    out = {}
    for p in sorted(AGG.glob("*_lambda_star.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        env = d.get("env_id", p.stem)
        out[env] = {r["learner"]: {"CI": r.get("lam_star_ci"), "점추정": r.get("lam_star_pt")}
                    for r in d.get("results_vs_best_rule", d.get("results", []))}
    return out


def main() -> None:
    print("=" * 78)
    print("원고 일관성 점검 — 사람이 판단할 목록 (자동 수정하지 않는다)")
    print("=" * 78)
    n_hits = 0
    for f in sorted(PAPER.glob("*.md")):
        if f.name.startswith("00_양식"):
            continue
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines()
        hits = []
        for name, pat, why in RISKS:
            for m in re.finditer(pat, text):
                ln = text[:m.start()].count("\n")
                hits.append((ln + 1, name, lines[ln].strip()[:100], why))
        if hits:
            print(f"\n[{f.name}]")
            for ln, name, ctx, why in sorted(hits):
                n_hits += 1
                print(f"  {ln:>4}행 ({name}) {ctx}")
                print(f"        └ {why}")
    print(f"\n확인할 곳 {n_hits}군데")
    print("\n현재 집계 기준 λ* (최강 규칙 포락선 대비) — 원고 수치와 대조할 것")
    for env, rows in numbers_in_use().items():
        bits = ", ".join(f"{k}: CI {v['CI']} / 점추정 {v['점추정']}" for k, v in rows.items())
        print(f"  {env}: {bits}")
    print("=" * 78)


if __name__ == "__main__":
    main()
