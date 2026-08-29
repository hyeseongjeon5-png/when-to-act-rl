"""손으로 옮겨 적은 표의 숫자가 집계 파일과 맞는지 본다.

왜 필요한가 (2026-08-29):
    Ⅳ장은 자동 생성이라 안전하지만, Ⅴ장(결론)은 사람이 쓰는 장이라 표를 손으로 옮겨 적었다.
    손으로 적은 숫자는 재측정하면 **조용히 낡는다.** 실제로 기준선 감사의 잣대를 고치자
    본문 세 곳의 숫자가 한꺼번에 낡았다.

    이 저장소의 규칙은 "숫자는 만들어내지 않고 results/에서만 인용한다"이다.
    그 규칙이 지켜지는지 기계가 확인한다.

실행: python -m src.report.check_hand_numbers
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "results" / "aggregate"
TOL = 0.06      # 원고는 반올림해 적으므로 소수점 한 자리 반올림 오차를 허용한다


def cells(md: str, header_starts: str) -> list[list[str]]:
    """머리글이 header_starts로 시작하는 표의 데이터 줄을 뽑는다."""
    out, on = [], False
    for line in md.split(chr(10)):
        t = line.strip()
        if t.startswith("| " + header_starts):
            on = True
            continue
        if on:
            if not t.startswith("|"):
                break
            row = [c.strip().replace("**", "") for c in t.strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in row):
                out.append(row)
    return out


def close(a: str, b: float, tol: float = TOL) -> bool:
    try:
        return abs(float(a.replace("배", "").replace(",", "")) - b) <= max(tol, abs(b) * 0.006)
    except Exception:
        return False


def check_switch_cost() -> list[str]:
    p = AGG / "switch_cost_rules.json"
    src = ROOT / "paper" / "05_결론.md"
    if not p.exists() or not src.exists():
        return ["  [건너뜀] 전환비용 집계나 Ⅴ장 원고가 없다"]
    data = json.loads(p.read_text(encoding="utf-8"))["results"]
    rows = cells(src.read_text(encoding="utf-8"), "환경 / 규칙")
    if not rows:
        return ["  [건너뜀] Ⅴ장에서 전환비용 표를 찾지 못했다"]
    msgs, bad = [], 0
    for row in rows:
        if len(row) < 6:
            continue
        env_key = row[0].split("/")[0].strip().replace("MinAtar Freeway", "MinAtar/Freeway-v1")
        rec = next((d for d in data if d["env_id"].startswith(env_key.split("-")[0])
                    and env_key.split("-")[0] in d["env_id"]), None)
        if rec is None:
            msgs.append(f"  [확인] '{row[0]}' 에 해당하는 집계 항목을 못 찾았다")
            bad += 1
            continue
        want = [("매 스텝 과금 횟수", row[1], rec["per_step"]["charged_mean"]),
                ("전환 과금 횟수", row[2], rec["per_switch"]["charged_mean"]),
                ("λ_교차(매 스텝)", row[3], rec["per_step"]["lam_cross"]),
                ("λ_교차(전환)", row[4], rec["per_switch"]["lam_cross"]),
                ("배율", row[5], rec["lam_cross_ratio"])]
        for label, got, exp in want:
            if exp is None:
                continue
            if not close(got, float(exp)):
                msgs.append(f"  [고칠 것] {row[0]} {label}: 원고 {got} · 집계 {float(exp):.3f}")
                bad += 1
    if not bad:
        msgs.append(f"  [맞음] Ⅴ장 전환비용 표의 숫자 {len(rows) * 5}개가 집계와 일치한다")
    return msgs


def check_collapse_ratio() -> list[str]:
    """Ⅴ장의 "행동이 완전히 멈추는 것은 그보다 N~M배 비싸진 뒤다"를 집계에서 다시 계산해 본다.

    2026-08-29에 이 자리에 '3~10배'가 적혀 있었으나 실제로는 4~27배였다.
    이 논문의 핵심 발견(붕괴가 두 단계다)에 붙은 숫자라 틀리면 안 된다.
    """
    import pandas as pd
    csv = AGG / "MountainCar-v0_iqm.csv"
    src = ROOT / "paper" / "05_결론.md"
    if not csv.exists() or not src.exists():
        return ["  [건너뜀] MountainCar 집계나 Ⅴ장 원고가 없다"]
    m = re.search(r"행동이 완전히 멈추는 것은 그보다 (\d+)~(\d+)배", src.read_text(encoding="utf-8"))
    if not m:
        return ["  [건너뜀] Ⅴ장에서 붕괴 배수 문장을 찾지 못했다"]
    a = pd.read_csv(csv)
    ratios = []
    for ag in ("dqn", "temporl", "lazy"):
        z = a[(a.agent == ag) & (a.lam == 0.0)]
        g = a[(a.agent == ag) & (a.lam > 0)].sort_values("lam")
        if z.empty or g.empty:
            continue
        z = z.iloc[0]
        half = g[g.solved_iqm < z.solved_iqm * 0.5]
        zero = g[g.n_actions_iqm <= 1e-9]
        if half.empty or zero.empty:
            continue
        ratios.append(float(zero.iloc[0].lam) / float(half.iloc[0].lam))
    if not ratios:
        return ["  [건너뜀] 배수를 계산할 조건이 모자란다"]
    lo, hi = round(min(ratios)), round(max(ratios))
    got = (int(m.group(1)), int(m.group(2)))
    if got == (lo, hi):
        return [f"  [맞음] 붕괴 배수 {lo}~{hi}배가 집계와 일치한다"]
    return [f"  [고칠 것] 붕괴 배수: 원고 {got[0]}~{got[1]}배 · 집계 {lo}~{hi}배",
            "        └ Ⅴ장 '넷째' 문단의 배수를 고칠 것"]


def main() -> None:
    print("=" * 74)
    print("손으로 옮겨 적은 숫자가 집계와 맞는가")
    print("=" * 74)
    for line in check_switch_cost():
        print(line)
    for line in check_collapse_ratio():
        print(line)
    print("=" * 74)


if __name__ == "__main__":
    main()
