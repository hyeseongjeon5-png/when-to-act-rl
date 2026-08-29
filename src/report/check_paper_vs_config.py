"""논문이 적은 설정이 실제 config·코드와 같은지 대조한다.

왜 필요한가:
    Ⅲ장은 "은닉층 50×50, 학습률 1e-3, 배치 64 …" 처럼 **손으로 적은 설정**을 담고 있다.
    config를 바꾸면 이 문장이 조용히 낡는다. 논문이 실제로 돌린 것과 다른 설정을 적고 있으면
    재현성 주장 자체가 무너진다 — 이 저장소가 가장 크게 내세우는 것이 그것이다.

    이 저장소에서 확인한 수치 결함 5건이 전부 사람이 쓴 문장에서 나왔다.
    설정 문장도 같은 위험에 있다.

무엇을 대조하나: 본실험 config 3개(MountainCar · LunarLander · MinAtar)의 학습 계열
초매개변수와, Ⅲ장 4절이 적은 값. 세 config가 서로 다르면 그것도 알린다.

실행: python -m src.report.check_paper_vs_config
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
METHOD = ROOT / "paper" / "03_방법.md"
CONFIGS = [
    ROOT / "experiments" / "configs" / "main_mountaincar.yaml",
    ROOT / "experiments" / "configs" / "main_lunarlander.yaml",
    ROOT / "experiments" / "configs" / "main_minatar_freeway.yaml",
]

# 논문 문장에서 찾아낼 값 → config의 어느 키인지
CLAIMS = [
    ("은닉층", r"은닉층\s*(\d+)×(\d+)", "hidden", lambda m: [int(m.group(1)), int(m.group(2))]),
    ("학습률", r"Adam 학습률\s*([\d.e-]+)", "lr", lambda m: float(m.group(1))),
    ("할인율", r"할인율\s*([\d.]+)", "gamma", lambda m: float(m.group(1))),
    ("재생 버퍼", r"재생 버퍼\s*(\d+)만", "buffer_size", lambda m: int(m.group(1)) * 10000),
    ("배치", r"배치\s*(\d+)", "batch_size", lambda m: int(m.group(1))),
    ("목표망 갱신", r"목표망\s*(\d+)\s*갱신마다", "target_update", lambda m: int(m.group(1))),
    ("최대 지속 길이", r"최대 지속 길이\s*J\s*=\s*(\d+)", "max_skip", lambda m: int(m.group(1))),
]


def _declared(text: str, value) -> bool:
    """논문이 이 값을 예외로 밝혀 두었는지 본다 (예: 은닉층 [128,128] → '128×128')."""
    if isinstance(value, list) and len(value) == 2:
        return any(f"{value[0]}×{value[1]}" in text or f"{value[0]}x{value[1]}" in text
                   for _ in (0,))
    return str(value) in text


def main() -> int:
    print("=" * 78)
    print("논문이 적은 설정이 실제 config와 같은가")
    print("=" * 78)
    if not METHOD.exists():
        print("Ⅲ장 원고가 없다")
        return 1
    text = METHOD.read_text(encoding="utf-8")

    cfgs = []
    for p in CONFIGS:
        if p.exists():
            cfgs.append((p.name, yaml.safe_load(p.read_text(encoding="utf-8"))))
    if not cfgs:
        print("본실험 config를 찾지 못했다")
        return 1

    n_bad = 0
    for label, pat, key, conv in CLAIMS:
        m = re.search(pat, text)
        if not m:
            print(f"  [못 찾음] 논문에서 '{label}' 문장을 찾지 못했다 — 표현이 바뀌었는지 볼 것")
            n_bad += 1
            continue
        said = conv(m)
        for cname, c in cfgs:
            hp = c.get("hyperparams", {})
            # max_skip은 temporl에만 있다
            agents = ["temporl"] if key == "max_skip" else list(hp)
            for ag in agents:
                if ag not in hp or key not in hp[ag]:
                    continue
                real = hp[ag][key]
                if real == said:
                    continue
                if _declared(text, real):
                    continue          # 논문이 그 예외를 밝혀 두었다
                print(f"  [어긋남] {label}: 논문 {said} · {cname}의 {ag} {real}")
                print("           논문에 이 예외가 적혀 있지 않다 — 밝히거나 config를 맞출 것")
                n_bad += 1
    # 세 config가 서로 같은지도 본다 (환경마다 달라지면 '완전히 동일'이 거짓이 된다)
    common = ["hidden", "lr", "gamma", "buffer_size", "batch_size", "target_update", "eps_const"]
    for cname, c in cfgs:
        hp = c.get("hyperparams", {})
        for key in common:
            vals = {str(v.get(key)) for v in hp.values() if key in v}
            if len(vals) > 1:
                print(f"  [어긋남] {cname}: 같은 환경 안에서 {key}가 계열마다 다르다 — {vals}")
                print("           이것이 깨지면 공정 비교 주장이 무너진다")
                n_bad += 1

    if not n_bad:
        print(f"  [맞음] 논문이 적은 설정 {len(CLAIMS)}가지가 본실험 config {len(cfgs)}개와 모두 같다")
        print("         같은 환경 안에서 세 계열의 초매개변수가 완전히 동일하다")
        print("         환경에 따라 다른 값은 논문이 예외로 밝혀 두었다")
    print("=" * 78)
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
