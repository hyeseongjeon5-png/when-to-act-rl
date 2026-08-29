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


def section_refs() -> list[str]:
    """원고가 가리키는 절 번호가 실제로 있는지 본다.

    왜 필요한가: 절을 하나 끼워 넣으면 뒤 번호가 밀리는데, **그 절을 가리키던 다른 장의 문장은
    그대로 남는다.** 실제로 Ⅲ장에 기준선 감사를 6.2로 넣자 인과 검사가 6.3이 되었고,
    Ⅴ장이 여전히 '6.2절'을 가리키고 있었다. 읽는 사람은 엉뚱한 절로 간다.
    """
    heads = {}
    for f in sorted(PAPER.glob("0[0-9]_*.md")):
        ch = None
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^# ([ⅠⅡⅢⅣⅤ])\.", line.strip())
            if m:
                ch = m.group(1)
            m2 = re.match(r"^#{2,3}\s*(\d+(?:\.\d+)?)[\.\s]", line.strip())
            if m2 and ch:
                heads.setdefault(ch, set()).add(m2.group(1))
    out, bad = [], 0
    for f in sorted(PAPER.glob("0[0-9]_*.md")):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"([ⅠⅡⅢⅣⅤ])장\s*(\d+(?:\.\d+)?)절", text):
            ch, sec = m.group(1), m.group(2)
            if sec not in heads.get(ch, set()):
                ln = text[:m.start()].count(chr(10)) + 1
                out.append(f"  [없는 절] {f.name} {ln}행: '{ch}장 {sec}절' — "
                           f"{ch}장에 있는 절: {sorted(heads.get(ch, []))}")
                bad += 1
    if not out:
        n = sum(len(v) for v in heads.values())
        out = [f"  [맞음] 원고가 가리키는 절 번호가 모두 실제로 있다 (절 {n}개 확인)"]
    return out


def numbering_rules() -> list[str]:
    """조립된 .docx에서 그림·표 번호가 지시받은 배치와 맞는지 본다.

    왜 필요한가: 번호는 등장 순서로 자동 부여되므로 **다른 장에 표를 하나 추가하면
    조용히 밀린다.** 실제로 Ⅲ장에 표를 넣었더니 '표 1 = 임계 비용 λ*'가 표 2로 밀렸다.
    사람 눈에는 잘 안 띄는 종류의 어긋남이라 기계가 본다.
    """
    docx = ROOT / "졸업논문_초안v1.docx"
    if not docx.exists():
        return ["  [건너뜀] .docx가 아직 없다"]
    try:
        import mammoth
        with docx.open("rb") as f:
            html = mammoth.convert_to_html(f).value
    except Exception as e:
        return [f"  [건너뜀] .docx를 읽지 못함: {e}"]
    text = re.sub(r"<[^>]+>", " ", html)
    caps = {}
    for m in re.finditer(r"(표|그림) (\d+)\.\s*([^|]{0,60})", text):
        key = f"{m.group(1)} {m.group(2)}"
        caps.setdefault(key, re.sub(r"\s+", " ", m.group(3)).strip())
    # 지시받은 배치 (사용자 요구사항)
    want = [("표 1", "임계 비용"), ("표 2", "실험 설정"),
            ("그림 1", "래퍼"), ("그림 2", "λ-성능 지도"), ("그림 3", "λ-성능 지도"),
            ("그림 4", "무행동 붕괴")]
    out = []
    for key, must in want:
        got = caps.get(key)
        if got is None:
            out.append(f"  [없음] {key} — 아직 만들어지지 않았다 (실험이 끝나면 생길 수 있다)")
        elif must not in got:
            out.append(f"  [어긋남] {key}가 '{got[:40]}' 이다 — '{must}'이어야 한다")
        else:
            out.append(f"  [맞음] {key}. {got[:46]}")
    return out


def check_dangling_headings() -> None:
    """제목 바로 뒤에 또 제목이 오는 곳을 찾는다.

    그런 절은 "무엇을 말하는 절인지" 한 문장도 없이 시작한다. 실제로 이 원고의
    Ⅳ장 2절(가장 중요한 절)이 아무 문장 없이 표부터 시작하고 있었다 —
    심사자가 표를 스스로 해석해야 했다. 이 저장소의 글쓰기 규칙은 "결론을 맨 위에 쓴다"이다.
    """
    print(chr(10) + "절이 문장 없이 시작하지 않는가")
    bad = []
    for f in sorted(PAPER.glob("*.md")):
        if f.name.startswith("00_양식"):
            continue
        lines = f.read_text(encoding="utf-8").split(chr(10))
        for i, l in enumerate(lines):
            if not re.match(r"^#{2,3} ", l):
                continue
            for j in range(i + 1, len(lines)):
                t = lines[j].strip()
                if not t:
                    continue
                if re.match(r"^#{2,3} ", t):
                    bad.append((f.name, i + 1, l.strip(), t))
                break
    if not bad:
        print("  [맞음] 모든 절이 문장이나 목록으로 시작한다")
        return
    for fn_, ln, h, nxt in bad:
        print(f"  [고칠 것] {fn_}:{ln} '{h[:38]}' 바로 뒤에 '{nxt[:30]}'")
        print("        └ 이 절이 무엇을 말하는지 한 문장을 앞에 둘 것")


def main() -> None:
    print("=" * 78)
    print("원고 일관성 점검 — 사람이 판단할 목록 (자동 수정하지 않는다)")
    print("=" * 78)
    n_hits = 0
    # 참고문헌은 원고가 아니라 조립 스크립트에 들어 있다 — 거기 남은 표시도 함께 본다
    targets = sorted(PAPER.glob("*.md")) + [ROOT / "src" / "report" / "make_thesis_docx.py"]
    for f in targets:
        if f.name.startswith("00_양식") or not f.exists():
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
    check_dangling_headings()
    print(f"\n절 참조가 맞는가")
    for line in section_refs():
        print(line)
    print(f"\n지시받은 그림·표 배치와 맞는가")
    for line in numbering_rules():
        print(line)
    print("\n현재 집계 기준 λ* (최강 규칙 포락선 대비) — 원고 수치와 대조할 것")
    for env, rows in numbers_in_use().items():
        bits = ", ".join(f"{k}: CI {v['CI']} / 점추정 {v['점추정']}" for k, v in rows.items())
        print(f"  {env}: {bits}")
    print("=" * 78)


if __name__ == "__main__":
    main()
