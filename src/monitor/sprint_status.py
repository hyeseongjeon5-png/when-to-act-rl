"""스프린트 상태 한 장 요약 — 실험 진행 · 자가 감시 · 논문 진행률을 함께 본다.

사용자가 "상태 확인"이라고 할 때 이 하나만 돌리면 된다.
숫자는 전부 실제 파일에서 읽는다 (progress_*.json, watchdog.log, sprint_queue_state.json,
results/raw/, paper/*.md, 졸업논문_초안v1.docx).

실행: python -m src.monitor.sprint_status
"""
from __future__ import annotations

import glob
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
DOCX = ROOT / "졸업논문_초안v1.docx"

SPRINT_START = None  # results/sprint_start.txt 가 있으면 거기서 읽는다


def _hours_since(ts: float) -> float:
    return (time.time() - ts) / 3600


def sprint_clock() -> str:
    f = RESULTS / "sprint_start.txt"
    if not f.exists():
        return "스프린트 시작 시각 미기록"
    try:
        t0 = float(f.read_text(encoding="utf-8").strip())
    except Exception:
        return "스프린트 시작 시각을 읽지 못함"
    used = _hours_since(t0)
    return (f"경과 {used:.1f}시간 / 85시간  (남은 시간 약 {max(0.0, 85 - used):.1f}시간, "
            f"시작 {time.strftime('%Y-%m-%d %H:%M', time.localtime(t0))} KST)")


def experiments() -> list[str]:
    out = []
    for f in sorted(glob.glob(str(RESULTS / "progress_*.json"))):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        state = "종료" if d.get("finished") else f"진행중({d['running']})"
        eta = d.get("eta_text") or ""
        skipped = f" · 건너뜀 {d['skipped']}" if d.get("skipped") else ""
        out.append(f"  {d['run_name']:<24} {d['done']:>4}/{d['total']:<4} {state:<10}"
                   f"{skipped} {eta}")
    return out or ["  (실행 중이거나 끝난 러너 없음)"]


def queue() -> list[str]:
    f = RESULTS / "sprint_queue_state.json"
    qf = ROOT / "experiments" / "sprint_queue.tsv"
    if not f.exists():
        return ["  (작업 대기열 상태 파일 없음)"]
    d = json.loads(f.read_text(encoding="utf-8"))
    labels = []
    if qf.exists():
        for line in qf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = [x for x in line.split("\t") if x.strip()]
                if len(parts) >= 3:
                    labels.append(parts[0])
    done = d.get("done", [])
    cur = (d.get("current") or {}).get("label")
    out = []
    for lb in labels:
        mark = "완료" if lb in done else ("▶ 진행중" if lb == cur else "대기")
        hrs = next((h["hours"] for h in d.get("history", []) if h["label"] == lb), None)
        out.append(f"  [{mark:<6}] {lb}" + (f"  ({hrs:.2f}시간)" if hrs is not None else ""))
    return out or ["  (대기열이 비어 있음)"]


def watchdog() -> list[str]:
    f = RESULTS / "watchdog.log"
    if not f.exists():
        return ["  (자가 감시 기록 없음)"]
    lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    tail = lines[-3:]
    levels = [l.split("[")[1].split("]")[0] for l in lines if "[" in l]
    n_warn = sum(1 for x in levels if x == "경고")
    n_bad = sum(1 for x in levels if x == "이상")
    out = [f"  점검 {len(levels)}회 · 경고 {n_warn} · 이상 {n_bad}"]
    for l in tail:
        # 한 줄이 길어 앞부분만 자르면 정작 중요한 '문제 내용'이 잘린다.
        # 시각·등급과 마지막 칸(문제 요약)만 남긴다.
        head = l[:19]
        level = l.split("[", 1)[1].split("]", 1)[0] if "[" in l else "?"
        issue = l.rsplit("|", 1)[-1].strip()
        out.append(f"  {head} [{level}] {issue[:110]}")
    warn_lines = [l for l in lines if "[경고]" in l or "[이상]" in l]
    if warn_lines:
        out.append(f"  최근 경고/이상: {warn_lines[-1][:19]} — {warn_lines[-1].rsplit('|', 1)[-1].strip()[:110]}")
    return out


def conditions_done() -> str:
    n = sum(1 for _ in (RESULTS / "raw").rglob("*_meta.json"))
    rooms = sorted({p.parts[len((RESULTS / "raw").parts)] for p in (RESULTS / "raw").rglob("*_meta.json")})
    return f"  완료 조건 누계 {n}개 · 결과 방 {len(rooms)}개: {', '.join(rooms)}"


def paper_progress() -> list[str]:
    files = [("00_초록.md", "국문요약·핵심어"), ("01_서론.md", "Ⅰ. Introduction"),
             ("02_관련연구.md", "Ⅱ. Related Works"), ("03_방법.md", "Ⅲ. Proposed Method"),
             ("04_결과.md", "Ⅳ. Experimental Results (자동 생성)"),
             ("05_결론.md", "Ⅴ. Conclusions")]
    out, total = [], 0
    for fn, label in files:
        p = PAPER / fn
        if not p.exists():
            out.append(f"  [빠짐] {label}")
            continue
        n = len(re.sub(r"\s", "", p.read_text(encoding="utf-8")))
        total += n
        out.append(f"  [있음] {label:<34} {n:>6}자")
    out.append(f"  본문 합계 {total:,}자")
    if DOCX.exists():
        try:
            import mammoth
            with DOCX.open("rb") as fh:
                h = mammoth.convert_to_html(fh).value
            body = len(re.sub(r"\s", "", re.sub(r"<[^>]+>", " ", h)))
            out.append(f"  [있음] {DOCX.name} — 본문 {body:,}자 · 그림 {h.count('<img')}개 "
                       f"· 표 {h.count('<table')}개 "
                       f"(갱신 {time.strftime('%m-%d %H:%M', time.localtime(DOCX.stat().st_mtime))})")
        except Exception as e:
            out.append(f"  [있음] {DOCX.name} (내용 확인 실패: {e})")
    else:
        out.append(f"  [빠짐] {DOCX.name} — 아직 조립되지 않았다")
    todo = []
    for fn, _ in files:
        p = PAPER / fn
        if p.exists() and "본인 확인 필요" in p.read_text(encoding="utf-8"):
            todo.append(fn)
    if todo:
        out.append(f"  [본인 확인 필요] {', '.join(todo)}")
    return out


def main() -> None:
    W = 78
    print("=" * W)
    print("스프린트 상태 요약 — " + time.strftime("%Y-%m-%d %H:%M:%S KST"))
    print("  " + sprint_clock())
    print("=" * W)
    for title, fn in [("1. 실험 진행 (러너별)", experiments), ("2. 작업 대기열", queue),
                      ("3. 자가 감시", watchdog), ("4. 논문 진행률", paper_progress)]:
        print("\n" + title)
        for line in fn():
            print(line)
    print("\n5. 결과 누계")
    print(conditions_done())
    print("\n" + "=" * W)


if __name__ == "__main__":
    main()
