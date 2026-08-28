"""원고 마크다운(paper/*.md)을 학교 양식 .docx 문단으로 옮기는 변환기.

**원고의 진짜 원본은 마크다운이다.** .docx는 거기서 조립된 결과물이며, 손으로 고치지 않는다.
손으로 고치기 시작하면 마크다운과 .docx가 갈라져 어느 쪽이 맞는지 알 수 없게 된다.

지원하는 문법 (내가 쓰는 것만 — 범용 변환기가 아니다):
  # 제목            → 장 제목 (Ⅰ. Introduction …)
  ## 1. 제목        → 절 제목
  ### 제목          → 소절 제목
  - 항목            → 글머리표 (들여쓰기 2칸이면 하위 항목)
  | 표 | 형식 |     → 표 (바로 앞 <!--TABCAP: 국문 | 영문 --> 를 제목으로 쓴다)
  **굵게** `코드` [1] → 인라인
  <!--FIG:fig1-->   → 그림 삽입 (캡션은 captions.json에서)
  <!--TABLE:tab1--> → 자동 생성 표 삽입 (집계 파일에서 만들어진 표)
  <!-- 그 밖의 주석 --> → 무시 (출처 표시용)
  > 인용            → 무시 (초안 메모용)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.report import docx_build as B

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "results" / "figures" / "paper"

FIG_MARK = re.compile(r"<!--\s*FIG:([A-Za-z0-9_]+)\s*-->")
TAB_MARK = re.compile(r"<!--\s*TABLE:([A-Za-z0-9_]+)\s*-->")
TABCAP_MARK = re.compile(r"<!--\s*TABCAP:\s*(.+?)\s*\|\s*(.+?)\s*-->")
COMMENT = re.compile(r"<!--.*?-->", re.S)


def _captions() -> dict:
    p = FIGDIR / "captions.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(doc, md_text: str, auto_tables: dict | None = None,
           fig_width: dict | None = None) -> None:
    """마크다운 한 장을 doc에 이어 붙인다."""
    caps = _captions()
    auto_tables = auto_tables or {}
    fig_width = fig_width or {}
    lines = md_text.splitlines()
    i = 0
    pending_cap: tuple[str, str] | None = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- 표 제목 예약 ----
        m = TABCAP_MARK.search(stripped)
        if m:
            pending_cap = (m.group(1), m.group(2))
            i += 1
            continue

        # ---- 그림 삽입 ----
        m = FIG_MARK.search(stripped)
        if m:
            key = m.group(1)
            c = caps.get(key)
            if c and (FIGDIR / c["file"]).exists():
                B.add_figure(doc, FIGDIR / c["file"], c["ko"], c["en"],
                             note=c.get("note", ""), width_cm=fig_width.get(key, 15.0))
            else:
                print(f"  [경고] 그림 {key}를 찾지 못해 건너뜀 (captions.json 확인)")
            i += 1
            continue

        # ---- 자동 생성 표 삽입 ----
        m = TAB_MARK.search(stripped)
        if m:
            key = m.group(1)
            t = auto_tables.get(key)
            if t:
                B.add_table(doc, t["header"], t["rows"], t["ko"], t["en"],
                            note=t.get("note", ""), widths=t.get("widths"))
            else:
                print(f"  [경고] 표 {key}를 찾지 못해 건너뜀")
            i += 1
            continue

        # ---- 그 밖의 주석·구분선·인용 ----
        if not stripped or stripped.startswith(">") or stripped.startswith("---"):
            i += 1
            continue
        if stripped.startswith("<!--"):
            i += 1
            continue

        # ---- 제목 ----
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            if level == 1:
                B.add_chapter_heading(doc, title)
            elif level == 2:
                B.add_section_heading(doc, title)
            else:
                B.add_sub_heading(doc, title)
            i += 1
            continue

        # ---- 표 ----
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            if len(block) >= 2:
                header = _split_row(block[0])
                rows = [_split_row(b) for b in block[2:]]   # block[1]은 구분선
                ko, en = pending_cap or ("", "")
                B.add_table(doc, header, rows, ko, en)
                pending_cap = None
            continue

        # ---- 글머리표 ----
        if stripped.startswith("- ") or stripped.startswith("* "):
            indent = len(line) - len(line.lstrip())
            B.add_bullet(doc, COMMENT.sub("", stripped[2:]).strip(), level=1 if indent >= 2 else 0)
            i += 1
            continue

        # ---- 번호 목록도 글머리표로 ----
        if re.match(r"^\d+\.\s", stripped):
            B.add_bullet(doc, COMMENT.sub("", stripped).strip())
            i += 1
            continue

        # ---- 일반 문단: 빈 줄이 나올 때까지 이어 붙인다 ----
        buf = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(
                ("#", "|", "- ", "* ", ">", "<!--", "---")) and not re.match(r"^\d+\.\s", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        text = COMMENT.sub("", " ".join(buf)).strip()
        if text:
            B.para(doc, text)


def render_file(doc, md_path: Path, **kw) -> None:
    print(f"  본문 삽입: {md_path.name}")
    render(doc, md_path.read_text(encoding="utf-8"), **kw)
