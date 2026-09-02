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
# 원고 안에서 만들어지는 표(TABCAP)에 이름표를 달아 본문이 가리킬 수 있게 한다.
TABTAG_MARK = re.compile(r"<!--\s*TABTAG:([A-Za-z0-9_]+)\s*-->")
# 워드 수식 객체(OMML) 자리. 글자로 쓰면 수식이 아니라 문자열이 된다.
EQ_MARK = re.compile(r"<!--\s*EQ:([A-Za-z0-9_]+)\s*-->")
COMMENT = re.compile(r"<!--.*?-->", re.S)


FIGREF_MARK = re.compile(r"<!--\s*FIGREF:([A-Za-z0-9_]+)(?:\|(\S+?))?\s*-->")
TABREF_MARK = re.compile(r"<!--\s*TABREF:([A-Za-z0-9_]+)(?:\|(\S+?))?\s*-->")

# 한국어 조사는 앞 글자의 받침 유무로 갈린다. 번호가 바뀌면 조사도 바뀌어야 한다.
#   1(일)·3(삼)·6(육)·7(칠)·8(팔)·10(십) → 받침 있음 → 은/이/을/과/으로
#   2(이)·4(사)·5(오)·9(구)             → 받침 없음 → 는/가/를/와/로
_HAS_FINAL = {0: True, 1: True, 3: True, 6: True, 7: True, 8: True}   # 0(영)·10(십)도 받침
_JOSA = {"은": ("은", "는"), "는": ("은", "는"), "이": ("이", "가"), "가": ("이", "가"),
         "을": ("을", "를"), "를": ("을", "를"), "과": ("과", "와"), "와": ("과", "와"),
         "으로": ("으로", "로"), "로": ("으로", "로")}


def _josa(n: int, want: str) -> str:
    """번호 n 뒤에 붙일 조사를 고른다. 10 이상은 마지막 자리로 판단한다(11=십일 …)."""
    pair = _JOSA.get(want)
    if not pair:
        return want
    last = n % 10 if n >= 10 and n % 10 else n
    if n >= 10 and n % 10 == 0:
        last = 0                      # 10·20 … = '십' → 받침 있음
    return pair[0] if _HAS_FINAL.get(last, False) else pair[1]


def scan_numbers(md_texts: list[str]) -> dict:
    """본문을 미리 훑어 '어떤 그림·표가 몇 번이 될지'를 알아낸다.

    왜 필요한가: 번호는 등장 순서로 매겨지는데, 본문이 그림을 **가리키는 문장**은
    그림보다 먼저 나온다. 한 번만 훑어서는 앞을 내다볼 수 없어 "그림 5는 …" 같은
    참조를 손으로 적게 되고, 그림을 하나 끼워 넣는 순간 조용히 어긋난다.
    (2026-08-29: 실제로 인과 실험 그림을 넣으려다 이 문제를 발견했다.)

    그래서 먼저 번호만 세고, 그 표를 들고 본문을 렌더한다.
    """
    st = {"fig": 0, "tab": 0}
    out = {}
    pending_tag = None
    for md in md_texts:
        for line in md.splitlines():
            t = line.strip()
            m = FIG_MARK.search(t)
            if m:
                st["fig"] += 1
                out[m.group(1)] = st["fig"]
                continue
            m = TAB_MARK.search(t)
            if m:
                st["tab"] += 1
                out[m.group(1)] = st["tab"]
                continue
            m = TABTAG_MARK.search(t)
            if m:
                pending_tag = m.group(1)
                continue
            if TABCAP_MARK.search(t):
                st["tab"] += 1
                if pending_tag:
                    out[pending_tag] = st["tab"]
                    pending_tag = None
    return out


def resolve_refs(md: str, numbers: dict) -> str:
    """<!--FIGREF:tag--> / <!--TABREF:tag--> 를 실제 번호로 바꾼다."""
    def _one(kind: str, m):
        n = numbers.get(m.group(1))
        if not n:
            return f"{kind} ?"
        want = m.group(2)
        return f"{kind} {n}" + (_josa(n, want) if want else "")

    def fig(m):
        return _one("그림", m)

    def tab(m):
        return _one("표", m)

    out = TABREF_MARK.sub(tab, FIGREF_MARK.sub(fig, md))
    # 형식이 틀린 참조 마커는 아래 COMMENT 규칙에 걸려 **조용히 지워진다.**
    # 사라진 자리는 "보면() 답이 갈린다"처럼 괄호만 남아 눈에 잘 안 띈다. 그래서 알린다.
    for bad in re.finditer(r"<!--\s*(FIGREF|TABREF):[^>]*-->", out):
        print("  [경고] 알아보지 못한 참조 마커: " + bad.group(0)
              + "  → <!--FIGREF:태그--> 또는 <!--FIGREF:태그|조사--> 형식이어야 한다")
    return out


def _captions() -> dict:
    p = FIGDIR / "captions.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def new_counter() -> dict:
    """그림·표 번호를 등장 순서로 매기기 위한 상태. 장을 여러 번 render해도 이어진다.

    왜 자동인가: 번호를 원고에 손으로 적으면 장을 하나 끼워 넣는 순간 전부 어긋난다.
    학교 양식은 '본문에서 언급된 쪽에' 싣도록 하므로 번호는 등장 순서를 따라야 한다.
    """
    return {"fig": 0, "tab": 0}


def _numbered(kind: str, st: dict, ko: str, en: str) -> tuple[str, str]:
    st[kind] += 1
    n = st[kind]
    if kind == "fig":
        return f"그림 {n}. {ko}", f"Fig. {n}. {en}"
    return f"표 {n}. {ko}", f"Table {n}. {en}"


def render(doc, md_text: str, auto_tables: dict | None = None,
           fig_width: dict | None = None, counter: dict | None = None,
           numbers: dict | None = None) -> None:
    """마크다운 한 장을 doc에 이어 붙인다. numbers를 주면 FIGREF/TABREF를 번호로 바꾼다."""
    if numbers:
        md_text = resolve_refs(md_text, numbers)
    # 각주 자리 표시를 '살아남는 글자'로 바꾼다. <!--FN1--> 그대로 두면 주석 제거에 지워져
    # 조립이 끝난 .docx에서 자리를 찾을 수 없다 (2026-09-02에 실제로 그랬다).
    md_text = md_text.replace("<!--FN1-->", "")
    caps = _captions()
    auto_tables = auto_tables or {}
    fig_width = fig_width or {}
    st = counter if counter is not None else new_counter()
    lines = md_text.splitlines()
    i = 0
    pending_cap: tuple[str, str] | None = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- 표 제목 예약 ----
        # 마커가 여러 줄에 걸쳐 있을 수 있다(긴 제목을 줄바꿈해 쓰는 경우).
        # 한 줄만 보면 조용히 놓쳐서 표에 제목이 안 붙는다 — 닫는 --> 까지 이어 붙여 본다.
        if stripped.startswith("<!--") and "TABCAP:" in stripped:
            block, j = stripped, i
            while "-->" not in block and j + 1 < len(lines):
                j += 1
                block += " " + lines[j].strip()
            m = TABCAP_MARK.search(block)
            if m:
                pending_cap = (m.group(1), m.group(2))
                i = j + 1
                continue

        # ---- 그림 삽입 ----
        m = FIG_MARK.search(stripped)
        if m:
            key = m.group(1)
            c = caps.get(key)
            if c and (FIGDIR / c["file"]).exists():
                ko, en = _numbered("fig", st, c["ko"], c["en"])
                B.add_figure(doc, FIGDIR / c["file"], ko, en,
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
                ko, en = _numbered("tab", st, t["ko"], t["en"])
                B.add_table(doc, t["header"], t["rows"], ko, en,
                            note=t.get("note", ""), widths=t.get("widths"))
            else:
                print(f"  [경고] 표 {key}를 찾지 못해 건너뜀")
            i += 1
            continue

        # ---- 수식 (워드 수식 객체) ----
        # 반드시 '주석 건너뛰기'보다 앞에 와야 한다. <!--EQ:...--> 도 주석 모양이라
        # 뒤에 두면 그냥 건너뛰어져 수식이 통째로 사라진다 (2026-09-02에 실제로 그랬다).
        m = EQ_MARK.search(stripped)
        if m:
            from src.report import omml
            omml.insert(doc, m.group(1))
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
                if pending_cap:
                    ko, en = _numbered("tab", st, pending_cap[0], pending_cap[1])
                else:
                    ko, en = "", ""
                    print("  [경고] 제목 없는 표가 있다 — 앞에 <!--TABCAP: 국문 | 영문 --> 를 넣을 것")
                B.add_table(doc, header, rows, ko, en)
                pending_cap = None
            continue

        def _take_item(start: int) -> tuple[str, int]:
            """목록 한 항목을 이어지는 줄까지 모아 온다.

            마크다운에서 항목의 둘째 줄부터는 들여쓴다. 그 줄을 따로 문단으로 만들면
            왼쪽 정렬이 0이 되어 항목 아래로 들어가지 않고 새 문단처럼 보인다
            (2026-08-29에 실제로 서론 기여 목록이 그렇게 쪼개져 있었다).
            """
            parts = [lines[start].strip()]
            j = start + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip():
                    break
                if not nxt[:1].isspace():          # 들여쓰기가 없으면 새 덩어리다
                    break
                t = nxt.strip()
                if t.startswith(("- ", "* ", "#", "|", ">", "<!--")) or re.match(r"^\d+\.\s", t):
                    break
                parts.append(t)
                j += 1
            return " ".join(parts), j

        # ---- 글머리표 ----
        if stripped.startswith("- ") or stripped.startswith("* "):
            indent = len(line) - len(line.lstrip())
            text, i = _take_item(i)
            B.add_bullet(doc, COMMENT.sub("", text[2:]).strip(), level=1 if indent >= 2 else 0)
            continue

        # ---- 번호 목록: 번호가 이미 있으므로 점(·)을 덧붙이지 않는다 ----
        if re.match(r"^\d+\.\s", stripped):
            text, i = _take_item(i)
            B.add_numbered(doc, COMMENT.sub("", text).strip())
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
