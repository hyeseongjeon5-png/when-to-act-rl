"""조립된 .docx가 A4에서 어떻게 보일지 확인하기 위한 미리보기 HTML.

왜 필요한가: 이 PC에는 Word도 LibreOffice도 없어 .docx를 실제로 렌더링해 볼 수 없다.
그래서 문서의 내용을 그대로 꺼내(mammoth) 학교 양식과 같은 판형(A4 · 여백 2.5cm ·
더블 스페이싱 · 바탕 10pt)의 HTML로 다시 그려 눈으로 확인한다.

**이것은 Word의 실제 조판이 아니다.** 표가 넘치는지, 그림이 너무 큰지, 캡션이 붙었는지 같은
'큰 사고'를 잡는 용도다. 최종 제출 전에는 사람이 Word로 한 번 열어 확인해야 한다.

실행: python -m src.report.docx_preview [--open]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCX = ROOT / "졸업논문_초안v1.docx"
OUT = ROOT / "results" / "reports" / "논문_미리보기.html"

CSS = """
:root { --ink:#111; --muted:#666; }
* { box-sizing: border-box; }
body { margin:0; background:#8a8a8a; font-family:"바탕","Batang",serif; color:var(--ink); }
.page {
  width: 21cm; min-height: 29.7cm; padding: 2.5cm;
  margin: 1.2cm auto; background:#fff; box-shadow: 0 2px 10px rgba(0,0,0,.35);
}
p { font-size: 10pt; line-height: 2.0; margin: 0 0 0 0; text-align: justify; text-indent: 10pt; }
h1,h2,h3 { font-family:"맑은 고딕","Malgun Gothic",sans-serif; line-height:1.3; margin:14pt 0 6pt; }
h1 { font-size: 13pt; } h2 { font-size: 11pt; } h3 { font-size: 10pt; }
img { max-width: 100%; display:block; margin: 8pt auto 2pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 9pt;
        font-family:"바탕","Batang",serif; table-layout: fixed; }
th, td { border: 1px solid #333; padding: 2pt 3pt; line-height: 1.15;
         word-break: break-word; text-align:center; }
th { background:#ededed; font-family:"맑은 고딕","Malgun Gothic",sans-serif; }
td:first-child { text-align:left; }
sup { font-size: 7.5pt; }
.banner { position:sticky; top:0; background:#222; color:#fff; padding:8px 14px;
          font-family:"맑은 고딕",sans-serif; font-size:12px; z-index:9; }
.overflow-warn { outline: 3px solid #d62728; }
"""

JS = """
// 표가 인쇄 폭을 넘치면 빨간 테두리로 표시한다 (Word에서 잘릴 위험 신호)
document.querySelectorAll('table').forEach(function (t) {
  if (t.scrollWidth > t.clientWidth + 2) { t.classList.add('overflow-warn'); }
});
"""


def build() -> Path:
    import mammoth
    if not DOCX.exists():
        raise SystemExit(f"{DOCX.name} 이 없다 — 먼저 make_thesis_docx 를 돌릴 것")
    with DOCX.open("rb") as f:
        res = mammoth.convert_to_html(f, convert_image=mammoth.images.data_uri)
    n_img = res.value.count("<img")
    n_tab = res.value.count("<table")
    banner = (f'<div class="banner">미리보기 — 실제 Word 조판이 아님 · 그림 {n_img}개 · 표 {n_tab}개 · '
              f'빨간 테두리 표 = 폭 넘침 위험 · 최종 제출 전 Word로 확인할 것</div>')
    html = ("<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
            "<title>졸업논문 초안 미리보기</title><style>" + CSS + "</style></head><body>"
            + banner + "<div class='page'>" + res.value + "</div>"
            + "<script>" + JS + "</script></body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"미리보기 저장: {OUT.relative_to(ROOT)} (그림 {n_img} · 표 {n_tab} · 변환 경고 {len(res.messages)}건)")
    for m in res.messages[:5]:
        print("  ", m)
    return OUT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="Microsoft Edge로 연다")
    a = ap.parse_args()
    p = build()
    if a.open:
        subprocess.run(["cmd", "/c", "start", "msedge", str(p)], check=False)
        print("  Edge로 열었다")


if __name__ == "__main__":
    main()
