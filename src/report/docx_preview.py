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
        font-family:"바탕","Batang",serif; table-layout: auto; }
th, td { border: 1px solid #333; padding: 2pt 3pt; line-height: 1.15;
         overflow-wrap: break-word; text-align:center; }
td:first-child, th:first-child { white-space: nowrap; }
th { background:#ededed; font-family:"맑은 고딕","Malgun Gothic",sans-serif; }
td:first-child { text-align:left; }
sup { font-size: 7.5pt; }
.banner { position:sticky; top:0; background:#222; color:#fff; padding:8px 14px;
          font-family:"맑은 고딕",sans-serif; font-size:12px; z-index:9; }
.overflow-warn { outline: 3px solid #d62728; }
/* 쪽 경계 — Word의 실제 조판은 아니지만 '그림·표가 페이지를 걸치는가'를 보는 데 쓴다 */
.pagebreak { position:absolute; left:0; right:0; border-top:2px dashed #c33; z-index:5; }
.pagebreak span { position:absolute; right:4px; top:2px; font:11px "맑은 고딕",sans-serif;
                  color:#c33; background:#fff; padding:0 4px; }
.page { position:relative; }
.straddle { outline: 3px solid #e08a00; outline-offset: 2px; }
"""

JS = """
// 표가 인쇄 폭을 넘치면 빨간 테두리로 표시한다 (Word에서 잘릴 위험 신호)
document.querySelectorAll('table').forEach(function (t) {
  if (t.scrollWidth > t.clientWidth + 2) { t.classList.add('overflow-warn'); }
});

// Word의 조판을 흉내 낸다: 쪽 경계를 걸치는 그림·표는 다음 쪽 맨 처음으로 밀어낸다
// (.docx에 넣은 cantSplit·keepNext가 Word에서 하는 일이 이것이다).
// 밀어내도 여전히 걸치는 것 = 한 쪽보다 큰 덩어리 = Word도 못 고치는 진짜 문제.
(function () {
  var page = document.querySelector('.page');
  var CM = 37.7952755906;                 // 1cm = 이만큼의 CSS 픽셀
  var H = 24.7 * CM;                      // 한 쪽에 들어가는 본문 높이 (29.7 − 2.5×2)
  var top0 = 2.5 * CM;

  function boundsNow() {
    var n = Math.ceil((page.scrollHeight - top0) / H), b = [];
    for (var i = 1; i <= n; i++) b.push(top0 + i * H);
    return b;
  }
  // 그림은 <p><img></p> + 캡션 2줄, 표는 캡션 2줄 + <table> 이 한 덩어리다.
  function blockOf(el) {
    if (el.tagName === 'IMG') {
      var a = el.closest('p') || el, list = [a];
      for (var i = 0; i < 2 && a.nextElementSibling; i++) { a = a.nextElementSibling; list.push(a); }
      return list;
    }
    var b = [el], q = el.previousElementSibling;
    for (var j = 0; j < 2 && q; j++) { b.unshift(q); q = q.previousElementSibling; }
    return b;
  }
  var pushed = 0, stuck = [];
  // docx_build.KEEP_WHOLE_MAX_CM 과 같은 기준이다. 여기서는 어림하지 않고 실제로 잰다.
  // 반 쪽을 넘는 표는 .docx에서도 통째로 붙들지 않으므로, 여기서도 밀지 않는다.
  var KEEP_WHOLE_MAX_CM = 13.5;
  document.querySelectorAll('table, img').forEach(function (el) {
    if (el.tagName === 'TABLE' &&
        el.getBoundingClientRect().height > KEEP_WHOLE_MAX_CM * CM) return;
    var blk = blockOf(el);
    for (var pass = 0; pass < 2; pass++) {
      var pr = page.getBoundingClientRect();
      var t = blk[0].getBoundingClientRect().top - pr.top;
      var bt = blk[blk.length - 1].getBoundingClientRect().bottom - pr.top;
      var bd = boundsNow(), hit = null;
      for (var k = 0; k < bd.length; k++) if (t < bd[k] && bt > bd[k]) { hit = bd[k]; break; }
      if (hit === null) return;
      if (bt - t > H) {                                   // 한 쪽보다 큰 덩어리 — 밀어내도 소용없다
        el.classList.add('straddle');
        stuck.push(el.tagName === 'TABLE'
          ? '표(' + el.rows.length + '행, ' + Math.round((bt - t) / H * 100) + '%쪽)'
          : '그림(' + Math.round((bt - t) / H * 100) + '%쪽)');
        return;
      }
      blk[0].style.marginTop = (parseFloat(getComputedStyle(blk[0]).marginTop) + (hit - t) + 2) + 'px';
      pushed++;
    }
  });
  var nb = boundsNow();
  nb.forEach(function (y, i) {
    var d = document.createElement('div');
    d.className = 'pagebreak'; d.style.top = y + 'px';
    d.innerHTML = '<span>— ' + (i + 2) + ' —</span>';
    page.appendChild(d);
  });
  document.querySelector('.banner').innerHTML +=
    ' · 총 ' + (nb.length + 1) + '쪽 · 다음 쪽으로 밀린 그림·표 ' + pushed + '개'
    + ' · 밀어도 안 되는 것 ' + stuck.length + '개' + (stuck.length ? ' (주황): ' + stuck.join(', ') : '');
})();
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
        from src.report.open_edge import open_in_edge
        open_in_edge(p)


if __name__ == "__main__":
    main()
