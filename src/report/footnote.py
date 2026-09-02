"""워드 각주(footnote)를 .docx에 넣는다.

왜 직접 만드나: python-docx에는 각주 API가 없다. 그래서 조립이 끝난 .docx 꾸러미를 열어
각주 부품을 직접 넣는다. 넣어야 할 것이 네 가지다.

  1. word/footnotes.xml            — 각주 본문 (구분선 0·1번은 워드가 요구하는 기본 항목)
  2. word/_rels/document.xml.rels  — 문서에서 footnotes.xml로 가는 관계
  3. [Content_Types].xml           — footnotes.xml의 형식 선언
  4. 본문의 <!--FN1--> 자리        — 각주 번호를 다는 참조 run

실행(자가 점검): python -m src.report.footnote
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MARK = ""   # md_to_docx가 <!--FN1--> 를 이 글자로 바꿔 둔다

FOOTNOTES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes xmlns:w="{ns}">
  <w:footnote w:type="separator" w:id="-1">
    <w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>
      <w:r><w:separator/></w:r></w:p>
  </w:footnote>
  <w:footnote w:type="continuationSeparator" w:id="0">
    <w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>
      <w:r><w:continuationSeparator/></w:r></w:p>
  </w:footnote>
  <w:footnote w:id="1">
    <w:p>
      <w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>
      <w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:footnoteRef/></w:r>
      <w:r><w:rPr><w:rFonts w:ascii="바탕" w:eastAsia="바탕" w:hAnsi="바탕"/><w:sz w:val="18"/></w:rPr>
        <w:t xml:space="preserve"> {text}</w:t></w:r>
    </w:p>
  </w:footnote>
</w:footnotes>""".replace("{ns}", NS_W)

REF_RUN = (
    '<w:r xmlns:w="{ns}">'
    '<w:rPr><w:vertAlign w:val="superscript"/></w:rPr>'
    '<w:footnoteReference w:id="1"/>'
    "</w:r>"
).replace("{ns}", NS_W)


def add(docx_path: Path, text: str) -> bool:
    """<!--FN1--> 자리에 각주를 단다. 자리가 없으면 아무것도 하지 않고 False."""
    docx_path = Path(docx_path)
    src = zipfile.ZipFile(docx_path)
    doc = src.read("word/document.xml").decode("utf-8")
    if MARK not in doc:
        src.close()
        print("  [건너뜀] 본문에 각주 자리 표시가 없다")
        return False

    # 1) 본문의 표시를 각주 참조 run으로 바꾼다.
    #    표시는 <w:t> 안에 글자로 들어와 있으므로, 그 run을 통째로 참조 run으로 교체한다.
    pat = re.compile(r"<w:r\b[^>]*>(?:(?!</w:r>).)*?<!--" + MARK + r"-->(?:(?!</w:r>).)*?</w:r>", re.S)
    new_doc, n = pat.subn(REF_RUN, doc)
    if n == 0:                       # run 경계를 못 잡으면 글자만 바꾼다
        new_doc = doc.replace(MARK, "")
        i = new_doc.find("</w:p>", new_doc.find("전체 코드"))
        new_doc = new_doc[:i] + REF_RUN + new_doc[i:] if i > 0 else new_doc

    rels = src.read("word/_rels/document.xml.rels").decode("utf-8")
    if "footnotes.xml" not in rels:
        rels = rels.replace(
            "</Relationships>",
            '<Relationship Id="rIdFootnotes" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" '
            'Target="footnotes.xml"/></Relationships>')
    ct = src.read("[Content_Types].xml").decode("utf-8")
    if "footnotes+xml" not in ct:
        ct = ct.replace(
            "</Types>",
            '<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.wordprocessingml.footnotes+xml"/></Types>')

    tmp = docx_path.with_suffix(".tmp.docx")
    out = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
    for it in src.infolist():
        if it.filename == "word/document.xml":
            out.writestr(it, new_doc)
        elif it.filename == "word/_rels/document.xml.rels":
            out.writestr(it, rels)
        elif it.filename == "[Content_Types].xml":
            out.writestr(it, ct)
        elif it.filename == "word/footnotes.xml":
            continue
        else:
            out.writestr(it, src.read(it.filename))
    out.writestr("word/footnotes.xml", FOOTNOTES_XML.replace("{text}", text))
    out.close()
    src.close()
    shutil.move(str(tmp), str(docx_path))
    print(f"  각주 1개 삽입: {text[:48]}…")
    return True


def main() -> int:
    p = ROOT / "졸업논문_초안v3.docx"
    ok = add(p, "전체 코드·설정·실험일지·집계 결과는 github.com/hyeseongjeon5-png/when-to-act-rl "
                "에 공개되어 있다.")
    z = zipfile.ZipFile(p)
    has = "word/footnotes.xml" in z.namelist()
    ref = z.read("word/document.xml").decode("utf-8").count("footnoteReference")
    print(f"footnotes.xml {'있음' if has else '없음'} · 본문의 각주 참조 {ref}개")
    return 0 if (ok and has and ref == 1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
