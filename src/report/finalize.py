"""스프린트 마감 — 집계부터 논문·보고서까지 한 번에 다시 만든다.

마감 3~4시간 구간에서 이 하나만 돌리면 모든 산출물이 최신 결과로 갱신된다.
순서가 중요하다: 집계가 먼저여야 그림·표·원고가 같은 숫자를 본다.

  1. 집계        results/aggregate/*.csv, *_lambda_star.json   (IQM + 95% 계층 부트스트랩 CI)
  2. 인과 비교   results/aggregate/causal_warmup.json          (비용을 켜는 시점만 바꾼 대조)
  3. 작업용 그림  results/figures/*.png
  4. 논문용 그림  results/figures/paper/*.png  (300dpi, 캡션 국문·영문)
  5. Ⅳ장 원고    paper/04_결과.md              (숫자는 집계 파일에서만)
  6. 논문 조립    졸업논문_초안v3.docx
  7. 미리보기     results/reports/논문_미리보기.html
  8. 원고 점검    낡은 표현·남은 [본인 확인 필요] 표시 목록
  9. README 갱신  README.md의 '결과' 절 (표시 구간 안쪽만)
 10. HTML 보고서  results/reports/{날짜}_본실험보고서.html

실행: python -m src.report.finalize [--reps 10000] [--open]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable


def run(title: str, args: list[str], required: bool = True) -> bool:
    print("\n" + "=" * 74)
    print(f"[{title}] " + " ".join(args[2:]))
    print("=" * 74, flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + args, cwd=str(ROOT))
    ok = r.returncode == 0
    print(f"  → {'완료' if ok else '실패(코드 ' + str(r.returncode) + ')'} · {time.time() - t0:.0f}초", flush=True)
    if not ok and required:
        print("  이 단계가 실패하면 뒤 단계의 숫자가 낡는다 — 원인을 보고 다시 돌릴 것")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000,
                    help="부트스트랩 반복 수 (마감에는 10000 권장, 중간 점검은 2000)")
    ap.add_argument("--open", action="store_true", help="HTML 보고서를 Microsoft Edge로 연다")
    a = ap.parse_args()

    t0 = time.time()
    steps = [
        ("1/10 집계", ["-m", "src.analysis.aggregate", "--env", "all", "--reps", str(a.reps)]),
        ("2/10 인과 비교", ["-m", "src.analysis.causal_compare"]),
        ("3/10 작업용 그림", ["-m", "src.analysis.plots", "--env", "all"]),
        ("4/10 논문용 그림", ["-m", "src.report.make_paper_figures"]),
        ("5/10 Ⅳ장 원고", ["-m", "src.report.make_results_chapter"]),
        ("6/10 논문 조립", ["-m", "src.report.make_thesis_docx"]),
        ("7/10 미리보기", ["-m", "src.report.docx_preview"]),
        ("8/10 원고 점검", ["-m", "src.report.check_manuscript"]),
        ("9/10 README 갱신", ["-m", "src.report.update_readme"]),
    ]
    results = {t: run(t, args) for t, args in steps}
    rep = ["-m", "src.report.make_experiment_report"] + (["--open"] if a.open else [])
    results["10/10 HTML 보고서"] = run("10/10 HTML 보고서", rep)

    print("\n" + "=" * 74)
    print(f"마감 처리 요약 · 총 {(time.time() - t0) / 60:.1f}분")
    for k, v in results.items():
        print(f"  {'✔' if v else '✖'} {k}")
    print("=" * 74)
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
