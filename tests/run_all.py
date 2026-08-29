"""테스트를 한 번에 돌린다 (pytest 없이).

이 환경에는 pytest가 없다. 테스트 파일마다 `main()`이 있으므로 그것을 부른다.
실행: python tests/run_all.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    files = sorted(p for p in (ROOT / "tests").glob("test_*.py"))
    bad = 0
    for f in files:
        print("=" * 62)
        print(f.name)
        print("=" * 62)
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.main() if hasattr(mod, "main") else 0
        bad += int(bool(rc))
        print()
    print("=" * 62)
    print(f"테스트 파일 {len(files)}개 중 {len(files) - bad}개 통과"
          if bad else f"테스트 파일 {len(files)}개 모두 통과")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
