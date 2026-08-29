"""Microsoft Edge로 파일을 연다 — **열렸는지 확인까지 한다.**

2026-08-29: `cmd /c start msedge` 가 조용히 아무것도 안 하는 경우가 있었는데
(실행 환경이 GUI 실행을 막고 있었다), 코드는 "Edge로 열었다"고 출력만 했다.
사용자는 아무것도 못 봤고 나는 열렸다고 보고했다.

이 저장소에서 두 번 데인 실수와 같은 뿌리다 — **한 일과 됐다고 말하는 것을 구분한다.**
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def _running() -> int:
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq msedge.exe", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return -1                      # 셀 수 없으면 '모른다' — 0과 구분한다
    return sum(1 for l in out.splitlines() if "msedge" in l.lower())


def open_in_edge(path: Path | str) -> bool:
    """연다. 실제로 떴으면 True. 못 열었으면 그 사실과 대안을 알린다."""
    p = Path(path).resolve()
    if not p.exists():
        print(f"  [못 엶] 파일이 없다: {p}")
        return False
    before = _running()
    exe = next((e for e in EDGE_PATHS if e.exists()), None)
    try:
        if exe:
            subprocess.Popen([str(exe), str(p)])
        else:
            subprocess.run(["cmd", "/c", "start", "msedge", str(p)], check=False)
    except Exception as e:
        print(f"  [못 엶] {e}")
        print(f"        └ 직접 열 것: {p}")
        return False
    time.sleep(2.5)
    after = _running()
    # before가 -1이면 셀 수 없었던 것이므로 판정하지 않는다
    if before >= 0 and after <= before and after == 0:
        print("  [못 엶] Edge를 띄우려 했으나 프로세스가 뜨지 않았다")
        print(f"        └ 직접 열 것: {p}")
        return False
    print(f"  Microsoft Edge로 열었다 (프로세스 {after}개 확인)")
    return True
