"""감시견 자가시험 — 검사가 '실제로 잡는지'를 일부러 고장 내어 확인한다.

왜 필요한가 (2026-08-29 사고 4):
    자동 점검을 하나 새로 만들고 [맞음]을 확인했는데, 일부러 어긋나게 만들어도
    [맞음]이 나왔다. 정규식이 망가져 아무것도 못 찾고 있었던 것이다.
    **아무것도 못 찾는 것과 문제가 없는 것은 다르다.**
    같은 뿌리의 사고가 이미 한 번 더 있었다(사고 3 — 감시자가 실험을 세 벌 돌림).

    감시견은 72시간을 무인으로 도는 실험을 지키는 장치다. 그 검사가 헛돌면
    "이상 없음"이라는 로그만 쌓이고 실험은 조용히 망가진다.

무엇을 하나:
    임시 폴더에 가짜 로그·결과 파일을 만들고 감시견의 경로를 그쪽으로 돌린 뒤,
    ① 고장을 넣으면 잡는가 ② 정상이면 조용한가 를 둘 다 확인한다.
    **실행 중인 실험 폴더는 건드리지 않는다.**

실행: python -m src.monitor.watchdog_selftest
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.monitor import watchdog as W

NL = chr(10)
PASS, FAIL = "통과", "실패 ← 헛도는 검사"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, PASS if ok else FAIL, detail))


def with_results(fn):
    """감시견이 보는 results/ 폴더를 임시 폴더로 갈아 끼우고 fn을 돌린다."""
    old_results, old_root = W.RESULTS, W.ROOT
    with tempfile.TemporaryDirectory(prefix="wd_selftest_") as d:
        base = Path(d)
        (base / "logs" / "train").mkdir(parents=True)
        (base / "raw").mkdir(parents=True)
        W.RESULTS, W.ROOT = base, base
        try:
            return fn(base)
        finally:
            W.RESULTS, W.ROOT = old_results, old_root


def write_meta(base: Path, name: str, **kw) -> None:
    p = base / "raw" / "env" / "agent" / "lam0.0"
    p.mkdir(parents=True, exist_ok=True)
    (p / (name + "_meta.json")).write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")


def test_traceback(base: Path) -> None:
    f = base / "logs" / "train" / "a.log"
    f.write_text("정상 줄" + NL + "Traceback (most recent call last):" + NL + "ValueError", encoding="utf-8")
    r = W.check_logs()
    record("③ 로그의 예외(Traceback)를 잡는가", bool(r["traceback_logs"]), str(r["traceback_logs"]))
    f.write_text("정상 줄만 있다" + NL, encoding="utf-8")
    r = W.check_logs()
    record("③ 정상 로그에는 조용한가", not r["traceback_logs"], "")


def test_nan_log(base: Path) -> None:
    f = base / "logs" / "train" / "b.log"
    f.write_text("step 100 loss nan" + NL, encoding="utf-8")
    r = W.check_logs()
    record("③ 로그의 NaN 손실을 잡는가", bool(r["nan_logs"]), str(r["nan_logs"]))


def test_nan_result(base: Path) -> None:
    write_meta(base, "seed0", agent="dqn", lam=0.0, env_id="env",
               final={"cost_return": float("nan"), "solved_rate": 0.5, "n_actions_mean": 10})
    r = W.check_curves()
    record("⑤ 결과 파일의 NaN을 잡는가", bool(r["nan_results"]), str(r["nan_results"]))


def test_floor_lam0(base: Path) -> None:
    write_meta(base, "seed1", agent="dqn", lam=0.0, env_id="env", seed=1,
               final={"solved_rate": 0, "n_actions_mean": 0})
    r = W.check_curves()
    record("⑤ λ=0인데 무행동으로 굳은 것을 잡는가", r["n_floor"] > 0, str(r["floor_conditions"]))


def test_no_false_positive_rule(base: Path) -> None:
    """무행동 규칙은 '행동 0회·성공 0%'가 정상이다. 여기 걸리면 오탐이다."""
    write_meta(base, "seed2", agent="rule_noop", lam=0.0, env_id="env", seed=2,
               final={"solved_rate": 0, "n_actions_mean": 0})
    r = W.check_curves()
    record("⑤ 무행동 '규칙'은 오탐하지 않는가", r["n_floor"] == 0, f"n_floor={r['n_floor']}")


def test_expected_floor(base: Path) -> None:
    """λ>0에서 무행동으로 굳는 것은 이 연구가 관찰하려는 현상이다 — 세되 경고하지 않는다."""
    write_meta(base, "seed3", agent="dqn", lam=0.5, env_id="env", seed=3,
               final={"solved_rate": 0, "n_actions_mean": 0})
    r = W.check_curves()
    record("⑤ λ>0의 무행동은 경고가 아니라 집계인가",
           r["n_floor"] == 0 and r["n_expected_floor"] > 0,
           f"경고 {r['n_floor']} · 집계 {r['n_expected_floor']}")


def test_fresh(base: Path) -> None:
    """'조건이 안 늘었다'와 '학습이 멈췄다'는 다르다 — 최근 갱신된 로그가 있으면 도는 중이다."""
    (base / "logs" / "train" / "c.log").write_text("도는 중" + NL, encoding="utf-8")
    r = W.check_logs()
    record("② 최근 갱신 로그를 '도는 중'으로 세는가", r["fresh"] > 0, f"fresh={r['fresh']}")


def test_pid_alive() -> None:
    record("① 살아 있는 프로세스를 살아 있다고 하는가", W.pid_alive(os.getpid()), f"pid={os.getpid()}")
    record("① 없는 프로세스를 죽었다고 하는가", not W.pid_alive(999999), "pid=999999")


def test_process_roots() -> None:
    """탐지가 '기능하는지'를 본다 — 0을 반환하는 것과 없는 것을 구분해야 한다(사고 3).

    자기 자신으로는 시험할 수 없다. process_roots는 **재는 쪽을 일부러 제외**하기 때문이다
    (2026-08-29: 재는 명령의 명령줄에 패턴이 들어 있어 스스로를 세는 바람에
    '대기열 2개'로 보였다). 그래서 표식이 든 자식 프로세스를 잠깐 띄워 확인한다.
    """
    import subprocess
    import sys
    import time as _t

    mark = "wd_selftest_marker_zq7"
    child = subprocess.Popen([sys.executable, "-c",
                              f"import time; _ = '{mark}'; time.sleep(6)"])
    try:
        found = []
        for _ in range(12):                 # 프로세스 목록에 뜰 때까지 잠깐 기다린다
            found = W.process_roots(mark)
            if found:
                break
            _t.sleep(0.5)
        record("⑦ 돌고 있는 다른 프로세스를 찾는가", len(found) >= 1, f"찾은 뿌리 {found}")
    finally:
        child.terminate()
        child.wait(timeout=10)

    record("⑦ 재는 쪽(자기 자신)은 세지 않는가",
           W.process_roots("watchdog_selftest") == [], "자기 제외가 동작한다")
    n_none = W.process_roots("존재하지않는패턴_zzzq")
    record("⑦ 없는 패턴에는 0을 반환하는가", n_none == [], str(n_none))


def main() -> int:
    with_results(test_traceback)
    with_results(test_nan_log)
    with_results(test_nan_result)
    with_results(test_floor_lam0)
    with_results(test_no_false_positive_rule)
    with_results(test_expected_floor)
    with_results(test_fresh)
    test_pid_alive()
    test_process_roots()

    print("=" * 76)
    print("감시견 자가시험 — 일부러 고장 내어 '잡는지'를 본다")
    print("=" * 76)
    n_fail = 0
    for name, verdict, detail in results:
        mark = "  " if verdict == PASS else "! "
        print(f"{mark}{name:<44} {verdict}")
        if detail:
            print(f"      └ {detail[:80]}")
        n_fail += verdict != PASS
    print("-" * 76)
    print(f"시험 {len(results)}개 · 실패 {n_fail}개")
    if n_fail:
        print("실패한 검사는 '문제가 없어서' 조용한 것이 아니라 '못 찾아서' 조용한 것이다.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
