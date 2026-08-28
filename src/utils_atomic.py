"""파일을 안전하게 바꿔치기하는 공용 도구.

왜 필요한가 (2026-08-29 실제 사고):
윈도우에서는 어떤 프로세스가 파일을 **읽는 중**이면 다른 프로세스의 os.replace가
PermissionError(WinError 5)로 거부된다. 이 저장소는 progress_*.json을 러너가 계속 쓰고
자가 감시·상태 요약·진행 확인이 계속 읽는다. 그 순간이 겹치면 러너가 통째로 죽는다.
실제로 공정성 파일럿 후보 C 러너가 이렇게 죽어 조건 2개를 잃었다.

원칙: **상태 기록이 실패했다고 실험이 죽어서는 안 된다.**
잠깐 기다렸다 다시 시도하고, 그래도 안 되면 건너뛴다(다음 갱신 때 어차피 다시 쓴다).
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def replace_with_retry(src: Path, dst: Path, attempts: int = 8, base_sleep: float = 0.05) -> bool:
    """src를 dst로 바꿔치기한다. 실패하면 조금씩 더 기다리며 다시 시도한다.

    반환: 성공하면 True, 끝내 실패하면 False (예외를 올리지 않는다).
    """
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            time.sleep(base_sleep * (i + 1))
        except FileNotFoundError:
            return False
    return False


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> bool:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    if replace_with_retry(tmp, path):
        return True
    # 바꿔치기가 끝내 안 되면 임시 파일을 치우고 조용히 넘어간다
    try:
        tmp.unlink()
    except OSError:
        pass
    return False
