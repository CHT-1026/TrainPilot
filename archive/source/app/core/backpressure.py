"""应用侧背压与依赖故障策略（S20）。

先限制本进程的在途重请求、给依赖失败明确返回码，再讨论 API 水平扩容。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

from enterprise_rag import config


class OverloadedError(RuntimeError):
    def __init__(self, message: str = "服务繁忙，请稍后重试", retry_after: int = 2):
        super().__init__(message)
        self.retry_after = retry_after


_LOCK = threading.Lock()
_SEMAPHORES: dict[str, threading.BoundedSemaphore] = {}
_INFLIGHT: dict[str, int] = {}


def _semaphore(key: str, limit: int) -> threading.BoundedSemaphore:
    with _LOCK:
        if key not in _SEMAPHORES:
            _SEMAPHORES[key] = threading.BoundedSemaphore(max(1, limit))
            _INFLIGHT[key] = 0
        return _SEMAPHORES[key]


def try_admit(key: str, limit: int) -> bool:
    """非阻塞获取一个并发槽；失败由调用方返回 503，而不是无限排队。"""
    semaphore = _semaphore(key, limit)
    acquired = semaphore.acquire(blocking=False)
    if acquired:
        with _LOCK:
            _INFLIGHT[key] = _INFLIGHT.get(key, 0) + 1
    return acquired


def release(key: str) -> None:
    with _LOCK:
        semaphore = _SEMAPHORES.get(key)
        if semaphore is None or _INFLIGHT.get(key, 0) <= 0:
            return
        _INFLIGHT[key] -= 1
    semaphore.release()


@contextmanager
def admit(key: str, limit: int):
    if not try_admit(key, limit):
        raise OverloadedError(retry_after=2)
    try:
        yield
    finally:
        release(key)


def inflight(key: str) -> int:
    with _LOCK:
        return _INFLIGHT.get(key, 0)


def snapshot() -> dict:
    with _LOCK:
        return {
            "inflight": dict(_INFLIGHT),
            "limits": {key: sem._initial_value for key, sem in _SEMAPHORES.items()},
            "rag_max_concurrent": config.RAG_MAX_CONCURRENT,
        }
