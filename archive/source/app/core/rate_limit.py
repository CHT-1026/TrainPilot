"""按用户/键的轻量限流：Redis 优先，Redis 不可用时退化为进程内计数。"""
import time
from collections import defaultdict
from threading import Lock

from app.database.connection import get_redis_client

_memory: dict[str, list[float]] = defaultdict(list)
_lock = Lock()


def _redis_check(key: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
    client = get_redis_client()
    redis_key = f"rate_limit:{key}"
    count = int(client.incr(redis_key))
    if count == 1:
        client.expire(redis_key, window_seconds)
    if count > limit:
        ttl = int(client.ttl(redis_key) or window_seconds)
        return False, 0, max(ttl, 1)
    return True, max(limit - count, 0), 0


def _memory_check(key: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
    now = time.time()
    cutoff = now - window_seconds
    with _lock:
        timestamps = _memory[key]
        timestamps[:] = [stamp for stamp in timestamps if stamp >= cutoff]
        if len(timestamps) >= limit:
            retry_after = int(timestamps[0] + window_seconds - now) + 1
            return False, 0, max(retry_after, 1)
        timestamps.append(now)
        return True, limit - len(timestamps), 0


def check_rate_limit(key: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
    """返回 (allowed, remaining, retry_after_seconds)。"""
    if limit <= 0:
        return True, limit, 0
    try:
        return _redis_check(key, limit, window_seconds)
    except Exception:
        return _memory_check(key, limit, window_seconds)


def reset_rate_limit(key: str) -> None:
    try:
        get_redis_client().delete(f"rate_limit:{key}")
    except Exception:
        pass
    with _lock:
        _memory.pop(key, None)
