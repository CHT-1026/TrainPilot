"""轻量进程内指标：请求量、错误量、延迟分位、阶段 span 与真实用量。"""
import json
import math
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from threading import Lock

_lock = Lock()
_counters: Counter[str] = Counter()
_latencies: dict[str, list[float]] = defaultdict(list)
_usage = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
_MAX_SAMPLES = 2000


def _key(name: str, labels: dict | None) -> str:
    if not labels:
        return name
    return name + "|" + json.dumps(labels, ensure_ascii=False, sort_keys=True)


def inc(name: str, labels: dict | None = None, value: int = 1) -> None:
    with _lock:
        _counters[_key(name, labels)] += value


def observe(name: str, value: float, labels: dict | None = None) -> None:
    with _lock:
        samples = _latencies[_key(name, labels)]
        samples.append(float(value))
        if len(samples) > _MAX_SAMPLES:
            del samples[: len(samples) - _MAX_SAMPLES]


@contextmanager
def span(name: str, labels: dict | None = None):
    """跨 SDK 的阶段计时；异常也会记录 ok=false 与耗时。"""
    started = time.monotonic()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        duration = time.monotonic() - started
        observe(f"{name}_seconds", duration, labels)
        inc(f"{name}_total", {**(labels or {}), "ok": str(ok).lower()})
        if duration > 60:
            inc("slow_stage_total", {"span": name})


def record_usage(kind: str, model: str = "", prompt_tokens: int = 0,
                 completion_tokens: int = 0, total_tokens: int = 0) -> None:
    """只记录计数与 token，不保存提示词/答案正文。"""
    key = f"{kind}:{model or 'unknown'}"
    with _lock:
        bucket = _usage[key]
        bucket["calls"] += 1
        bucket["prompt_tokens"] += int(prompt_tokens or 0)
        bucket["completion_tokens"] += int(completion_tokens or 0)
        bucket["total_tokens"] += int(total_tokens or (prompt_tokens or 0) + (completion_tokens or 0))


def _percentile(samples: list[float], ratio: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * ratio) - 1))
    return round(ordered[index], 4)


def snapshot() -> dict:
    with _lock:
        counters = dict(_counters)
        latencies = {}
        for key, samples in _latencies.items():
            latencies[key] = {
                "count": len(samples),
                "avg": round(sum(samples) / len(samples), 4),
                "p50": _percentile(samples, 0.50),
                "p95": _percentile(samples, 0.95),
                "max": round(max(samples), 4),
            }
        usage = {key: dict(value) for key, value in _usage.items()}
    return {"counters": counters, "latencies": latencies, "usage": usage}


def reset() -> None:
    with _lock:
        _counters.clear()
        _latencies.clear()
        _usage.clear()
