"""轻量进程内指标：请求量、错误量、延迟分位。"""
import json
import math
from collections import Counter, defaultdict
from threading import Lock

_lock = Lock()
_counters: Counter[str] = Counter()
_latencies: dict[str, list[float]] = defaultdict(list)
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
    return {"counters": counters, "latencies": latencies}


def reset() -> None:
    with _lock:
        _counters.clear()
        _latencies.clear()
