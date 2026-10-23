"""离线可复现向量：字符 n-gram 哈希 + 子线性词频 + L2 归一化。

默认使用 hash 提供方，保证评测不依赖外部服务；设置
RAG_EMBEDDING_PROVIDER=dashscope 可切换为通义千问 text-embedding-v3。
"""
import hashlib
import math
import os
import re
import threading
import time
from collections import Counter, OrderedDict

import numpy as np

from app.capabilities.retrieval.tokenizer import tokenize
from enterprise_rag import config

class EmbeddingUnavailableError(RuntimeError):
    """Configured embedding service failed; never substitute a different vector space."""


class EmbeddingIntegrityError(ValueError):
    """Embedding dimensions, metadata, or numeric values violate the index contract."""


class EmbeddingBudgetExceeded(RuntimeError):
    """The request-level embedding budget or deadline does not allow another call."""


def validate_vectors(vectors: list, expected_count: int) -> list[list[float]]:
    """Validate an entire batch before ingestion or similarity search can use it."""
    if not isinstance(vectors, (list, tuple)) or len(vectors) != expected_count:
        raise EmbeddingIntegrityError("向量服务返回数量与输入数量不一致")
    validated = []
    for vector in vectors:
        try:
            values = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingIntegrityError("向量必须是一维有限数值数组") from exc
        if values.shape != (config.EMBEDDING_DIM,) or not np.isfinite(values).all():
            raise EmbeddingIntegrityError(
                f"向量维度或数值无效：需要 {config.EMBEDDING_DIM} 维有限数值"
            )
        validated.append(values.tolist())
    return validated


_ASCII_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _features(text: str) -> Counter[str]:
    """检索词 + 汉字三元组：二元组负责召回，三元组补充短语区分度。"""
    text = (text or "").lower()
    features: Counter[str] = Counter(tokenize(text))
    for seq in _CJK_RE.findall(text):
        for index in range(len(seq) - 2):
            features[seq[index : index + 3]] += 0.6
    return features


def hash_embed(text: str, dim: int | None = None) -> list[float]:
    dim = config.EMBEDDING_DIM if dim is None else dim
    if dim <= 0:
        raise EmbeddingIntegrityError("向量维度必须大于零")
    vector = np.zeros(dim, dtype=np.float32)
    features = _features(text)
    if not features:
        return vector.tolist()
    weights = {term: 1.0 + math.log(count) for term, count in features.items()}
    for term, weight in weights.items():
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * weight
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector.tolist()


def _dashscope_embed(texts: list[str]) -> list[list[float]]:
    """通义千问兼容接口；失败时抛出异常，由调用方决定是否回退。

    向量服务与对话模型可以指向不同网关，因此单独读取
    RAG_EMBEDDING_API_BASE / RAG_EMBEDDING_API_KEY，未配置时回退到 OPENAI_*。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("RAG_EMBEDDING_API_BASE") or os.getenv("OPENAI_API_BASE") or None,
        timeout=60,
    )
    # DashScope text-embedding-v3 单请求最多 10 条；配置值只作上限，不能直接透传。
    batch_size = max(1, min(10, config.EMBEDDING_BATCH))
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=texts[start:start + batch_size],
            dimensions=config.EMBEDDING_DIM,
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if config.EMBEDDING_PROVIDER == "dashscope":
        try:
            vectors = _dashscope_embed(texts)
        except Exception as exc:
            # Hash and semantic vectors have unrelated coordinate systems.
            raise EmbeddingUnavailableError("DashScope 向量服务不可用；未替换为 hash 向量，请恢复服务后重试") from exc
        return validate_vectors(vectors, len(texts))
    if config.EMBEDDING_PROVIDER != "hash":
        raise EmbeddingIntegrityError(f"不支持的向量提供方：{config.EMBEDDING_PROVIDER}")
    return validate_vectors([hash_embed(text) for text in texts], len(texts))


def embed_query(text: str) -> np.ndarray:
    return np.asarray(embed_texts([text])[0], dtype=np.float32)


# ---- 查询向量缓存与 single-flight（S19） ---------------------------------
_QUERY_CACHE: "OrderedDict[str, tuple[float, list[float]]]" = OrderedDict()
_INFLIGHT: dict[str, threading.Event] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_STATS = {"hits": 0, "misses": 0, "merged": 0, "evictions": 0, "errors": 0}


def _query_cache_key(text: str) -> str:
    raw = "|".join(
        (
            config.EMBEDDING_PROVIDER,
            config.EMBEDDING_MODEL,
            str(config.EMBEDDING_DIM),
            str(config.EMBEDDING_VERSION),
            text,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> list[float] | None:
    ttl = config.QUERY_CACHE_TTL_SECONDS
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _QUERY_CACHE.get(key)
        if entry is None:
            return None
        stored_at, vector = entry
        if ttl > 0 and now - stored_at > ttl:
            _QUERY_CACHE.pop(key, None)
            return None
        _QUERY_CACHE.move_to_end(key)
        _CACHE_STATS["hits"] += 1
        return vector


def _cache_put(key: str, vector: list[float]) -> None:
    with _CACHE_LOCK:
        _QUERY_CACHE[key] = (time.monotonic(), vector)
        _QUERY_CACHE.move_to_end(key)
        while len(_QUERY_CACHE) > max(config.QUERY_CACHE_SIZE, 0):
            _QUERY_CACHE.popitem(last=False)
            _CACHE_STATS["evictions"] += 1


def embed_query_cached(text: str, budget=None, stage: str = "embedding") -> np.ndarray:
    """有界缓存 + 合并相同查询；缓存只保存文本向量，不保存权限/时点证据。"""
    key = _query_cache_key(text)
    cached = _cache_get(key)
    if cached is not None:
        return np.asarray(cached, dtype=np.float32)

    with _CACHE_LOCK:
        _CACHE_STATS["misses"] += 1
        event = _INFLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _INFLIGHT[key] = event
            leader = True
        else:
            _CACHE_STATS["merged"] += 1
            leader = False

    if not leader:
        timeout = max(0.1, min(60.0, budget.remaining_seconds() if budget else 60.0))
        event.wait(timeout=timeout)
        cached = _cache_get(key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)
        # 前一个请求失败或超时：当前请求自行计算，避免无限等待。
        with _CACHE_LOCK:
            _INFLIGHT[key] = threading.Event()
        event = _INFLIGHT[key]

    try:
        if budget is not None and not budget.can_call_embedding():
            raise EmbeddingBudgetExceeded("请求预算不允许继续调用向量服务")
        if budget is not None:
            budget.charge_embedding(stage=stage, detail=f"attempt:{key[:12]}")
        try:
            vector = embed_query(text)
        except Exception:
            if budget is not None:
                budget.note_call_result(ok=False, detail="embedding_failed")
            raise
        _cache_put(key, vector.tolist())
        return vector
    except Exception:
        with _CACHE_LOCK:
            _CACHE_STATS["errors"] += 1
        raise
    finally:
        with _CACHE_LOCK:
            current = _INFLIGHT.get(key)
            if current is event:
                _INFLIGHT.pop(key, None)
            event.set()


def cache_stats() -> dict:
    with _CACHE_LOCK:
        return {
            **_CACHE_STATS,
            "size": len(_QUERY_CACHE),
            "inflight": len(_INFLIGHT),
            "capacity": config.QUERY_CACHE_SIZE,
            "ttl_seconds": config.QUERY_CACHE_TTL_SECONDS,
        }


def reset_query_cache() -> None:
    with _CACHE_LOCK:
        _QUERY_CACHE.clear()
        _INFLIGHT.clear()
        for key in _CACHE_STATS:
            _CACHE_STATS[key] = 0



def to_json(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"
