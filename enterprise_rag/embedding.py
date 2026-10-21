"""离线可复现向量：字符 n-gram 哈希 + 子线性词频 + L2 归一化。

默认使用 hash 提供方，保证评测不依赖外部服务；设置
RAG_EMBEDDING_PROVIDER=dashscope 可切换为通义千问 text-embedding-v3。
"""
import hashlib
import math
import os
import re
from collections import Counter

import numpy as np

from app.capabilities.retrieval.tokenizer import tokenize
from enterprise_rag import config

class EmbeddingUnavailableError(RuntimeError):
    """Configured embedding service failed; never substitute a different vector space."""


class EmbeddingIntegrityError(ValueError):
    """Embedding dimensions, metadata, or numeric values violate the index contract."""


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
    """通义千问兼容接口；失败时抛出异常，由调用方决定是否回退。"""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_API_BASE") or None,
        timeout=60,
    )
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
        dimensions=config.EMBEDDING_DIM,
    )
    return [item.embedding for item in response.data]


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


def to_json(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"
