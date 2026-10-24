"""命名空间索引：BM25 + 向量矩阵按需加载，进程内缓存并按 Redis 版本失效。"""
import json
import logging
import math
from collections import Counter, defaultdict

import numpy as np

from app.capabilities.retrieval.index import BM25Index
from app.capabilities.retrieval.tokenizer import tokenize
from app.database.connection import get_redis_client
from app.infra.mysql import connection
from enterprise_rag import config
from enterprise_rag.embedding import EmbeddingIntegrityError, hash_embed, validate_vectors

logger = logging.getLogger(__name__)

_indexes: dict[str, "RagIndex"] = {}
_versions: dict[str, str] = {}


def _representation_text(chunk: dict) -> str:
    """多表示索引：标题、章节、表头/表名与正文首句，帮助短问句命中文档主题。"""
    stored = (chunk.get("representation") or "").strip()
    if stored:
        return stored
    text = chunk.get("text", "") or ""
    body = text.split("\n", 1)[1] if "\n" in text else text
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    header = ""
    if chunk.get("chunk_type") == "table" and lines:
        cells = [cell.strip() for cell in lines[0].strip("|").split("|") if cell.strip()]
        header = " ".join(cells[:8])
    first = lines[1] if chunk.get("chunk_type") == "table" and len(lines) > 1 else (lines[0] if lines else "")
    return " ".join(part for part in (
        chunk.get("title", ""),
        chunk.get("section", ""),
        header,
        first[:80],
    ) if part)



class RagIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # 加载时绑定的索引版本；检索前与发布快照核对，避免同块数换版被漏检。
        self.index_version = ""
        self.by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.by_doc_ord: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            self.by_doc_ord[chunk.get("doc_id", "")].append(chunk)
        for doc_chunks in self.by_doc_ord.values():
            doc_chunks.sort(key=lambda item: item.get("ord", 0))
        self.bm25 = BM25Index(chunks)
        self.representation_chunks = [
            dict(chunk, text=_representation_text(chunk)) for chunk in chunks
        ]
        self.representation_index = BM25Index(self.representation_chunks)
        self.matrix = self._build_matrix(chunks)
        self.family_docs: dict[str, list[dict]] = defaultdict(list)
        seen = set()
        for chunk in chunks:
            family = chunk.get("doc_family") or ""
            if family and chunk["doc_id"] not in seen:
                seen.add(chunk["doc_id"])
                self.family_docs[family].append(
                    {
                        "doc_id": chunk["doc_id"],
                        "title": chunk.get("title", ""),
                        "version": chunk.get("version", ""),
                        "effective_date": chunk.get("effective_date") or "",
                        "doc_updated_at": chunk.get("doc_updated_at") or "",
                    }
                )
        self.versioned_families = {
            family for family, docs in self.family_docs.items() if len(docs) > 1
        }
        # 词项 IDF：让 P1、AMR、LT-R20 等稀有词在重排中占更高权重。
        document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            document_frequency.update(set(tokenize(chunk.get("text", ""))))
        total = len(chunks) or 1
        self.term_idf = {
            term: math.log(1 + (total + 1) / (freq + 1))
            for term, freq in document_frequency.items()
        }
        self.corpus_text = "\n".join(chunk.get("text", "") for chunk in chunks).lower()

    @staticmethod
    def _build_matrix(chunks: list[dict]) -> np.ndarray:
        if not chunks:
            return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)
        rows = []
        for chunk in chunks:
            vector = chunk.get("embedding")
            if vector is None:
                if config.EMBEDDING_PROVIDER != "hash":
                    raise EmbeddingIntegrityError("索引缺少已配置模型的向量，请重新入库，禁止混入 hash 向量")
                vector = hash_embed(chunk.get("text", ""))
            else:
                _validate_embedding_metadata(chunk)
            rows.append(vector)
        return np.asarray(validate_vectors(rows, len(chunks)), dtype=np.float32)

    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        return self.bm25.search(query, top_k=top_k)

    def representation_search(self, query: str, top_k: int) -> list[dict]:
        """标题/章节/表头表示召回；返回真实块（不带表示文本）。"""
        hits = self.representation_index.search(query, top_k=top_k)
        results = []
        for hit in hits:
            chunk = self.by_id.get(hit["chunk_id"])
            if chunk is not None:
                current = dict(chunk)
                current["representation_score"] = hit.get("score", 0.0)
                results.append(current)
        return results

    def vector_search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        if not len(self.chunks):
            return []
        query_vector = np.asarray(validate_vectors([query_vector], 1)[0], dtype=np.float32)
        scores = self.matrix @ query_vector.astype(np.float32)
        order = np.argsort(-scores)[:top_k]
        results = []
        for position in order:
            score = float(scores[position])
            if score <= 0:
                continue
            chunk = dict(self.chunks[int(position)])
            chunk["vector_score"] = round(score, 6)
            results.append(chunk)
        return results

    def allowed_doc_ids(self, as_of: str | None) -> set[str] | None:
        """同族多版本消解：默认取最新版本；给定时间点时取该时点有效的最新版本。"""
        if not self.versioned_families:
            return None
        allowed = set()
        for family in self.versioned_families:
            docs = self.family_docs[family]
            if as_of:
                candidates = [doc for doc in docs if (doc["effective_date"] or "9999") <= as_of]
                if not candidates:
                    continue
                chosen = max(candidates, key=lambda item: (item["effective_date"], _version_key(item["version"])))
            else:
                chosen = max(
                    docs,
                    key=lambda item: (item["doc_updated_at"], _version_key(item["version"])),
                )
            allowed.add(chosen["doc_id"])
        return allowed


def _version_key(version: str) -> tuple[int, ...]:
    digits = "".join(char if char.isdigit() else "." for char in (version or ""))
    return tuple(int(part) for part in digits.split(".") if part.isdigit())


def _validate_embedding_metadata(chunk: dict) -> None:
    """Stored vectors must match the model, dimension, and revision used by queries."""
    chunk_id = chunk.get("chunk_id", "unknown")
    required = ("embedding_dim", "embedding_model", "embedding_version")
    if any(chunk.get(field) in (None, "") for field in required):
        raise EmbeddingIntegrityError(f"索引块 {chunk_id} 缺少向量元数据，请重新入库")
    try:
        dimension_matches = int(chunk["embedding_dim"]) == config.EMBEDDING_DIM
    except (TypeError, ValueError, OverflowError):
        dimension_matches = False
    if (
        not dimension_matches
        or chunk["embedding_model"] != config.EMBEDDING_MODEL
        or str(chunk["embedding_version"]) != str(config.EMBEDDING_VERSION)
    ):
        raise EmbeddingIntegrityError(f"索引块 {chunk_id} 的向量模型、维度或版本与当前配置不一致，请重新入库")


def _current_version(namespace: str) -> str:
    try:
        return get_redis_client().get(f"rag_index_version:{namespace}") or ""
    except Exception as exc:
        from app.capabilities.retrieval.revision import IndexRevisionUnavailable
        raise IndexRevisionUnavailable("索引版本存储不可用") from exc


def _load_chunks(namespace: str) -> list[dict]:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.id AS chunk_id, c.doc_id, c.section, c.chunk_type, c.ord,
                       c.text, c.token_count, c.embedding_json,
                       c.embedding_dim, c.embedding_model, c.embedding_version,
                       c.metadata_json, c.representation,
                       d.title, d.version, d.effective_date, d.visibility, d.department,
                       d.namespace, d.author, d.doc_family, d.category, d.format,
                       d.doc_created_at, d.doc_updated_at, d.source_path
                FROM kb_chunks AS c
                JOIN kb_documents AS d ON c.doc_id = d.id
                WHERE d.status = 'active' AND d.namespace = %s
                ORDER BY d.id, c.ord
                """,
                (namespace,),
            )
            rows = cursor.fetchall()
    for row in rows:
        for field in ("effective_date", "doc_created_at", "doc_updated_at"):
            value = row.get(field)
            if value and hasattr(value, "isoformat"):
                row[field] = value.isoformat()
        metadata_raw = row.pop("metadata_json", None)
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
            except (TypeError, ValueError):
                metadata = {}
            if isinstance(metadata, dict):
                for field in ("page", "paragraph", "anchor", "extractor_version", "ocr",
                              "parent_id", "prev_id", "next_id", "source_hash"):
                    if field in metadata:
                        row[field] = metadata[field]
        payload = row.pop("embedding_json", None)
        _validate_embedding_metadata(row)
        if not payload:
            raise EmbeddingIntegrityError(f"索引块 {row.get('chunk_id')} 缺少持久化向量，请重新入库")
        try:
            row["embedding"] = validate_vectors([json.loads(payload)], 1)[0]
        except (TypeError, ValueError) as exc:
            raise EmbeddingIntegrityError(f"索引块 {row.get('chunk_id')} 的向量 JSON 无效，请重新入库") from exc
    return rows


def load_index(namespace: str | None = None, force: bool = False) -> RagIndex:
    namespace = namespace or config.RAG_NAMESPACE
    from app.capabilities.retrieval.revision import access_revision

    acl_revision = access_revision(namespace)
    version = _current_version(namespace)
    if not version:
        from enterprise_rag.release import active_release
        from app.capabilities.retrieval.revision import IndexRevisionUnavailable

        published = active_release(namespace)
        if published.get("status") == "active" and published.get("index_version"):
            raise IndexRevisionUnavailable("已发布索引缺少权威版本，不能按旧快照提供内容")
    index = _indexes.get(namespace)
    if force or index is None or version != _versions.get(namespace, "") \
            or acl_revision != getattr(index, "access_revision", ""):
        chunks = _load_chunks(namespace)
        index = RagIndex(chunks)
        index.index_version = version
        index.access_revision = acl_revision
        _indexes[namespace] = index
        _versions[namespace] = version
        logger.info("RAG 索引已加载：namespace=%s 文档族=%s 块=%s", namespace, len(index.family_docs), len(chunks))
    return index


def reload_index(namespace: str | None = None) -> RagIndex:
    return load_index(namespace=namespace, force=True)


def index_stats(namespace: str | None = None) -> dict:
    namespace = namespace or config.RAG_NAMESPACE
    index = load_index(namespace=namespace)
    chunk_docs = {chunk["doc_id"] for chunk in index.chunks}
    return {
        "namespace": namespace,
        "documents": len(chunk_docs),
        "chunks": len(index.chunks),
        "families": len(index.family_docs),
        "versioned_families": len(index.versioned_families),
        "vector_dim": int(index.matrix.shape[1]) if len(index.matrix) else 0,
        "embedding_model": config.EMBEDDING_MODEL,
        "index_version": index.index_version,
        "categories": sorted({chunk.get("category") for chunk in index.chunks if chunk.get("category")}),
        "formats": sorted({chunk.get("format") for chunk in index.chunks if chunk.get("format")}),
        "ann": ann_recommendation(index),
        "query_cache": _cache_stats_safe(),
    }


def ann_recommendation(index: "RagIndex | None" = None, threshold: int | None = None) -> dict:
    """按规模与过滤能力决定是否值得迁移 ANN/向量库（S35）。"""
    threshold = threshold or config.ANN_CHUNK_THRESHOLD
    chunks = len(index.chunks) if index is not None else 0
    dim = int(index.matrix.shape[1]) if index is not None and len(index.matrix) else config.EMBEDDING_DIM
    estimated_mb = round(chunks * dim * 4 / (1024 * 1024), 3)
    return {
        "chunks": chunks,
        "dim": dim,
        "estimated_vector_mb": estimated_mb,
        "threshold": threshold,
        "recommended": chunks >= threshold,
        "reason": (
            f"当前 {chunks} 块低于 {threshold} 块阈值，进程内精确扫描仍可控；"
            "达到阈值后再评估带过滤能力的 ANN 或向量数据库"
        ) if chunks < threshold else (
            f"当前 {chunks} 块达到 {threshold} 块阈值，应评估 ANN/向量库的过滤下推与一致性"
        ),
    }


def _cache_stats_safe() -> dict:
    try:
        from enterprise_rag.embedding import cache_stats

        return cache_stats()
    except Exception:
        return {}



def document_list(namespace: str | None = None) -> list[dict]:
    namespace = namespace or config.RAG_NAMESPACE
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id AS doc_id, title, category, department, author, doc_family,
                       version, format, effective_date, doc_created_at, doc_updated_at,
                       visibility, status
                FROM kb_documents
                WHERE namespace = %s
                ORDER BY category, doc_family, doc_updated_at DESC
                """,
                (namespace,),
            )
            rows = cursor.fetchall()
    for row in rows:
        for field in ("effective_date", "doc_created_at", "doc_updated_at"):
            value = row.get(field)
            if value and hasattr(value, "isoformat"):
                row[field] = value.isoformat()
    return rows


def query_vector(text: str) -> np.ndarray:
    from enterprise_rag.budget import current_budget
    from enterprise_rag.embedding import embed_query_cached

    return embed_query_cached(text, budget=current_budget())

