"""进程内 BM25 索引：语料小，先不上向量库。"""
import logging
import math
from collections import Counter, defaultdict

from app.capabilities.retrieval.tokenizer import tokenize
from app.database.connection import get_redis_client
from app.infra.mysql import connection

logger = logging.getLogger(__name__)

K1 = 1.5
B = 0.75
TITLE_BOOST = 0.7

DEFAULT_NAMESPACE = "default"

_indexes: dict[str, "BM25Index"] = {}
_index_versions: dict[str, str] = {}


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(chunk.get("text", "")) for chunk in chunks]
        self.doc_tf = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        self.df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.df.update(set(tokens))
        total = len(chunks)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }
        self.title_tokens = [
            set(tokenize(f"{chunk.get('title', '')} {chunk.get('section', '')}"))
            for chunk in chunks
        ]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_tokens = [token for token in tokenize(query) if token]
        if not query_tokens or not self.chunks:
            return []
        unique_terms = set(query_tokens)
        scores: dict[int, float] = defaultdict(float)
        for index, tf in enumerate(self.doc_tf):
            length = self.doc_len[index] or 1
            score = 0.0
            for term in unique_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = freq + K1 * (1 - B + B * length / self.avgdl)
                score += idf * freq * (K1 + 1) / denominator
            if not score:
                continue
            for term in unique_terms:
                if term in self.title_tokens[index]:
                    score += TITLE_BOOST * self.idf.get(term, 0.0)
            scores[index] = score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for index, score in ranked:
            chunk = dict(self.chunks[index])
            chunk["score"] = round(score, 4)
            results.append(chunk)
        return results


def _load_chunks(namespace: str = DEFAULT_NAMESPACE) -> list[dict]:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.id AS chunk_id, c.doc_id, c.section, c.chunk_type, c.ord, c.text,
                       d.title, d.version, d.effective_date, d.visibility, d.department,
                       d.namespace, d.author, d.doc_family, d.category, d.format,
                       d.doc_created_at, d.doc_updated_at
                FROM kb_chunks AS c
                JOIN kb_documents AS d ON c.doc_id = d.id
                WHERE d.status = 'active' AND d.namespace = %s
                ORDER BY d.title, c.ord
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


def _current_version(namespace: str = DEFAULT_NAMESPACE) -> str:
    try:
        client = get_redis_client()
        value = client.get(f"kb_index_version:{namespace}")
        if value is None and namespace == DEFAULT_NAMESPACE:
            value = client.get("kb_index_version")
        return value or ""
    except Exception:
        return ""


def get_index(namespace: str = DEFAULT_NAMESPACE, force: bool = False) -> BM25Index:
    version = _current_version(namespace)
    index = _indexes.get(namespace)
    if force or index is None or version != _index_versions.get(namespace, ""):
        chunks = _load_chunks(namespace)
        index = BM25Index(chunks)
        _indexes[namespace] = index
        _index_versions[namespace] = version
        logger.info("知识库索引已加载：namespace=%s 块=%s", namespace, len(chunks))
    return index


def reload_index(namespace: str = DEFAULT_NAMESPACE) -> BM25Index:
    return get_index(namespace=namespace, force=True)


def index_stats(namespace: str = DEFAULT_NAMESPACE) -> dict:
    index = get_index(namespace=namespace)
    return {
        "namespace": namespace,
        "chunks": len(index.chunks),
        "documents": len({chunk["doc_id"] for chunk in index.chunks}),
        "version": _index_versions.get(namespace, ""),
    }
