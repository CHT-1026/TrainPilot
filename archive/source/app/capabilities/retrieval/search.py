"""检索入口：BM25 召回 + 权限过滤 + 上下文/引用格式化。"""
from typing import Any

from app.capabilities.retrieval.index import DEFAULT_NAMESPACE, get_index
from app.domain.user.models import UserContext


def _allowed(chunk: dict, user: UserContext) -> bool:
    if user.is_admin:
        return True
    department = chunk.get("department")
    if department and not user.is_hr and user.department != department:
        return False
    visibility = chunk.get("visibility") or "all"
    if visibility == "all":
        return True
    if visibility == "manager":
        return user.is_manager
    if visibility == "hr":
        return user.is_hr
    if visibility == "finance":
        return user.role == "finance"
    if visibility == "admin":
        return user.is_admin
    return False


def chunk_allowed(chunk: dict, user: UserContext) -> bool:
    """对外暴露的权限判断，供企业 RAG 复用。"""
    return _allowed(chunk, user)


def retrieve(
    query: str,
    user: UserContext,
    top_k: int = 6,
    namespace: str = DEFAULT_NAMESPACE,
) -> list[dict]:
    index = get_index(namespace=namespace)
    candidates = index.search(query, top_k=max(top_k * 4, 20))
    return [chunk for chunk in candidates if _allowed(chunk, user)][:top_k]


def _body(chunk: dict) -> str:
    text = chunk.get("text", "")
    return text.split("\n", 1)[1] if "\n" in text else text


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, 1):
        version = chunk.get("version") or "未标注版本"
        blocks.append(
            f"[{index}] 文档：{chunk.get('title', '')}（{version}） | 章节：{chunk.get('section', '')}\n"
            f"{_body(chunk)}"
        )
    return "\n\n".join(blocks)


def to_citations(chunks: list[dict]) -> list[dict[str, Any]]:
    citations = []
    for position, chunk in enumerate(chunks, 1):
        body = _body(chunk)
        index = chunk.get("citation_index", position)
        citations.append(
            {
                "index": index,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "version": chunk.get("version", ""),
                "effective_date": chunk.get("effective_date", ""),
                "snippet": body[:180],
                "score": chunk.get("score", 0.0),
            }
        )
    return citations
