"""引用与上下文构造：编号、正文清洗、来源元数据。"""
from typing import Any

from enterprise_rag import config

MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS


def chunk_body(chunk: dict) -> str:
    text = chunk.get("text", "")
    prefix = f"{chunk.get('title', '')} · "
    return text.split("\n", 1)[1] if text.startswith(prefix) and "\n" in text else text


def select_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[list[dict], bool]:
    """Budget the exact chunks passed to generation and citation validation."""
    selected, used = [], 0
    for chunk in chunks:
        cost = len(_context_block(len(selected) + 1, chunk)) + 2
        if used + cost > max_chars:
            continue
        selected.append(chunk)
        used += cost
    return selected, len(selected) < len(chunks)


def _context_block(index, chunk):
    return (f"[{index}] 《{chunk.get('title', '')}》 版本 {chunk.get('version', '')} "
            f"| 部门 {chunk.get('department', '')} | 更新 {chunk.get('doc_updated_at', '')} "
            f"| 章节 {chunk.get('section', '')}\n{chunk_body(chunk)}")


def build_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    blocks = []
    used = 0
    for index, chunk in enumerate(chunks, 1):
        block = _context_block(index, chunk)
        if used + len(block) > max_chars:
            raise ValueError("上下文预算不一致：请先调用 select_context")
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def to_citations(chunks: list[dict]) -> list[dict[str, Any]]:
    citations = []
    for position, chunk in enumerate(chunks, 1):
        citations.append(
            {
                "index": position,
                "doc_id": chunk.get("doc_id"),
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "department": chunk.get("department", ""),
                "author": chunk.get("author", ""),
                "version": chunk.get("version", ""),
                "format": chunk.get("format", ""),
                "effective_date": chunk.get("effective_date", ""),
                "doc_updated_at": chunk.get("doc_updated_at", ""),
                "source_path": chunk.get("source_path", ""),
                "snippet": chunk_body(chunk)[:240],
                "score": chunk.get("score", 0.0),
                "vector_score": chunk.get("vector_score", 0.0),
            }
        )
    return citations


def cited_sources(chunks: list[dict], indices: list[int]) -> list[dict]:
    selected = []
    for index in sorted(set(indices)):
        if 1 <= index <= len(chunks):
            chunk = dict(chunks[index - 1])
            chunk["citation_index"] = index
            selected.append(chunk)
    return selected


def source_summary(chunks: list[dict]) -> str:
    lines = ["来源："]
    for index, chunk in enumerate(chunks, 1):
        lines.append(
            f"- [{index}] 《{chunk.get('title', '')}》 · {chunk.get('section', '')}"
            f"（{chunk.get('version', '')}，更新于 {str(chunk.get('doc_updated_at', ''))[:10]}）"
        )
    return "\n".join(lines)


def min_score() -> float:
    return config.MIN_SCORE
