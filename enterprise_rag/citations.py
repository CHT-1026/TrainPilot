"""引用与上下文构造：编号、正文清洗、分层组装、可逆压缩与来源元数据。"""
from __future__ import annotations

import re
from typing import Any

from enterprise_rag import config

MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS

_SENTENCE = re.compile(r"(?<=[。！？；;\n])")

_ROLE_TITLES = {
    "primary": "核心证据",
    "table": "表格证据",
    "graph": "图谱支撑证据",
    "supporting": "补充证据",
}


def chunk_body(chunk: dict) -> str:
    text = chunk.get("text", "")
    prefix = f"{chunk.get('title', '')} · "
    return text.split("\n", 1)[1] if text.startswith(prefix) and "\n" in text else text


def sentence_spans(text: str) -> list[dict]:
    """返回原文字符偏移，压缩后可逆地回到原句。"""
    spans = []
    start = 0
    for match in _SENTENCE.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            spans.append({"start": left, "end": end, "text": sentence})
        start = end
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            spans.append({"start": start, "end": len(text), "text": sentence})
    return spans


def evidence_role(chunk: dict, position: int = 0) -> str:
    if chunk.get("graph_paths"):
        return "graph"
    if chunk.get("chunk_type") == "table":
        return "table"
    if position < 3:
        return "primary"
    return "supporting"


def compress_chunk(chunk: dict, terms: list[str], max_chars: int = 900) -> dict:
    """按问题词覆盖度保留证据句，并记录保留句的原偏移。"""
    body = chunk_body(chunk)
    if len(body) <= max_chars:
        return chunk
    spans = sentence_spans(body)
    scored = []
    for span in spans:
        hits = sum(1 for term in terms if term and term in span["text"])
        scored.append((hits, len(span["text"]), span))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    kept, used = [], 0
    for hits, length, span in scored:
        if used + length > max_chars:
            continue
        kept.append(span)
        used += length
        if used >= max_chars * 0.85:
            break
    if not kept:
        return chunk
    kept.sort(key=lambda span: span["start"])
    compressed = "……".join(span["text"].strip() for span in kept)
    item = dict(chunk)
    prefix = f"{chunk.get('title', '')} · "
    item["text"] = prefix + compressed if chunk.get("text", "").startswith(prefix) else compressed
    item["compressed_spans"] = [
        {"start": span["start"], "end": span["end"], "text": span["text"][:120]} for span in kept
    ]
    item["context_compressed"] = True
    return item


def select_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS,
                   compress: bool = False, question: str = "") -> tuple[list[dict], bool]:
    """Budget the exact chunks passed to generation and citation validation."""
    selected, used = [], 0
    terms = []
    if compress and question:
        from enterprise_rag.query import query_terms

        terms = query_terms(question, question)
    for original in chunks:
        chunk = original
        cost = len(_context_block(len(selected) + 1, chunk)) + 2
        if used + cost > max_chars and compress:
            chunk = compress_chunk(original, terms, max_chars=max(300, max_chars // 2))
            cost = len(_context_block(len(selected) + 1, chunk)) + 2
        if used + cost > max_chars:
            continue
        selected.append(chunk)
        used += cost
    return selected, len(selected) < len(chunks)


def _context_block(index, chunk):
    role = evidence_role(chunk, index - 1)
    page = chunk.get("page")
    anchor = chunk.get("anchor") or ""
    location = f" | 页 {page}" if page not in (None, "", 0) else ""
    if anchor:
        location += f" | 定位 {anchor}"
    return (f"[{index}] 《{chunk.get('title', '')}》 版本 {chunk.get('version', '')} "
            f"| 部门 {chunk.get('department', '')} | 更新 {chunk.get('doc_updated_at', '')} "
            f"| 章节 {chunk.get('section', '')}{location} | 证据角色 {_ROLE_TITLES.get(role, role)}\n"
            f"{chunk_body(chunk)}")


def build_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """按证据角色分层；常驻规则留在系统提示，不重复占用证据预算。"""
    layers: dict[str, list[str]] = {}
    used = 0
    for index, chunk in enumerate(chunks, 1):
        block = _context_block(index, chunk)
        if used + len(block) > max_chars:
            raise ValueError("上下文预算不一致：请先调用 select_context")
        role = evidence_role(chunk, index - 1)
        layers.setdefault(role, []).append(block)
        used += len(block)
    sections = []
    for role in ("primary", "table", "graph", "supporting"):
        if layers.get(role):
            sections.append(f"【{_ROLE_TITLES[role]}】\n" + "\n\n".join(layers[role]))
    return "\n\n".join(sections)


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
                "visibility": chunk.get("visibility"),
                "author": chunk.get("author", ""),
                "version": chunk.get("version", ""),
                "format": chunk.get("format", ""),
                "effective_date": chunk.get("effective_date", ""),
                "doc_updated_at": chunk.get("doc_updated_at", ""),
                "source_path": chunk.get("source_path", ""),
                "page": chunk.get("page"),
                "anchor": chunk.get("anchor", ""),
                "evidence_role": evidence_role(chunk, position - 1),
                "snippet": chunk_body(chunk)[:240],
                "compressed_spans": chunk.get("compressed_spans", []),
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
