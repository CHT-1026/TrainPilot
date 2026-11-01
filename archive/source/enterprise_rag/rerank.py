"""受约束的精排实验（S34）。

当前没有引入外部 cross-encoder 依赖，这里实现一个确定性的“查询-证据”
配对打分器，检验精排是否值得引入真实模型。默认关闭，开启后只改变排序，
不绕过权限、时点或证据预算。
"""
from __future__ import annotations

import re

from enterprise_rag import config
from enterprise_rag.citations import chunk_body
from enterprise_rag.query import query_terms


def _proximity(question_terms: list[str], text: str) -> float:
    positions = []
    lowered = text.lower()
    for term in question_terms:
        found = lowered.find(term.lower())
        if found >= 0:
            positions.append(found)
    if len(positions) < 2:
        return 0.0
    spread = max(positions) - min(positions)
    return 1.0 / (1.0 + spread / 80.0)


def lexical_pair_score(question: str, chunk: dict) -> float:
    body = chunk_body(chunk).lower()
    title = f"{chunk.get('title', '')} {chunk.get('section', '')}".lower()
    terms = query_terms(question, question)
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term.lower() in body)
    coverage = hits / len(terms)
    title_hits = sum(1 for term in terms if term.lower() in title)
    title_score = min(title_hits / max(len(terms), 1), 1.0)
    numbers = set(re.findall(r"\d[\d,\.]*", question))
    numeric_match = 1.0 if numbers and any(number in body for number in numbers) else 0.0
    proximity = _proximity(terms, body)
    return round(
        0.45 * coverage + 0.20 * title_score + 0.15 * proximity + 0.10 * numeric_match
        + 0.10 * (1.0 if chunk.get("chunk_type") == "table" else 0.0),
        6,
    )


def lexical_pair_rerank(question: str, chunks: list[dict], weight: float = 0.25) -> list[dict]:
    """在原检索分上叠加词法配对分后重排；不是神经 cross-encoder。"""
    reranked = []
    for chunk in chunks:
        item = dict(chunk)
        lexical = lexical_pair_score(question, item)
        item["lexical_pair_score"] = lexical
        item["score"] = round(float(item.get("score", 0.0)) + weight * lexical, 6)
        reranked.append(item)
    reranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return reranked
