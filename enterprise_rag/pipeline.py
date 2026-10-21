"""RAG 编排：查询理解 → 混合检索 → 重排 → 答案生成 → 引用溯源。"""
import time

from app.domain.user.models import UserContext
from enterprise_rag.answer import generate_answer
from enterprise_rag.citations import to_citations
from enterprise_rag.retrieval import retrieve


def _trace_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": chunk.get("doc_id"),
            "title": chunk.get("title"),
            "section": chunk.get("section"),
            "department": chunk.get("department"),
            "version": chunk.get("version"),
            "doc_updated_at": chunk.get("doc_updated_at"),
            "chunk_type": chunk.get("chunk_type"),
            "score": chunk.get("score"),
            "rrf_score": chunk.get("rrf_score"),
            "bm25_rank": chunk.get("bm25_rank"),
            "vector_rank": chunk.get("vector_rank"),
            "vector_score": chunk.get("vector_score"),
            "features": chunk.get("features"),
        }
        for chunk in chunks
    ]


def query(
    question: str,
    user: UserContext | None = None,
    top_k: int | None = None,
    as_of: str | None = None,
    use_llm: bool | None = None,
) -> dict:
    started = time.monotonic()
    result = retrieve(question, user=user, top_k=top_k, as_of=as_of)
    chunks = result["chunks"]
    generated = generate_answer(question, chunks, use_llm=use_llm,
                                evidence_complete=result["debug"].get("evidence_budget_complete", True))
    latency_ms = int((time.monotonic() - started) * 1000)

    return {
        "question": question,
        "answer": generated["answer"],
        "grounded": generated["grounded"],
        "refused": generated["refused"],
        "method": generated["method"],
        "confidence": generated["confidence"],
        "model_error": generated.get("model_error"),
        "reasoning": generated.get("reasoning"),
        "context_truncated": generated.get("context_truncated", False),
        "context_chunk_ids": generated.get("context_chunk_ids", []),
        "citations": to_citations(generated["citations"]),
        "query_rewrite": {
            "original": result["query"]["original"],
            "rewritten": result["query"]["rewritten"],
            "matched_synonyms": result["query"]["matched_synonyms"],
            "expansions": result["query"]["expansions"],
            "time_label": result["query"]["time_label"],
        },
        "retrieval": {
            "as_of": result["as_of"],
            "terms": result["terms"],
            "candidate_count": result["candidates"],
            "top_score": result["debug"]["top_score"],
            "debug": result["debug"],
            "chunks": _trace_chunks(chunks),
        },
        "latency_ms": latency_ms,
    }
