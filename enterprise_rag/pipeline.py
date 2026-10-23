"""RAG 编排：查询理解 → 混合检索 → 有效证据补查 → 答案生成 → 引用溯源。"""
import re
import time
from collections.abc import Callable

from app.domain.user.models import UserContext
from enterprise_rag.answer import generate_answer
from enterprise_rag.budget import RequestCancelled, bind_budget, new_budget
from enterprise_rag.citations import to_citations
from enterprise_rag.retrieval import retrieve
from enterprise_rag.scoring import describe as describe_scores, threshold_for


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
            "page": chunk.get("page"),
            "anchor": chunk.get("anchor"),
            "score": chunk.get("score"),
            "rrf_score": chunk.get("rrf_score"),
            "bm25_rank": chunk.get("bm25_rank"),
            "vector_rank": chunk.get("vector_rank"),
            "vector_score": chunk.get("vector_score"),
            "features": chunk.get("features"),
        }
        for chunk in chunks
    ]


def _evidence_gap(question: str, chunks: list[dict], debug: dict, threshold: float) -> str:
    """只判断“检索侧缺证”；格式/引用失败不触发重新检索。"""
    if not chunks:
        return "no_evidence"
    if float(debug.get("top_score") or 0.0) < threshold:
        return "low_score"
    if debug.get("plan", {}).get("strict_effective_date") and debug.get("plan", {}).get("policy_as_of"):
        years = set(re.findall(r"(20\d{2})", question))
        if years and not any(str(year) in str(chunk.get("text", "")) for year in years for chunk in chunks):
            return "missing_period_evidence"
    return ""


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RequestCancelled("cancelled")


def query(
    question: str,
    user: UserContext | None = None,
    top_k: int | None = None,
    as_of: str | None = None,
    use_llm: bool | None = None,
    retrieval_mode: str = "hybrid",
    offline: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    started = time.monotonic()
    budget = new_budget(use_llm=use_llm, offline=offline)
    with bind_budget(budget):
        with budget.stage("pipeline"):
            _check_cancel(cancel_check)
            result = retrieve(question, user=user, top_k=top_k, as_of=as_of,
                              retrieval_mode=retrieval_mode, budget=budget, offline=offline)
            chunks = result["chunks"]
            debug = result["debug"]
            threshold = threshold_for(bool(debug.get("comparison_mode")))
            correction = {"attempted": False, "reason": "", "added": 0}
            gap = _evidence_gap(question, chunks, debug, threshold)
            _check_cancel(cancel_check)
            if gap and not offline and not budget.expired():
                correction["attempted"] = True
                correction["reason"] = gap
                with budget.stage("corrective_retrieval"):
                    retry = retrieve(
                        f"{question} 相关制度 规定 原文 数据",
                        user=user, top_k=top_k, as_of=as_of,
                        retrieval_mode=retrieval_mode, budget=budget, offline=False,
                    )
                known = {chunk["chunk_id"] for chunk in chunks}
                merged = list(chunks)
                for chunk in retry["chunks"]:
                    if chunk["chunk_id"] not in known:
                        merged.append(chunk)
                        known.add(chunk["chunk_id"])
                merged.sort(key=lambda item: item.get("score", 0.0), reverse=True)
                correction["added"] = len(merged) - len(chunks)
                first_scope = result.get("scope", {})
                retry_scope = retry.get("scope", {})
                same_scope = (
                    retry_scope.get("snapshot") == first_scope.get("snapshot")
                    and retry_scope.get("acl_version") == first_scope.get("acl_version")
                )
                if same_scope and merged and len(merged) > len(chunks):
                    chunks = merged
                    result["debug"]["evidence_budget_complete"] = retry["debug"].get(
                        "evidence_budget_complete", True
                    )
                    result["debug"]["corrective_retrieval"] = correction
                elif not same_scope:
                    changed = ("snapshot_changed" if first_scope.get("snapshot") != retry_scope.get("snapshot")
                               else "acl_changed")
                    correction["reason"] = f"{gap}:{changed}"
                    result["debug"]["evidence_budget_complete"] = False
                    result["debug"]["corrective_retrieval"] = correction
            graph_paths = (result["debug"].get("graph") or {}).get("paths", [])
            effective_use_llm = False if offline else use_llm
            _check_cancel(cancel_check)
            generated = generate_answer(
                question,
                chunks,
                use_llm=effective_use_llm,
                evidence_complete=result["debug"].get("evidence_budget_complete", True),
                graph_paths=graph_paths,
                budget=budget,
            )
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
        "fact_check": generated.get("fact_check"),
        "prompt_version": generated.get("prompt_version"),
        "model_used": generated.get("model_used"),
        "skill_packs": generated.get("skill_packs", []),
        "context_truncated": generated.get("context_truncated", False),
        "context_chunk_ids": generated.get("context_chunk_ids", []),
        "graph_paths": generated.get("graph_paths", []),
        "citations": to_citations(generated["citations"]),
        "evidence_scope": result.get("scope"),
        "score_semantics": describe_scores(),
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
        "observability": budget.snapshot(),
        "latency_ms": latency_ms,
    }
