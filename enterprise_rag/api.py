"""企业 RAG HTTP 接口：查询、统计、文档目录、索引刷新、评测结果。"""
import asyncio
import json
import threading
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.core.backpressure import OverloadedError, release, try_admit
from app.domain.user.models import UserContext
from app.routes.auth import get_user_context
from enterprise_rag import config
from enterprise_rag.feedback import feedback_summary, read_feedback, record_feedback
from enterprise_rag.pipeline import query as run_query
from enterprise_rag.store import document_list, index_stats, reload_index

router = APIRouter(prefix="/api/rag", tags=["企业RAG"])


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)
    as_of: date | None = None
    use_llm: bool | None = None
    offline: bool = False
    retrieval_mode: str = Field(default="hybrid", pattern="^(vector|hybrid|graph)$")

    @field_validator("question")
    @classmethod
    def valid_question(cls, value):
        from enterprise_rag.query import extract_time
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空白")
        extract_time(value)
        return value


def _require_admin(user: UserContext) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可执行该操作")


@router.post("/query")
async def rag_query(payload: RagQueryRequest, user: UserContext = Depends(get_user_context)):
    if not try_admit("rag", config.RAG_MAX_CONCURRENT):
        raise OverloadedError()

    cancelled = threading.Event()

    def work():
        try:
            return run_query(
                payload.question,
                user,
                payload.top_k,
                payload.as_of.isoformat() if payload.as_of else None,
                payload.use_llm,
                payload.retrieval_mode,
                payload.offline,
                cancel_check=cancelled.is_set,
            )
        finally:
            # HTTP 协程被取消时线程仍在执行；容量必须绑定到真实 worker 结束。
            release("rag")
    # shield also keeps a queued executor job alive long enough to release its slot.
    # Cancelling the await must not cancel that Future before work() can run.
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        return {"code": 200, "data": await asyncio.shield(task)}
    except asyncio.CancelledError:
        cancelled.set()
        # Retrieve the eventual exception even when the HTTP waiter is gone.
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        raise


@router.get("/stats")
async def rag_stats(user: UserContext = Depends(get_user_context)):
    return {"code": 200, "data": await asyncio.to_thread(index_stats)}


@router.get("/documents")
async def rag_documents(
    category: str | None = Query(default=None, max_length=32),
    user: UserContext = Depends(get_user_context),
):
    _require_admin(user)
    documents = await asyncio.to_thread(document_list)
    if category:
        documents = [item for item in documents if item.get("category") == category]
    return {"code": 200, "data": {"total": len(documents), "items": documents}}


@router.post("/reindex")
async def rag_reindex(user: UserContext = Depends(get_user_context)):
    _require_admin(user)
    index = await asyncio.to_thread(reload_index)
    return {
        "code": 200,
        "data": {
            "namespace": config.RAG_NAMESPACE,
            "documents": len({chunk["doc_id"] for chunk in index.chunks}),
            "chunks": len(index.chunks),
        },
    }


@router.get("/eval/latest")
async def rag_eval_latest(user: UserContext = Depends(get_user_context)):
    path = config.REPORT_DIR / "latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="尚未生成评测报告")
    return {"code": 200, "data": json.loads(path.read_text(encoding="utf-8"))}


@router.get("/graph/stats")
async def graph_stats(user: UserContext = Depends(get_user_context)):
    from enterprise_rag.graph.retrieve import load_graph

    return {"code": 200, "data": await asyncio.to_thread(lambda: load_graph().stats())}


@router.get("/graph/quality")
async def graph_quality(user: UserContext = Depends(get_user_context)):
    path = config.PACKAGE_ROOT / "graph" / "data" / "graph_meta.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="尚未构建知识图谱")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"code": 200, "data": {"version": data.get("version"),
                                  "stats": data.get("stats"),
                                  "quality": data.get("quality")}}


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    answer_excerpt: str = Field(default="", max_length=2000)
    rating: int | None = Field(default=None, ge=-1, le=1)
    correction: str = Field(default="", max_length=2000)
    citation_ids: list[str] = Field(default_factory=list, max_length=40)
    trace_id: str = Field(default="", max_length=64)
    snapshot: str = Field(default="", max_length=64)
    failure_stage: str = Field(default="", max_length=32)


@router.post("/feedback")
async def rag_feedback(payload: FeedbackRequest, user: UserContext = Depends(get_user_context)):
    try:
        record = await asyncio.to_thread(
            record_feedback,
            user,
            question=payload.question,
            answer_excerpt=payload.answer_excerpt,
            rating=payload.rating,
            correction=payload.correction,
            citation_ids=payload.citation_ids,
            trace_id=payload.trace_id,
            snapshot=payload.snapshot,
            failure_stage=payload.failure_stage,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"code": 200, "data": {"feedback_id": record["feedback_id"],
                                  "review_status": record["review_status"]}}


@router.get("/feedback")
async def rag_feedback_list(user: UserContext = Depends(get_user_context),
                            limit: int = Query(default=100, ge=1, le=500)):
    _require_admin(user)
    return {"code": 200, "data": {"summary": feedback_summary(),
                                  "items": read_feedback(limit)}}
