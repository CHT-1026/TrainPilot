"""知识库运维接口（仅管理员）。"""
import asyncio
from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.capabilities.retrieval.index import index_stats, reload_index
from app.capabilities.retrieval.kb_admin import list_documents, update_document
from app.capabilities.retrieval.search import retrieve, to_citations
from app.domain.user.models import UserContext
from app.routes.auth import get_user_context

router = APIRouter(prefix="/api/kb", tags=["知识库"])
Namespace = Annotated[str, Query(pattern=r"^[A-Za-z0-9_-]{1,32}$")]


def _require_admin(user: UserContext) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可操作知识库")


class DocumentUpdate(BaseModel):
    status: Literal["active", "superseded"] | None = None
    visibility: Literal["all", "manager", "hr", "finance", "admin"] | None = None
    department: str | None = Field(default=None, max_length=64)
    effective_date: date | None = None


@router.get("/documents")
async def documents(user: UserContext = Depends(get_user_context), namespace: Namespace = "default"):
    _require_admin(user)
    return {"code": 200, "data": await asyncio.to_thread(list_documents, namespace)}


@router.patch("/documents/{doc_id}")
async def patch_document(
    doc_id: str,
    payload: DocumentUpdate,
    user: UserContext = Depends(get_user_context),
    namespace: Namespace = "default",
):
    _require_admin(user)
    fields = payload.model_dump(exclude_unset=True)
    try:
        updated = await asyncio.to_thread(update_document, doc_id, fields, namespace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="文档不存在")
    # Index readers compare the authoritative metadata revision on every request.
    return {"code": 200, "data": updated}


@router.get("/stats")
async def stats(user: UserContext = Depends(get_user_context), namespace: Namespace = "default"):
    _require_admin(user)
    return {"code": 200, "data": await asyncio.to_thread(index_stats, namespace)}


@router.post("/reindex")
async def reindex(user: UserContext = Depends(get_user_context), namespace: Namespace = "default"):
    _require_admin(user)
    if namespace == "default":
        index = await asyncio.to_thread(reload_index, namespace)
    else:
        from enterprise_rag.store import reload_index as reload_enterprise_index
        index = await asyncio.to_thread(reload_enterprise_index, namespace)
    return {"code": 200, "data": {"namespace": namespace, "chunks": len(index.chunks)}}


@router.get("/search")
async def search(
    q: str = Query(min_length=1, max_length=200),
    top_k: int = Query(default=5, ge=1, le=20),
    user: UserContext = Depends(get_user_context),
    namespace: Namespace = "default",
):
    _require_admin(user)
    chunks = await asyncio.to_thread(retrieve, q, user, top_k, namespace)
    return {"code": 200, "data": {"items": to_citations(chunks)}}
