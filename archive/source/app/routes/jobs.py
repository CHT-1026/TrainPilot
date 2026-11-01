"""后台任务接口：创建、查询、取消；owner 隔离由服务端用户上下文决定。"""
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.domain.user.models import UserContext
from app.jobs.rag_jobs import cancel_job, create_job, get_job, list_jobs
from app.routes.auth import get_user_context

router = APIRouter(prefix="/api/jobs", tags=["后台任务"])


class RagJobRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_llm: bool = False
    top_k: int | None = Field(default=None, ge=1, le=50)


@router.post("/rag")
async def create_rag_job(payload: RagJobRequest, user: UserContext = Depends(get_user_context)):
    try:
        job = await asyncio.to_thread(create_job, user, payload.question,
                                      use_llm=payload.use_llm, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"code": 200, "data": job.to_dict(include_result=False)}


@router.get("")
async def jobs(user: UserContext = Depends(get_user_context)):
    items = await asyncio.to_thread(list_jobs, user)
    return {"code": 200, "data": {"items": [job.to_dict(include_result=False) for job in items]}}


@router.get("/{job_id}")
async def job_detail(job_id: str, user: UserContext = Depends(get_user_context)):
    job = await asyncio.to_thread(get_job, user, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或不属于当前用户")
    return {"code": 200, "data": job.to_dict(include_result=True)}


@router.post("/{job_id}/cancel")
async def job_cancel(job_id: str, user: UserContext = Depends(get_user_context)):
    job = await asyncio.to_thread(cancel_job, user, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或不属于当前用户")
    return {"code": 200, "data": job.to_dict(include_result=False)}
