"""事实记忆接口：用户隔离、可纠正、可删除。"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.domain.user.models import UserContext
from app.routes.auth import get_user_context
from app.workflows.fact_memory import (
    MemoryUnavailable,
    correct_fact,
    forget_fact,
    list_facts,
    remember_fact,
)

router = APIRouter(prefix="/api/memory", tags=["事实记忆"])

_UNAVAILABLE = "记忆存储暂不可用，请稍后重试"


class MemoryText(BaseModel):
    text: str = Field(min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def normalize(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("记忆内容不能为空")
        return value


class MemoryCreate(MemoryText):
    pass


class MemoryUpdate(MemoryText):
    pass


def _unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail=_UNAVAILABLE)


@router.get("")
async def memory_list(user: UserContext = Depends(get_user_context)):
    try:
        items = list_facts(user)
    except MemoryUnavailable:
        raise _unavailable()
    return {"code": 200, "data": {"items": items}}


@router.post("")
async def memory_create(payload: MemoryCreate, user: UserContext = Depends(get_user_context)):
    try:
        fact = remember_fact(user, payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MemoryUnavailable:
        raise _unavailable()
    return {"code": 200, "data": fact}


@router.patch("/{fact_id}")
async def memory_update(fact_id: str, payload: MemoryUpdate,
                        user: UserContext = Depends(get_user_context)):
    try:
        fact = correct_fact(user, fact_id, payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MemoryUnavailable:
        raise _unavailable()
    if not fact:
        raise HTTPException(status_code=404, detail="记忆不存在或不属于当前用户")
    return {"code": 200, "data": fact}


@router.delete("/{fact_id}")
async def memory_delete(fact_id: str, user: UserContext = Depends(get_user_context)):
    try:
        deleted = forget_fact(user, fact_id)
    except MemoryUnavailable:
        raise _unavailable()
    if not deleted:
        raise HTTPException(status_code=404, detail="记忆不存在或不属于当前用户")
    return {"code": 200, "message": "已删除"}
