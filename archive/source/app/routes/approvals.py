"""审批只读接口：待办、建议、审计。写操作仍走对话确认。"""
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.domain.common.results import ToolResult
from app.domain.user.models import UserContext
from app.routes.auth import get_user_context
from app.services.leave_service import leave_service

router = APIRouter(prefix="/api/approvals", tags=["审批"])

_ERROR_STATUS = {
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "CONFLICT": 409,
    "INVALID_ARGS": 400,
}


def _raise_if_failed(result: ToolResult) -> None:
    if not result.ok:
        raise HTTPException(
            status_code=_ERROR_STATUS.get(result.code, 500),
            detail=result.message,
        )


@router.get("/pending")
async def pending(user: UserContext = Depends(get_user_context)):
    result = await asyncio.to_thread(leave_service.pending, user, limit=20)
    _raise_if_failed(result)
    return {"code": 200, "message": result.message, "data": result.data}


@router.post("/{app_id}/suggestion")
async def suggestion(app_id: str, user: UserContext = Depends(get_user_context)):
    result = await asyncio.to_thread(leave_service.suggest, user, app_id)
    _raise_if_failed(result)
    return {"code": 200, "message": result.message, "data": result.data}


@router.get("/{app_id}/audit")
async def audit(
    app_id: str,
    limit: int = Query(default=100, ge=1, le=200),
    user: UserContext = Depends(get_user_context),
):
    result = await asyncio.to_thread(leave_service.audit, user, app_id, limit=limit)
    _raise_if_failed(result)
    return {"code": 200, "message": result.message, "data": result.data}
