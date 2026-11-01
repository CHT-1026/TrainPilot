"""用户与汇报线管理（仅管理员）。"""
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.domain.user.models import UserContext
from app.routes.auth import get_user_context
from app.services.user_service import list_users, set_manager

router = APIRouter(prefix="/api/users", tags=["用户管理"])


class ManagerUpdate(BaseModel):
    manager_id: str | None = Field(default=None, max_length=64)


def _require_admin(user: UserContext) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可管理用户")


@router.get("")
async def users(user: UserContext = Depends(get_user_context)):
    _require_admin(user)
    return {"code": 200, "data": await asyncio.to_thread(list_users)}


@router.put("/{username}/manager")
async def update_manager(
    username: str,
    payload: ManagerUpdate,
    user: UserContext = Depends(get_user_context),
):
    _require_admin(user)
    manager_id = payload.manager_id.strip() if payload.manager_id else None
    try:
        updated = await asyncio.to_thread(set_manager, username, manager_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"code": 200, "data": updated}
