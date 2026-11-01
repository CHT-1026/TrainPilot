"""Authenticated discovery and read-only direct tool invocation."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.capabilities.tools.registry import build_registry
from app.domain.user.models import UserContext
from app.routes.auth import get_user_context

router = APIRouter(prefix="/api/tools", tags=["工具调用"])


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)
    arguments: dict = Field(default_factory=dict)


@router.get("")
async def list_tools(user: UserContext = Depends(get_user_context)):
    registry = await build_registry(user)
    return {"code": 200, "data": registry.describe()}


@router.post("/call")
async def call_tool(payload: ToolCallRequest, user: UserContext = Depends(get_user_context)):
    registry = await build_registry(user)
    result = await registry.invoke(payload.name, payload.arguments)
    return {"code": 200, "data": result.to_dict()}
