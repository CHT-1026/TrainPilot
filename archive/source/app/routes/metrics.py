"""指标查看接口（仅管理员）。"""
from fastapi import APIRouter, Depends, HTTPException

from app.core.backpressure import snapshot as backpressure_snapshot
from app.core.metrics import snapshot
from app.domain.user.models import UserContext
from app.routes.auth import get_user_context

router = APIRouter(prefix="/api/metrics", tags=["可观测性"])


@router.get("")
async def metrics(user: UserContext = Depends(get_user_context)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可查看指标")
    return {"code": 200, "data": {**snapshot(), "backpressure": backpressure_snapshot()}}
