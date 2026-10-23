"""取消申请节点：生成待确认草稿，不直接改库。"""
import logging
from datetime import datetime
from uuid import uuid4

from app.services.leave_service import leave_service
from app.workflows.chat_nodes.common import actor_from_state
from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)


def create_delete_nodes():
    nodes = {}

    def delete_node(state: ChatState):
        info = state.get("extracted_info", {}) or {}
        app_id = str(
            info.get("app_id")
            or info.get("application_id")
            or info.get("申请编号")
            or info.get("操作对象")
            or ""
        ).strip()
        if not app_id:
            return {"task_result": "请提供要取消的申请编号（格式：leave_xxx）。", "pending_action": None}

        result = leave_service.preview_cancel(actor_from_state(state), app_id)
        if not result.ok:
            return {"task_result": result.message, "pending_action": None}

        pending_action = {
            "action_id": f"act_{uuid4().hex[:12]}",
            "action_type": "cancel_leave",
            "draft": {"app_id": app_id, "application": result.data.get("application")},
            "created_at": datetime.now().isoformat(),
        }
        draft_version = int(state.get("draft_version", 0) or 0) + 1
        pending_action["draft_version"] = draft_version
        return {"task_result": result.message, "pending_action": pending_action,
                "draft_version": draft_version}

    nodes["delete"] = delete_node
    return nodes
