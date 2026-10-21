"""审批草稿与审批建议：只读校验，写入统一交给确认后的执行节点。"""
import logging
from datetime import datetime
from uuid import uuid4

from app.services.leave_service import leave_service
from app.workflows.chat_nodes.common import actor_from_state, pick
from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)


def create_approval_nodes():
    nodes = {}

    def approve_node(state: ChatState):
        info = state.get("extracted_info", {}) or {}
        app_id = str(pick(info, "app_id", "application_id", "申请编号", "操作对象")).strip()
        decision = str(pick(info, "decision", "审批决策")).strip()
        comments = str(pick(info, "comments", "comment", "reason", "审批意见")).strip()

        if not app_id:
            return {"task_result": "请提供要审批的申请编号（格式：leave_xxx）。", "pending_action": None}
        if not decision:
            return {"task_result": "请明确是「通过」还是「驳回」。", "pending_action": None}

        result = leave_service.preview_approval(actor_from_state(state), app_id, decision, comments)
        if not result.ok:
            return {"task_result": result.message, "pending_action": None}

        pending_action = {
            "action_id": f"act_{uuid4().hex[:12]}",
            "action_type": "approve",
            "draft": {
                "app_id": app_id,
                "decision": result.data.get("decision"),
                "decision_label": result.data.get("decision_label"),
                "comments": comments,
                "application": result.data.get("application"),
            },
            "created_at": datetime.now().isoformat(),
        }
        return {"task_result": result.message, "pending_action": pending_action}

    nodes["approve"] = approve_node

    def suggestion_node(state: ChatState):
        info = state.get("extracted_info", {}) or {}
        app_id = str(pick(info, "app_id", "application_id", "申请编号", "操作对象")).strip()
        if not app_id:
            return {"task_result": "请提供要生成建议的申请编号（格式：leave_xxx）。", "pending_action": None}
        result = leave_service.suggest(actor_from_state(state), app_id)
        return {
            "task_result": result.message,
            "citations": result.data.get("citations", []) if result.data else [],
            "pending_action": None,
        }

    nodes["suggestion"] = suggestion_node
    return nodes
