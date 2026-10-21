"""待确认动作的执行/放弃节点：所有写操作只在这里发生。"""
import logging

from app.core.metrics import inc
from app.services.leave_service import leave_service
from app.workflows.chat_nodes.common import actor_from_state
from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)


def create_execution_nodes():
    nodes = {}

    def execute_pending_node(state: ChatState):
        pending = state.get("pending_action")
        if not pending:
            return {
                "task_result": "当前没有待确认的操作。",
                "pending_action": None,
            }

        action_type = pending.get("action_type")
        action_id = pending.get("action_id") or ""
        draft = pending.get("draft") or {}
        actor = actor_from_state(state)
        trace_id = state.get("trace_id", "")

        if action_type == "submit_leave":
            result = leave_service.submit(
                actor, draft, idempotency_key=action_id, trace_id=trace_id
            )
        elif action_type == "approve":
            result = leave_service.approve(
                actor,
                draft.get("app_id", ""),
                draft.get("decision", ""),
                draft.get("comments", ""),
                idempotency_key=action_id,
                trace_id=trace_id,
            )
        elif action_type == "cancel_leave":
            result = leave_service.cancel(
                actor, draft.get("app_id", ""), idempotency_key=action_id, trace_id=trace_id
            )
        else:
            return {
                "task_result": "无法识别的待确认操作，已放弃。",
                "pending_action": None,
            }

        inc(
            "leave_actions_total",
            {"action": action_type, "ok": str(result.ok).lower(), "code": result.code},
        )
        if result.ok:
            return {
                "task_result": result.message,
                "last_result": result.to_dict(),
                "pending_action": None,
            }

        # 可重试的临时失败保留草稿，方便用户再次确认；业务冲突/越权则清除。
        keep_pending = result.retryable
        return {
            "task_result": result.message,
            "last_result": result.to_dict(),
            "pending_action": pending if keep_pending else None,
        }

    nodes["execute_pending"] = execute_pending_node

    def clear_pending_node(state: ChatState):
        return {
            "task_result": "已取消本次操作，未提交任何变更。",
            "pending_action": None,
        }

    nodes["clear_pending"] = clear_pending_node
    return nodes
