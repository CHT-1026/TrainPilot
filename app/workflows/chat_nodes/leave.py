"""提交请假节点：收集槽位、确定性校验、生成待确认草稿。"""
import logging
from datetime import datetime
from uuid import uuid4

from app.capabilities.policy_engine import blocking_failures, validate_leave_draft
from app.domain.leave.presenter import format_draft
from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)


def _get(info: dict, *keys, default=""):
    for key in keys:
        if key in info and info[key] not in (None, ""):
            return info[key]
    return default


def create_leave_nodes():
    nodes = {}

    def submit_leave_node(state: ChatState):
        extracted_info = state.get("extracted_info", {}) or {}
        context = state.get("conversation_context", {}) or {}
        accumulated = context.get("accumulated_extracted_info", {}) or {}
        merged = {**accumulated, **extracted_info}

        leave_type = _get(merged, "leave_type", "请假类型")
        start_date = _get(merged, "start_date", "开始日期")
        end_date = _get(merged, "end_date", "结束日期")
        reason = _get(merged, "reason", "leave_reason", "请假原因")

        missing = []
        if not leave_type:
            missing.append("请假类型（年假/病假/事假/婚假/产假/丧假）")
        if not start_date or not end_date:
            missing.append("请假时间（格式：2026-06-23 到 2026-06-28）")
        if not reason:
            missing.append("请假原因")

        if missing:
            iteration = int(state.get("iteration_count", 0))
            if iteration >= 5:
                return {
                    "task_result": "信息收集次数过多，请重新发起请假申请。",
                    "pending_action": None,
                    "iteration_count": 0,
                }
            question = "请提供以下信息：\n" + "\n".join(f"- {item}" for item in missing)
            return {
                "task_result": question,
                "extracted_info": merged,
                "conversation_context": {**context, "accumulated_extracted_info": merged},
                "iteration_count": iteration + 1,
                "pending_action": None,
            }

        checks, draft = validate_leave_draft(
            {
                "leave_type": leave_type,
                "start_date": start_date,
                "end_date": end_date,
                "reason": reason,
            }
        )
        failures = blocking_failures(checks)
        if failures:
            return {
                "task_result": "请假信息未通过校验：\n"
                + "\n".join(f"- {item.message}" for item in failures),
                "extracted_info": merged,
                "conversation_context": {**context, "accumulated_extracted_info": merged},
                "pending_action": None,
                "iteration_count": 0,
            }

        display_draft = {
            "leave_type": draft["leave_type"],
            "start_date": draft["start_date"].isoformat(),
            "end_date": draft["end_date"].isoformat(),
            "working_days": draft["working_days"],
            "reason": draft["reason"],
            "requires_hr": draft["requires_hr"],
        }
        pending_action = {
            "action_id": f"act_{uuid4().hex[:12]}",
            "action_type": "submit_leave",
            "draft": display_draft,
            "policy_checks": [check.model_dump() for check in checks],
            "created_at": datetime.now().isoformat(),
        }

        logger.info("生成请假待确认草稿: %s", pending_action["action_id"])
        return {
            "task_result": format_draft(display_draft, checks),
            "extracted_info": merged,
            "conversation_context": {**context, "accumulated_extracted_info": merged},
            "pending_action": pending_action,
            "iteration_count": 0,
        }

    nodes["submit_leave"] = submit_leave_node
    return nodes
