"""Execution boundary for saved business drafts; completeness belongs to preview."""
from app.domain.common.results import ToolResult

BUSINESS_ACTIONS = {"submit_leave", "approve", "cancel_leave"}


def validate_execution(action_type: str, draft: dict, action_id: str) -> ToolResult | None:
    if action_type not in BUSINESS_ACTIONS:
        return ToolResult.failure("UNKNOWN_ACTION", "无法识别的待确认操作类型")
    if not isinstance(draft, dict) or not action_id:
        return ToolResult.failure("INVALID_ARGS", "待确认操作必须有结构化草稿与action_id")
    return None
