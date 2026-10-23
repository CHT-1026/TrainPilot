"""工具契约与错误恢复边界（S12）。

可执行动作、必填字段、幂等与超时由代码声明；模型只能生成草稿，
不能自行扩大动作集合或跳过服务端校验。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.common.results import ToolResult

TOOL_SPECS = {
    "submit_leave": {
        "required": ("leave_type", "start_date", "end_date", "reason"),
    },
    "approve": {
        "required": ("app_id", "decision"),
    },
    "cancel_leave": {
        "required": ("app_id",),
    },
}


@dataclass(frozen=True)
class ContractCheck:
    ok: bool
    code: str = "OK"
    message: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "code": self.code, "message": self.message}


def validate_action(action_type: str, draft: dict) -> ContractCheck:
    spec = TOOL_SPECS.get(action_type)
    if spec is None:
        return ContractCheck(False, "UNKNOWN_ACTION", "无法识别的待确认操作类型")
    if not isinstance(draft, dict):
        return ContractCheck(False, "INVALID_ARGS", "待确认草稿必须是结构化对象")
    missing = [field for field in spec["required"] if draft.get(field) in (None, "")]
    if missing:
        return ContractCheck(False, "INVALID_ARGS", "待确认草稿缺少字段：" + "、".join(missing))
    return ContractCheck(True)


def validate_execution(action_type: str, draft: dict, action_id: str) -> ContractCheck:
    """执行前只校验不可绕过边界；字段完整性在草稿创建时已校验。

    这样既能阻止模型伪造动作类型/空 action_id，也不会让历史草稿因新增字段
    被误判为不可执行。
    """
    if action_type not in TOOL_SPECS:
        return ContractCheck(False, "UNKNOWN_ACTION", "无法识别的待确认操作类型")
    if not isinstance(draft, dict):
        return ContractCheck(False, "INVALID_ARGS", "待确认草稿必须是结构化对象")
    if not action_id:
        return ContractCheck(False, "INVALID_ARGS", "待确认操作缺少 action_id")
    return ContractCheck(True)


def contract_error(action_type: str, check: ContractCheck) -> ToolResult:
    return ToolResult.failure(check.code, check.message, data={"action_type": action_type})
