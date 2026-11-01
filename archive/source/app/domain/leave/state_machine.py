"""请假审批状态机与权限判断（纯函数，便于单元测试）。"""
from dataclasses import dataclass
from typing import Any

from app.domain.user.models import UserContext

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_CANCELLED = "cancelled"
STATUS_FINISHED = "finished"

STEP_MANAGER = "manager"
STEP_HR = "hr"
STEP_DONE = "done"

DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"

DECISION_LABELS = {
    "通过": DECISION_APPROVED,
    "批准": DECISION_APPROVED,
    "同意": DECISION_APPROVED,
    "approved": DECISION_APPROVED,
    "驳回": DECISION_REJECTED,
    "拒绝": DECISION_REJECTED,
    "不同意": DECISION_REJECTED,
    "rejected": DECISION_REJECTED,
}


@dataclass(frozen=True)
class PermissionDecision:
    ok: bool
    code: str = "OK"
    message: str = ""


@dataclass(frozen=True)
class Transition:
    status: str
    step: str
    manager_approval: str
    hr_approval: str


def normalize_decision(value: str) -> str | None:
    return DECISION_LABELS.get((value or "").strip())


def _step(application: dict[str, Any]) -> str:
    return application.get("approval_step") or STEP_MANAGER


def check_approval_permission(
    actor: UserContext,
    application: dict[str, Any],
) -> PermissionDecision:
    if application.get("status") != STATUS_PENDING:
        return PermissionDecision(False, "CONFLICT", "该申请已不在待审批状态，请刷新后重试")

    step = _step(application)
    if step not in (STEP_MANAGER, STEP_HR):
        return PermissionDecision(False, "CONFLICT", "该申请当前没有待审批环节")

    if application.get("employee_id") == actor.user_id:
        return PermissionDecision(False, "FORBIDDEN", "不能审批本人提交的申请")

    if actor.is_admin:
        return PermissionDecision(True)

    if step == STEP_MANAGER:
        if not actor.is_manager:
            return PermissionDecision(False, "FORBIDDEN", "只有部门主管可以执行主管审批")
        expected_manager = application.get("manager_id")
        if expected_manager:
            if actor.user_id != expected_manager:
                return PermissionDecision(False, "FORBIDDEN", "只能由申请人的直属主管审批")
            return PermissionDecision(True)
        # 未配置汇报线时回退部门权限，兼容现有数据。
        if not actor.department or application.get("department") != actor.department:
            return PermissionDecision(False, "FORBIDDEN", "只能审批本部门员工的申请")
        return PermissionDecision(True)

    if step == STEP_HR:
        if not actor.is_hr:
            return PermissionDecision(False, "FORBIDDEN", "当前环节需要 HR 审批")
        return PermissionDecision(True)

    return PermissionDecision(False, "FORBIDDEN", "没有审批权限")


def check_view_permission(actor: UserContext, application: dict[str, Any]) -> PermissionDecision:
    if actor.is_admin or actor.is_hr:
        return PermissionDecision(True)
    if application.get("employee_id") == actor.user_id:
        return PermissionDecision(True)
    if actor.is_manager:
        expected_manager = application.get("manager_id")
        if expected_manager:
            if expected_manager == actor.user_id:
                return PermissionDecision(True)
        elif actor.department and application.get("department") == actor.department:
            return PermissionDecision(True)
    return PermissionDecision(False, "FORBIDDEN", "只能查看本人或权限范围内员工的申请")


def check_cancel_permission(actor: UserContext, application: dict[str, Any]) -> PermissionDecision:
    if application.get("employee_id") != actor.user_id:
        return PermissionDecision(False, "FORBIDDEN", "只能取消本人的申请")
    if application.get("status") != STATUS_PENDING:
        return PermissionDecision(False, "CONFLICT", "只有待审批状态的申请可以取消")
    if _step(application) not in (STEP_MANAGER, STEP_HR):
        return PermissionDecision(False, "CONFLICT", "该申请当前不可取消")
    return PermissionDecision(True)


def transition_for(application: dict[str, Any], decision: str) -> Transition:
    """根据当前环节与决策计算下一状态。decision 必须是 approved/rejected。"""
    step = _step(application)
    manager_approval = application.get("manager_approval") or "pending"
    hr_approval = application.get("hr_approval") or "pending"

    if decision == DECISION_REJECTED:
        if step == STEP_MANAGER:
            manager_approval = "rejected"
        elif step == STEP_HR:
            hr_approval = "rejected"
        return Transition(STATUS_REJECTED, STEP_DONE, manager_approval, hr_approval)

    if step == STEP_MANAGER:
        manager_approval = "approved"
        if application.get("requires_hr"):
            return Transition(STATUS_PENDING, STEP_HR, manager_approval, "pending")
        return Transition(STATUS_APPROVED, STEP_DONE, manager_approval, "not_required")

    if step == STEP_HR:
        hr_approval = "approved"
        return Transition(STATUS_APPROVED, STEP_DONE, manager_approval, hr_approval)

    return Transition(STATUS_APPROVED, STEP_DONE, manager_approval, hr_approval)


def cancel_transition(application: dict[str, Any]) -> Transition:
    step = _step(application)
    manager_approval = application.get("manager_approval") or "pending"
    hr_approval = application.get("hr_approval") or "pending"
    if step == STEP_MANAGER:
        manager_approval = "cancelled"
    elif step == STEP_HR:
        hr_approval = "cancelled"
    return Transition(STATUS_CANCELLED, STEP_DONE, manager_approval, hr_approval)
