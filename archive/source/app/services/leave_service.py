"""请假领域服务：事务、权限、状态机、审计、幂等。"""
import logging
from datetime import date
from functools import wraps
from typing import Any
from uuid import uuid4

from app.capabilities.policy_engine import (
    PolicyCheck,
    blocking_failures,
    requires_hr,
    validate_leave_draft,
)
from app.capabilities.retrieval.search import retrieve, to_citations
from app.core.config import get_settings
from app.domain.common.results import ToolResult
from app.domain.leave import repository as repo
from app.domain.leave.presenter import (
    format_approval_draft,
    format_approval_receipt,
    format_cancel_draft,
    format_my_applications,
    format_pending,
    format_receipt,
    format_suggestion,
    serialize_application,
)
from app.domain.leave.state_machine import (
    DECISION_APPROVED,
    DECISION_REJECTED,
    STATUS_PENDING,
    STEP_MANAGER,
    cancel_transition,
    check_approval_permission,
    check_cancel_permission,
    check_view_permission,
    normalize_decision,
    transition_for,
)
from app.domain.user.models import UserContext
from app.infra.mysql import connection, transaction

logger = logging.getLogger(__name__)


def _tool_boundary(action: str, reply: str):
    """工具边界只处理一次：异常统一映射为可重试失败，业务分支保持显式返回。"""
    # ponytail:shortcut 工具边界保留一次宽泛捕获，避免 SSE/HTTP 层拿到裸异常。
    def decorate(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as exc:
                logger.error("%s失败: %s", action, exc, exc_info=True)
                return ToolResult.failure("TOOL_FAILED", reply, retryable=True)
        return wrapper
    return decorate


class LeaveApplicationService:
    """所有写操作都在事务内完成；权限只使用服务端 UserContext。"""

    @_tool_boundary("提交请假", "提交请假失败，请稍后重试")
    def submit(
        self,
        actor: UserContext,
        raw_draft: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        trace_id: str = "",
    ) -> ToolResult:
        checks, draft = validate_leave_draft(raw_draft)
        failures = blocking_failures(checks)
        if failures:
            return ToolResult.failure(
                "INVALID_ARGS",
                "请假信息未通过校验：" + "；".join(item.message for item in failures),
                data={"checks": [item.model_dump() for item in checks]},
            )

        with transaction() as conn:
            if idempotency_key:
                cached = repo.get_idempotent(
                    conn, idempotency_key, user_id=actor.user_id, endpoint="leave.submit"
                )
                if cached:
                    return ToolResult(**cached)

            overlap = repo.find_overlap(
                conn,
                actor.user_id,
                draft["start_date"],
                draft["end_date"],
            )
            if overlap:
                return ToolResult.failure(
                    "CONFLICT",
                    (
                        f"时间冲突：您在 {overlap['start_date']} 至 {overlap['end_date']} "
                        f"已有{overlap['leave_type']}申请"
                    ),
                )

            app_id = f"leave_{uuid4().hex[:12]}"
            repo.insert_application(
                conn,
                app_id=app_id,
                employee_id=actor.user_id,
                employee_name=actor.name or actor.user_id,
                employee_role=actor.role,
                manager_id=actor.manager_id,
                department=actor.department,
                leave_type=draft["leave_type"],
                leave_days=draft["working_days"],
                start_date=draft["start_date"],
                end_date=draft["end_date"],
                reason=draft["reason"],
                idempotency_key=idempotency_key,
            )
            repo.insert_audit(
                conn,
                application_id=app_id,
                action="submit",
                actor=actor,
                from_status="",
                to_status=STATUS_PENDING,
                comment=draft["reason"],
                trace_id=trace_id,
            )
            row = repo.get_application(conn, app_id)
            application = serialize_application(row or {})
            response = ToolResult.success(
                data={
                    "application": application,
                    "checks": [item.model_dump() for item in checks],
                },
                message=format_receipt(application),
            )
            if idempotency_key:
                repo.save_idempotent(
                    conn,
                    key=idempotency_key,
                    user_id=actor.user_id,
                    endpoint="leave.submit",
                    response=response.to_dict(),
                )
            return response

    @_tool_boundary("预览审批", "读取申请失败，请稍后重试")
    def preview_approval(
        self,
        actor: UserContext,
        app_id: str,
        decision_raw: str,
        comments: str = "",
    ) -> ToolResult:
        decision = normalize_decision(decision_raw)
        if not decision:
            return ToolResult.failure("INVALID_ARGS", "请明确是「通过」还是「驳回」")
        if decision == DECISION_REJECTED and not comments.strip():
            return ToolResult.failure("INVALID_ARGS", "驳回时必须填写审批意见")

        with connection() as conn:
            row = repo.get_application(conn, app_id)
            if not row:
                return ToolResult.failure("NOT_FOUND", f"未找到申请 {app_id}")
            application = serialize_application(row)
            permission = check_approval_permission(actor, row)
            if not permission.ok:
                return ToolResult.failure(permission.code, permission.message)
            application["requires_hr"] = requires_hr(
                row["leave_type"], int(row["leave_days"])
            )
            decision_label = "通过" if decision == DECISION_APPROVED else "驳回"
            return ToolResult.success(
                data={
                    "application": application,
                    "decision": decision,
                    "decision_label": decision_label,
                    "comments": comments,
                },
                message=format_approval_draft(application, decision_label, comments),
            )

    @_tool_boundary("审批", "审批失败，请稍后重试")
    def approve(
        self,
        actor: UserContext,
        app_id: str,
        decision_raw: str,
        comments: str = "",
        *,
        idempotency_key: str | None = None,
        trace_id: str = "",
    ) -> ToolResult:
        decision = normalize_decision(decision_raw)
        if not decision:
            return ToolResult.failure("INVALID_ARGS", "请明确是「通过」还是「驳回」")
        if decision == DECISION_REJECTED and not comments.strip():
            return ToolResult.failure("INVALID_ARGS", "驳回时必须填写审批意见")

        with transaction() as conn:
            if idempotency_key:
                cached = repo.get_idempotent(
                    conn, idempotency_key, user_id=actor.user_id, endpoint="leave.approve"
                )
                if cached:
                    return ToolResult(**cached)

            row = repo.get_application(conn, app_id, for_update=True)
            if not row:
                return ToolResult.failure("NOT_FOUND", f"未找到申请 {app_id}")

            permission = check_approval_permission(actor, row)
            if not permission.ok:
                return ToolResult.failure(permission.code, permission.message)

            row["requires_hr"] = requires_hr(row["leave_type"], int(row["leave_days"]))
            current_step = row.get("approval_step") or STEP_MANAGER
            transition = transition_for(row, decision)
            updated = repo.transition_application(
                conn,
                app_id=app_id,
                expected_status=row["status"],
                expected_step=current_step,
                expected_version=int(row.get("version") or 0),
                new_status=transition.status,
                new_step=transition.step,
                manager_approval=transition.manager_approval,
                hr_approval=transition.hr_approval,
                comments=comments,
            )
            if not updated:
                return ToolResult.failure("CONFLICT", "申请状态已变化，请刷新后重试")

            repo.insert_audit(
                conn,
                application_id=app_id,
                action="approve" if decision == DECISION_APPROVED else "reject",
                actor=actor,
                from_status=row["status"],
                to_status=transition.status,
                comment=comments,
                trace_id=trace_id,
            )
            fresh = repo.get_application(conn, app_id)
            application = serialize_application(fresh or {})
            decision_label = "审批通过" if decision == DECISION_APPROVED else "驳回"
            response = ToolResult.success(
                data={"application": application, "decision": decision},
                message=format_approval_receipt(application, decision_label),
            )
            if idempotency_key:
                repo.save_idempotent(
                    conn,
                    key=idempotency_key,
                    user_id=actor.user_id,
                    endpoint="leave.approve",
                    response=response.to_dict(),
                )
            return response

    @_tool_boundary("预览取消", "读取申请失败，请稍后重试")
    def preview_cancel(self, actor: UserContext, app_id: str) -> ToolResult:
        with connection() as conn:
            row = repo.get_application(conn, app_id)
            if not row:
                return ToolResult.failure("NOT_FOUND", f"未找到申请 {app_id}")
            permission = check_cancel_permission(actor, row)
            if not permission.ok:
                return ToolResult.failure(permission.code, permission.message)
            application = serialize_application(row)
            return ToolResult.success(
                data={"application": application},
                message=format_cancel_draft(application),
            )

    @_tool_boundary("取消申请", "取消申请失败，请稍后重试")
    def cancel(
        self,
        actor: UserContext,
        app_id: str,
        *,
        idempotency_key: str | None = None,
        trace_id: str = "",
    ) -> ToolResult:
        with transaction() as conn:
            if idempotency_key:
                cached = repo.get_idempotent(
                    conn, idempotency_key, user_id=actor.user_id, endpoint="leave.cancel"
                )
                if cached:
                    return ToolResult(**cached)

            row = repo.get_application(conn, app_id, for_update=True)
            if not row:
                return ToolResult.failure("NOT_FOUND", f"未找到申请 {app_id}")
            permission = check_cancel_permission(actor, row)
            if not permission.ok:
                return ToolResult.failure(permission.code, permission.message)

            if not repo.cancel_application(
                conn, app_id=app_id, expected_version=int(row.get("version") or 0)
            ):
                return ToolResult.failure("CONFLICT", "申请状态已变化，请刷新后重试")

            repo.insert_audit(
                conn,
                application_id=app_id,
                action="cancel",
                actor=actor,
                from_status=row["status"],
                to_status="cancelled",
                comment="用户取消",
                trace_id=trace_id,
            )
            fresh = repo.get_application(conn, app_id)
            application = serialize_application(fresh or {})
            response = ToolResult.success(
                data={"application": application},
                message=format_receipt(application),
            )
            if idempotency_key:
                repo.save_idempotent(
                    conn,
                    key=idempotency_key,
                    user_id=actor.user_id,
                    endpoint="leave.cancel",
                    response=response.to_dict(),
                )
            return response

    @_tool_boundary("查询我的申请", "查询失败，请稍后重试")
    def mine(self, actor: UserContext, *, limit: int = 50, offset: int = 0) -> ToolResult:
        with connection() as conn:
            rows = repo.list_mine(conn, actor.user_id, limit=limit, offset=offset)
            total = repo.count_mine(conn, actor.user_id)
        data = {
            "items": [serialize_application(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
        return ToolResult.success(data=data, message=format_my_applications(data))

    @_tool_boundary("查询待审批", "查询失败，请稍后重试")
    def pending(self, actor: UserContext, *, limit: int = 50, offset: int = 0) -> ToolResult:
        if not actor.is_approver:
            return ToolResult.failure("FORBIDDEN", "您没有审批权限")
        with connection() as conn:
            rows = repo.list_pending(conn, actor, limit=limit, offset=offset)
            total = repo.count_pending(conn, actor)
        data = {
            "items": [serialize_application(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
        return ToolResult.success(data=data, message=format_pending(data))

    @_tool_boundary("查询审计", "查询审批记录失败，请稍后重试")
    def audit(self, actor: UserContext, app_id: str, *, limit: int = 100) -> ToolResult:
        with connection() as conn:
            row = repo.get_application(conn, app_id)
            if not row:
                return ToolResult.failure("NOT_FOUND", f"未找到申请 {app_id}")
            permission = check_view_permission(actor, row)
            if not permission.ok:
                return ToolResult.failure(permission.code, permission.message)
            rows = repo.list_audit(conn, app_id, limit=limit)
        items = []
        for item in rows:
            created = item.get("created_at")
            items.append(
                {
                    "id": item.get("id"),
                    "application_id": item.get("application_id"),
                    "action": item.get("action"),
                    "actor_id": item.get("actor_id"),
                    "actor_role": item.get("actor_role"),
                    "from_status": item.get("from_status"),
                    "to_status": item.get("to_status"),
                    "comment": item.get("comment"),
                    "trace_id": item.get("trace_id"),
                    "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
                }
            )
        return ToolResult.success(
            data={"items": items, "total": len(items)},
            message=f"申请 {app_id} 共 {len(items)} 条审批记录。",
        )

    @_tool_boundary("生成审批建议", "生成建议失败，请稍后重试")
    def suggest(self, actor: UserContext, app_id: str) -> ToolResult:
        if not app_id:
            return ToolResult.failure("INVALID_ARGS", "请提供要生成建议的申请编号")
        with connection() as conn:
            row = repo.get_application(conn, app_id)
            if not row:
                return ToolResult.failure("NOT_FOUND", f"未找到申请 {app_id}")
            permission = check_view_permission(actor, row)
            if not permission.ok:
                return ToolResult.failure(permission.code, permission.message)

            app_date = row.get("start_date")
            today = app_date if isinstance(app_date, date) and app_date < date.today() else date.today()
            checks, draft = validate_leave_draft(
                {
                    "leave_type": row["leave_type"],
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "reason": row["reason"],
                },
                today=today,
            )
            overlap = repo.find_overlap(
                conn,
                row["employee_id"],
                row["start_date"],
                row["end_date"],
                exclude_id=row["id"],
            )
            if overlap:
                checks.append(
                    PolicyCheck(
                        rule="overlap",
                        ok=False,
                        message=(
                            f"与既有申请时间冲突：{overlap['start_date']} 至 {overlap['end_date']}"
                        ),
                        blocking=True,
                    )
                )

            failures = blocking_failures(checks)
            needs_hr = requires_hr(row["leave_type"], int(row["leave_days"]))
            if failures:
                recommendation = "建议驳回"
            elif needs_hr:
                recommendation = "建议批准，需 HR 审批"
            else:
                recommendation = "建议批准"

            reasons = [
                check.message
                for check in checks
                if check.ok and check.rule in {"leave_type", "dates", "max_days", "reason"}
            ]
            if not failures:
                reasons.append("规则校验未发现阻断项")
            risks = [
                check.message
                for check in checks
                if not check.blocking and check.ok and check.rule in {"hr_step", "sick_certificate"}
            ]
            risks.append("年假余额与历史请假记录需由 HR 系统核对")

            application = serialize_application(row)
            try:
                policy_query = f"{row['leave_type']} 请假 审批 规定 天数 上限 HR"
                chunks = [
                    chunk
                    for chunk in retrieve(policy_query, actor, top_k=3)
                    if chunk.get("score", 0.0) >= get_settings().kb_min_score
                ]
                citations = to_citations(chunks)
            except Exception as exc:
                logger.warning("审批建议检索制度失败: %s", exc)
                citations = []
            data = {
                "application": application,
                "recommendation": recommendation,
                "reasons": reasons,
                "risks": risks,
                "checks": [check.model_dump() for check in checks],
                "citations": citations,
            }
            return ToolResult.success(data=data, message=format_suggestion(data))


leave_service = LeaveApplicationService()
