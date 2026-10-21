"""Phase 0 核心规则单元测试：权限、状态机、政策、结果模型。"""
import unittest
from datetime import date

from app.capabilities.policy_engine import (
    blocking_failures,
    count_working_days,
    requires_hr,
    validate_leave_draft,
)
from app.domain.common.results import ToolResult
from app.domain.leave.state_machine import (
    DECISION_APPROVED,
    DECISION_REJECTED,
    STEP_HR,
    STEP_MANAGER,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
    cancel_transition,
    check_approval_permission,
    check_cancel_permission,
    transition_for,
)
from app.domain.user.models import UserContext
from app.workflows.chat_nodes.common import actor_from_state


def actor(user_id: str, role: str, department: str, manager_id: str | None = None) -> UserContext:
    return UserContext(
        user_id=user_id,
        name=user_id,
        role=role,
        department=department,
        manager_id=manager_id,
    )


def application(**overrides):
    base = {
        "id": "leave_test001",
        "employee_id": "employee_zhao",
        "employee_name": "赵工",
        "manager_id": None,
        "department": "技术部",
        "leave_type": "事假",
        "leave_days": 2,
        "start_date": date(2026, 10, 19),
        "end_date": date(2026, 10, 20),
        "status": STATUS_PENDING,
        "approval_step": STEP_MANAGER,
        "manager_approval": "pending",
        "hr_approval": "pending",
        "version": 0,
    }
    base.update(overrides)
    return base


class PolicyEngineTest(unittest.TestCase):
    def test_working_days_excludes_weekend(self):
        self.assertEqual(count_working_days(date(2026, 10, 19), date(2026, 10, 23)), 5)
        self.assertEqual(count_working_days(date(2026, 10, 24), date(2026, 10, 25)), 0)

    def test_valid_draft(self):
        checks, draft = validate_leave_draft(
            {
                "leave_type": "事假",
                "start_date": "2026-10-19",
                "end_date": "2026-10-20",
                "reason": "家里有事",
            },
            today=date(2026, 10, 1),
        )
        self.assertEqual(draft["working_days"], 2)
        self.assertFalse(blocking_failures(checks))

    def test_weekend_only_is_blocked(self):
        checks, _ = validate_leave_draft(
            {
                "leave_type": "事假",
                "start_date": "2026-10-24",
                "end_date": "2026-10-25",
                "reason": "周末",
            },
            today=date(2026, 10, 1),
        )
        self.assertTrue(blocking_failures(checks))

    def test_end_before_start_is_blocked(self):
        checks, _ = validate_leave_draft(
            {
                "leave_type": "事假",
                "start_date": "2026-10-21",
                "end_date": "2026-10-20",
                "reason": "测试",
            },
            today=date(2026, 10, 1),
        )
        self.assertTrue(blocking_failures(checks))

    def test_annual_leave_over_three_days_requires_hr(self):
        self.assertTrue(requires_hr("年假", 4))
        self.assertFalse(requires_hr("年假", 3))
        self.assertFalse(requires_hr("事假", 8))


class ApprovalPermissionTest(unittest.TestCase):
    def test_employee_cannot_approve(self):
        decision = check_approval_permission(actor("employee_qian", "employee", "技术部"), application())
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "FORBIDDEN")

    def test_manager_can_approve_same_department(self):
        decision = check_approval_permission(actor("manager_zhang", "manager", "技术部"), application())
        self.assertTrue(decision.ok)

    def test_manager_cannot_approve_other_department(self):
        decision = check_approval_permission(actor("manager_li", "manager", "财务部"), application())
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "FORBIDDEN")

    def test_manager_cannot_approve_self(self):
        decision = check_approval_permission(
            actor("manager_zhang", "manager", "技术部"),
            application(employee_id="manager_zhang"),
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "FORBIDDEN")

    def test_direct_manager_can_approve_by_manager_id(self):
        decision = check_approval_permission(
            actor("manager_zhang", "manager", "技术部"),
            application(manager_id="manager_zhang"),
        )
        self.assertTrue(decision.ok)

    def test_same_department_non_direct_manager_is_blocked(self):
        decision = check_approval_permission(
            actor("manager_other", "manager", "技术部"),
            application(manager_id="manager_zhang"),
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "FORBIDDEN")
        self.assertIn("直属主管", decision.message)

    def test_manager_id_null_falls_back_to_department(self):
        decision = check_approval_permission(
            actor("manager_other", "manager", "技术部"),
            application(manager_id=None),
        )
        self.assertTrue(decision.ok)

    def test_hr_cannot_approve_manager_step(self):
        decision = check_approval_permission(actor("hr_liu", "hr", "人事部"), application())
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "FORBIDDEN")

    def test_hr_can_approve_hr_step(self):
        decision = check_approval_permission(
            actor("hr_liu", "hr", "人事部"),
            application(approval_step=STEP_HR, manager_approval="approved"),
        )
        self.assertTrue(decision.ok)

    def test_admin_can_approve(self):
        decision = check_approval_permission(actor("admin", "admin", "管理部"), application())
        self.assertTrue(decision.ok)

    def test_already_approved_is_conflict(self):
        decision = check_approval_permission(
            actor("manager_zhang", "manager", "技术部"),
            application(status=STATUS_APPROVED, approval_step="done"),
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "CONFLICT")


class TransitionTest(unittest.TestCase):
    def test_manager_approve_short_leave_completes(self):
        transition = transition_for(application(leave_type="事假", leave_days=2), DECISION_APPROVED)
        self.assertEqual(transition.status, STATUS_APPROVED)
        self.assertEqual(transition.step, "done")
        self.assertEqual(transition.manager_approval, "approved")
        self.assertEqual(transition.hr_approval, "not_required")

    def test_manager_approve_long_annual_goes_to_hr(self):
        transition = transition_for(
            application(leave_type="年假", leave_days=4, requires_hr=True),
            DECISION_APPROVED,
        )
        self.assertEqual(transition.status, STATUS_PENDING)
        self.assertEqual(transition.step, STEP_HR)
        self.assertEqual(transition.manager_approval, "approved")

    def test_reject_ends_flow(self):
        transition = transition_for(application(), DECISION_REJECTED)
        self.assertEqual(transition.status, STATUS_REJECTED)
        self.assertEqual(transition.step, "done")
        self.assertEqual(transition.manager_approval, "rejected")

    def test_hr_approve_completes(self):
        transition = transition_for(
            application(
                approval_step=STEP_HR,
                manager_approval="approved",
                leave_type="年假",
                leave_days=4,
                requires_hr=True,
            ),
            DECISION_APPROVED,
        )
        self.assertEqual(transition.status, STATUS_APPROVED)
        self.assertEqual(transition.hr_approval, "approved")

    def test_cancel_transition(self):
        transition = cancel_transition(application())
        self.assertEqual(transition.status, "cancelled")
        self.assertEqual(transition.manager_approval, "cancelled")


class CancelPermissionTest(unittest.TestCase):
    def test_owner_can_cancel_pending(self):
        self.assertTrue(
            check_cancel_permission(actor("employee_zhao", "employee", "技术部"), application()).ok
        )

    def test_other_user_cannot_cancel(self):
        decision = check_cancel_permission(
            actor("employee_qian", "employee", "技术部"), application()
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "FORBIDDEN")

    def test_cannot_cancel_approved(self):
        decision = check_cancel_permission(
            actor("employee_zhao", "employee", "技术部"),
            application(status=STATUS_APPROVED, approval_step="done"),
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "CONFLICT")


class ActorFromStateTest(unittest.TestCase):
    def test_prefers_server_user_and_keeps_manager_id(self):
        state = {
            "user": {
                "user_id": "employee_qian",
                "name": "钱工",
                "role": "employee",
                "department": "技术部",
                "manager_id": "manager_zhang",
            },
            "conversation_context": {"user_id": "employee_qian"},
        }
        actor = actor_from_state(state)
        self.assertEqual(actor.manager_id, "manager_zhang")
        self.assertEqual(actor.department, "技术部")


class ToolResultTest(unittest.TestCase):
    def test_success_and_failure(self):
        success = ToolResult.success(data={"x": 1}, message="ok")
        self.assertTrue(success.ok)
        self.assertEqual(success.data, {"x": 1})
        failure = ToolResult.failure("FORBIDDEN", "no")
        self.assertFalse(failure.ok)
        self.assertEqual(failure.code, "FORBIDDEN")


if __name__ == "__main__":
    unittest.main()
