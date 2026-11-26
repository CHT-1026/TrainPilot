"""隔离存储适配器的业务语义检查；不计入模型能力成绩。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent.app.domain.user.models import UserContext
from agent.app.services.leave_service import leave_service
from agent.local_runtime.business_eval_support import SQLiteSandbox


class SandboxChecks(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.db = SQLiteSandbox(Path(self.temp.name) / "business.sqlite")
        self.actor = UserContext(user_id="eval_employee", name="隔离员工", role="employee",
                                 department="技术部", manager_id="eval_manager")
        self.draft = {"leave_type":"事假","start_date":"2027-03-01","end_date":"2027-03-02","reason":"隔离自检"}
        self.enterContext(patch("agent.app.services.leave_service.connection", self.db.connection))
        self.enterContext(patch("agent.app.services.leave_service.transaction", self.db.transaction))

    def tearDown(self):
        self.db.db.close()
        self.temp.cleanup()

    def test_idempotent_submit_has_one_record_and_one_audit(self):
        first = leave_service.submit(self.actor, self.draft, idempotency_key="same-action")
        second = leave_service.submit(self.actor, self.draft, idempotency_key="same-action")
        self.assertTrue(first.ok, first)
        self.assertEqual(first, second)
        self.assertEqual(len(self.db.rows()), 1)
        self.assertEqual(len(self.db.audit()), 1)
        cancelled = leave_service.cancel(self.actor, first.data["application"]["app_id"], idempotency_key="cancel-action")
        self.assertTrue(cancelled.ok, cancelled)
        self.assertEqual(self.db.rows()[0]["status"], "cancelled")

    def test_transaction_rollback_preserves_state(self):
        with self.assertRaisesRegex(RuntimeError, "rollback-check"):
            with self.db.transaction():
                self.db.seed("leave_rollback")
                raise RuntimeError("rollback-check")
        self.assertEqual(self.db.rows(), [])

    def test_original_permission_and_transition_are_preserved(self):
        self.db.seed("leave_permission")
        forbidden = leave_service.preview_approval(self.actor, "leave_permission", "通过", "同意")
        self.assertEqual(forbidden.code, "FORBIDDEN")
        manager = UserContext(user_id="eval_manager", name="隔离主管", role="manager", department="技术部")
        result = leave_service.approve(manager, "leave_permission", "通过", "同意", idempotency_key="manager-action")
        self.assertTrue(result.ok, result)
        self.assertEqual(self.db.rows()[0]["status"], "approved")
        self.assertEqual(self.db.rows()[0]["comments"], "同意")


if __name__ == "__main__": unittest.main()
