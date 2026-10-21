"""两级审批冒烟：年假 > 3 个工作日 -> 主管批准后转 HR。"""
import sys
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env.local", override=True)

from app.database.connection import get_sync_db_connection  # noqa: E402
from app.domain.user.models import UserContext  # noqa: E402
from app.domain.leave.repository import _idempotency_storage_key  # noqa: E402
from app.services.leave_service import leave_service  # noqa: E402


def next_monday(start: date) -> date:
    cursor = start
    while cursor.weekday() != 0:
        cursor += timedelta(days=1)
    return cursor


def main():
    employee = UserContext(user_id="employee_qian", name="钱工", role="employee", department="技术部")
    manager = UserContext(user_id="manager_zhang", name="张主管", role="manager", department="技术部")
    hr = UserContext(user_id="hr_liu", name="刘HR", role="hr", department="人事部")

    conn = get_sync_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT MAX(end_date) AS last_date FROM leave_applications WHERE employee_id=%s",
            ("employee_qian",),
        )
        last = cursor.fetchone()["last_date"]
    conn.close()

    start = next_monday(max(date(2027, 10, 1), (last + timedelta(days=14)) if last else date(2027, 10, 1)))
    end = start + timedelta(days=3)
    key = f"hr-smoke-{uuid4().hex[:10]}"
    app_id = None

    try:
        created = leave_service.submit(
            employee,
            {
                "leave_type": "年假",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "reason": "HR two-step smoke test",
            },
            idempotency_key=key,
            trace_id="hr-smoke-trace",
        )
        assert created.ok, created
        app_id = created.data["application"]["app_id"]
        assert created.data["application"]["leave_days"] == 4, created.data
        print("submit ok", app_id, "days", created.data["application"]["leave_days"])

        manager_approved = leave_service.approve(
            manager, app_id, "通过", "主管同意", trace_id="hr-smoke-trace"
        )
        assert manager_approved.ok, manager_approved
        assert manager_approved.data["application"]["status"] == "pending"
        assert manager_approved.data["application"]["approval_step"] == "hr"
        assert manager_approved.data["application"]["manager_approval"] == "approved"
        print("manager approved, moved to HR")

        again = leave_service.approve(manager, app_id, "通过", "重复审批")
        assert not again.ok and again.code in ("CONFLICT", "FORBIDDEN"), again
        print("manager repeat blocked", again.code)

        hr_approved = leave_service.approve(
            hr, app_id, "通过", "HR同意", trace_id="hr-smoke-trace"
        )
        assert hr_approved.ok, hr_approved
        assert hr_approved.data["application"]["status"] == "approved"
        assert hr_approved.data["application"]["hr_approval"] == "approved"
        print("HR approved, final status", hr_approved.data["application"]["status"])

        audit = leave_service.audit(hr, app_id)
        assert audit.ok and audit.data["total"] >= 3, audit
        actions = [item["action"] for item in audit.data["items"]]
        traces = {item["trace_id"] for item in audit.data["items"]}
        assert actions.count("submit") == 1 and actions.count("approve") == 2, actions
        assert "hr-smoke-trace" in traces, traces
        print("audit ok", actions, traces)
    finally:
        cleanup(app_id, key)

    print("HR_SMOKE_PASS")


def cleanup(app_id, key):
    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            if app_id:
                cursor.execute("DELETE FROM approval_audit WHERE application_id=%s", (app_id,))
                cursor.execute("DELETE FROM leave_applications WHERE id=%s", (app_id,))
            cursor.execute(
                "DELETE FROM idempotency_keys WHERE idempotency_key LIKE %s OR idempotency_key = %s",
                (f"{key}%", _idempotency_storage_key(key, "employee_qian", "leave.submit")),
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
