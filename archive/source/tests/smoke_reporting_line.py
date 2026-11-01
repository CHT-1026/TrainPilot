"""汇报线权限冒烟：有 manager_id 时只允许直属主管审批，不再按部门兜底。"""
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
from app.services.user_service import get_user_by_username  # noqa: E402


def main():
    conn = get_sync_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT manager_id FROM users WHERE username=%s", ("employee_qian",))
        old_manager = cursor.fetchone()["manager_id"]
        cursor.execute("UPDATE users SET manager_id=%s WHERE username=%s", ("manager_li", "employee_qian"))
    conn.commit()
    conn.close()

    app_id = None
    key = f"reportline-{uuid4().hex[:10]}"
    try:
        user = get_user_by_username("employee_qian")
        assert user["manager_id"] == "manager_li", user
        employee = UserContext.from_dict(user)
        manager_zhang = UserContext(user_id="manager_zhang", name="张主管", role="manager", department="技术部")
        manager_li = UserContext.from_dict(get_user_by_username("manager_li"))

        start = date(2028, 3, 6)
        while start.weekday() >= 5:
            start += timedelta(days=1)
        end = start + timedelta(days=1)
        created = leave_service.submit(
            employee,
            {
                "leave_type": "年假",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "reason": "Reporting line smoke",
            },
            idempotency_key=key,
        )
        assert created.ok, created
        app_id = created.data["application"]["app_id"]
        assert created.data["application"]["manager_id"] == "manager_li", created.data
        print("submit snapshot manager_id", created.data["application"]["manager_id"])

        blocked = leave_service.approve(manager_zhang, app_id, "通过", "same dept but not direct manager")
        assert not blocked.ok and blocked.code == "FORBIDDEN", blocked
        assert "直属主管" in blocked.message, blocked.message
        print("same-department non-direct manager blocked")

        approved = leave_service.approve(manager_li, app_id, "通过", "direct manager approve")
        assert approved.ok, approved
        assert approved.data["application"]["status"] == "approved", approved
        print("direct manager approved", approved.data["application"]["status"])
    finally:
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
                cursor.execute("UPDATE users SET manager_id=%s WHERE username=%s", (old_manager, "employee_qian"))
            conn.commit()
        finally:
            conn.close()

    print("REPORTING_LINE_PASS")


if __name__ == "__main__":
    main()
