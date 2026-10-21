"""针对本地数据库的服务层冒烟测试：创建合成记录并在结束时清理。

运行（backend 目录）：
    .venv312\\Scripts\\python.exe tests\\smoke_service.py
"""
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


def next_weekday(start: date) -> date:
    cursor = start
    while cursor.weekday() >= 5:
        cursor += timedelta(days=1)
    return cursor


def main() -> None:
    employee = UserContext(
        user_id="employee_qian", name="钱工", role="employee", department="技术部"
    )
    manager = UserContext(
        user_id="manager_zhang", name="张主管", role="manager", department="技术部"
    )
    other_manager = UserContext(
        user_id="manager_li", name="李经理", role="manager", department="财务部"
    )

    conn = get_sync_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT MAX(end_date) AS last_date FROM leave_applications WHERE employee_id=%s",
            ("employee_qian",),
        )
        last = cursor.fetchone()["last_date"]
    conn.close()

    start = next_weekday(max(date(2027, 6, 1), (last + timedelta(days=10)) if last else date(2027, 6, 1)))
    end = next_weekday(start + timedelta(days=1))
    reason = "Phase0 service smoke test"
    key = f"smoke-{uuid4().hex[:12]}"
    app_id = None

    try:
        created = leave_service.submit(
            employee,
            {
                "leave_type": "事假",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "reason": reason,
            },
            idempotency_key=key,
        )
        assert created.ok, created
        app_id = created.data["application"]["app_id"]
        assert created.data["application"]["status"] == "pending"
        print("submit ok", app_id)

        duplicate = leave_service.submit(
            employee,
            {
                "leave_type": "事假",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "reason": reason,
            },
            idempotency_key=key,
        )
        assert duplicate.ok and duplicate.data["application"]["app_id"] == app_id
        print("idempotent submit ok")

        forbidden = leave_service.approve(employee, app_id, "通过", "self approve")
        assert not forbidden.ok and forbidden.code == "FORBIDDEN", forbidden
        print("employee approve blocked", forbidden.code)

        other_dept = leave_service.approve(other_manager, app_id, "通过", "cross dept")
        assert not other_dept.ok and other_dept.code == "FORBIDDEN", other_dept
        print("cross-department approve blocked", other_dept.code)

        suggestion = leave_service.suggest(manager, app_id)
        assert suggestion.ok, suggestion
        print("suggestion ok", suggestion.data["recommendation"])

        approved = leave_service.approve(manager, app_id, "通过", "smoke approve", idempotency_key=f"{key}-approve")
        assert approved.ok, approved
        assert approved.data["application"]["status"] == "approved"
        print("manager approve ok", approved.data["application"]["status"])

        again = leave_service.approve(manager, app_id, "通过", "again")
        assert not again.ok and again.code == "CONFLICT", again
        print("repeat approve blocked", again.code)

        cancel = leave_service.cancel(employee, app_id)
        assert not cancel.ok and cancel.code == "CONFLICT", cancel
        print("approved cancel blocked", cancel.code)

        mine = leave_service.mine(employee)
        assert mine.ok and any(item["app_id"] == app_id for item in mine.data["items"])
        print("mine query ok", mine.data["total"])

        audit_count = _audit_count(app_id)
        assert audit_count >= 2, audit_count
        print("audit rows", audit_count)
    finally:
        _cleanup(app_id, key)

    print("SMOKE_PASS")


def _audit_count(app_id):
    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS total FROM approval_audit WHERE application_id=%s",
                (app_id,),
            )
            return int(cursor.fetchone()["total"])
    finally:
        conn.close()


def _cleanup(app_id, key):
    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            if app_id:
                cursor.execute("DELETE FROM approval_audit WHERE application_id=%s", (app_id,))
                cursor.execute("DELETE FROM leave_applications WHERE id=%s", (app_id,))
            cursor.execute(
                "DELETE FROM idempotency_keys WHERE idempotency_key LIKE %s OR idempotency_key IN (%s, %s)",
                (f"{key}%", _idempotency_storage_key(key, "employee_qian", "leave.submit"),
                 _idempotency_storage_key(f"{key}-approve", "manager_zhang", "leave.approve")),
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
