"""Phase 2 HTTP 冒烟：审批待办、建议引用、审计、trace 头与权限。"""
import json
import os
import sys
import urllib.request
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

BASE = os.getenv("OA_BASE", "http://127.0.0.1:8080")


def api(path, token=None, payload=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response), response.headers


def login(username):
    data, _ = api("/api/auth/login", payload={"username": username, "password": "admin123"})
    return data["data"]["token"]


def main():
    start = date(2028, 1, 3)
    while start.weekday() >= 5:
        start += timedelta(days=1)
    end = start + timedelta(days=1)
    key = f"phase2-{uuid4().hex[:10]}"
    employee = UserContext(user_id="employee_qian", name="钱工", role="employee", department="技术部")
    created = leave_service.submit(
        employee,
        {
            "leave_type": "年假",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "reason": "Phase2 HTTP smoke",
        },
        idempotency_key=key,
        trace_id="phase2-http-trace",
    )
    assert created.ok, created
    app_id = created.data["application"]["app_id"]

    try:
        manager = login("manager_zhang")
        other_employee = login("employee_zhao")

        pending, headers = api("/api/approvals/pending", manager)
        assert pending["data"]["total"] >= 1
        assert any(item["app_id"] == app_id for item in pending["data"]["items"]), pending
        trace_header = headers.get("X-Trace-ID")
        assert trace_header, headers
        print("pending ok, trace header", trace_header)

        suggestion, _ = api(f"/api/approvals/{app_id}/suggestion", manager, method="POST")
        assert suggestion["data"]["citations"], suggestion
        print("suggestion citations", len(suggestion["data"]["citations"]))

        audit, _ = api(f"/api/approvals/{app_id}/audit", manager)
        assert audit["data"]["total"] >= 1
        assert any(item["trace_id"] == "phase2-http-trace" for item in audit["data"]["items"]), audit
        print("audit ok", audit["data"]["total"])

        try:
            api(f"/api/approvals/{app_id}/audit", other_employee)
            raise AssertionError("other employee should be forbidden")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403, exc.code
            print("audit forbidden for other employee")

        print("PHASE2_HTTP_PASS")
    finally:
        conn = get_sync_db_connection()
        try:
            with conn.cursor() as cursor:
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
