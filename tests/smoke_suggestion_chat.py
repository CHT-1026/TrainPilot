"""对话内审批建议冒烟：规则建议 + 制度引用进入 SSE complete。"""
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
    with urllib.request.urlopen(
        urllib.request.Request(BASE + path, data=data, headers=headers, method=method), timeout=180
    ) as response:
        return json.load(response)


def chat(token, thread, text):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
    data = json.dumps({"thread_id": thread, "user_message": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(BASE + "/api/chat/stream", data=data, headers=headers, method="POST")
    name, lines, final = "", [], None
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].lstrip())
            elif not line and lines:
                data = json.loads("\n".join(lines))
                lines = []
                if name == "error":
                    raise RuntimeError(data)
                if name == "complete":
                    final = data
    return final


def main():
    start = date(2028, 2, 7)
    while start.weekday() >= 5:
        start += timedelta(days=1)
    end = start + timedelta(days=1)
    key = f"suggest-{uuid4().hex[:10]}"
    employee = UserContext(user_id="employee_qian", name="钱工", role="employee", department="技术部")
    created = leave_service.submit(
        employee,
        {
            "leave_type": "年假",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "reason": "Suggestion chat smoke",
        },
        idempotency_key=key,
    )
    assert created.ok, created
    app_id = created.data["application"]["app_id"]
    try:
        login = api("/api/auth/login", payload={"username": "employee_qian", "password": "admin123"})
        token = login["data"]["token"]
        thread = api("/api/chat/session", token, method="POST")["data"]["thread_id"]
        try:
            final = chat(token, thread, f"给 {app_id} 一个审批建议")
            assert final["intent"] == "get_suggestion", final["intent"]
            assert "制度依据" in final["response"], final["response"]
            assert final.get("citations"), final
            print("suggestion chat citations", len(final["citations"]))
            print("SUGGESTION_CHAT_PASS")
        finally:
            api("/api/chat/session/" + thread, token, method="DELETE")
    finally:
        conn = get_sync_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM approval_audit WHERE application_id=%s", (app_id,))
                cursor.execute("DELETE FROM leave_applications WHERE id=%s", (app_id,))
                cursor.execute(
                    "DELETE FROM idempotency_keys WHERE idempotency_key IN (%s, %s) AND user_id=%s AND endpoint=%s",
                    (key, _idempotency_storage_key(key, employee.user_id, "leave.submit"),
                     employee.user_id, "leave.submit"),
                )
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    main()
