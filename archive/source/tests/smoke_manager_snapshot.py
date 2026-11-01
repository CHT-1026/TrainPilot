"""HTTP 路径验证：确认提交时仍保留直属主管快照。"""
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
from app.domain.leave.repository import _idempotency_storage_key  # noqa: E402

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
                payload = json.loads("\n".join(lines))
                lines = []
                if name == "error":
                    raise RuntimeError(payload)
                if name == "complete":
                    final = payload
    return final


def main():
    conn = get_sync_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT manager_id FROM users WHERE username=%s", ("employee_qian",))
        old_manager = cursor.fetchone()["manager_id"]
        cursor.execute("UPDATE users SET manager_id=%s WHERE username=%s", ("manager_li", "employee_qian"))
    conn.commit()
    conn.close()

    start = date(2028, 4, 3)
    while start.weekday() >= 5:
        start += timedelta(days=1)
    end = start + timedelta(days=1)
    app_id = None
    action_id = None
    try:
        login = api("/api/auth/login", payload={"username": "employee_qian", "password": "admin123"})
        token = login["data"]["token"]
        thread = api("/api/chat/session", token, method="POST")["data"]["thread_id"]
        try:
            draft = chat(
                token,
                thread,
                f"我要提交请假申请。请假类型：年假；开始日期：{start}；结束日期：{end}；"
                f"请假原因：manager snapshot smoke。",
            )
            action_id = (draft.get("pending_action") or {}).get("action_id")
            assert draft["intent"] == "submit_leave", draft
            confirmed = chat(token, thread, "确认")
            assert confirmed["intent"] == "confirm_action", confirmed

            conn = get_sync_db_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, manager_id FROM leave_applications "
                        "WHERE employee_id=%s AND start_date=%s ORDER BY created_at DESC LIMIT 1",
                        ("employee_qian", start),
                    )
                    row = cursor.fetchone()
            finally:
                conn.close()
            app_id = row["id"] if row else None
            assert row and row["manager_id"] == "manager_li", row
            print("HTTP manager snapshot", row["manager_id"])
            print("MANAGER_SNAPSHOT_PASS")
        finally:
            api("/api/chat/session/" + thread, token, method="DELETE")
    finally:
        conn = get_sync_db_connection()
        try:
            with conn.cursor() as cursor:
                if app_id:
                    cursor.execute("DELETE FROM approval_audit WHERE application_id=%s", (app_id,))
                    cursor.execute("DELETE FROM leave_applications WHERE id=%s", (app_id,))
                if action_id:
                    cursor.execute(
                        "DELETE FROM idempotency_keys WHERE idempotency_key IN (%s, %s)",
                        (action_id, _idempotency_storage_key(action_id, "employee_qian", "leave.submit")),
                    )
                cursor.execute("UPDATE users SET manager_id=%s WHERE username=%s", (old_manager, "employee_qian"))
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    main()
