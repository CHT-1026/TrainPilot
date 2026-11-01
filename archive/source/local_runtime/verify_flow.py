"""One bounded end-to-end check using explicitly synthetic local demo records."""
import json
import os
import pathlib
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone

import pymysql
from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.domain.leave.repository import _idempotency_storage_key  # noqa: E402

BASE = os.getenv("OA_BASE", "http://127.0.0.1:8080")
REPORT = ROOT / "local_runtime" / "verification.json"


def request(path, token=None, payload=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return urllib.request.Request(BASE + path, data=data, headers=headers, method=method)


def api(path, token=None, payload=None, method=None):
    with urllib.request.urlopen(request(path, token, payload, method), timeout=30) as response:
        return json.load(response)


def login(username):
    return api("/api/auth/login", payload={"username": username, "password": "admin123"})["data"]["token"]


def chat(token, thread, text):
    start = time.monotonic()
    events, name, lines, final = [], "", [], None
    with urllib.request.urlopen(request("/api/chat/stream", token,
         {"thread_id": thread, "user_message": text}), timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].lstrip())
            elif not line and lines:
                data = json.loads("\n".join(lines)); lines = []
                events.append(name)
                if name == "error":
                    raise RuntimeError("SSE error: " + data.get("error", "unknown"))
                if name == "complete":
                    final = data
    if not final:
        raise RuntimeError("Missing SSE complete event")
    final["elapsed_seconds"] = round(time.monotonic() - start, 2)
    final["event_types"] = sorted(set(events))
    return final


def next_weekday(start):
    cursor = start
    while cursor.weekday() >= 5:
        cursor += timedelta(days=1)
    return cursor


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    values = dotenv_values(ROOT / ".env.local")
    if values["MYSQL_HOST"] not in ("127.0.0.1", "localhost"):
        raise RuntimeError("Verification must use the local database")
    report = {"checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "kind": "synthetic_local_leave_workflow_with_confirmation", "steps": []}
    def step(name, **extra):
        record = {"name": name, "result": "pass", **extra}
        report["steps"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    conn = pymysql.connect(host=values["MYSQL_HOST"], port=int(values["MYSQL_PORT"]),
        user=values["MYSQL_USER"], password=values["MYSQL_PASSWORD"], database=values["MYSQL_DATABASE"],
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor, autocommit=True)
    app_id = None
    cancel_id = None
    employee = manager = thread = manager_thread = None
    owned_actions = set()

    def remember_action(response, owner, endpoint):
        action_id = (response.get("pending_action") or {}).get("action_id")
        if action_id:
            owned_actions.add((owner, endpoint, action_id))

    try:
        assert api("/local/status")["model_ready"]
        employee = login("employee_qian"); manager = login("manager_zhang")
        step("employee_and_manager_login")
        thread = api("/api/chat/session", employee, method="POST")["data"]["thread_id"]
        manager_thread = api("/api/chat/session", manager, method="POST")["data"]["thread_id"]
        report.update(employee_thread=thread, manager_thread=manager_thread)
        with conn.cursor() as cursor:
            cursor.execute("SELECT MAX(end_date) AS last_date FROM leave_applications WHERE employee_id=%s", ("employee_qian",))
            last = cursor.fetchone()["last_date"]
        start = next_weekday(max(date(2027, 1, 4), date.today() + timedelta(days=30), (last + timedelta(days=3)) if last else date.min))
        end = next_weekday(start + timedelta(days=1))
        reason = "本机运行验收测试（虚拟记录）"

        draft = chat(employee, thread,
            f"我要提交请假申请。请假类型：事假；开始日期：{start}；结束日期：{end}；请假原因：{reason}。")
        remember_action(draft, "employee_qian", "leave.submit")
        assert draft["intent"] == "submit_leave", draft["intent"]
        assert draft.get("pending_action", {}).get("action_type") == "submit_leave", draft
        assert "确认" in draft["response"], draft["response"]
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM leave_applications WHERE employee_id=%s AND start_date=%s", ("employee_qian", start))
            assert cursor.fetchone()["total"] == 0
        step("leave_draft_requires_confirmation", elapsed_seconds=draft["elapsed_seconds"])

        submitted = chat(employee, thread, "确认")
        assert submitted["intent"] == "confirm_action", submitted["intent"]
        assert submitted.get("pending_action") is None, submitted
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM leave_applications WHERE employee_id=%s AND start_date=%s ORDER BY created_at DESC LIMIT 1", ("employee_qian", start))
            record = cursor.fetchone()
        cancel_id = record["id"] if record else None
        assert record and record["status"] == "pending", record
        assert record["approval_step"] == "manager", record
        step("submit_leave_confirmed_and_persisted", application_id=cancel_id, status=record["status"], elapsed_seconds=submitted["elapsed_seconds"])

        cancel_draft = chat(employee, thread, f"取消申请 {cancel_id}。")
        remember_action(cancel_draft, "employee_qian", "leave.cancel")
        assert cancel_draft["intent"] == "delete_application", cancel_draft["intent"]
        assert cancel_draft.get("pending_action", {}).get("action_type") == "cancel_leave", cancel_draft
        cancelled = chat(employee, thread, "确认")
        assert cancelled["intent"] == "confirm_action", cancelled["intent"]
        with conn.cursor() as cursor:
            cursor.execute("SELECT status FROM leave_applications WHERE id=%s", (cancel_id,))
            record = cursor.fetchone()
        assert record["status"] == "cancelled", record
        step("cancel_confirmed", status=record["status"], elapsed_seconds=cancelled["elapsed_seconds"])

        start2 = next_weekday(start + timedelta(days=7))
        end2 = next_weekday(start2 + timedelta(days=1))
        draft2 = chat(employee, thread,
            f"我要提交请假申请。请假类型：事假；开始日期：{start2}；结束日期：{end2}；请假原因：{reason}。")
        remember_action(draft2, "employee_qian", "leave.submit")
        assert draft2["intent"] == "submit_leave", draft2["intent"]
        submitted2 = chat(employee, thread, "确认")
        assert submitted2["intent"] == "confirm_action", submitted2["intent"]
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM leave_applications WHERE employee_id=%s AND start_date=%s ORDER BY created_at DESC LIMIT 1", ("employee_qian", start2))
            record = cursor.fetchone()
        app_id = record["id"] if record else None
        assert record and record["status"] == "pending", record
        report["application_id"] = app_id
        step("second_leave_confirmed", application_id=app_id, status=record["status"], elapsed_seconds=submitted2["elapsed_seconds"])

        forbidden = chat(employee, thread, f"通过 {app_id}。")
        assert forbidden["intent"] == "approve", forbidden["intent"]
        assert forbidden.get("pending_action") is None, forbidden
        assert any(word in forbidden["response"] for word in ("权限", "主管", "本人")), forbidden["response"]
        step("employee_approval_blocked", elapsed_seconds=forbidden["elapsed_seconds"])

        pending = chat(manager, manager_thread, "查看待审批申请。")
        assert pending["intent"] == "query_pending", pending["intent"]
        assert app_id in pending["response"], pending["response"]
        assert cancel_id not in pending["response"], pending["response"]
        step("pending_list_visible", application_visible=True, elapsed_seconds=pending["elapsed_seconds"])

        approval_draft = chat(manager, manager_thread, f"通过 {app_id}，审批意见：本机运行验收测试批准。")
        remember_action(approval_draft, "manager_zhang", "leave.approve")
        assert approval_draft["intent"] == "approve", approval_draft["intent"]
        assert approval_draft.get("pending_action", {}).get("action_type") == "approve", approval_draft
        with conn.cursor() as cursor:
            cursor.execute("SELECT status FROM leave_applications WHERE id=%s", (app_id,))
            assert cursor.fetchone()["status"] == "pending"
        step("approval_draft_requires_confirmation", elapsed_seconds=approval_draft["elapsed_seconds"])

        approved = chat(manager, manager_thread, "确认")
        assert approved["intent"] == "confirm_action", approved["intent"]
        with conn.cursor() as cursor:
            cursor.execute("SELECT status,manager_approval FROM leave_applications WHERE id=%s", (app_id,))
            record = cursor.fetchone()
        assert record["status"] == "approved" and record["manager_approval"] == "approved", record
        step("manager_approval_confirmed_and_persisted", status=record["status"], elapsed_seconds=approved["elapsed_seconds"])

        mine = chat(employee, thread, "查询我的请假申请。")
        assert mine["intent"] == "query_status", mine["intent"]
        assert app_id in mine["response"], mine["response"]
        step("employee_reads_approved_record", elapsed_seconds=mine["elapsed_seconds"])

        history = api("/api/chat/history?thread_id=" + thread, employee)["data"]["messages"]
        assert len(history) >= 10
        step("redis_history_roundtrip", message_count=len(history))

        report["result"] = "pass"
    except Exception as exc:
        report.update(result="failed", error_type=type(exc).__name__, error=str(exc)[:300])
        raise
    finally:
        cleanup_errors, deleted_sessions, deleted_cache_rows = [], [], 0
        try:
            with conn.cursor() as cursor:
                owned_record_ids = {app_id, cancel_id}
                # A confirmed write may commit before an SSE connection drops.
                # Recover only this run's submissions by their exact action IDs.
                for owner, endpoint, action_id in owned_actions:
                    if endpoint == "leave.submit":
                        cursor.execute(
                            "SELECT id FROM leave_applications WHERE employee_id=%s AND idempotency_key=%s",
                            (owner, action_id),
                        )
                        owned_record_ids.update(row["id"] for row in cursor.fetchall())
                for record_id in owned_record_ids:
                    if record_id:
                        cursor.execute("DELETE FROM approval_audit WHERE application_id=%s", (record_id,))
                        cursor.execute("DELETE FROM leave_applications WHERE id=%s", (record_id,))
                for owner, endpoint, action_id in owned_actions:
                    cursor.execute(
                        "DELETE FROM idempotency_keys WHERE idempotency_key IN (%s, %s) AND user_id=%s AND endpoint=%s",
                        (action_id, _idempotency_storage_key(action_id, owner, endpoint), owner, endpoint),
                    )
                    deleted_cache_rows += cursor.rowcount
        except Exception as exc:
            cleanup_errors.append({"kind": "database", "error_type": type(exc).__name__})
        finally:
            conn.close()
        for owner_token, owned_thread in ((employee, thread), (manager, manager_thread)):
            if owner_token and owned_thread:
                try:
                    api("/api/chat/session/" + owned_thread, owner_token, method="DELETE")
                    deleted_sessions.append(owned_thread)
                except Exception as exc:
                    cleanup_errors.append({"kind": "session", "thread_id": owned_thread, "error_type": type(exc).__name__})
        report["cleanup"] = {"deleted_sessions": deleted_sessions, "idempotency_actions": len(owned_actions),
                             "deleted_cache_rows": deleted_cache_rows, "errors": cleanup_errors}
        cleanup_failed_after_pass = bool(cleanup_errors and report.get("result") == "pass")
        if cleanup_failed_after_pass:
            report.update(result="failed", error_type="CleanupFailed", error="Synthetic verification cleanup failed")
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if cleanup_failed_after_pass:
            raise RuntimeError("Synthetic verification cleanup failed; see verification.json")
    print("END_TO_END_PASS", flush=True)


if __name__ == "__main__":
    main()
