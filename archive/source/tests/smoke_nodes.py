"""节点级冒烟测试：草稿 -> 确认提交 -> 草稿取消 -> 确认取消。"""
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env.local", override=True)

from app.database.connection import get_sync_db_connection  # noqa: E402
from app.workflows.chat_nodes.delete import create_delete_nodes  # noqa: E402
from app.workflows.chat_nodes.execute import create_execution_nodes  # noqa: E402
from app.workflows.chat_nodes.leave import create_leave_nodes  # noqa: E402


def next_weekday(start: date) -> date:
    cursor = start
    while cursor.weekday() >= 5:
        cursor += timedelta(days=1)
    return cursor


def base_state():
    return {
        "messages": [],
        "user_message": "",
        "intent": "",
        "extracted_info": {},
        "current_task": "",
        "task_result": "",
        "user": {
            "user_id": "employee_qian",
            "name": "钱工",
            "role": "employee",
            "department": "技术部",
        },
        "conversation_context": {
            "user_id": "employee_qian",
            "user_name": "钱工",
            "user_role": "employee",
            "user_department": "技术部",
            "accumulated_extracted_info": {},
        },
        "pending_action": None,
        "last_result": None,
        "iteration_count": 0,
        "turn": 1,
        "response": "",
    }


def main():
    leave_nodes = create_leave_nodes()
    execution_nodes = create_execution_nodes()
    delete_nodes = create_delete_nodes()

    start = next_weekday(date(2027, 8, 2))
    end = next_weekday(start + timedelta(days=1))
    state = base_state()
    state["extracted_info"] = {
        "leave_type": "事假",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "reason": "Phase0 node smoke test",
    }

    draft = leave_nodes["submit_leave"](state)
    assert draft["pending_action"]["action_type"] == "submit_leave", draft
    assert "确认" in draft["task_result"], draft["task_result"]
    print("draft ok", draft["pending_action"]["action_id"])

    state.update(draft)
    executed = execution_nodes["execute_pending"](state)
    assert executed["last_result"]["ok"], executed
    app_id = executed["last_result"]["data"]["application"]["app_id"]
    print("submit confirmed", app_id)

    state.update(executed)
    state["extracted_info"] = {"app_id": app_id}
    cancel_draft = delete_nodes["delete"](state)
    assert cancel_draft["pending_action"]["action_type"] == "cancel_leave", cancel_draft
    print("cancel draft ok", cancel_draft["pending_action"]["action_id"])

    state.update(cancel_draft)
    cancelled = execution_nodes["execute_pending"](state)
    assert cancelled["last_result"]["ok"], cancelled
    assert cancelled["last_result"]["data"]["application"]["status"] == "cancelled", cancelled
    print("cancel confirmed", cancelled["last_result"]["data"]["application"]["status"])

    cleanup(app_id)
    print("NODE_SMOKE_PASS")


def cleanup(app_id):
    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM approval_audit WHERE application_id=%s", (app_id,))
            cursor.execute("DELETE FROM leave_applications WHERE id=%s", (app_id,))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
