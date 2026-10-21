"""请假数据访问层：所有 SQL 集中在这里，服务层只处理规则与事务。"""
import json
import hashlib
from datetime import date
from typing import Any

from app.domain.leave.state_machine import STEP_DONE, STEP_MANAGER, STATUS_PENDING
from app.domain.user.models import UserContext


def get_application(conn, app_id: str, *, for_update: bool = False) -> dict[str, Any] | None:
    sql = "SELECT * FROM leave_applications WHERE id = %s"
    if for_update:
        sql += " FOR UPDATE"
    with conn.cursor() as cursor:
        cursor.execute(sql, (app_id,))
        return cursor.fetchone()


def find_overlap(
    conn,
    employee_id: str,
    start_date: date,
    end_date: date,
    *,
    exclude_id: str | None = None,
) -> dict[str, Any] | None:
    sql = """
        SELECT id, start_date, end_date, leave_type
        FROM leave_applications
        WHERE employee_id = %s
          AND status NOT IN ('rejected', 'cancelled', 'finished')
          AND start_date <= %s
          AND end_date >= %s
    """
    params: list[Any] = [employee_id, end_date, start_date]
    if exclude_id:
        sql += " AND id <> %s"
        params.append(exclude_id)
    sql += " ORDER BY created_at DESC LIMIT 1"
    with conn.cursor() as cursor:
        cursor.execute(sql, tuple(params))
        return cursor.fetchone()


def insert_application(
    conn,
    *,
    app_id: str,
    employee_id: str,
    employee_name: str,
    employee_role: str,
    manager_id: str | None,
    department: str | None,
    leave_type: str,
    leave_days: int,
    start_date: date,
    end_date: date,
    reason: str,
    idempotency_key: str | None,
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO leave_applications
                (id, employee_id, employee_name, employee_role, manager_id, department,
                 leave_type, leave_days, start_date, end_date, reason, status,
                 manager_approval, hr_approval, approval_step, version,
                 idempotency_key, submitted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    'pending', 'pending', %s, 0, %s, NOW())
            """,
            (
                app_id,
                employee_id,
                employee_name,
                employee_role,
                manager_id,
                department,
                leave_type,
                leave_days,
                start_date,
                end_date,
                reason,
                STATUS_PENDING,
                STEP_MANAGER,
                idempotency_key,
            ),
        )


def list_mine(conn, employee_id: str, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT * FROM leave_applications
            WHERE employee_id = %s AND status <> 'finished'
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (employee_id, limit, offset),
        )
        return cursor.fetchall()


def count_mine(conn, employee_id: str) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) AS total FROM leave_applications WHERE employee_id = %s AND status <> 'finished'",
            (employee_id,),
        )
        row = cursor.fetchone()
        return int(row["total"]) if row else 0


def list_pending(conn, actor: UserContext, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    base = """
        SELECT l.*, u.name AS real_name
        FROM leave_applications AS l
        LEFT JOIN users AS u ON l.employee_id = u.username
        WHERE l.status = 'pending'
    """
    params: list[Any]

    if actor.is_admin:
        sql = base + " ORDER BY l.created_at ASC LIMIT %s OFFSET %s"
        params = [limit, offset]
    elif actor.is_manager:
        sql = base + """ AND l.approval_step = 'manager'
            AND l.employee_id <> %s
            AND (l.manager_id = %s OR
                 ((l.manager_id IS NULL OR l.manager_id = '') AND l.department = %s))
            ORDER BY l.created_at ASC LIMIT %s OFFSET %s"""
        params = [actor.user_id, actor.user_id, actor.department, limit, offset]
    elif actor.is_hr:
        sql = base + " AND l.approval_step = 'hr' ORDER BY l.created_at ASC LIMIT %s OFFSET %s"
        params = [limit, offset]
    else:
        return []

    with conn.cursor() as cursor:
        cursor.execute(sql, tuple(params))
        return cursor.fetchall()


def count_pending(conn, actor: UserContext) -> int:
    base = "SELECT COUNT(*) AS total FROM leave_applications WHERE status = 'pending'"
    if actor.is_admin:
        sql, params = base, ()
    elif actor.is_manager:
        sql = base + """ AND approval_step = 'manager'
            AND employee_id <> %s
            AND (manager_id = %s OR
                 ((manager_id IS NULL OR manager_id = '') AND department = %s))"""
        params = (actor.user_id, actor.user_id, actor.department)
    elif actor.is_hr:
        sql, params = base + " AND approval_step = 'hr'", ()
    else:
        return 0
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return int(row["total"]) if row else 0


def transition_application(
    conn,
    *,
    app_id: str,
    expected_status: str,
    expected_step: str,
    expected_version: int,
    new_status: str,
    new_step: str,
    manager_approval: str,
    hr_approval: str,
    comments: str,
) -> bool:
    """带乐观锁的状态迁移；返回 False 表示状态已被其他请求改变。"""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE leave_applications
            SET status = %s,
                approval_step = %s,
                manager_approval = %s,
                hr_approval = %s,
                comments = %s,
                version = version + 1,
                updated_at = NOW()
            WHERE id = %s
              AND status = %s
              AND approval_step = %s
              AND version = %s
            """,
            (
                new_status,
                new_step,
                manager_approval,
                hr_approval,
                comments,
                app_id,
                expected_status,
                expected_step,
                expected_version,
            ),
        )
        return cursor.rowcount == 1


def cancel_application(conn, *, app_id: str, expected_version: int) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE leave_applications
            SET status = 'cancelled',
                approval_step = %s,
                cancelled_at = NOW(),
                version = version + 1,
                updated_at = NOW()
            WHERE id = %s
              AND status = 'pending'
              AND version = %s
            """,
            (STEP_DONE, app_id, expected_version),
        )
        return cursor.rowcount == 1


def insert_audit(
    conn,
    *,
    application_id: str,
    action: str,
    actor: UserContext,
    from_status: str,
    to_status: str,
    comment: str = "",
    trace_id: str = "",
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO approval_audit
                (application_id, action, actor_id, actor_role,
                 from_status, to_status, comment, trace_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                application_id,
                action,
                actor.user_id,
                actor.role,
                from_status,
                to_status,
                comment,
                trace_id,
            ),
        )


def list_audit(conn, application_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, application_id, action, actor_id, actor_role,
                   from_status, to_status, comment, trace_id, created_at
            FROM approval_audit
            WHERE application_id = %s
            ORDER BY created_at ASC, id ASC
            LIMIT %s
            """,
            (application_id, limit),
        )
        return cursor.fetchall()


def _idempotency_storage_key(key: str, user_id: str, endpoint: str) -> str:
    """Fit the existing VARCHAR(128) key while separating actors and actions."""
    scope = json.dumps([user_id, endpoint, key], ensure_ascii=False, separators=(",", ":"))
    return "v2:" + hashlib.sha256(scope.encode("utf-8")).hexdigest()


def get_idempotent(conn, key: str, *, user_id: str, endpoint: str) -> dict[str, Any] | None:
    scoped_key = _idempotency_storage_key(key, user_id, endpoint)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT response_json FROM idempotency_keys
            WHERE idempotency_key IN (%s, %s)
              AND user_id = %s AND endpoint = %s
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY (idempotency_key = %s) DESC LIMIT 1
            """,
            (scoped_key, key, user_id, endpoint, scoped_key),
        )
        row = cursor.fetchone()
        if not row:
            return None
        value = row["response_json"]
        return json.loads(value) if isinstance(value, str) else value


def save_idempotent(
    conn,
    *,
    key: str,
    user_id: str,
    endpoint: str,
    response: dict[str, Any],
    ttl_hours: int = 24,
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO idempotency_keys
                (idempotency_key, user_id, endpoint, response_json, expires_at)
            VALUES (%s, %s, %s, %s, DATE_ADD(NOW(), INTERVAL %s HOUR))
            ON DUPLICATE KEY UPDATE
                response_json = VALUES(response_json),
                expires_at = VALUES(expires_at)
            """,
            (_idempotency_storage_key(key, user_id, endpoint), user_id, endpoint,
             json.dumps(response, ensure_ascii=False, default=str), ttl_hours),
        )
