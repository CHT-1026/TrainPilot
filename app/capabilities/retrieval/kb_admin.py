"""知识库文档管理：列表、上下线、可见性与生效日期。"""
from datetime import date, datetime
from typing import Any

from app.infra.mysql import connection, transaction

ALLOWED_STATUS = {"active", "superseded"}
ALLOWED_VISIBILITY = {"all", "manager", "hr", "finance", "admin"}
UPDATABLE_FIELDS = ("status", "visibility", "department", "effective_date")


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in ("effective_date", "updated_at"):
        value = result.get(key)
        if isinstance(value, (date, datetime)):
            result[key] = value.isoformat()
    result["chunks"] = int(result.get("chunks") or 0)
    return result


def _get_document(doc_id: str, namespace: str = "default") -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT d.id, d.namespace, d.title, d.source_path, d.doc_type, d.version,
                       d.effective_date, d.status, d.visibility, d.department,
                       d.updated_at, COUNT(c.id) AS chunks
                FROM kb_documents AS d
                LEFT JOIN kb_chunks AS c ON c.doc_id = d.id
                WHERE d.id = %s AND d.namespace = %s
                GROUP BY d.id
                """,
                (doc_id, namespace),
            )
            row = cursor.fetchone()
    return _serialize(row) if row else None


def list_documents(namespace: str = "default") -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT d.id, d.namespace, d.title, d.source_path, d.doc_type, d.version,
                       d.effective_date, d.status, d.visibility, d.department,
                       d.updated_at, COUNT(c.id) AS chunks
                FROM kb_documents AS d
                LEFT JOIN kb_chunks AS c ON c.doc_id = d.id
                WHERE d.namespace = %s
                GROUP BY d.id
                ORDER BY d.title
                """, (namespace,),
            )
            rows = cursor.fetchall()
    return [_serialize(row) for row in rows]


def update_document(doc_id: str, fields: dict[str, Any], namespace: str = "default") -> dict[str, Any] | None:
    if not fields:
        raise ValueError("没有可更新的字段")
    unknown = set(fields) - set(UPDATABLE_FIELDS)
    if unknown:
        raise ValueError("不支持更新字段: " + ", ".join(sorted(unknown)))
    if "status" in fields and fields["status"] not in ALLOWED_STATUS:
        raise ValueError("status 只能是 active 或 superseded")
    if "visibility" in fields and fields["visibility"] not in ALLOWED_VISIBILITY:
        raise ValueError("visibility 取值不合法")

    assignments = []
    params = []
    for key in UPDATABLE_FIELDS:
        if key in fields:
            assignments.append(f"{key} = %s")
            params.append(fields[key])
    params.append(doc_id)
    params.append(namespace)

    with transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"UPDATE kb_documents SET {', '.join(assignments)} WHERE id = %s AND namespace = %s", tuple(params)
            )
    # No-op updates are still valid; the scoped read distinguishes missing rows.
    return _get_document(doc_id, namespace)
