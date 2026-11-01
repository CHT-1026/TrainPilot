"""Read the authoritative document access revision; no Redis success fallback.

The corpus is small, so one metadata query per request is preferable to serving
revoked ACLs indefinitely when an invalidation notification is missed.
"""
import hashlib
import json

from app.infra.mysql import connection


class IndexRevisionUnavailable(RuntimeError):
    """The authoritative index identity cannot be obtained."""


def access_revision(namespace: str) -> str:
    try:
        with connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, status, visibility, department, effective_date, doc_family, version "
                    "FROM kb_documents WHERE namespace = %s ORDER BY id", (namespace,),
                )
                rows = cursor.fetchall()
    except Exception as exc:
        raise IndexRevisionUnavailable("知识文档权限元数据暂不可用") from exc
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
