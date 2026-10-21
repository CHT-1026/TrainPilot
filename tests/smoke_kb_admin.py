"""知识库文档管理冒烟：上下线、可见性、部门过滤与索引刷新。"""
import json
import os
import sys
import urllib.error
import urllib.request
from uuid import uuid4

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env.local"), override=True)

from app.capabilities.retrieval.index import reload_index  # noqa: E402
from app.capabilities.retrieval.search import retrieve  # noqa: E402
from app.database.connection import get_sync_db_connection  # noqa: E402
from app.domain.user.models import UserContext  # noqa: E402

BASE = os.getenv("OA_BASE", "http://127.0.0.1:8080")


def api(path, token=None, payload=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with urllib.request.urlopen(
        urllib.request.Request(BASE + path, data=data, headers=headers, method=method), timeout=60
    ) as response:
        return json.load(response)


def login(username):
    return api("/api/auth/login", payload={"username": username, "password": "admin123"})["data"]["token"]


def hits(query, user):
    return [item["doc_id"] for item in retrieve(query, user, top_k=5)]


def main():
    admin = login("admin")
    employee = login("employee_qian")
    employee_ctx = UserContext(user_id="employee_qian", name="钱工", role="employee", department="技术部")
    manager_ctx = UserContext(user_id="manager_zhang", name="张主管", role="manager", department="技术部")
    finance_ctx = UserContext(user_id="manager_li", name="李经理", role="manager", department="财务部")

    doc_id = "doc_smoke_" + uuid4().hex[:8]
    chunk_id = doc_id + "#c000"
    token = "zzsmoketoken"

    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO kb_documents
                    (id, title, source_path, doc_type, version, effective_date,
                     status, visibility, department, checksum)
                VALUES (%s, %s, 'smoke', 'txt', '2099', '2099-01-01',
                        'active', 'all', NULL, 'smoke')
                """,
                (doc_id, "烟雾测试制度"),
            )
            cursor.execute(
                """
                INSERT INTO kb_chunks (id, doc_id, section, ord, text, token_count)
                VALUES (%s, %s, '一、测试', 0, %s, 0)
                """,
                (chunk_id, doc_id, f"烟雾测试制度 · 一、测试\n{token} 特别规定：测试期间每天补贴 123 元。"),
            )
        conn.commit()
    finally:
        conn.close()

    try:
        try:
            api("/api/kb/documents", employee)
            raise AssertionError("employee should not list documents")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403, exc.code
            print("employee documents blocked")

        documents = api("/api/kb/documents", admin)["data"]
        assert any(item["id"] == doc_id for item in documents), documents
        print("documents list ok", len(documents))

        api("/api/kb/reindex", admin, method="POST")
        reload_index()
        assert doc_id in hits(token, employee_ctx)
        print("active doc retrievable")

        api(f"/api/kb/documents/{doc_id}", admin, {"status": "superseded"}, method="PATCH")
        reload_index()
        assert doc_id not in hits(token, employee_ctx)
        print("superseded doc hidden")

        api(
            f"/api/kb/documents/{doc_id}",
            admin,
            {"status": "active", "visibility": "manager"},
            method="PATCH",
        )
        reload_index()
        assert doc_id not in hits(token, employee_ctx)
        assert doc_id in hits(token, manager_ctx)
        print("visibility manager enforced")

        api(
            f"/api/kb/documents/{doc_id}",
            admin,
            {"status": "active", "visibility": "all", "department": "财务部"},
            method="PATCH",
        )
        reload_index()
        assert doc_id not in hits(token, employee_ctx)
        assert doc_id in hits(token, finance_ctx)
        print("department filter enforced")

        print("KB_ADMIN_PASS")
    finally:
        conn = get_sync_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM kb_chunks WHERE doc_id=%s", (doc_id,))
                cursor.execute("DELETE FROM kb_documents WHERE id=%s", (doc_id,))
            conn.commit()
        finally:
            conn.close()
        try:
            api("/api/kb/reindex", admin, method="POST")
        except Exception:
            pass


if __name__ == "__main__":
    main()
