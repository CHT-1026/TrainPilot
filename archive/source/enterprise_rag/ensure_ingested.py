"""确保企业语料已完成入库（供便携启动器调用）。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe -m enterprise_rag.ensure_ingested
    .venv312\\Scripts\\python.exe -m enterprise_rag.ensure_ingested --force
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env.local", override=False)
load_dotenv(BACKEND_ROOT / ".env", override=False)

from app.database.connection import get_sync_db_connection  # noqa: E402
from enterprise_rag import config  # noqa: E402
from enterprise_rag.ingest import ingest, load_manifest  # noqa: E402


def current_count(namespace: str) -> tuple[int, int]:
    conn = get_sync_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS documents FROM kb_documents WHERE namespace = %s",
                (namespace,),
            )
            documents = int(cursor.fetchone()["documents"])
            cursor.execute(
                """
                SELECT COUNT(*) AS chunks FROM kb_chunks AS c
                JOIN kb_documents AS d ON c.doc_id = d.id
                WHERE d.namespace = %s
                """,
                (namespace,),
            )
            chunks = int(cursor.fetchone()["chunks"])
            return documents, chunks
    finally:
        conn.close()


def ensure_oa_docs(force: bool = False) -> dict:
    """OA 制度库（default 命名空间）随包携带，首次启动自动入库。"""
    from app.capabilities.retrieval.ingest import default_docs_dir, ingest as ingest_oa

    documents, chunks = current_count("default")
    if force or documents == 0:
        docs_dir = default_docs_dir()
        summary = ingest_oa(docs_dir)
        return {
            "action": "ingested",
            "docs_dir": str(docs_dir),
            "documents": summary["documents"],
            "chunks": summary["chunks"],
            "failures": summary["failures"],
        }
    return {"action": "skipped", "documents": documents, "chunks": chunks}


def main():
    parser = argparse.ArgumentParser(description="检查并按需入库企业语料与 OA 制度库")
    parser.add_argument("--force", action="store_true", help="强制全量重建 lantu 语料")
    parser.add_argument("--force-oa", action="store_true", help="强制全量重建 default 制度库")
    args = parser.parse_args()

    expected = len(load_manifest())
    documents, chunks = current_count(config.RAG_NAMESPACE)
    if args.force or documents == 0:
        summary = ingest(clean=args.force)
        enterprise = {
            "action": "ingested",
            "expected_documents": expected,
            "documents": summary["documents"],
            "chunks": summary["chunks"],
            "failures": summary["failures"],
        }
    else:
        enterprise = {
            "action": "skipped",
            "expected_documents": expected,
            "documents": documents,
            "chunks": chunks,
        }
    oa = ensure_oa_docs(force=args.force_oa)
    print(json.dumps({"enterprise": enterprise, "oa": oa}, ensure_ascii=False))


if __name__ == "__main__":
    main()
