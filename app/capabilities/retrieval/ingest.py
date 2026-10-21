"""知识库入库 CLI：解析 -> 切分 -> 写入 MySQL -> 刷新索引版本。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe -m app.capabilities.retrieval.ingest
可用 KB_DOCS_DIR 指定文档目录。
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from app.capabilities.retrieval.chunker import chunk_document  # noqa: E402
from app.capabilities.retrieval.parser import parse_document  # noqa: E402
from app.database.connection import get_redis_client  # noqa: E402
from app.infra.mysql import transaction  # noqa: E402

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".docx", ".pdf"}


def default_docs_dir() -> Path:
    configured = os.getenv("KB_DOCS_DIR")
    if configured:
        return Path(configured)
    local = ROOT / "data" / "oa_docs"
    if local.exists():
        return local
    return ROOT.parent / "enterprise-kb-qa" / "data" / "docs"


def _upsert_document(conn, document: dict) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO kb_documents
                (id, title, source_path, doc_type, version, effective_date,
                 status, visibility, department, checksum)
            VALUES (%s, %s, %s, %s, %s, %s, 'active', 'all', NULL, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                source_path = VALUES(source_path),
                doc_type = VALUES(doc_type),
                version = VALUES(version),
                effective_date = VALUES(effective_date),
                checksum = VALUES(checksum)
            """,
            (
                document["doc_id"],
                document["title"],
                document["source_path"],
                document["doc_type"],
                document["version"],
                document["effective_date"],
                document["checksum"],
            ),
        )


def _replace_chunks(conn, document: dict, chunks: list[dict]) -> None:
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM kb_chunks WHERE doc_id = %s", (document["doc_id"],))
        if not chunks:
            return
        cursor.executemany(
            """
            INSERT INTO kb_chunks
                (id, doc_id, section, ord, text, token_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk["chunk_id"],
                    chunk["doc_id"],
                    chunk["section"],
                    chunk["ord"],
                    chunk["text"],
                    chunk["token_count"],
                )
                for chunk in chunks
            ],
        )


def ingest(docs_dir: Path | None = None) -> dict:
    docs_dir = docs_dir or default_docs_dir()
    if not docs_dir.exists():
        raise RuntimeError(f"知识库目录不存在: {docs_dir}")

    files = sorted(
        path for path in docs_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = 0
    chunks_total = 0
    failures = []

    for path in files:
        try:
            document = parse_document(path)
            chunks = chunk_document(document)
            with transaction() as conn:
                _upsert_document(conn, document)
                _replace_chunks(conn, document, chunks)
            documents += 1
            chunks_total += len(chunks)
            logger.info("已入库 %s -> %s 块", path.name, len(chunks))
        except Exception as exc:
            logger.error("入库失败 %s: %s", path.name, exc, exc_info=True)
            failures.append({"file": path.name, "error": str(exc)[:200]})

    version = datetime.now().isoformat()
    try:
        client = get_redis_client()
        client.set("kb_index_version", version)
        client.set("kb_index_version:default", version)
    except Exception as exc:
        logger.warning("刷新索引版本失败: %s", exc)
        version = ""

    summary = {
        "documents": documents,
        "chunks": chunks_total,
        "files": len(files),
        "failures": failures,
        "index_version": version,
        "docs_dir": str(docs_dir),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ingest()
