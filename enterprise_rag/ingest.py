"""企业语料入库 CLI：解析多格式 → 结构切分 → 向量化 → 写入 MySQL → 刷新索引版本。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe -m enterprise_rag.ingest
    .venv312\\Scripts\\python.exe -m enterprise_rag.ingest --limit 10
    .venv312\\Scripts\\python.exe -m enterprise_rag.ingest --keep
"""
import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env.local", override=False)
load_dotenv(BACKEND_ROOT / ".env", override=False)

from app.database.connection import get_redis_client  # noqa: E402
from app.infra.mysql import transaction  # noqa: E402
from enterprise_rag import config  # noqa: E402
from enterprise_rag.chunking import chunk_document  # noqa: E402
from enterprise_rag.embedding import embed_texts, to_json  # noqa: E402
from enterprise_rag.loaders import load_document  # noqa: E402

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("doc_id", "title", "department", "author", "created_at", "updated_at", "version")


def load_manifest(path: Path | None = None) -> list[dict]:
    path = path or config.MANIFEST_PATH
    if not path.exists():
        raise RuntimeError(f"manifest 不存在：{path}，请先运行 python -m enterprise_rag.corpus_gen")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_datetime(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except (TypeError, ValueError):
            continue
    return datetime.now()


def _parse_date(value: str | None):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _upsert_document(conn, record: dict, path: Path) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO kb_documents
                (id, namespace, title, source_path, doc_type, format, version, effective_date,
                 status, visibility, department, author, doc_family, category, checksum,
                 doc_created_at, doc_updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                namespace = VALUES(namespace),
                title = VALUES(title),
                source_path = VALUES(source_path),
                doc_type = VALUES(doc_type),
                format = VALUES(format),
                version = VALUES(version),
                effective_date = VALUES(effective_date),
                status = 'active',
                visibility = VALUES(visibility),
                department = VALUES(department),
                author = VALUES(author),
                doc_family = VALUES(doc_family),
                category = VALUES(category),
                checksum = VALUES(checksum),
                doc_created_at = VALUES(doc_created_at),
                doc_updated_at = VALUES(doc_updated_at)
            """,
            (
                record["doc_id"],
                record.get("namespace", config.RAG_NAMESPACE),
                record["title"],
                record.get("path") or str(path),
                record.get("format"),
                record.get("format"),
                record.get("version"),
                _parse_date(record.get("effective_date")),
                record.get("visibility", "all"),
                record.get("department"),
                record.get("author"),
                record.get("doc_family"),
                record.get("category"),
                record.get("checksum"),
                _parse_datetime(record.get("created_at", "")),
                _parse_datetime(record.get("updated_at", "")),
            ),
        )


def _replace_chunks(conn, doc_id: str, chunks: list[dict]) -> None:
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM kb_chunks WHERE doc_id = %s", (doc_id,))
        if not chunks:
            return
        cursor.executemany(
            """
            INSERT INTO kb_chunks
                (id, doc_id, section, chunk_type, ord, text, token_count,
                 embedding_json, embedding_dim, embedding_model, embedding_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk["chunk_id"],
                    chunk["doc_id"],
                    chunk["section"],
                    chunk["chunk_type"],
                    chunk["ord"],
                    chunk["text"],
                    chunk["token_count"],
                    chunk["embedding_json"],
                    config.EMBEDDING_DIM,
                    config.EMBEDDING_MODEL,
                    config.EMBEDDING_VERSION,
                )
                for chunk in chunks
            ],
        )


def _clear_namespace(namespace: str) -> None:
    with transaction() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM kb_chunks WHERE doc_id IN (SELECT id FROM kb_documents WHERE namespace = %s)", (namespace,))
            cursor.execute("DELETE FROM kb_documents WHERE namespace = %s", (namespace,))


def ingest(limit: int | None = None, clean: bool = True, namespace: str | None = None) -> dict:
    namespace = namespace or config.RAG_NAMESPACE
    records = load_manifest()
    if limit:
        records = records[:limit]
    if clean and not limit:
        _clear_namespace(namespace)

    documents = 0
    chunks_total = 0
    failures = []
    for index, record in enumerate(records, 1):
        missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
        if missing:
            failures.append({"doc_id": record.get("doc_id"), "error": f"缺少元数据 {missing}"})
            continue
        path = config.PACKAGE_ROOT / "corpus" / record["path"]
        if not path.exists():
            failures.append({"doc_id": record["doc_id"], "error": f"文件不存在 {path}"})
            continue
        try:
            loaded = load_document(path)
            document = {
                "doc_id": record["doc_id"],
                "title": record["title"],
                "text": "\n".join(block["text"] for block in loaded["blocks"]),
            }
            chunks = chunk_document(document, loaded["blocks"])
            if not chunks:
                raise RuntimeError("切分结果为空")

            texts = [chunk["text"] for chunk in chunks]
            vectors = []
            for start in range(0, len(texts), config.EMBEDDING_BATCH):
                vectors.extend(embed_texts(texts[start : start + config.EMBEDDING_BATCH]))
            for chunk, vector in zip(chunks, vectors):
                chunk["embedding_json"] = to_json(vector)

            with transaction() as conn:
                _upsert_document(conn, record, path)
                _replace_chunks(conn, record["doc_id"], chunks)
            documents += 1
            chunks_total += len(chunks)
            if index % 10 == 0:
                logger.info("已入库 %s/%s，当前块数 %s", index, len(records), chunks_total)
        except Exception as exc:
            logger.error("入库失败 %s: %s", record.get("doc_id"), exc, exc_info=True)
            failures.append({"doc_id": record.get("doc_id"), "error": str(exc)[:200]})

    version = datetime.now().isoformat()
    try:
        get_redis_client().set(f"rag_index_version:{namespace}", version)
    except Exception as exc:
        logger.warning("刷新索引版本失败: %s", exc)
        version = ""

    summary = {
        "namespace": namespace,
        "documents": documents,
        "chunks": chunks_total,
        "manifest_records": len(records),
        "failures": failures,
        "index_version": version,
        "embedding_model": config.EMBEDDING_MODEL,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="澜图企业语料入库")
    parser.add_argument("--limit", type=int, default=0, help="仅入库前 N 篇，用于分批抽样验收")
    parser.add_argument("--keep", action="store_true", help="保留既有文档，仅增量覆盖")
    parser.add_argument("--namespace", default=None, help="命名空间，默认 lantu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    summary = ingest(limit=args.limit or None, clean=not args.keep, namespace=args.namespace)
    if summary["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
