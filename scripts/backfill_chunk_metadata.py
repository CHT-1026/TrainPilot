"""回填块级页码/锚点/父子关系与多表示索引，不重新调用向量服务。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe scripts\\backfill_chunk_metadata.py --dry-run
    .venv312\\Scripts\\python.exe scripts\\backfill_chunk_metadata.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from app.database.connection import get_redis_client  # noqa: E402
from app.infra.mysql import connection, transaction  # noqa: E402
from enterprise_rag import config  # noqa: E402
from enterprise_rag.chunking import chunk_document  # noqa: E402
from enterprise_rag.ingest import load_manifest  # noqa: E402
from enterprise_rag.loaders import load_document  # noqa: E402
from enterprise_rag.store import _representation_text  # noqa: E402


def _existing_ids(cursor, namespace: str) -> dict[str, str]:
    cursor.execute(
        "SELECT c.id AS chunk_id FROM kb_chunks c JOIN kb_documents d ON c.doc_id = d.id "
        "WHERE d.namespace = %s",
        (namespace,),
    )
    return {row["chunk_id"] for row in cursor.fetchall()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--namespace", default=config.RAG_NAMESPACE)
    args = parser.parse_args()

    records = load_manifest()
    result = {"documents": 0, "chunks": 0, "missing_chunks": 0, "dry_run": args.dry_run,
              "namespace": args.namespace}
    missing = []
    with connection() as conn:
        with conn.cursor() as cursor:
            existing = _existing_ids(cursor, args.namespace)
    for record in records:
        path = config.PACKAGE_ROOT / "corpus" / record["path"]
        if not path.exists():
            continue
        loaded = load_document(path)
        chunks = chunk_document({"doc_id": record["doc_id"], "title": record["title"]},
                                loaded["blocks"])
        result["documents"] += 1
        result["chunks"] += len(chunks)
        if args.dry_run:
            continue
        with transaction() as conn:
            with conn.cursor() as cursor:
                for chunk in chunks:
                    if chunk["chunk_id"] not in existing:
                        missing.append(chunk["chunk_id"])
                        continue
                    metadata = {
                        "page": chunk.get("page"),
                        "paragraph": chunk.get("paragraph"),
                        "anchor": chunk.get("anchor", ""),
                        "extractor_version": chunk.get("extractor_version", ""),
                        "ocr": bool(chunk.get("ocr", False)),
                        "parent_id": chunk.get("parent_id", ""),
                        "prev_id": chunk.get("prev_id", ""),
                        "next_id": chunk.get("next_id", ""),
                        "source_hash": record.get("checksum", ""),
                    }
                    cursor.execute(
                        "UPDATE kb_chunks SET metadata_json = %s, representation = %s WHERE id = %s",
                        (json.dumps(metadata, ensure_ascii=False), _representation_text(chunk),
                         chunk["chunk_id"]),
                    )
    result["missing_chunks"] = len(missing)
    if missing:
        result["missing_sample"] = missing[:10]
    if not args.dry_run:
        try:
            get_redis_client().set(f"rag_index_version:{args.namespace}",
                                   __import__("datetime").datetime.now().isoformat())
        except Exception:
            pass
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
