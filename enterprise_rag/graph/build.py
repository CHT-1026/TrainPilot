"""图谱构建 CLI：全量/增量抽取、版本备份与质量统计。"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from enterprise_rag import config  # noqa: E402
from enterprise_rag.graph.extract import extract_entities, extract_relations  # noqa: E402
from enterprise_rag.graph.quality import quality_report  # noqa: E402
from enterprise_rag.graph.store import GraphStore  # noqa: E402
from enterprise_rag.store import load_index  # noqa: E402

GRAPH_DIR = ROOT / "enterprise_rag" / "graph" / "data"
GRAPH_PATH = GRAPH_DIR / "graph.json"
META_PATH = GRAPH_DIR / "graph_meta.json"
VERSIONS_DIR = GRAPH_DIR / "versions"

CORE_PERSONS = {
    "顾承澜", "沈亦舟", "周敏仪", "韩立恒", "罗嘉树", "陈依然", "潘文博", "邵岚",
    "程知远", "周启航", "林晓", "赵一鸣", "唐婧", "许博", "陈璐", "方磊", "郭涛",
    "蒋雯", "文梓萱", "葛俊", "孟凡", "林可", "苏婉清", "陆绍",
}


def _manifest_checksums() -> dict[str, str]:
    path = config.MANIFEST_PATH
    checksums = {}
    if not path.exists():
        return checksums
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        checksums[record["doc_id"]] = record.get("checksum", "")
    return checksums


def _backup_current() -> None:
    if not GRAPH_PATH.exists():
        return
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    shutil.copy2(GRAPH_PATH, VERSIONS_DIR / f"graph_{stamp}.json")
    backups = sorted(VERSIONS_DIR.glob("graph_*.json"))
    for old in backups[:-5]:
        old.unlink(missing_ok=True)


def build_graph(force: bool = False, incremental: bool = True, namespace: str | None = None) -> dict:
    index = load_index(namespace=namespace)
    checksums = _manifest_checksums()
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    previous_checksums = meta.get("doc_checksums", {})
    store = GraphStore() if force or not GRAPH_PATH.exists() else GraphStore.load(GRAPH_PATH)

    changed_docs = {
        doc_id for doc_id, checksum in checksums.items()
        if force or not incremental or previous_checksums.get(doc_id) != checksum
    }
    known_docs = {doc_id for chunk in store.relations.values() for doc_id in [chunk.get("doc_id")] if doc_id}
    for chunk in index.chunks:
        if chunk["doc_id"] not in known_docs:
            changed_docs.add(chunk["doc_id"])

    if changed_docs and not force and incremental:
        for doc_id in changed_docs:
            store.remove_doc(doc_id)

    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk in index.chunks:
        chunks_by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    known_persons = set(CORE_PERSONS)
    for chunk in index.chunks:
        author = (chunk.get("author") or "").strip()
        if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", author):
            known_persons.add(author)

    processed_chunks = 0
    for doc_id in sorted(changed_docs):
        for chunk in chunks_by_doc.get(doc_id, []):
            entities = extract_entities(chunk, known_persons)
            relations = extract_relations(chunk, entities)
            for entity in entities:
                store.add_entity(entity)
            for relation in relations:
                store.add_relation(relation)
            processed_chunks += 1

    # Cross-document version edges.
    family_docs: dict[str, list[dict]] = {}
    for doc_id, chunks in chunks_by_doc.items():
        if not chunks:
            continue
        first = chunks[0]
        family = first.get("doc_family") or ""
        if not family:
            continue
        family_docs.setdefault(family, []).append({
            "doc_id": doc_id,
            "title": first.get("title", ""),
            "version": first.get("version", ""),
            "effective_date": first.get("effective_date") or "9999",
            "chunk_id": first.get("chunk_id", ""),
            "section": first.get("section", ""),
        })
    from enterprise_rag.graph.align import canonical_id
    from enterprise_rag.graph.schema import relation_payload

    for family, docs in family_docs.items():
        docs.sort(key=lambda item: (str(item["effective_date"]), item["version"]))
        for older, newer in zip(docs, docs[1:]):
            store.add_relation(relation_payload(
                canonical_id("document", newer["title"]),
                canonical_id("document", older["title"]),
                "supersedes",
                doc_id=newer["doc_id"],
                chunk_id=newer["chunk_id"],
                section=newer["section"],
                evidence=f"版本演进：{older['title']}（{older['version']}）→ {newer['title']}（{newer['version']}）",
                confidence=1.0,
            ))

    store.version = datetime.now(timezone(timedelta(hours=8))).isoformat()
    pruned_dangling = store.prune_dangling()
    _backup_current()
    store.save(GRAPH_PATH)
    stats = store.stats()
    quality = quality_report(store, chunks_by_doc)
    META_PATH.write_text(json.dumps({
        "version": store.version,
        "stats": stats,
        "quality": quality,
        "doc_checksums": checksums,
        "changed_docs": sorted(changed_docs),
        "processed_chunks": processed_chunks,
        "pruned_dangling_relations": pruned_dangling,
        "namespace": namespace or config.RAG_NAMESPACE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stats": stats, "quality": quality, "changed_docs": sorted(changed_docs),
            "processed_chunks": processed_chunks, "pruned_dangling_relations": pruned_dangling,
            "graph_path": str(GRAPH_PATH)}


def main() -> None:
    parser = argparse.ArgumentParser(description="构建澜图企业知识图谱")
    parser.add_argument("--force", action="store_true", help="全量重建")
    parser.add_argument("--full", action="store_true", help="等同 --force")
    parser.add_argument("--no-incremental", action="store_true", help="关闭增量")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--list-versions", action="store_true", help="列出可回滚的图谱版本")
    parser.add_argument("--rollback", default="", help="回滚到指定备份文件名")
    args = parser.parse_args()
    if args.list_versions:
        versions = sorted(VERSIONS_DIR.glob("graph_*.json"))
        print(json.dumps([item.name for item in versions], ensure_ascii=False))
        return
    if args.rollback:
        source = VERSIONS_DIR / args.rollback
        if not source.exists():
            raise SystemExit(f"备份不存在：{source}")
        shutil.copy2(source, GRAPH_PATH)
        print(json.dumps({"rolled_back": source.name, "graph_path": str(GRAPH_PATH)}, ensure_ascii=False))
        return
    result = build_graph(force=args.force or args.full,
                         incremental=not args.no_incremental,
                         namespace=args.namespace)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
