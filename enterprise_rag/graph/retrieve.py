"""图谱检索：实体链接 → 邻域扩展 → 证据块与推理路径。"""
from __future__ import annotations

import re
from pathlib import Path

from enterprise_rag.graph.store import GraphStore

GRAPH_PATH = Path(__file__).resolve().parent / "data" / "graph.json"
_cache: dict = {"mtime": None, "store": GraphStore()}


def _doc_name(store: GraphStore, doc_id: str) -> str:
    for entity in store.entities.values():
        if entity.get("type") == "document" and (entity.get("attributes") or {}).get("doc_id") == doc_id:
            return entity.get("name", "")
    return ""


def load_graph() -> GraphStore:
    if not GRAPH_PATH.exists():
        return GraphStore()
    mtime = GRAPH_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["store"] = GraphStore.load(GRAPH_PATH)
        _cache["mtime"] = mtime
    return _cache["store"]


def graph_candidates(question: str, index, hops: int = 2) -> tuple[list[dict], dict]:
    store = load_graph()
    comparison_like = bool(re.search(r"架构|版本|新旧|对比|变化|差异|演进", question))
    entities = store.find_entities(question, include_documents=comparison_like)
    if not entities:
        return [], {"entities": [], "paths": [], "candidates": 0, "enabled": store.stats()["entities"] > 0}
    entity_ids = [item["id"] for item in entities[:5]]
    refs = store.chunk_refs(entity_ids, hops=hops, question=question)

    # Document entities are expanded to their key sections: graph edges point to
    # the first chunk, but version comparisons need the detail sections too.
    document_ids: set[str] = set()
    for item in entities:
        entity = store.entity(item["id"]) or {}
        if entity.get("type") == "document":
            document_ids.add((entity.get("attributes") or {}).get("doc_id", ""))
    for relation in store.expand(entity_ids, hops=hops):
        for node in (relation["source"], relation["target"]):
            entity = store.entity(node) or {}
            if entity.get("type") == "document":
                document_ids.add((entity.get("attributes") or {}).get("doc_id", ""))
    document_ids.discard("")
    if re.search(r"组织", question):
        document_ids = {
            doc_id for doc_id in document_ids
            if "组织" in _doc_name(store, doc_id)
        }
    elif re.search(r"控制|架构图|旧架构|新架构|系统架构", question):
        document_ids = {
            doc_id for doc_id in document_ids
            if "控制" in _doc_name(store, doc_id)
        }
    section_priority = ("版本演进", "版本说明", "控制器", "可靠性", "总体架构", "伺服", "软件", "安全")
    for doc_id in document_ids:
        chunks = [chunk for chunk in index.chunks if chunk.get("doc_id") == doc_id]
        chunks.sort(key=lambda chunk: (
            next((position for position, word in enumerate(section_priority)
                  if word in chunk.get("section", "")), len(section_priority)),
            chunk.get("ord", 0),
        ))
        for chunk in chunks[:6]:
            current = refs.setdefault(chunk["chunk_id"], {"score": 0.08, "paths": []})
            current["score"] = max(current["score"], 0.08)
            current["document_expanded"] = True
            current["section_priority"] = next(
                (position for position, word in enumerate(section_priority)
                 if word in chunk.get("section", "")), len(section_priority)
            )
            if len(current["paths"]) < 2:
                current["paths"].append(f"文档实体扩展：{chunk.get('title', '')} / {chunk.get('section', '')}")

    candidates = []
    for chunk_id, payload in refs.items():
        chunk = index.by_id.get(chunk_id)
        if not chunk:
            continue
        item = dict(chunk)
        item["graph_score"] = payload["score"]
        item["graph_paths"] = payload["paths"]
        item["graph_document"] = payload.get("document_expanded", False)
        item["graph_priority"] = payload.get("section_priority", 99)
        candidates.append(item)
    candidates.sort(key=lambda item: item.get("graph_score", 0), reverse=True)
    paths = []
    for item in candidates[:8]:
        paths.extend(item.get("graph_paths", []))
    debug = {
        "entities": entities,
        "paths": list(dict.fromkeys(paths))[:12],
        "candidates": len(candidates),
        "enabled": True,
    }
    return candidates, debug
