"""图谱检索：实体链接 → 邻域扩展 → 证据块与推理路径。

所有实体、关系、文档扩展与路径都必须经过同一个 EvidenceScope：
只从本次合法的 chunk_id 重建图视图，非法关系既不会返回，也不会被当作桥接边。
"""
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


def _entity_legal(store: GraphStore, entity: dict, legal_docs: set[str]) -> bool:
    if entity.get("type") == "document":
        doc_id = (entity.get("attributes") or {}).get("doc_id", "")
        return bool(doc_id) and doc_id in legal_docs
    cited = set(entity.get("doc_ids") or [])
    return bool(cited & legal_docs)


def _alias_legal(match: dict, entity: dict, index, legal_docs: set[str]) -> bool:
    """别名命中必须能在本次合法文档原文中找到，避免受限来源的别名授权公共实体。"""
    matched = str(match.get("matched") or "")
    if not matched or matched == str(entity.get("name") or ""):
        return True
    return any(
        matched in str(chunk.get("text") or "")
        for chunk in index.chunks
        if chunk.get("doc_id") in legal_docs
    )


def graph_candidates(question: str, index, hops: int = 2, scope=None) -> tuple[list[dict], dict]:
    store = load_graph()
    comparison_like = bool(re.search(r"架构|版本|新旧|对比|变化|差异|演进", question))
    legal_ids = scope.legal_chunk_ids(index) if scope is not None else None
    legal_docs = scope.legal_doc_ids(index) if scope is not None else {
        chunk.get("doc_id") for chunk in index.chunks if chunk.get("doc_id")
    }
    entities = store.find_entities(question, include_documents=comparison_like)
    unfiltered_entities = len(entities)
    entities = [item for item in entities if _entity_legal(store, store.entities.get(item["id"], {}), legal_docs)]
    entities = [item for item in entities
                if _alias_legal(item, store.entities.get(item["id"], {}), index, legal_docs)]
    if not entities:
        return [], {
            "entities": [], "paths": [], "path_evidence": [], "candidates": 0,
            "enabled": store.stats()["entities"] > 0,
            "graph_version": store.version,
            "legal_entities": 0,
            "filtered_entities": unfiltered_entities,
            "scope_fingerprint": scope.fingerprint() if scope else "",
        }
    entity_ids = [item["id"] for item in entities[:5]]
    refs = store.chunk_refs(entity_ids, hops=hops, question=question,
                            allowed_chunk_ids=legal_ids)

    # Document entities are expanded to their key sections: graph edges point to
    # the first chunk, but version comparisons need the detail sections too.
    document_ids: set[str] = set()
    for item in entities:
        entity = store.entity(item["id"]) or {}
        if entity.get("type") == "document":
            document_ids.add((entity.get("attributes") or {}).get("doc_id", ""))
    for relation in store.expand(entity_ids, hops=hops, allowed_chunk_ids=legal_ids):
        for node in (relation["source"], relation["target"]):
            entity = store.entity(node) or {}
            if entity.get("type") == "document":
                document_ids.add((entity.get("attributes") or {}).get("doc_id", ""))
    document_ids.discard("")
    document_ids &= legal_docs
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
        chunks = [
            chunk for chunk in index.chunks
            if chunk.get("doc_id") == doc_id
            and (legal_ids is None or chunk["chunk_id"] in legal_ids)
        ]
        chunks.sort(key=lambda chunk: (
            next((position for position, word in enumerate(section_priority)
                  if word in chunk.get("section", "")), len(section_priority)),
            chunk.get("ord", 0),
        ))
        for chunk in chunks[:6]:
            current = refs.setdefault(chunk["chunk_id"], {"score": 0.08, "paths": [],
                                                          "path_evidence": []})
            current["score"] = max(current["score"], 0.08)
            current["document_expanded"] = True
            current["section_priority"] = next(
                (position for position, word in enumerate(section_priority)
                 if word in chunk.get("section", "")), len(section_priority)
            )
            if len(current["paths"]) < 2:
                path = f"文档实体扩展：{chunk.get('title', '')} / {chunk.get('section', '')}（{chunk['chunk_id']}）"
                current["paths"].append(path)
                current["path_evidence"].append({
                    "path": path,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": doc_id,
                    "section": chunk.get("section", ""),
                    "relation": "document_expansion",
                    "confidence": 0.8,
                    "evidence": chunk.get("text", "")[:200],
                })

    candidates = []
    filtered_chunks = 0
    for chunk_id, payload in refs.items():
        if legal_ids is not None and chunk_id not in legal_ids:
            filtered_chunks += 1
            continue
        chunk = index.by_id.get(chunk_id)
        if not chunk:
            filtered_chunks += 1
            continue
        item = dict(chunk)
        item["graph_score"] = payload["score"]
        item["graph_paths"] = payload["paths"]
        item["graph_path_evidence"] = payload.get("path_evidence", [])
        item["graph_document"] = payload.get("document_expanded", False)
        item["graph_priority"] = payload.get("section_priority", 99)
        candidates.append(item)
    candidates.sort(key=lambda item: (item.get("graph_score", 0), -item.get("graph_priority", 99)),
                    reverse=True)
    paths, path_evidence = [], []
    for item in candidates[:8]:
        paths.extend(item.get("graph_paths", []))
        path_evidence.extend(item.get("graph_path_evidence", []))
    debug = {
        "entities": entities,
        "paths": list(dict.fromkeys(paths))[:12],
        "path_evidence": path_evidence[:12],
        "candidates": len(candidates),
        "enabled": True,
        "graph_version": store.version,
        "legal_entities": len(entities),
        "filtered_entities": unfiltered_entities - len(entities),
        "filtered_chunks": filtered_chunks,
        "scope_fingerprint": scope.fingerprint() if scope else "",
    }
    return candidates, debug
