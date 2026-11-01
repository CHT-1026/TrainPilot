"""发布绑定：校验语料、索引、图谱和 ACL 的资源身份。

候选校验失败不改 current 指针；显式失效操作更新 ACL 版本标记。
本模块不替代鉴权、不可变资源或并发读隔离；原子写 current 指针不等于资源切换隔离。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from enterprise_rag import config

RELEASE_DIR = config.PACKAGE_ROOT / "data" / "releases"
CURRENT_DIR = config.PACKAGE_ROOT / "data" / "releases" / "current"
_TZ = timezone(timedelta(hours=8))


def _now() -> str:
    return datetime.now(_TZ).isoformat()


def _digest(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def release_path(namespace: str | None = None) -> Path:
    name = namespace or config.RAG_NAMESPACE
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name):
        raise ValueError("命名空间必须为1至32位字母、数字、下划线或连字符")
    return CURRENT_DIR / f"{name}.json"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def active_release(namespace: str | None = None) -> dict:
    path = release_path(namespace)
    if not path.exists():
        return {
            "snapshot_id": "unpublished",
            "namespace": namespace or config.RAG_NAMESPACE,
            "status": "unpublished",
            "acl_version": "0",
            "index_version": "",
            "graph_version": "",
            "corpus_checksum": "",
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dim": config.EMBEDDING_DIM,
            "published_at": "",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "snapshot_id": "unreadable",
            "namespace": namespace or config.RAG_NAMESPACE,
            "status": "unreadable",
            "acl_version": "0",
            "index_version": "",
            "graph_version": "",
            "corpus_checksum": "",
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dim": config.EMBEDDING_DIM,
            "published_at": "",
        }
    payload.setdefault("namespace", namespace or config.RAG_NAMESPACE)
    payload.setdefault("status", "active")
    payload.setdefault("acl_version", "0")
    return payload


def _write_release(namespace: str, payload: dict) -> dict:
    _atomic_write(release_path(namespace), payload)
    try:
        from app.database.connection import get_redis_client

        get_redis_client().set(
            f"rag_release:{namespace}", json.dumps(payload, ensure_ascii=False), ex=60 * 60 * 24 * 30
        )
    except Exception:
        # 文件快照是权威回退；Redis 只是加速与跨进程可见性。
        pass
    return payload


def prepare_release(
    namespace: str | None = None,
    *,
    index_version: str = "",
    graph_version: str = "",
    corpus_checksum: str = "",
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    index_documents: int | None = None,
    index_chunks: int | None = None,
    graph_entities: int | None = None,
    graph_relations: int | None = None,
) -> dict:
    """Build a candidate without changing the active pointer."""
    namespace = namespace or config.RAG_NAMESPACE
    previous = active_release(namespace)
    payload = {
        "namespace": namespace,
        "status": "active",
        "index_version": index_version,
        "graph_version": graph_version,
        "corpus_checksum": corpus_checksum,
        "embedding_model": embedding_model or config.EMBEDDING_MODEL,
        "embedding_dim": embedding_dim or config.EMBEDDING_DIM,
        "embedding_version": config.EMBEDDING_VERSION,
        "index_documents": index_documents,
        "index_chunks": index_chunks,
        "graph_entities": graph_entities,
        "graph_relations": graph_relations,
        "published_at": _now(),
    }
    payload["snapshot_id"] = _digest(payload)
    payload["acl_version"] = hashlib.sha256(
        f"{previous.get('acl_version', '0')}|{payload['snapshot_id']}|{corpus_checksum}".encode("utf-8")
    ).hexdigest()[:12]
    return payload


def activate_release(candidate: dict, *, index_stats: dict, graph_stats: dict,
                     corpus_checksum: str) -> dict:
    """Verify actual resources before the only mutation of the current pointer."""
    issues = verify_release(candidate, index_stats=index_stats, graph_stats=graph_stats,
                            corpus_checksum=corpus_checksum)
    if not candidate.get("namespace") or index_stats.get("namespace") != candidate["namespace"]:
        issues.append({"code": "NAMESPACE_MISMATCH", "detail": "实际索引命名空间与候选不同"})
    if not index_stats.get("index_version") or not graph_stats.get("version"):
        issues.append({"code": "RESOURCE_VERSION_MISSING", "detail": "发布必须验证实际索引和图谱版本"})
    if "namespace" not in graph_stats:
        issues.append({"code": "GRAPH_NAMESPACE_MISSING", "detail": "发布必须验证实际图谱命名空间"})
    if issues:
        raise ValueError("发布候选校验失败: " + ", ".join(item["code"] for item in issues))
    return _write_release(candidate["namespace"], candidate)


def invalidate_release(namespace: str | None = None, reason: str = "") -> dict:
    """语料删除/停用/权限变化时调用：保留快照内容，但更换 ACL 版本。"""
    namespace = namespace or config.RAG_NAMESPACE
    current = active_release(namespace)
    current["acl_version"] = hashlib.sha256(
        f"{current.get('acl_version', '0')}|{_now()}|{reason}".encode("utf-8")
    ).hexdigest()[:12]
    current["invalidated_at"] = _now()
    current["invalidation_reason"] = reason[:160]
    current["status"] = "active"
    return _write_release(namespace, current)


def verify_release(release: dict, *, index_stats: dict | None = None,
                   graph_stats: dict | None = None, corpus_checksum: str | None = None) -> list[dict]:
    """返回不一致项；空列表才表示可安全发布。"""
    issues: list[dict] = []
    if release.get("status") != "active":
        issues.append({"code": "STATUS_INVALID", "detail": release.get("status")})
    expected_model = release.get("embedding_model") or config.EMBEDDING_MODEL
    expected_dim = release.get("embedding_dim") or config.EMBEDDING_DIM
    if expected_model != config.EMBEDDING_MODEL or int(expected_dim) != int(config.EMBEDDING_DIM):
        issues.append({"code": "EMBEDDING_MISMATCH",
                       "detail": f"{expected_model}/{expected_dim} != {config.EMBEDDING_MODEL}/{config.EMBEDDING_DIM}"})
    if release.get("embedding_version", config.EMBEDDING_VERSION) != config.EMBEDDING_VERSION:
        issues.append({"code": "EMBEDDING_VERSION_MISMATCH", "detail": "向量revision与当前配置不一致"})
    if corpus_checksum is not None and (not corpus_checksum or release.get("corpus_checksum") != corpus_checksum):
        issues.append({"code": "CORPUS_CHECKSUM_MISMATCH", "detail": "语料清单校验和与发布快照不一致"})
    if index_stats:
        if "index_version" in index_stats and (not release.get("index_version") or \
                release["index_version"] != index_stats["index_version"]):
            issues.append({"code": "INDEX_VERSION_MISMATCH", "detail": "索引已变化，需重新发布"})
        if release.get("index_chunks") is not None and index_stats.get("chunks") is not None \
                and int(release["index_chunks"]) != int(index_stats["chunks"]):
            issues.append({"code": "INDEX_CHUNK_COUNT_MISMATCH", "detail": "块数与发布快照不一致"})
    if graph_stats:
        if "namespace" in graph_stats:
            if not graph_stats["namespace"]:
                issues.append({"code": "GRAPH_NAMESPACE_MISSING", "detail": "无法确定实际图谱命名空间"})
            elif graph_stats["namespace"] != release.get("namespace"):
                issues.append({"code": "GRAPH_NAMESPACE_MISMATCH", "detail": "实际图谱命名空间与发布快照不一致"})
        if "version" in graph_stats and (not release.get("graph_version") or \
                release["graph_version"] != graph_stats["version"]):
            issues.append({"code": "GRAPH_VERSION_MISMATCH", "detail": "图谱已变化，需重新发布"})
        if release.get("graph_relations") is not None and graph_stats.get("relations") is not None \
                and int(release["graph_relations"]) != int(graph_stats["relations"]):
            issues.append({"code": "GRAPH_RELATION_COUNT_MISMATCH", "detail": "关系数与发布快照不一致"})
    for saved, actual, stats, code in (
        ("index_documents", "documents", index_stats, "INDEX_DOCUMENT_COUNT_MISMATCH"),
        ("graph_entities", "entities", graph_stats, "GRAPH_ENTITY_COUNT_MISMATCH"),
    ):
        if stats and release.get(saved) is not None and stats.get(actual) is not None \
                and int(release[saved]) != int(stats[actual]):
            issues.append({"code": code, "detail": "资源数量与发布快照不一致"})
    return issues


def manifest_checksum(path: Path | None = None) -> str:
    path = path or config.MANIFEST_PATH
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
