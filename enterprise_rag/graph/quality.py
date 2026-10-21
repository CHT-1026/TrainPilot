"""图谱质量评估：来源覆盖率、实体/关系分布、冲突边。"""
from __future__ import annotations

from collections import Counter, defaultdict


def quality_report(store, chunks_by_doc: dict[str, list[dict]]) -> dict:
    relations = list(store.relations.values())
    sourced = [relation for relation in relations
               if relation.get("doc_id") and relation.get("chunk_id") and relation.get("evidence")]
    managers: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if relation.get("relation") == "manages":
            managers[relation["target"]].add(relation["source"])
    conflicts = {
        person: sorted(bosses) for person, bosses in managers.items()
        if len(bosses) > 1
    }
    documented = {doc_id for relation in relations for doc_id in [relation.get("doc_id")] if doc_id}
    all_docs = set(chunks_by_doc)
    entity_type_counts = Counter(entity.get("type") for entity in store.entities.values())
    relation_type_counts = Counter(relation.get("relation") for relation in relations)
    return {
        "entities": len(store.entities),
        "relations": len(relations),
        "source_coverage": round(len(sourced) / len(relations), 4) if relations else 0.0,
        "doc_coverage": round(len(documented & all_docs) / len(all_docs), 4) if all_docs else 0.0,
        "entity_types": dict(sorted(entity_type_counts.items())),
        "relation_types": dict(sorted(relation_type_counts.items())),
        "conflict_persons": conflicts,
        "conflict_count": len(conflicts),
        "low_confidence_edges": sum(1 for relation in relations if float(relation.get("confidence", 0)) < 0.75),
        "entities_without_source": sum(1 for entity in store.entities.values() if not entity.get("doc_ids")),
    }
