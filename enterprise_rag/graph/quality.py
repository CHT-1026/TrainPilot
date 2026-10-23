"""图谱质量评估：来源覆盖率、实体/关系分布、冲突边、悬空边与别名碰撞。"""
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
    dangling = [
        {"source": relation["source"], "target": relation["target"],
         "relation": relation.get("relation", ""), "chunk_id": relation.get("chunk_id", "")}
        for relation in relations
        if relation["source"] not in store.entities or relation["target"] not in store.entities
    ]
    direction_conflicts = []
    seen_pairs = set()
    for relation in relations:
        if relation.get("relation") != "manages":
            continue
        pair = (relation["source"], relation["target"])
        reverse = (relation["target"], relation["source"])
        if reverse in seen_pairs:
            direction_conflicts.append({"a": pair[0], "b": pair[1]})
        seen_pairs.add(pair)
    alias_collisions: dict[str, set[str]] = defaultdict(set)
    for entity in store.entities.values():
        for alias in [entity.get("name", "")] + list(entity.get("aliases", [])):
            if alias:
                alias_collisions[alias].add(entity["id"])
    collisions = {
        alias: sorted(ids) for alias, ids in alias_collisions.items() if len(ids) > 1
    }
    documented = {doc_id for relation in relations for doc_id in [relation.get("doc_id")] if doc_id}
    all_docs = set(chunks_by_doc)
    entity_type_counts = Counter(entity.get("type") for entity in store.entities.values())
    relation_type_counts = Counter(relation.get("relation") for relation in relations)
    confidence_buckets = Counter()
    for relation in relations:
        confidence = float(relation.get("confidence", 0) or 0)
        if confidence >= 0.9:
            confidence_buckets["high_0.9+"] += 1
        elif confidence >= 0.75:
            confidence_buckets["medium_0.75-0.9"] += 1
        else:
            confidence_buckets["low_below_0.75"] += 1
    return {
        "entities": len(store.entities),
        "relations": len(relations),
        "source_coverage": round(len(sourced) / len(relations), 4) if relations else 0.0,
        "doc_coverage": round(len(documented & all_docs) / len(all_docs), 4) if all_docs else 0.0,
        "entity_types": dict(sorted(entity_type_counts.items())),
        "relation_types": dict(sorted(relation_type_counts.items())),
        "conflict_persons": conflicts,
        "conflict_count": len(conflicts),
        "direction_conflict_count": len(direction_conflicts),
        "direction_conflicts": direction_conflicts[:20],
        "dangling_relation_count": len(dangling),
        "dangling_relations": dangling[:20],
        "alias_collision_count": len(collisions),
        "alias_collisions": dict(list(collisions.items())[:20]),
        "low_confidence_edges": sum(1 for relation in relations if float(relation.get("confidence", 0)) < 0.75),
        "confidence_buckets": dict(confidence_buckets),
        "entities_without_source": sum(1 for entity in store.entities.values() if not entity.get("doc_ids")),
        "source_coverage_note": "来源字段完整不等于边语义、方向或时间有效性正确",
    }
