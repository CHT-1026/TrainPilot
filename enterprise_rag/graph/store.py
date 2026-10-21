"""轻量知识图谱存储：JSON 持久化、实体链接、邻域扩展与路径序列化。"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

from enterprise_rag.graph.schema import entity_id, normalize_name

RELATION_LABELS = {
    "manages": "汇报/管理",
    "responsible_for": "负责",
    "belongs_to": "属于",
    "version_of": "版本属于",
    "supersedes": "替代",
    "uses": "配套/采用",
    "mentions_org": "提及",
}

RELATION_HINTS = {
    "manages": ("上级", "下属", "汇报", "管理", "领导", "关系"),
    "responsible_for": ("负责", "分管", "主管", "管理"),
    "belongs_to": ("属于", "部门", "归属"),
    "version_of": ("版本", "架构", "新旧", "对比", "变化"),
    "supersedes": ("版本", "替代", "升级", "新旧", "对比"),
    "uses": ("配套", "采用", "使用", "组成"),
    "mentions_org": ("客户", "项目", "合作"),
}


class GraphStore:
    def __init__(self, data: dict | None = None):
        data = data or {}
        self.version = data.get("version", "")
        self.entities: dict[str, dict] = {item["id"]: dict(item) for item in data.get("entities", [])}
        self.relations: dict[str, dict] = {
            f"{item['source']}|{item['relation']}|{item['target']}": dict(item)
            for item in data.get("relations", [])
        }
        self.out: dict[str, list[str]] = defaultdict(list)
        self.incoming: dict[str, list[str]] = defaultdict(list)
        self._alias_index: dict[str, list[str]] = defaultdict(list)
        self.rebuild_index()

    def rebuild_index(self) -> None:
        self.out = defaultdict(list)
        self.incoming = defaultdict(list)
        self._alias_index = defaultdict(list)
        for key, relation in self.relations.items():
            self.out[relation["source"]].append(key)
            self.incoming[relation["target"]].append(key)
        for entity in self.entities.values():
            candidates = [entity.get("name", "")] + list(entity.get("aliases", []))
            for alias in candidates:
                normalized = normalize_name(alias)
                if normalized and entity["id"] not in self._alias_index[normalized]:
                    self._alias_index[normalized].append(entity["id"])

    def add_entity(self, entity: dict) -> None:
        current = self.entities.get(entity["id"])
        if current:
            current["doc_ids"] = sorted(set(current.get("doc_ids", [])) | set(entity.get("doc_ids", [])))
            current["aliases"] = sorted(set(current.get("aliases", [])) | set(entity.get("aliases", [])))
            current["attributes"].update(entity.get("attributes", {}))
        else:
            self.entities[entity["id"]] = dict(entity)
        for alias in [entity.get("name", "")] + list(entity.get("aliases", [])):
            normalized = normalize_name(alias)
            if normalized and entity["id"] not in self._alias_index[normalized]:
                self._alias_index[normalized].append(entity["id"])

    def add_relation(self, relation: dict) -> None:
        key = f"{relation['source']}|{relation['relation']}|{relation['target']}"
        current = self.relations.get(key)
        if current is None:
            self.relations[key] = dict(relation)
            self.out[relation["source"]].append(key)
            self.incoming[relation["target"]].append(key)
            return
        current["confidence"] = max(float(current.get("confidence", 0)), float(relation.get("confidence", 0)))
        if relation.get("evidence") and not current.get("evidence"):
            current.update({
                "doc_id": relation.get("doc_id", ""),
                "chunk_id": relation.get("chunk_id", ""),
                "section": relation.get("section", ""),
                "evidence": relation.get("evidence", ""),
            })

    def remove_doc(self, doc_id: str) -> None:
        self.relations = {
            key: relation for key, relation in self.relations.items()
            if relation.get("doc_id") != doc_id
        }
        for entity in self.entities.values():
            if doc_id in entity.get("doc_ids", []):
                entity["doc_ids"] = [item for item in entity["doc_ids"] if item != doc_id]
        self.rebuild_index()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version or datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "entities": sorted(self.entities.values(), key=lambda item: item["id"]),
            "relations": sorted(self.relations.values(), key=lambda item: (item["source"], item["relation"], item["target"])),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GraphStore":
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def entity(self, entity_id_value: str) -> dict | None:
        return self.entities.get(entity_id_value)

    def name(self, entity_id_value: str) -> str:
        entity = self.entities.get(entity_id_value)
        return entity["name"] if entity else entity_id_value.split(":", 1)[-1]

    def find_entities(self, text: str, limit: int = 8, include_documents: bool = False) -> list[dict]:
        normalized = normalize_name(text)
        lowered = (text or "").lower()
        matches: dict[str, dict] = {}
        for alias, ids in self._alias_index.items():
            if len(alias) < 2 or alias not in normalized:
                continue
            for entity_id_value in ids:
                entity = self.entities.get(entity_id_value)
                if not entity or entity.get("type") == "document":
                    continue
                score = min(len(alias), 12) / 12 + (0.2 if entity.get("type") in {"person", "product"} else 0)
                current = matches.get(entity_id_value)
                if current is None or score > current["score"]:
                    matches[entity_id_value] = {
                        "id": entity_id_value,
                        "name": entity.get("name", ""),
                        "type": entity.get("type", ""),
                        "score": round(score, 4),
                        "matched": alias,
                    }
        if include_documents:
            family_docs: dict[str, list[dict]] = defaultdict(list)
            for entity in self.entities.values():
                if entity.get("type") == "document":
                    family = (entity.get("attributes") or {}).get("doc_family", "")
                    if family:
                        family_docs[family].append(entity)
            keywords = [word for word in ("架构", "控制器", "伺服", "云平台", "数据平台",
                                           "安全", "路线图", "专利", "流程", "组织", "服务",
                                           "财务", "合同", "招聘", "培训")
                        if word in text]
            for family, docs in family_docs.items():
                if len(docs) < 2:
                    continue
                for entity in docs:
                    title = entity.get("name", "")
                    matched = [word for word in keywords if word in title]
                    if not matched:
                        continue
                    score = min(len(matched[0]) + 3, 8) / 10
                    matches[entity["id"]] = {
                        "id": entity["id"],
                        "name": title,
                        "type": "document",
                        "score": round(score, 4),
                        "matched": matched[0],
                    }
        # Model codes are strong signals even when punctuation differs.
        for model in re.findall(r"\b(?:LT|Lantu)[-A-Za-z0-9]*[A-Za-z0-9]+(?:[-A-Za-z0-9]+)*\b", text or "", re.I):
            for entity_id_value, entity in self.entities.items():
                if entity.get("type") == "product" and normalize_name(model) == normalize_name(entity.get("name", "")):
                    matches[entity_id_value] = {
                        "id": entity_id_value,
                        "name": entity.get("name", ""),
                        "type": "product",
                        "score": 1.0,
                        "matched": model,
                    }
        return sorted(matches.values(), key=lambda item: item["score"], reverse=True)[:limit]

    def expand(self, entity_ids: list[str], hops: int = 2, max_relations: int = 120) -> list[dict]:
        """Bounded BFS over relations; returns relations annotated with distance."""
        results: list[dict] = []
        seen_relations: set[str] = set()
        visited_nodes: set[str] = set(entity_ids)
        frontier = [(entity_id_value, 0) for entity_id_value in entity_ids]
        while frontier and len(results) < max_relations:
            current, distance = frontier.pop(0)
            if distance >= hops:
                continue
            for key in self.out.get(current, []) + self.incoming.get(current, []):
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                relation = dict(self.relations[key])
                relation["distance"] = distance + 1
                results.append(relation)
                for node in (relation["source"], relation["target"]):
                    if node not in visited_nodes:
                        visited_nodes.add(node)
                        frontier.append((node, distance + 1))
                if len(results) >= max_relations:
                    break
        return results

    def paths_between(self, source: str, target: str, max_hops: int = 4, max_paths: int = 5) -> list[dict]:
        if source == target:
            return []
        queue = deque([(source, [], [])])
        paths = []
        while queue and len(paths) < max_paths:
            current, nodes, relations = queue.popleft()
            if len(relations) >= max_hops:
                continue
            for key in self.out.get(current, []):
                relation = self.relations[key]
                next_node = relation["target"]
                if next_node in nodes or next_node == current:
                    continue
                next_relations = relations + [relation]
                next_nodes = nodes + [current]
                if next_node == target:
                    paths.append({"nodes": next_nodes + [next_node], "relations": next_relations})
                    continue
                queue.append((next_node, next_nodes, next_relations))
            for key in self.incoming.get(current, []):
                relation = self.relations[key]
                next_node = relation["source"]
                if next_node in nodes or next_node == current:
                    continue
                next_relations = relations + [relation]
                next_nodes = nodes + [current]
                if next_node == target:
                    paths.append({"nodes": next_nodes + [next_node], "relations": next_relations})
                    continue
                queue.append((next_node, next_nodes, next_relations))
        return paths

    def chunk_refs(self, entity_ids: list[str], hops: int = 2, question: str = "") -> dict[str, dict]:
        """Map source chunks to graph scores and human-readable paths."""
        relations = self.expand(entity_ids, hops=hops)
        refs: dict[str, dict] = {}
        for relation in relations:
            chunk_id = relation.get("chunk_id")
            if not chunk_id:
                continue
            hint_bonus = 0.0
            if any(word in question for word in RELATION_HINTS.get(relation["relation"], ())):
                hint_bonus = 0.08
            base = float(relation.get("confidence", 0.8)) * (0.22 if relation["distance"] == 1 else 0.12)
            score = base + hint_bonus
            current = refs.setdefault(chunk_id, {"score": 0.0, "paths": []})
            current["score"] = round(current["score"] + score, 4)
            if len(current["paths"]) < 6:
                current["paths"].append(self.serialize_relation(relation))
        return refs

    def serialize_relation(self, relation: dict) -> str:
        label = RELATION_LABELS.get(relation["relation"], relation["relation"])
        return (f"{self.name(relation['source'])} -[{label}]-> {self.name(relation['target'])}"
                f"（{relation.get('doc_id', '')} / {relation.get('section', '')}）")

    def serialize_path(self, path: dict) -> str:
        parts = []
        for index, relation in enumerate(path["relations"]):
            if index == 0:
                parts.append(self.name(relation["source"]))
            label = RELATION_LABELS.get(relation["relation"], relation["relation"])
            parts.append(f"-[{label}]-> {self.name(relation['target'])}")
        return " ".join(parts)

    def stats(self) -> dict:
        type_counts = defaultdict(int)
        for entity in self.entities.values():
            type_counts[entity.get("type", "unknown")] += 1
        relation_counts = defaultdict(int)
        sourced = 0
        for relation in self.relations.values():
            relation_counts[relation.get("relation", "unknown")] += 1
            if relation.get("doc_id") and relation.get("chunk_id") and relation.get("evidence"):
                sourced += 1
        return {
            "version": self.version,
            "entities": len(self.entities),
            "relations": len(self.relations),
            "entity_types": dict(sorted(type_counts.items())),
            "relation_types": dict(sorted(relation_counts.items())),
            "source_coverage": round(sourced / len(self.relations), 4) if self.relations else 0.0,
        }
