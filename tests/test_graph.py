"""GraphRAG 实体、关系与路径查询的单元测试。"""
import unittest

from enterprise_rag.graph.extract import extract_entities, extract_relations
from enterprise_rag.graph.quality import quality_report
from enterprise_rag.graph.schema import entity_payload, relation_payload
from enterprise_rag.graph.store import GraphStore


def _chunk(text: str, **overrides) -> dict:
    payload = {
        "doc_id": "lantu_org_reporting_lines",
        "chunk_id": "lantu_org_reporting_lines#c000",
        "title": "澜图智能员工汇报关系与业务分工",
        "section": "一、研发体系汇报链",
        "text": text,
        "department": "人力资源部",
        "author": "陈依然",
        "version": "v1.2",
        "doc_family": "org_reporting_lines",
        "effective_date": "2025-09-01",
        "category": "org",
    }
    payload.update(overrides)
    return payload


class GraphExtractionTest(unittest.TestCase):
    def test_known_person_and_department_extracted(self):
        chunk = _chunk("顾承澜是沈亦舟的上级。沈亦舟担任 CTO，负责研发中心。")
        entities = extract_entities(chunk, {"顾承澜", "沈亦舟"})
        names = {entity["name"] for entity in entities}
        self.assertIn("顾承澜", names)
        self.assertIn("沈亦舟", names)
        self.assertIn("研发中心", names)
        self.assertIn("人力资源部", names)

    def test_reporting_chain_relations(self):
        chunk = _chunk("顾承澜是沈亦舟的上级。沈亦舟是程知远的上级。")
        entities = extract_entities(chunk, {"顾承澜", "沈亦舟", "程知远"})
        relations = extract_relations(chunk, entities)
        edges = {(item["source"], item["target"]) for item in relations if item["relation"] == "manages"}
        self.assertIn(("person:顾承澜", "person:沈亦舟"), edges)
        self.assertIn(("person:沈亦舟", "person:程知远"), edges)
        for relation in relations:
            self.assertTrue(relation["doc_id"])
            self.assertTrue(relation["chunk_id"])
            self.assertTrue(relation["evidence"])

    def test_unknown_person_not_extracted(self):
        chunk = _chunk("本办法负责解释具体流程。")
        entities = extract_entities(chunk, set())
        names = {entity["name"] for entity in entities}
        self.assertNotIn("办法", names)
        self.assertNotIn("本办法", names)


class GraphStoreTest(unittest.TestCase):
    def _store(self) -> GraphStore:
        store = GraphStore()
        for name, value in (("顾承澜", "person"), ("沈亦舟", "person"), ("程知远", "person")):
            store.add_entity(entity_payload(value, name))
        store.add_relation(relation_payload("person:顾承澜", "person:沈亦舟", "manages",
                                            doc_id="doc", chunk_id="doc#c0", section="一、", evidence="顾承澜是沈亦舟的上级"))
        store.add_relation(relation_payload("person:沈亦舟", "person:程知远", "manages",
                                            doc_id="doc", chunk_id="doc#c0", section="一、", evidence="沈亦舟是程知远的上级"))
        return store

    def test_paths_between_two_hops(self):
        store = self._store()
        paths = store.paths_between("person:顾承澜", "person:程知远", max_hops=3)
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0]["relations"]), 2)

    def test_find_entities_and_chunk_refs(self):
        store = self._store()
        found = store.find_entities("程知远的上级是谁")
        self.assertEqual(found[0]["name"], "程知远")
        refs = store.chunk_refs([item["id"] for item in found], hops=2)
        self.assertIn("doc#c0", refs)
        self.assertGreater(refs["doc#c0"]["score"], 0)

    def test_quality_report_source_coverage(self):
        store = self._store()
        report = quality_report(store, {"doc": [{"text": "x"}]})
        self.assertEqual(report["source_coverage"], 1.0)
        self.assertEqual(report["doc_coverage"], 1.0)
        self.assertEqual(report["conflict_count"], 0)


if __name__ == "__main__":
    unittest.main()
