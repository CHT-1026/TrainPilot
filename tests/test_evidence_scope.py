"""S01 证据边界：块、图路径、时点与 namespace 的回归测试。"""
import unittest
from unittest.mock import patch

import numpy as np

from app.domain.user.models import UserContext
from enterprise_rag import config
from enterprise_rag.graph.retrieve import graph_candidates
from enterprise_rag.graph.schema import entity_payload, relation_payload
from enterprise_rag.graph.store import GraphStore
from enterprise_rag.retrieval import retrieve
from enterprise_rag.scope import EvidenceScope
from enterprise_rag.store import RagIndex

EMPLOYEE = UserContext(user_id="emp", name="员工", role="employee", department="技术部")
MANAGER = UserContext(user_id="mgr", name="主管", role="manager", department="技术部")
ADMIN = UserContext(user_id="adm", name="管理员", role="admin", department="管理部")


def row(doc_id, text, *, visibility="all", effective="2023-01-01", family="", version="v1.0",
        updated="2023-01-01", chunk_id=None):
    return {
        "chunk_id": chunk_id or f"{doc_id}#0",
        "doc_id": doc_id,
        "title": f"文档 {doc_id}",
        "section": "正文",
        "text": f"文档 {doc_id} · 正文\n{text}",
        "chunk_type": "text",
        "doc_family": family,
        "version": version,
        "effective_date": effective,
        "doc_updated_at": updated,
        "visibility": visibility,
        "category": "company",
        "score": 0.9,
    }


def build_index():
    rows = [
        row("legal", "顾承澜是沈亦舟的上级。"),
        row("restricted", "沈亦舟是程知远的上级。", visibility="manager"),
        row("future", "未来单版本制度：采购标准为 900 元。", effective="2025-01-01",
            updated="2025-01-01"),
    ]
    with patch.object(config, "EMBEDDING_PROVIDER", "hash"), \
         patch.object(config, "EMBEDDING_DIM", 384):
        return RagIndex(rows)


def build_graph():
    store = GraphStore()
    store.add_entity(entity_payload("person", "顾承澜", doc_ids=["legal"]))
    store.add_entity(entity_payload("person", "沈亦舟", doc_ids=["legal", "restricted", "future"]))
    store.add_entity(entity_payload("person", "程知远", doc_ids=["restricted"]))
    store.add_entity(entity_payload("document", "受限文档", doc_ids=["restricted"],
                                    attributes={"doc_id": "restricted"}))
    store.add_entity(entity_payload("document", "未来文档", doc_ids=["future"],
                                    attributes={"doc_id": "future"}))
    store.add_relation(relation_payload(
        "person:顾承澜", "person:沈亦舟", "manages",
        doc_id="legal", chunk_id="legal#0", section="正文",
        evidence="顾承澜是沈亦舟的上级", confidence=0.95,
    ))
    store.add_relation(relation_payload(
        "person:沈亦舟", "person:程知远", "manages",
        doc_id="restricted", chunk_id="restricted#0", section="正文",
        evidence="沈亦舟是程知远的上级", confidence=0.95,
    ))
    store.add_relation(relation_payload(
        "document:未来文档", "person:沈亦舟", "mentions_org",
        doc_id="future", chunk_id="future#0", section="正文",
        evidence="未来文档提及沈亦舟", confidence=0.9,
    ))
    return store


class ScopeFilterTest(unittest.TestCase):
    def setUp(self):
        self.index = build_index()

    def test_employee_cannot_see_manager_only_chunk(self):
        scope = EvidenceScope(user=EMPLOYEE)
        visible = {chunk["chunk_id"] for chunk in scope.filter(self.index.chunks, self.index)}
        self.assertEqual(visible, {"legal#0", "future#0"})

    def test_manager_and_admin_can_see_restricted_chunk(self):
        for user in (MANAGER, ADMIN):
            scope = EvidenceScope(user=user)
            visible = {chunk["chunk_id"] for chunk in scope.filter(self.index.chunks, self.index)}
            self.assertIn("restricted#0", visible)

    def test_future_single_version_is_excluded_at_historical_as_of(self):
        scope = EvidenceScope(user=ADMIN, as_of="2023-12-31", strict_effective_date=True)
        visible = {chunk["chunk_id"] for chunk in scope.filter(self.index.chunks, self.index)}
        self.assertEqual(visible, {"legal#0", "restricted#0"})

    def test_namespace_is_part_of_fingerprint(self):
        first = EvidenceScope(user=EMPLOYEE, namespace="lantu")
        second = EvidenceScope(user=EMPLOYEE, namespace="other")
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_fingerprint_changes_with_acl_version(self):
        first = EvidenceScope(user=EMPLOYEE, acl_version="1")
        second = EvidenceScope(user=EMPLOYEE, acl_version="2")
        self.assertNotEqual(first.fingerprint(), second.fingerprint())


class GraphBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.index = build_index()
        self.graph = build_graph()

    def test_graph_candidates_drop_restricted_and_future_relations(self):
        scope = EvidenceScope(user=EMPLOYEE, as_of="2023-12-31", strict_effective_date=True)
        with patch("enterprise_rag.graph.retrieve.load_graph", return_value=self.graph), \
             patch.object(config, "EMBEDDING_PROVIDER", "hash"):
            candidates, debug = graph_candidates("顾承澜的上级是谁", self.index, hops=2, scope=scope)
        ids = {chunk["chunk_id"] for chunk in candidates}
        self.assertEqual(ids, {"legal#0"})
        serialized = " ".join(debug["paths"])
        self.assertNotIn("restricted", serialized)
        self.assertNotIn("future", serialized)
        for evidence in debug["path_evidence"]:
            self.assertIn(evidence["chunk_id"], {"legal#0"})

    def test_admin_history_still_cannot_reenter_future_document(self):
        scope = EvidenceScope(user=ADMIN, as_of="2023-12-31", strict_effective_date=True)
        with patch("enterprise_rag.graph.retrieve.load_graph", return_value=self.graph):
            candidates, debug = graph_candidates("沈亦舟的上级是谁", self.index, hops=2, scope=scope)
        self.assertNotIn("future#0", {chunk["chunk_id"] for chunk in candidates})
        self.assertNotIn("future", " ".join(debug["paths"]))

    def test_retrieval_rebuilds_debug_paths_from_legal_hits(self):
        index = self.index
        restricted_hit = {
            **row("restricted", "沈亦舟是程知远的上级。", visibility="manager"),
            "score": 0.9,
            "graph_score": 0.9,
            "graph_paths": ["受限路径：沈亦舟 -> 程知远（restricted#0）"],
            "graph_path_evidence": [{"path": "受限路径", "chunk_id": "restricted#0"}],
        }
        debug = {"entities": [], "paths": ["受限路径"], "path_evidence": [],
                 "candidates": 1, "enabled": True}
        with patch("enterprise_rag.retrieval.load_index", return_value=index), \
             patch("enterprise_rag.retrieval.query_vector",
                   return_value=np.zeros(config.EMBEDDING_DIM)), \
             patch("enterprise_rag.retrieval.graph_candidates", return_value=([restricted_hit], debug)):
            result = retrieve("顾承澜的上级是谁", user=EMPLOYEE, retrieval_mode="hybrid")
        self.assertNotIn("restricted", {chunk["chunk_id"] for chunk in result["chunks"]})
        self.assertNotIn("future", {chunk["chunk_id"] for chunk in result["chunks"]})
        self.assertEqual(result["debug"]["graph"]["paths"], [])
        self.assertEqual(result["debug"]["graph"]["path_evidence"], [])


if __name__ == "__main__":
    unittest.main()
