"""编排、纠错检索、离线预算与动态评测生成的回归。"""
import unittest
from unittest.mock import patch

from app.domain.user.models import UserContext
from enterprise_rag import config
from enterprise_rag.pipeline import query as pipeline_query
from enterprise_rag.pipeline import _evidence_gap
from enterprise_rag.store import RagIndex
from scripts.generate_reasoning_cases import generate_cases
from enterprise_rag.graph.schema import entity_payload, relation_payload
from enterprise_rag.graph.store import GraphStore

USER = UserContext(user_id="u_pipe", name="测试", role="admin", department="管理部")


def chunk(text, doc_id="d1", **overrides):
    payload = {
        "chunk_id": overrides.pop("chunk_id", f"{doc_id}#0"),
        "doc_id": doc_id,
        "title": overrides.pop("title", "测试制度"),
        "section": overrides.pop("section", "正文"),
        "text": f"{overrides.get('title', '测试制度')} · 正文\n{text}",
        "chunk_type": "text",
        "doc_family": doc_id,
        "version": "v1.0",
        "effective_date": "2024-01-01",
        "doc_updated_at": "2024-01-01",
        "visibility": "all",
        "category": "policy",
        "score": 0.9,
    }
    payload.update(overrides)
    return payload


def empty_result(question):
    return {
        "query": {"original": question, "rewritten": question, "matched_synonyms": [],
                  "expansions": [], "time_label": ""},
        "as_of": None,
        "terms": [],
        "candidates": 0,
        "chunks": [],
        "scope": {"fingerprint": "x"},
        "debug": {"top_score": 0.0, "comparison_mode": False, "graph": {"paths": []},
                  "plan": {}, "evidence_budget_complete": True, "retrieval_mode": "hybrid"},
    }


def hit_result(question):
    result = empty_result(question)
    result.update({"candidates": 1, "chunks": [chunk("住宿标准为 600 元。")]})
    result["debug"]["top_score"] = 0.9
    return result


class CorrectiveRetrievalTest(unittest.TestCase):
    def test_one_corrective_retrieval_merges_new_evidence(self):
        calls = []

        def fake_retrieve(question, **kwargs):
            calls.append(question)
            return empty_result(question) if len(calls) == 1 else hit_result(question)

        generated = {
            "answer": "住宿标准为 600 元 [1]。", "citations": [], "grounded": True,
            "refused": False, "method": "llm", "confidence": 1.0,
            "fact_check": {"verified": True}, "prompt_version": "v", "model_used": "m",
            "context_truncated": False, "context_chunk_ids": ["d1#0"],
            "graph_paths": [], "reasoning": None,
        }
        with patch("enterprise_rag.pipeline.retrieve", side_effect=fake_retrieve), \
             patch("enterprise_rag.pipeline.generate_answer", return_value=generated):
            result = pipeline_query("住宿标准是多少？", user=USER, use_llm=False)
        self.assertEqual(len(calls), 2)
        self.assertTrue(result["retrieval"]["debug"]["corrective_retrieval"]["attempted"])
        self.assertEqual(result["retrieval"]["debug"]["corrective_retrieval"]["added"], 1)
        self.assertTrue(result["answer"])

    def test_no_correction_when_first_retrieval_is_sufficient(self):
        with patch("enterprise_rag.pipeline.retrieve", return_value=hit_result("住宿标准")) as retrieve, \
             patch("enterprise_rag.pipeline.generate_answer", return_value={
                 "answer": "600 元 [1]。", "citations": [], "grounded": True, "refused": False,
                 "method": "llm", "confidence": 1.0, "fact_check": {"verified": True},
                 "prompt_version": "v", "model_used": "m", "context_truncated": False,
                 "context_chunk_ids": [], "graph_paths": [], "reasoning": None,
             }):
            result = pipeline_query("住宿标准是多少？", user=USER, use_llm=False)
        self.assertEqual(retrieve.call_count, 1)
        self.assertNotIn("corrective_retrieval", result["retrieval"]["debug"])

    def test_acl_change_is_not_merged(self):
        first = empty_result("住宿标准")
        first["scope"] = {"snapshot": "s1", "acl_version": "acl-old"}
        second = hit_result("住宿标准")
        second["scope"] = {"snapshot": "s1", "acl_version": "acl-new"}
        generated = {
            "answer": "住宿标准为 600 元 [1]。", "citations": [], "grounded": True,
            "refused": False, "method": "llm", "confidence": 1.0,
            "fact_check": {"verified": True}, "prompt_version": "v", "model_used": "m",
            "context_truncated": False, "context_chunk_ids": [], "graph_paths": [], "reasoning": None,
        }
        with patch("enterprise_rag.pipeline.retrieve", side_effect=[first, second]), \
             patch("enterprise_rag.pipeline.generate_answer", return_value=generated) as answer:
            result = pipeline_query("住宿标准是多少？", user=USER, use_llm=False)
        self.assertEqual(answer.call_args.args[1], [])
        self.assertIn("acl_changed",
                      result["retrieval"]["debug"]["corrective_retrieval"]["reason"])

    def test_offline_forces_extractive_answer(self):
        generated = {
            "answer": "600 元 [1]。", "citations": [], "grounded": True, "refused": False,
            "method": "extractive", "confidence": 1.0, "fact_check": {"verified": True},
            "prompt_version": "v", "model_used": "", "context_truncated": False,
            "context_chunk_ids": [], "graph_paths": [], "reasoning": None,
        }
        with patch("enterprise_rag.pipeline.retrieve", return_value=hit_result("住宿标准")), \
             patch("enterprise_rag.pipeline.generate_answer", return_value=generated) as answer:
            pipeline_query("住宿标准是多少？", user=USER, use_llm=True, offline=True)
        self.assertIs(answer.call_args.kwargs["use_llm"], False)

    def test_evidence_gap_distinguishes_missing_period(self):
        debug = {"top_score": 0.5, "plan": {"strict_effective_date": True, "policy_as_of": True}}
        gap = _evidence_gap("2023年差旅标准是什么？", [chunk("2024年标准 600 元。")], debug, 0.18)
        self.assertEqual(gap, "missing_period_evidence")


class DynamicCaseGenerationTest(unittest.TestCase):
    def _index(self):
        rows = [
            chunk("住宿 500 元。", doc_id="travel_v1", doc_family="travel", version="v1.0"),
            chunk("住宿 600 元。", doc_id="travel_v2", doc_family="travel", version="v2.0"),
        ]
        with patch.object(config, "EMBEDDING_PROVIDER", "hash"):
            return RagIndex(rows)

    def test_generate_version_multihop_and_counterexample(self):
        store = GraphStore()
        store.add_entity(entity_payload("person", "顾承澜", doc_ids=["org"]))
        store.add_entity(entity_payload("person", "沈亦舟", doc_ids=["org"]))
        store.add_entity(entity_payload("person", "程知远", doc_ids=["org"]))
        store.add_relation(relation_payload("person:顾承澜", "person:沈亦舟", "manages",
                                            doc_id="org", chunk_id="org#0",
                                            evidence="顾承澜是沈亦舟的上级"))
        store.add_relation(relation_payload("person:沈亦舟", "person:程知远", "manages",
                                            doc_id="org", chunk_id="org#1",
                                            evidence="沈亦舟是程知远的上级"))
        cases = generate_cases(index=self._index(), store=store, seed=7,
                               version_limit=5, hop_limit=5, counter_limit=2)
        categories = {case["category"] for case in cases}
        self.assertIn("version_comparison", categories)
        self.assertIn("multi_hop_relation", categories)
        self.assertIn("counterexample", categories)
        # 固定 seed 的生成结果可复现
        again = generate_cases(index=self._index(), store=store, seed=7,
                               version_limit=5, hop_limit=5, counter_limit=2)
        self.assertEqual([case["id"] for case in cases], [case["id"] for case in again])


class EvalMetricLayerTest(unittest.TestCase):
    def test_aggregate_reports_fact_support_separately(self):
        from enterprise_rag.eval.run_eval import aggregate

        results = [{
            "expect_refusal": False, "top5": ["d1"], "top10": ["d1"],
            "expected_doc_ids": ["d1"], "first_relevant_rank": 1, "ndcg@10": 1.0,
            "answer_correct": True, "citation_hit": True,
            "expected_citation_sources_complete": True, "citation_markers_valid": True,
            "grounded": True, "refused": False, "latency_ms": 100, "fact_support": True,
        }]
        report = aggregate(results)
        self.assertEqual(report["fact_support_rate"], 1.0)
        self.assertEqual(report["keyword_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
