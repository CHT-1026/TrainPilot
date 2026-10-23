"""检索、缓存、预算、发布、工具与观测等优化项的单元回归。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.core.backpressure import OverloadedError, admit, inflight, release
from app.core.metrics import record_usage, reset as reset_metrics, snapshot as metrics_snapshot
from enterprise_rag import config
from enterprise_rag.budget import RequestBudget
from enterprise_rag.chunking import chunk_document
from enterprise_rag.citations import compress_chunk, sentence_spans
from enterprise_rag.embedding import EmbeddingBudgetExceeded, embed_query_cached, reset_query_cache
from enterprise_rag.evidence import unsupported_numeric_tokens
from enterprise_rag.facts import extract_facts
from enterprise_rag.feedback import record_feedback, read_feedback
from enterprise_rag.model_router import route_for
from enterprise_rag.release import (
    active_release,
    invalidate_release,
    publish_release,
    verify_release,
)
from enterprise_rag.rerank import lexical_pair_rerank
from enterprise_rag.scoring import describe as describe_scores
from enterprise_rag.store import RagIndex, ann_recommendation
from enterprise_rag.verification import verify_answer_facts
from app.domain.user.models import UserContext

USER = UserContext(user_id="u1", name="测试", role="employee", department="技术部")


def row(doc_id, text, **overrides):
    title = overrides.pop("title", f"文档 {doc_id}")
    payload = {
        "chunk_id": f"{doc_id}#0",
        "doc_id": doc_id,
        "title": title,
        "section": overrides.pop("section", "正文"),
        "text": f"{title} · {overrides.get('section', '正文')}\n{text}",
        "chunk_type": "text",
        "doc_family": doc_id,
        "version": "v1.0",
        "effective_date": "2024-01-01",
        "doc_updated_at": "2024-01-01",
        "visibility": "all",
        "category": "company",
        "score": 0.9,
    }
    payload.update(overrides)
    return payload


class QueryCacheAndBudgetTest(unittest.TestCase):
    def setUp(self):
        reset_query_cache()

    def test_same_query_vector_is_cached(self):
        calls = []
        with patch("enterprise_rag.embedding.embed_query",
                   side_effect=lambda text: calls.append(text) or np.ones(config.EMBEDDING_DIM)):
            first = embed_query_cached("差旅标准")
            second = embed_query_cached("差旅标准")
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.array_equal(first, second))

    def test_embedding_budget_stops_second_call(self):
        budget = RequestBudget(max_embedding_calls=1, allow_model=False)
        with patch("enterprise_rag.embedding.embed_query",
                   side_effect=lambda text: np.ones(config.EMBEDDING_DIM)):
            embed_query_cached("a", budget=budget)
            with self.assertRaises(EmbeddingBudgetExceeded):
                embed_query_cached("b", budget=budget)
        self.assertEqual(budget.embedding_calls, 1)
        # 局部额度耗尽只禁止后续向量调用，不把整个请求标记为全局终止。
        self.assertEqual(budget.stopped_reason, "")

    def test_offline_budget_blocks_model_calls(self):
        budget = RequestBudget(offline=True, allow_model=True)
        self.assertFalse(budget.can_call_model())
        self.assertFalse(budget.can_call_embedding())

    def test_budget_snapshot_records_stages_and_usage(self):
        budget = RequestBudget()
        with budget.stage("answer"):
            budget.charge_model(stage="answer", model="m", prompt_tokens=10, completion_tokens=5)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["model_calls"], 1)
        self.assertEqual(snapshot["usage"]["total_tokens"], 15)
        self.assertEqual(snapshot["stages"][0]["name"], "answer")


class ReleaseTest(unittest.TestCase):
    def test_publish_invalidate_and_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with patch("enterprise_rag.release.CURRENT_DIR", base), \
                 patch("enterprise_rag.release.RELEASE_DIR", base):
                first = publish_release("test", index_version="v1", graph_version="g1",
                                        corpus_checksum="c1", index_chunks=10, graph_relations=20)
                self.assertTrue(first["snapshot_id"])
                current = active_release("test")
                self.assertEqual(current["snapshot_id"], first["snapshot_id"])
                self.assertEqual(verify_release(current, index_stats={"index_version": "v2"})[0]["code"],
                                 "INDEX_VERSION_MISMATCH")
                invalidated = invalidate_release("test", "acl_changed")
                self.assertNotEqual(invalidated["acl_version"], first["acl_version"])


class ScoringAndRoutingTest(unittest.TestCase):
    def test_score_semantics_declared(self):
        semantics = describe_scores()["semantics"]
        self.assertEqual(semantics["vector"]["range"], [-1, 1])
        self.assertEqual(semantics["rrf"]["direction"], "higher_is_better")

    def test_model_route_uses_configured_aliases(self):
        with patch.object(config, "MODEL_ROUTE_COMPLEX", "strong-model"), \
             patch.object(config, "MODEL_ROUTE_SIMPLE", "fast-model"):
            self.assertEqual(route_for({"complexity": "high"})["model"], "strong-model")
            self.assertEqual(route_for({"complexity": "low"})["model"], "fast-model")

    def test_cross_encoder_changes_order_without_dropping(self):
        chunks = [
            row("a", "完全无关的行政通知。", title="行政通知"),
            row("b", "差旅住宿标准为 600 元每晚。", title="差旅标准"),
        ]
        reranked = lexical_pair_rerank("差旅住宿标准是多少", chunks)
        self.assertEqual(reranked[0]["doc_id"], "b")
        self.assertEqual(len(reranked), 2)

    def test_multi_representation_index_finds_title_topic(self):
        with patch.object(config, "EMBEDDING_PROVIDER", "hash"):
            index = RagIndex([row("travel", "住宿标准为 600 元。", title="差旅管理制度")])
        hits = index.representation_search("差旅管理制度", 5)
        self.assertEqual(hits[0]["chunk_id"], "travel#0")

    def test_ann_recommendation_respects_threshold(self):
        with patch.object(config, "EMBEDDING_PROVIDER", "hash"):
            index = RagIndex([row("a", "文本")])
        self.assertFalse(ann_recommendation(index, threshold=100)["recommended"])
        self.assertTrue(ann_recommendation(index, threshold=1)["recommended"])


class ContextAndFactsTest(unittest.TestCase):
    def test_sentence_spans_are_reversible(self):
        text = "第一句。第二句；第三句。"
        spans = sentence_spans(text)
        self.assertEqual("".join(span["text"] for span in spans), text)

    def test_compression_keeps_source_offsets(self):
        body = "无关内容。" * 60 + "住宿标准为 600 元每晚。"
        chunk = row("travel", body, title="差旅制度")
        compressed = compress_chunk(chunk, ["住宿标准"], max_chars=120)
        self.assertTrue(compressed.get("compressed_spans"))
        self.assertIn("住宿标准", compressed["text"])
        for span in compressed["compressed_spans"]:
            self.assertIn(span["text"][:4], body)

    def test_typed_facts_from_table_and_text(self):
        table = row("travel", "| 城市 | 住宿标准 |\n| --- | --- |\n| 苏州 | 600 元 |",
                    chunk_type="table", title="差旅标准")
        text = row("finance", "公司2025年营业收入为12亿元。")
        facts = extract_facts([table, text])
        self.assertTrue(any(fact.subject == "苏州" and fact.value == "600" for fact in facts))
        self.assertTrue(any(fact.value == "12" and fact.unit == "亿元" for fact in facts))

    def test_fact_check_reports_missing_numeric_and_conflict(self):
        chunk = row("policy", "差旅住宿标准为 600 元。", title="差旅制度")
        answer = "住宿标准是 900 元 [1]。不得报销住宿费 [1]。"
        report = verify_answer_facts("住宿标准", answer, [chunk])
        self.assertFalse(report["verified"])
        self.assertTrue(report["numeric_missing"])
        self.assertTrue(report["negation_suspects"])
        self.assertFalse(report["conflicts"])

    def test_fact_check_refuses_only_provable_modal_conflict(self):
        chunk = row("policy", "设备支持远程重启。", title="运维制度")
        report = verify_answer_facts("设备是否支持远程重启", "设备不支持远程重启 [1]。", [chunk])
        self.assertTrue(report["conflicts"])

    def test_chunk_metadata_keeps_page_and_anchor(self):
        document = {"doc_id": "d1", "title": "测试文档"}
        blocks = [{"section": "一、范围", "text": "第一条。", "type": "text",
                   "page": 3, "paragraph": 0, "anchor": "md:p3#b0",
                   "extractor_version": "2026.10", "ocr": False}]
        chunks = chunk_document(document, blocks)
        self.assertEqual(chunks[0]["page"], 3)
        self.assertEqual(chunks[0]["anchor"], "md:p3#b0")
        self.assertEqual(chunks[0]["extractor_version"], "2026.10")


class FeedbackAndBackpressureTest(unittest.TestCase):
    def test_feedback_is_appended_without_auto_training(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("enterprise_rag.feedback.FEEDBACK_DIR", Path(temp)):
                record = record_feedback(USER, question="差旅标准", rating=-1,
                                         correction="应为 600 元", citation_ids=["chunk#0"])
                records = read_feedback()
        self.assertEqual(records[-1]["feedback_id"], record["feedback_id"])
        self.assertEqual(records[-1]["review_status"], "pending")
        self.assertEqual(records[-1]["user_id"], "u1")

    def test_backpressure_rejects_when_slot_taken(self):
        with admit("test", 1):
            with self.assertRaises(OverloadedError):
                with admit("test", 1):
                    pass
            self.assertEqual(inflight("test"), 1)
        self.assertEqual(inflight("test"), 0)

    def test_family_contrast_quotes_old_and_new_versions(self):
        from enterprise_rag.answer import _family_contrast_answer
        chunks = [
            row("org_2024", "2024 年公司设 11 个一级部门；售后服务部隶属销售中心。",
                title="组织架构（2024版）", doc_family="org_chart", version="v1.0",
                effective_date="2024-01-01", chunk_id="org_2024#0", score=0.9),
            row("org_2025", "2025 年 4 月起公司设 12 个一级部门；售后服务部升级为一级部门。",
                title="组织架构（2025版）", doc_family="org_chart", version="v2.0",
                effective_date="2025-04-01", chunk_id="org_2025#0", score=0.8),
        ]
        result = _family_contrast_answer("公司架构有哪些升级", chunks)
        self.assertIsNotNone(result)
        self.assertIn("11 个一级部门", result[0])
        self.assertIn("12 个一级部门", result[0])
        self.assertIn("旧版 v1.0", result[0])
        self.assertIn("新版 v2.0", result[0])

    def test_document_metadata_answers_update_time_without_model(self):
        from enterprise_rag.answer import _metadata_answer
        chunks = [
            row("lantu_org_chart_2025", "2025 年 4 月起公司设 12 个一级部门。",
                title="澜图智能组织架构（2025版）", chunk_type="text",
                version="v2.0", doc_updated_at="2025-04-10T10:00:00",
                effective_date="2025-04-10", chunk_id="lantu_org_chart_2025#c000"),
            row("lantu_tech_control_architecture", "控制系统采用 EtherCAT。",
                title="澜图智能机器人控制系统架构", chunk_type="text",
                version="v3.3", doc_updated_at="2026-10-22T09:00:00",
                effective_date="2025-10-22", chunk_id="lantu_tech_control_architecture#c000"),
        ]
        result = _metadata_answer("现在的公司架构是什么时候更新的", chunks)
        self.assertIsNotNone(result)
        self.assertIn("2025-04-10", result[0])
        self.assertIn("v2.0", result[0])
        tech = _metadata_answer("控制系统架构是什么时候更新的", chunks)
        self.assertIsNotNone(tech)
        self.assertIn("v3.3", tech[0])

    def test_broad_version_comparison_uses_deterministic_table(self):
        from enterprise_rag.answer import _comparison_answer
        chunks = [
            row("d1", "现场总线采用 CANopen，总线周期 2 ms。", doc_family="arch",
                version="v2.4", effective_date="2024-01-01", chunk_id="d1#0"),
            row("d2", "现场总线采用 EtherCAT，总线周期 1 ms。", doc_family="arch",
                version="v3.3", effective_date="2025-01-01", chunk_id="d2#0"),
        ]
        result = _comparison_answer("旧架构和新架构有什么区别？", chunks)
        self.assertIsNotNone(result)
        self.assertIn("CANopen", result[0])
        self.assertIn("EtherCAT", result[0])
        self.assertIsNone(_comparison_answer("总线周期缩短了多少？", chunks))

    def test_derived_arithmetic_result_requires_verified_expression(self):
        from enterprise_rag.evidence import _derived_numbers

        chunks = [row("d1", "苏州员工 2,450 人；东莞员工 780 人；成都员工 430 人。")]
        good = "2,450+780+430=3,660 人 [1]。三个基地合计 3,660 人 [1]。"
        bad = "2,450+780+430=4,000 人 [1]。三个基地合计 4,000 人 [1]。"
        self.assertEqual(unsupported_numeric_tokens(good, chunks), [])
        self.assertTrue(unsupported_numeric_tokens(bad, chunks))
        self.assertIn(6.83, _derived_numbers("2025年42.63亿元，2024年35.80亿元，相差6.83亿元。"))

    def test_table_cell_number_with_header_unit_is_supported(self):
        chunks = [row("d1", "| 基地 | 员工数 |\n| 苏州总部 | 2,450 |", chunk_type="table")]
        self.assertEqual(
            unsupported_numeric_tokens("苏州总部员工数为2,450人 [1]。", chunks), []
        )

    def test_release_without_admission_is_noop(self):
        release("never-admitted")
        self.assertEqual(inflight("never-admitted"), 0)

    def test_dashscope_embedding_splits_batches_at_ten(self):
        from types import SimpleNamespace
        from enterprise_rag.embedding import _dashscope_embed

        sizes = []

        class FakeEmbeddings:
            def create(self, model, input, dimensions):
                sizes.append(len(input))
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 4) for _ in input])

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.embeddings = FakeEmbeddings()

        with patch("openai.OpenAI", FakeClient), patch.object(config, "EMBEDDING_BATCH", 16):
            vectors = _dashscope_embed(["文本" + str(index) for index in range(25)])
        self.assertEqual(sizes, [10, 10, 5])
        self.assertEqual(len(vectors), 25)

    def test_metrics_usage_records_tokens_without_text(self):
        reset_metrics()
        record_usage("llm", "m", 10, 5, 15)
        snapshot = metrics_snapshot()
        self.assertEqual(snapshot["usage"]["llm:m"]["total_tokens"], 15)
        self.assertNotIn("差旅", json.dumps(snapshot, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
