"""Evidence selection regressions with in-memory indexes and a fake model transport."""
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from enterprise_rag import config
from enterprise_rag.answer import generate_answer
from enterprise_rag.citations import MAX_CONTEXT_CHARS, build_context, select_context
from enterprise_rag.embedding import EmbeddingUnavailableError
from enterprise_rag.query import reasoning_plan
from enterprise_rag.retrieval import retrieve
from enterprise_rag.store import RagIndex


def document(doc_id, body, **overrides):
    title = overrides.pop("title", "澜图公司资料")
    row = {
        "chunk_id": f"{doc_id}#0",
        "doc_id": doc_id,
        "title": title,
        "section": "正文",
        "text": f"{title} · 正文\n{body}",
        "chunk_type": "text",
        "doc_family": doc_id,
        "version": "v1.0",
        "effective_date": "2024-01-01",
        "doc_updated_at": "2024-01-01",
        "visibility": "all",
        "category": "company",
        "score": 0.9,
    }
    row.update(overrides)
    return row


def local_retrieve(question, rows, **kwargs):
    with patch.object(config, "EMBEDDING_PROVIDER", "hash"):
        index = RagIndex(rows)
    with patch("enterprise_rag.retrieval.load_index", return_value=index):
        with patch("enterprise_rag.retrieval.query_vector", return_value=np.zeros(config.EMBEDDING_DIM)):
            return retrieve(question, **kwargs)


class EffectiveDatePlanTest(unittest.TestCase):
    def test_implicit_policy_snapshot_rejects_single_future_document(self):
        rows = [document(
            "future-policy", "差旅住宿标准为600元。", title="差旅制度",
            category="policy", effective_date="2025-01-01", doc_updated_at="2025-01-01",
        )]
        result = local_retrieve("2023年差旅住宿标准是什么？", rows)
        self.assertEqual(result["as_of"], "2023-12-31")
        self.assertTrue(result["debug"]["plan"]["strict_effective_date"])
        self.assertEqual(result["chunks"], [])

    def test_financial_data_period_is_not_report_publication_date(self):
        rows = [document(
            "annual-2025", "公司2025年营业收入为12亿元。", title="2025年度报告",
            category="finance", doc_family="annual_report_2025",
            effective_date="2026-01-28", doc_updated_at="2026-01-28",
        )]
        result = local_retrieve("公司2025年营业收入是多少？", rows)
        self.assertFalse(result["debug"]["plan"]["strict_effective_date"])
        self.assertIn("annual-2025", {c["doc_id"] for c in result["chunks"]})

    def test_explicit_as_of_still_excludes_later_publications(self):
        rows = [document(
            "annual-2025", "公司2025年营业收入为12亿元。", title="2025年度报告",
            category="finance", effective_date="2026-01-28", doc_updated_at="2026-01-28",
        )]
        result = local_retrieve("公司2025年营业收入是多少？", rows, as_of="2025-12-31")
        self.assertEqual(result["chunks"], [])

    def test_organization_family_uses_requested_2024_snapshot(self):
        rows = [
            document("org-2024", "2024年公司设有研发部、财务部。", title="组织架构2024版",
                     doc_family="org_chart", category="org", effective_date="2024-01-01"),
            document("org-2025", "2025年公司设有研发部、财务部、销售部。", title="组织架构2025版",
                     doc_family="org_chart", category="org", version="v2.0",
                     effective_date="2025-04-01", doc_updated_at="2025-04-10"),
        ]
        result = local_retrieve("2024年公司共有几个部门？", rows)
        self.assertEqual(result["as_of"], "2024-12-31")
        self.assertEqual({c["doc_id"] for c in result["chunks"]}, {"org-2024"})

    def test_two_year_financial_comparison_retains_both_periods(self):
        # Match the real corpus convention: annual reports have one family per data year.
        rows = [
            document("annual-2024", "公司2024年全年营业收入为10亿元。", title="2024年度报告",
                     doc_family="annual_report_2024", category="finance",
                     effective_date="2025-01-30", doc_updated_at="2025-01-30"),
            document("annual-2025", "公司2025年全年营业收入为12亿元。", title="2025年度报告",
                     doc_family="annual_report_2025", category="finance",
                     effective_date="2026-01-28", doc_updated_at="2026-01-28"),
        ]
        question = "公司2024年和2025年全年营业收入相差多少亿元？"
        plan = reasoning_plan(question)
        self.assertFalse(plan["policy_as_of"])
        self.assertTrue(any("2024年" in probe for probe in plan["queries"]))
        self.assertTrue(any("2025年" in probe for probe in plan["queries"]))
        result = local_retrieve(question, rows)
        self.assertIsNone(result["as_of"])
        self.assertEqual({c["doc_id"] for c in result["chunks"]}, {"annual-2024", "annual-2025"})


class RetrievalBudgetAndFallbackTest(unittest.TestCase):
    def test_budget_completeness_tracks_omitted_chunk_ids_not_total_count(self):
        rows = [document("many-sections", "公司组织部门资料。", chunk_id=f"many-sections#{i}", score=0.9)
                for i in range(6)]
        rows += [document(f"unrelated-{i}", "其他组织部门资料。", score=0.01) for i in range(4)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(config, "REASONING_TOP_K", 8))
            stack.enter_context(patch.object(config, "MAX_CHUNKS_PER_DOC", 4))
            stack.enter_context(patch("enterprise_rag.retrieval._rerank", side_effect=lambda *args: [dict(c) for c in rows]))
            result = local_retrieve("公司共有几个部门？", rows, top_k=8)
        self.assertEqual(len(result["chunks"]), 8)
        self.assertEqual(sum(c["score"] >= config.MIN_SCORE for c in result["chunks"]), 4)
        self.assertEqual(result["debug"]["omitted_relevant_chunks"], 2)
        self.assertFalse(result["debug"]["evidence_budget_complete"])

    def test_vector_fault_uses_bm25_with_explicit_fallback_trace(self):
        rows = [document("travel", "差旅住宿标准为600元。", title="差旅制度", category="policy")]
        with patch.object(config, "EMBEDDING_PROVIDER", "hash"):
            index = RagIndex(rows)
        with patch("enterprise_rag.retrieval.load_index", return_value=index):
            with patch("enterprise_rag.retrieval.query_vector", side_effect=EmbeddingUnavailableError("controlled vector fault")):
                with self.assertLogs("enterprise_rag.retrieval", level="WARNING"):
                    result = retrieve("差旅住宿标准是什么？")
        self.assertIn("travel", {c["doc_id"] for c in result["chunks"]})
        self.assertTrue(result["debug"]["vector_fallback"])
        self.assertEqual(result["debug"]["vector_errors"], ["EmbeddingUnavailableError"])
        self.assertEqual(result["debug"]["vector_hits"], 0)
        self.assertGreater(result["debug"]["bm25_hits"], 0)


class ModelContextVisibilityTest(unittest.TestCase):
    def model_result(self, rows, sources):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "refused": False,
                "claims": [{"text": "核对代码是42。", "sources": sources}],
            }, ensure_ascii=False))),
        ])
        settings = SimpleNamespace(openai_api_key="fake-test-key", openai_api_base=None, openai_enable_thinking="false")
        with patch("app.core.config.get_settings", return_value=settings):
            with patch("openai.OpenAI", return_value=client):
                with patch("enterprise_rag.answer._focus_supported", return_value=True):
                    result = generate_answer("核对代码是什么？", rows, use_llm=True)
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        return result, messages[-1]["content"]

    def test_oversize_first_chunk_is_not_sent_or_exposed_as_a_citation(self):
        huge = document("hidden-first", "首块隐去标识" + "长" * (MAX_CONTEXT_CHARS + 100))
        visible = document("visible", "核对代码是42。")
        result, prompt = self.model_result([huge, visible], sources=[1])
        self.assertNotIn("首块隐去标识", prompt)
        self.assertIn("核对代码是42", prompt)
        self.assertEqual(result["context_chunk_ids"], ["visible#0"])
        self.assertEqual({c["doc_id"] for c in result["citations"]}, {"visible"})
        self.assertTrue(result["context_truncated"])

    def test_model_cannot_cite_original_later_index_after_context_exclusion(self):
        visible = document("visible", "核对代码是42。")
        hidden = document("hidden-later", "后段隐去标识" + "长" * (MAX_CONTEXT_CHARS + 100))
        result, prompt = self.model_result([visible, hidden], sources=[2])
        self.assertNotIn("后段隐去标识", prompt)
        self.assertEqual(result["context_chunk_ids"], ["visible#0"])
        self.assertEqual(result["model_error"], "ModelAnswerValidationError")
        self.assertNotIn("hidden-later", {c["doc_id"] for c in result["citations"]})
        self.assertNotIn("[2]", result["answer"])

    def test_context_builder_rejects_unbudgeted_chunks_instead_of_silent_truncation(self):
        rows = [document("oversize", "长" * 500), document("short", "核对代码是42。")]
        selected, truncated = select_context(rows, max_chars=200)
        self.assertTrue(truncated)
        self.assertEqual([c["doc_id"] for c in selected], ["short"])
        self.assertIn("[1]", build_context(selected, max_chars=200))
        with self.assertRaises(ValueError):
            build_context(rows, max_chars=200)


if __name__ == "__main__":
    unittest.main()
