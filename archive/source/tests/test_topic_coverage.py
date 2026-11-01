"""Multi-topic retrieval reserves legal evidence without claiming an answered task."""
from collections import Counter
from unittest.mock import patch
import unittest

from app.domain.user.models import UserContext
from enterprise_rag import config
from enterprise_rag.budget import RequestBudget
from enterprise_rag.query import reasoning_plan
from enterprise_rag.retrieval import retrieve
from enterprise_rag.store import RagIndex

QUESTION = "结合2025年经营情况和研发投入，说明增长与技术投入之间有哪些有依据的联系，区分事实与推断。"
USER = UserContext(user_id="topic-employee", name="Synthetic", role="employee", department="研发部")


def row(doc, text, **kw):
    value = {"chunk_id": doc+"#0", "doc_id": doc, "title": "2025年企业资料", "section": "正文", "text": text,
             "visibility": "all", "namespace": "lantu", "effective_date": "2024-01-01",
             "doc_updated_at": "2024-01-01", "version": "v1", "doc_family": doc}
    value.update(kw)
    return value


def run(question, rows, **kwargs):
    with patch.object(config, "EMBEDDING_PROVIDER", "hash"):
        index = RagIndex(rows)
    with patch("enterprise_rag.retrieval.load_index", return_value=index), \
         patch("enterprise_rag.retrieval.active_release", return_value={}), \
         patch("enterprise_rag.retrieval.query_vector", side_effect=AssertionError("Offline must never embed")):
        return retrieve(question, user=USER, namespace="lantu", offline=True, **kwargs)


class TopicCoverageTest(unittest.TestCase):
    def test_decomposes_independent_topics_and_preserves_subject_period(self):
        plan = reasoning_plan("结合北辰公司2023年经营情况和研发投入说明联系。")
        self.assertEqual([t["id"] for t in plan["topics"]], ["business", "research"])
        self.assertTrue(all("北辰公司" in t["query"] and "2023年" in t["query"] for t in plan["topics"]))
        self.assertFalse(plan["policy_as_of"])
        self.assertTrue(plan["bounded_topic_only"])

    def test_research_and_production_variant_is_not_a_question_specific_rule(self):
        plan = reasoning_plan("结合2024年研发投入和生产效率说明两者联系。")
        self.assertEqual({t["id"] for t in plan["topics"]}, {"research", "production"})
        result = run("结合2024年研发投入和生产效率说明两者联系。", [
            row("rd", "2024年研发投入为37万元。"), row("plant", "2024年生产效率提升18%。")], top_k=2)
        self.assertTrue(result["debug"]["topic_coverage"]["complete"])

    def test_single_or_explicitly_excluded_topic_does_not_expand(self):
        self.assertEqual(reasoning_plan("2025年研发投入和研发人员分别是多少？")["topics"], [])
        self.assertEqual(reasoning_plan("只问研发投入，不谈营业收入和订单。")["topics"], [])
        # No more than three independent probes even when four domains occur.
        plan = reasoning_plan("结合经营业绩、研发投入和生产效率及员工总数分析联系。")
        self.assertLessEqual(len(plan["topics"]), 3)

    def test_business_is_not_crowded_out_by_many_research_documents(self):
        rows = [row(f"rd-{i}", f"2025年研发投入为{80+i}万元，研发人员{40+i}人，技术投入支持创新。",
                    title="2025年研发投入与技术创新") for i in range(18)]
        rows += [row("ops", "2025年营业收入为23亿元，同比增长15%；净利润为3亿元。", title="2025年度经营报告",
                     effective_date="2026-02-01", doc_updated_at="2026-02-01")]
        result = run(QUESTION, rows, top_k=8)
        self.assertIn("ops", {c["doc_id"] for c in result["chunks"]})
        self.assertTrue(result["debug"]["topic_coverage"]["complete"])
        self.assertEqual(len(result["chunks"]), 8)
        self.assertIsNone(result["as_of"])

    def test_percentage_mention_does_not_count_as_operating_revenue(self):
        result = run(QUESTION, [row("rd", "2025年研发投入为5亿元，占营业收入13.8%；研发人员100人。")], top_k=2)
        coverage = result["debug"]["topic_coverage"]
        self.assertEqual(coverage["covered"], ["research"])
        self.assertEqual(coverage["missing"], ["business"])
        self.assertFalse(coverage["complete"])

    def test_exact_duplicates_do_not_spend_multiple_context_slots(self):
        rows = [row(f"rd-copy-{i}", "2025年研发投入为37万元。") for i in range(6)]
        rows += [row("ops", "2025年营业收入为23亿元。")]
        result = run(QUESTION, rows, top_k=8)
        self.assertEqual(len(result["chunks"]), 2)
        self.assertTrue(result["debug"]["topic_coverage"]["complete"])
        self.assertEqual(len({c["chunk_id"] for c in result["chunks"]}), 2)

    def test_coverage_never_reenters_restricted_future_or_other_namespace(self):
        rows = [row("rd", "2025年研发投入为37万元。"),
                row("secret", "2025年营业收入为90亿元。", visibility="admin"),
                row("future", "2025年营业收入为91亿元。", effective_date="2026-02-01"),
                row("foreign", "2025年营业收入为92亿元。", namespace="default")]
        result = run(QUESTION, rows, top_k=8, as_of="2025-12-31")
        self.assertEqual({c["doc_id"] for c in result["chunks"]}, {"rd"})
        self.assertEqual(result["debug"]["topic_coverage"]["missing"], ["business"])
        self.assertTrue(result["scope"]["strict_effective_date"])

    def test_latest_eligible_version_is_preserved(self):
        rows = [row("rd", "2025年研发投入为37万元。"),
                row("ops-old", "2025年营业收入为20亿元。", doc_family="operating-policy", version="v1"),
                row("ops-new", "2025年营业收入为23亿元。", doc_family="operating-policy", version="v2",
                    effective_date="2025-01-01", doc_updated_at="2025-01-01")]
        result = run(QUESTION, rows, top_k=8)
        self.assertIn("ops-new", {c["doc_id"] for c in result["chunks"]})
        self.assertNotIn("ops-old", {c["doc_id"] for c in result["chunks"]})

    def test_same_named_subject_and_statistical_period_required_for_coverage(self):
        question = "结合北辰公司2025年经营情况和研发投入说明联系。"
        rows = [row("rd", "北辰公司2025年研发投入为37万元。"),
                row("other-company", "南星公司2025年营业收入为90亿元。"),
                row("past", "北辰公司2024年营业收入为20亿元。", title="2024年度报告"),
                row("current", "北辰公司2025年营业收入为23亿元。")]
        result = run(question, rows, top_k=4)
        self.assertEqual(result["debug"]["topic_coverage"]["source_chunk_ids"]["business"], ["current#0"])

    def test_query_count_topk_and_document_limits_remain_bounded(self):
        rows = [row("shared", "2025年研发投入为37万元。"),
                row("shared", "2025年营业收入为23亿元。", chunk_id="shared#1"),
                row("alternative", "2025年营业收入为23亿元。")]
        with patch.object(config, "MAX_QUERY_VARIANTS", 1), patch.object(config, "MAX_CHUNKS_PER_DOC", 1):
            result = run(QUESTION, rows, top_k=2, budget=RequestBudget(offline=True))
        self.assertEqual(len(result["debug"]["queries_executed"]), 1)
        self.assertLessEqual(len(result["chunks"]), 2)
        self.assertTrue(all(n <= 1 for n in Counter(c["doc_id"] for c in result["chunks"]).values()))
        self.assertEqual(result["debug"]["budget"]["model_calls"], 0)
        self.assertEqual(result["debug"]["budget"]["embedding_calls"], 0)


if __name__ == "__main__":
    unittest.main()
