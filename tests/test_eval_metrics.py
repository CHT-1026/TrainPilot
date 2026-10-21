"""Honest regression metrics: distinguish keyword hits, complete evidence, and marker syntax."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from enterprise_rag.eval import run_eval as evaluation


def case(expected=None, **overrides):
    row = {
        "id": "metric-test",
        "category": "cross_document",
        "question": "公司成立于哪一年？",
        "expected_doc_ids": ["d1"] if expected is None else expected,
        "answer_keywords": ["2011"],
    }
    row.update(overrides)
    return row


def response(answer="公司成立于2011年。[1]", citations=None, documents=None, refused=False):
    return {
        "answer": answer,
        "citations": [{"index": 1, "doc_id": "d1"}] if citations is None else citations,
        "grounded": not refused,
        "refused": refused,
        "method": "controlled-test",
        "retrieval": {
            "chunks": [{"doc_id": doc_id} for doc_id in (documents or ["d1"])],
            "as_of": None,
            "top_score": 0.9,
        },
        "query_rewrite": {"rewritten": "公司成立于哪一年？"},
    }


def evaluate(test_case=None, result=None):
    with patch.object(evaluation, "query", return_value=result or response()):
        return evaluation.evaluate_case(test_case or case(), use_llm=False)


class RetrievalMetricTest(unittest.TestCase):
    def test_top10_metric_actually_requests_at_least_ten_chunks(self):
        with patch.object(evaluation.config, "TOP_K", 4):
            with patch.object(evaluation, "query", return_value=response()) as run_query:
                result = evaluation.evaluate_case(case(), use_llm=False)
        self.assertEqual(run_query.call_args.kwargs["top_k"], 10)
        self.assertEqual(result["requested_top_k"], 10)

    def test_larger_configured_recall_budget_is_preserved(self):
        with patch.object(evaluation.config, "TOP_K", 16):
            with patch.object(evaluation, "query", return_value=response()) as run_query:
                evaluation.evaluate_case(case(), use_llm=False)
        self.assertEqual(run_query.call_args.kwargs["top_k"], 16)

    def test_any_source_hit_is_not_reported_as_complete_multihop_recall(self):
        result = evaluate(case(expected=["d1", "d2"]))
        metrics = evaluation.aggregate([result])
        self.assertEqual(metrics["any_expected_doc_hit@5"], 1.0)
        self.assertEqual(metrics["any_expected_doc_hit@10"], 1.0)
        self.assertEqual(metrics["all_expected_docs_recalled@10"], 0.0)
        self.assertEqual(metrics["all_expected_citation_coverage"], 0.0)
        self.assertEqual(metrics["recall@5"], metrics["any_expected_doc_hit@5"])
        self.assertEqual(metrics["recall@10"], metrics["any_expected_doc_hit@10"])

    def test_topn_window_keeps_legacy_chunk_unit_explicit(self):
        chunks = [{"doc_id": "d1"}] * 5 + [{"doc_id": "d2"}]
        self.assertEqual(evaluation.doc_ranking(chunks, 5), ["d1"])
        self.assertEqual(evaluation.doc_ranking(chunks, 10), ["d1", "d2"])

    def test_missing_expected_labels_do_not_vacuously_pass_complete_recall(self):
        metrics = evaluation.aggregate([evaluate(case(expected=[]))])
        self.assertEqual(metrics["all_expected_docs_recalled@10"], 0.0)
        self.assertEqual(metrics["all_expected_citation_coverage"], 0.0)


class CitationMetricTest(unittest.TestCase):
    def test_two_expected_sources_must_both_be_marked_in_answer(self):
        citations = [{"index": 1, "doc_id": "d1"}, {"index": 2, "doc_id": "d2"}]
        partial = evaluate(case(expected=["d1", "d2"]), response(citations=citations, documents=["d1", "d2"]))
        self.assertEqual(partial["cited_expected_doc_ids"], ["d1"])
        self.assertEqual(partial["missing_expected_citation_doc_ids"], ["d2"])
        self.assertFalse(partial["expected_citation_sources_complete"])
        complete = evaluate(case(expected=["d1", "d2"]), response(
            answer="公司成立于2011年。[1][2]", citations=citations, documents=["d1", "d2"],
        ))
        self.assertTrue(complete["expected_citation_sources_complete"])
        self.assertEqual(evaluation.aggregate([complete])["all_expected_citation_coverage"], 1.0)

    def test_out_of_range_markers_are_recorded_as_invalid(self):
        result = evaluate(result=response(answer="公司成立于2011年。[1][99]"))
        self.assertEqual(result["valid_citation_markers"], [1])
        self.assertEqual(result["invalid_citation_markers"], [99])
        self.assertFalse(result["citation_markers_valid"])

    def test_returned_sources_without_answer_markers_do_not_count_as_cited(self):
        result = evaluate(result=response(answer="公司成立于2011年。"))
        self.assertEqual(result["citation_docs"], ["d1"])
        self.assertEqual(result["cited_doc_ids"], [])
        self.assertFalse(result["citation_hit"])

    def test_citation_document_absent_from_retrieval_is_invalid(self):
        result = evaluate(result=response(citations=[{"index": 1, "doc_id": "d2"}]))
        self.assertEqual(result["invalid_citation_markers"], [1])
        self.assertFalse(result["citation_hit"])

    def test_duplicate_citation_indices_are_ambiguous(self):
        result = evaluate(result=response(citations=[
            {"index": 1, "doc_id": "d1"}, {"index": 1, "doc_id": "d1"},
        ]))
        self.assertEqual(result["invalid_citation_markers"], [1])
        self.assertFalse(result["expected_citation_sources_complete"])


class MetricDisclosureTest(unittest.TestCase):
    def test_keyword_proxy_can_pass_a_wrong_negated_claim_and_is_labeled(self):
        result = evaluate(result=response(answer="公司不是2011年成立。[1]"))
        metrics = evaluation.aggregate([result])
        self.assertEqual(metrics["keyword_accuracy"], 1.0)
        self.assertEqual(metrics["answer_accuracy"], metrics["keyword_accuracy"])
        self.assertIn("不是语义准确率", " ".join(evaluation.SCORING_LIMITATIONS))

    def test_refusal_cases_do_not_inflate_keyword_or_source_metrics(self):
        refused = evaluate(case(expected=[], expect_refusal=True), response(
            answer="现有企业资料未覆盖该问题。", citations=[], refused=True,
        ))
        metrics = evaluation.aggregate([evaluate(), refused])
        self.assertEqual(metrics["retrieval_cases"], 1)
        self.assertEqual(metrics["keyword_accuracy"], 1.0)
        self.assertEqual(metrics["refusal_accuracy"], 1.0)

    def test_reports_disclose_proxy_limits_and_separate_reasoning_evaluation(self):
        result = evaluate()
        metrics = evaluation.aggregate([result])
        report = {
            "corpus": {"namespace": "test", "documents": 1, "chunks": 1},
            "config": {"embedding_model": "lantu-hash-384", "answer_method": {"controlled-test": 1}, "top_k": 10},
            "metrics": metrics,
            "by_category": {"cross_document": metrics},
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evaluation.config, "REPORT_DIR", Path(directory)):
                evaluation.write_report(report, [result])
            markdown = (Path(directory) / "report.md").read_text(encoding="utf-8")
            saved = json.loads((Path(directory) / "latest.json").read_text(encoding="utf-8"))
        self.assertIn("scoring_limitations", saved)
        self.assertIn("独立 reasoning 评测", markdown)
        self.assertIn("全部关键词命中率", markdown)
        self.assertNotIn("语义近邻召回", markdown)
        self.assertNotIn("默认抽取式生成", markdown)

    def test_gate_thresholds_are_not_lowered(self):
        result = evaluate()
        with patch.object(evaluation, "load_cases", return_value=[case()]):
            with patch.object(evaluation, "load_manifest", return_value=[{"doc_id": "d1"}]):
                with patch.object(evaluation, "evaluate_case", return_value=result):
                    with patch.object(evaluation, "index_stats", return_value={}):
                        with patch.object(evaluation, "write_report") as write:
                            with patch("sys.argv", ["run_eval", "--gate"]), redirect_stdout(io.StringIO()):
                                with self.assertRaises(SystemExit):
                                    evaluation.main()
        report = write.call_args.args[0]
        self.assertEqual(report["gate"]["thresholds"], {
            "recall@5": 0.85, "answer_accuracy": 0.75, "citation_coverage": 0.85, "refusal_accuracy": 0.85,
        })
        self.assertFalse(report["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
