"""分维度断言评分器：拒答不计入答案/来源/引用分母，缺权限元数据不当通过。"""
import unittest

from scripts.eval_assertions import evaluate_case


def result(**overrides):
    payload = {
        "refused": False,
        "fact_check": {"verified": True},
        "answer": "公司名称为澜图智能装备股份有限公司 [1]。",
        "citations": [{"index": 1, "doc_id": "lantu_company_profile", "visibility": "all"}],
        "retrieval": {"chunks": [{}]},
        "observability": {"model_calls": 0, "embedding_calls": 1,
                          "max_model_calls": 4, "max_embedding_calls": 8},
    }
    payload.update(overrides)
    return payload


class EvalScoringTest(unittest.TestCase):
    def test_correct_refusal_is_not_counted_as_answer_source_citation_failure(self):
        case = {"id": "R01", "category": "refusal", "question": "公司食堂菜单？",
                "expected_doc_ids": [], "keywords": []}
        record = evaluate_case(case, result(refused=True, answer="现有资料不足。", citations=[]), 10)
        self.assertTrue(record["passed"])
        self.assertTrue(record["dimensions"]["refusal"])
        self.assertIsNone(record["dimensions"]["answer"])
        self.assertIsNone(record["dimensions"]["source"])
        self.assertIsNone(record["dimensions"]["citation"])
        self.assertIsNone(record["dimensions"]["permission"])

    def test_answered_refusal_case_fails_refusal_only(self):
        case = {"id": "R02", "category": "refusal", "question": "创始人完整生日？",
                "expected_doc_ids": [], "keywords": []}
        record = evaluate_case(case, result(), 10)
        self.assertFalse(record["dimensions"]["refusal"])
        self.assertFalse(record["passed"])

    def test_missing_visibility_is_unknown_not_pass(self):
        case = {"id": "S01", "category": "single_doc", "question": "公司名称？",
                "expected_doc_ids": ["lantu_company_profile"],
                "keywords": ["澜图智能装备股份有限公司"]}
        citations = [{"index": 1, "doc_id": "lantu_company_profile", "visibility": None}]
        record = evaluate_case(case, result(citations=citations), 10)
        self.assertIsNone(record["dimensions"]["permission"])
        self.assertIn("permission",record["unknown_dimensions"])
        self.assertFalse(record["passed"])

    def test_keyword_match_cannot_override_failed_or_missing_factuality(self):
        case={"question":"公司名称？","keywords":["澜图智能装备股份有限公司"],"expected_doc_ids":["lantu_company_profile"]}
        record=evaluate_case(case,result(fact_check={"verified":False}),10)
        self.assertTrue(record["dimensions"]["answer"])
        self.assertFalse(record["dimensions"]["factuality"])
        self.assertFalse(record["passed"])
        unknown=evaluate_case(case,result(fact_check={}),10)
        self.assertIn("factuality",unknown["unknown_dimensions"])
        self.assertFalse(unknown["passed"])


if __name__ == "__main__":
    unittest.main()
