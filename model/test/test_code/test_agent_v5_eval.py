"""检查评分器区分格式、内容与引用，不修复模型输出来提高得分。"""
import json
import unittest

from model.test.test_code.evaluate_agent_v5 import grade


class V5EvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.case = {"mode": "enterprise_rag", "category": "fact_query", "evidence": [{"text": "负载20kg"}],
                     "expected_claims": [{"text": "负载20 kg。", "sources": [1]}]}

    def test_correct_content_and_reference(self):
        result = grade(self.case, json.dumps({"refused": False, "claims": self.case["expected_claims"]}))
        self.assertTrue(result["reference_exact"])

    def test_schema_does_not_mean_correct_content(self):
        result = grade(self.case, '{"refused":false,"claims":[{"text":"负载200 kg。","sources":[1]}]}')
        self.assertTrue(result["schema_and_refs"])
        self.assertFalse(result["reference_exact"])

    def test_refusal_is_wrong_when_answer_is_in_evidence(self):
        result = grade(self.case, '{"refused":true,"claims":[]}')
        self.assertTrue(result["schema_and_refs"])
        self.assertFalse(result["refusal_decision_correct"])

    def test_out_of_range_and_boolean_reference_rejected(self):
        for source in ([2], [True]):
            result = grade(self.case, json.dumps({"refused": False, "claims": [{"text": "负载20 kg。", "sources": source}]}))
            self.assertFalse(result["schema_and_refs"])

    def test_empty_evidence_refusal(self):
        self.case.update(evidence=[], expected_claims=[])
        self.assertTrue(grade(self.case, '{"refused":true,"claims":[]}')["reference_exact"])


if __name__ == "__main__":
    unittest.main()
