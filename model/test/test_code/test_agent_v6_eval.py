"""V6对照评测的冻结题目、成对比较和开放回答统计检查。"""
import unittest

from model.test.test_code.evaluate_agent_v6 import compare, freeze_cases, summarize


class V6EvaluationChecks(unittest.TestCase):
    def test_new_and_old_suites_are_separate(self):
        cases, hashes = freeze_cases()
        self.assertEqual(sum(r["suite"] == "v6_holdout" for r in cases), 156)
        self.assertEqual(sum(r["suite"] == "old_regression" for r in cases), 165)
        self.assertEqual(len(hashes), 2)

    def test_paired_changes_preserve_unscored_natural_answers(self):
        case = {"id": "natural", "mode": "agent_reply", "category": "identity"}
        def row(text):
            return {"suite": "v6_holdout", "case": case, "text": text,
                    "stop_reason": "eos", "passed": None, "seconds": 0,
                    "mentions_trainpilot": "TrainPilot" in text}
        result = compare([row("我是助手")], [row("我是TrainPilot")])
        self.assertEqual(result["improved"], 0)
        self.assertEqual(result["regressed"], 0)
        self.assertEqual(len(result["changed_outputs"]), 1)
        result = summarize([row("我是TrainPilot")])
        self.assertEqual(result["v6_holdout"]["agent_reply"]["mentions_trainpilot"], 1)
        self.assertNotIn("passed", result["v6_holdout"]["agent_reply"])

    def test_pairing_does_not_compare_different_cases(self):
        with self.assertRaises(AssertionError):
            compare([{"case": {"id": "a"}}], [{"case": {"id": "b"}}])


if __name__ == "__main__":
    unittest.main()
