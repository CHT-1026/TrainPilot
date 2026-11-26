"""回归评分与配对比较的语义检查，不加载大模型。"""
import json
from pathlib import Path
import unittest

from model.test.test_code.evaluate_grpo_regression import compare, grade, load_cases

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "model/dataset/datasets/trainpilot_agent/grpo_v2_regression/regression_panel.jsonl"


class RegressionChecks(unittest.TestCase):
    def test_frozen_panel_and_reference_answers(self):
        cases, manifest = load_cases(PANEL)
        self.assertEqual(len(cases), 165)
        graded = 0
        for case in cases:
            if case["mode"] == "intent_parse":
                expected = json.dumps(case["expected"], ensure_ascii=False)
            elif case["mode"].endswith("_rag"):
                expected = case["messages"][-1]["content"]
            else:
                continue
            self.assertTrue(grade(case, expected)["passed"], case["id"])
            graded += 1
        self.assertEqual(graded, 147)
        self.assertEqual(manifest["selection"], "original_frozen_v5_panel_all_165")

    def test_correct_format_does_not_reward_wrong_action(self):
        case = {"mode": "intent_parse", "expected": {"intent": "clarify", "extracted_info": {}}}
        score = grade(case, '{"intent":"confirm_action","extracted_info":{}}')
        self.assertTrue(score["strict_json"])
        self.assertFalse(score["intent_ok"])
        self.assertFalse(score["passed"])
        duplicate = '{"intent":"clarify","intent":"clarify","extracted_info":{}}'
        self.assertFalse(grade(case, duplicate)["passed"])

    def test_rag_wrong_reference_and_truncation(self):
        cases, _ = load_cases(PANEL)
        case = next(c for c in cases if c["mode"] == "enterprise_rag" and c["expected_claims"])
        text = case["messages"][-1]["content"]
        score = grade(case, text, ended=False)
        self.assertFalse(score["passed"])
        self.assertFalse(score["reference_exact"])
        wrong = json.loads(text)
        wrong["claims"][0]["sources"] = [999]
        self.assertFalse(grade(case, json.dumps(wrong, ensure_ascii=False))["passed"])

    def test_pairing_counts_stop_change_and_rejects_wrong_ids(self):
        case = {"id":"a", "mode":"intent_parse", "category":"state",
                "expected":{"intent":"clarify","extracted_info":{}}}
        text = '{"intent":"clarify","extracted_info":{}}'
        def item(ended):
            return {"case":case,"text":text,"seconds":0.,
                    "stop_reason":"eos" if ended else "max_new_tokens", **grade(case,text,ended)}
        result = compare([item(False)], [item(True)])
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["regressed"], 0)
        self.assertEqual(len(result["changed_outputs"]), 1)
        with self.assertRaises(ValueError):
            compare([item(True),item(True)], [item(True)])
        bad=item(True)
        bad["case"]={**case,"id":"different"}
        with self.assertRaises(ValueError):
            compare([item(True)],[bad])


if __name__ == "__main__":
    unittest.main()
