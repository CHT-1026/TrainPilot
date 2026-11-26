"""字段差异分类检查；不加载模型、不改变训练奖励。"""
import unittest

from model.trainer.reward_v2 import field_differences


class RewardFieldChecks(unittest.TestCase):
    def test_correct_example(self):
        result = field_differences(
            {"reason": "办理证件", "start_date": "2027-04-02"},
            {"leave_type": "事假", "reason": "办理证件", "start_date": "2027-04-01"},
        )
        self.assertEqual(result["correct"], ["reason"])

    def test_four_kinds_are_distinct(self):
        expected = {"app_id": "leave_A", "decision": "通过", "comments": "材料已核对"}
        predicted = {"app_id": "leave_A", "decision": "驳回", "reason": "自填原因"}
        self.assertEqual(field_differences(predicted, expected), {
            "correct": ["app_id"], "missing": ["comments"],
            "wrong": ["decision"], "extra": ["reason"],
        })

    def test_exact_match_and_key_order(self):
        self.assertEqual(field_differences({"decision": "通过", "app_id": "leave_A"},
                                         {"app_id": "leave_A", "decision": "通过"}),
                         {"correct": ["app_id", "decision"], "missing": [], "wrong": [], "extra": []})

    def test_present_null_is_wrong_not_missing(self):
        self.assertEqual(field_differences({"app_id": None}, {"app_id": "leave_A"}),
                         {"correct": [], "missing": [], "wrong": ["app_id"], "extra": []})

    def test_unrequested_null_is_extra(self):
        self.assertEqual(field_differences({"comments": None}, {}),
                         {"correct": [], "missing": [], "wrong": [], "extra": ["comments"]})

    def test_empty_input_does_not_invent_fields(self):
        self.assertEqual(field_differences({}, {}),
                         {"correct": [], "missing": [], "wrong": [], "extra": []})


if __name__ == "__main__":
    unittest.main()
