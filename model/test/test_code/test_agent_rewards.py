"""奖励契约测试：不能靠 JSON、删字段、改动作或伪造执行结果拿到奖励。"""
import json
import unittest

from model.trainer.agent_rewards import intent_reward


class IntentRewardChecks(unittest.TestCase):
    def setUp(self):
        self.gold = {"intent": "approve", "extracted_info": {
            "app_id": "leave_test", "decision": "通过", "comments": "同意安排"}}

    def score(self, value):
        return intent_reward(json.dumps(value, ensure_ascii=False), self.gold)

    def test_complete_correct_protocol_is_rewarded(self):
        self.assertEqual(self.score(self.gold)["reward"], 1.)

    def test_valid_json_alone_is_not_rewarded(self):
        result = self.score({"intent": "general_chat", "extracted_info": {}})
        self.assertTrue(result["format_ok"])
        self.assertEqual(result["reward"], 0.)

    def test_approval_comment_must_preserve_original_text(self):
        for comments in ("安排", "同意安排。", "已批准"):
            result = self.score({**self.gold, "extracted_info": {**self.gold["extracted_info"], "comments": comments}})
            self.assertEqual(result["reward"], 0.)

    def test_wrong_or_missing_object_and_extra_fields_fail(self):
        for fields in ({"decision": "通过", "comments": "同意安排"},
                       {**self.gold["extracted_info"], "app_id": "leave_other"},
                       {**self.gold["extracted_info"], "user_role": "admin"}):
            self.assertEqual(self.score({**self.gold, "extracted_info": fields})["reward"], 0.)

    def test_confirm_and_cancel_are_not_interchangeable(self):
        gold = {"intent": "cancel_action", "extracted_info": {}}
        self.assertEqual(intent_reward('{"intent":"confirm_action","extracted_info":{}}', gold)["reward"], 0.)

    def test_duplicate_keys_fenced_output_and_trailing_text_fail(self):
        raw = json.dumps(self.gold, ensure_ascii=False)
        for text in ('{"intent":"cancel_action","intent":"approve","extracted_info":{}}',
                     '{"intent":"approve","extracted_info":{"app_id":"wrong","app_id":"leave_test",'
                     '"decision":"通过","comments":"同意安排"}}',
                     '```json\n' + raw + '\n```', raw + " 已执行审批"):
            self.assertEqual(intent_reward(text, self.gold)["reward"], 0.)

    def test_invalid_field_types_and_fabricated_results_fail(self):
        for payload in ([], {"intent": [], "extracted_info": {}},
                        {"intent": "approve", "extracted_info": {"comments": None}},
                        {**self.gold, "success": True}):
            self.assertEqual(self.score(payload)["reward"], 0.)

    def test_invalid_gold_is_not_silently_treated_as_model_failure(self):
        with self.assertRaises(ValueError):
            intent_reward("{}", {"intent": "unknown", "extracted_info": {}})


if __name__ == "__main__":
    unittest.main()
