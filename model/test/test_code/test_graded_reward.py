import json
import unittest

from model.trainer.reward_v2 import graded_intent_reward


class GradedRewardChecks(unittest.TestCase):
    def setUp(self):
        self.gold = {"intent": "approve", "extracted_info": {"app_id": "leave_A", "decision": "通过", "comments": "材料已核对"}}

    def score(self, info, intent="approve", **kw):
        return graded_intent_reward(json.dumps({"intent": intent, "extracted_info": info}, ensure_ascii=False), self.gold, **kw)

    def test_ranking_and_strict_criterion(self):
        full = self.score(self.gold["extracted_info"])
        partial = self.score({"app_id": "leave_A", "decision": "通过"})
        missing = self.score({"decision": "通过", "comments": "材料已核对"})
        wrong = self.score({"app_id": "leave_B", "decision": "通过", "comments": "材料已核对"})
        self.assertEqual(full["reward"], 1.)
        self.assertGreater(full["reward"], partial["reward"])
        self.assertGreater(partial["reward"], missing["reward"])
        self.assertGreater(missing["reward"], wrong["reward"])
        self.assertTrue(wrong["severe_error"])
        self.assertFalse(wrong["eligible"])
        self.assertEqual([x["strict_reward"] for x in (full, partial, missing, wrong)], [1.,0.,0.,0.])

    def test_wrong_action_cannot_collect_field_bonus(self):
        self.assertEqual(self.score(self.gold["extracted_info"], intent="delete_application")["reward"], -1.)
        self.assertFalse(self.score({}, intent="clarify")["severe_error"])
        gold = {"intent": "clarify", "extracted_info": {}}
        self.assertTrue(graded_intent_reward('{"intent":"confirm_action","extracted_info":{}}', gold)["severe_error"])

    def test_protocol_truncation_and_duplicate_fail(self):
        self.assertEqual(self.score(self.gold["extracted_info"], ended=False)["reward"], -1.)
        for text in ('not JSON', '{"intent":"approve","intent":"approve","extracted_info":{}}'):
            self.assertEqual(graded_intent_reward(text, self.gold)["reward"], -1.)
        self.assertEqual(self.score({"app_id":None})["level"], "invalid_schema")

    def test_extras_and_wrong_fields_cannot_improve_score(self):
        clean=self.score({"app_id":"leave_A","decision":"通过"})
        extra=self.score({"app_id":"leave_A","decision":"通过","reason":"编造"})
        self.assertLess(extra["reward"],clean["reward"])
        self.assertEqual(self.score({"app_id":"leave_A","decision":"驳回","comments":"材料已核对"})["reward"],-1.)

    def test_empty_expected_fields_are_complete_only_when_empty(self):
        gold={"intent":"general_chat","extracted_info":{}}
        good=graded_intent_reward('{"intent":"general_chat","extracted_info":{}}',gold)
        self.assertEqual(good["reward"],1.)
        bad=graded_intent_reward('{"intent":"general_chat","extracted_info":{"app_id":"leave_A"}}',gold)
        self.assertTrue(bad["severe_error"])


if __name__ == "__main__":
    unittest.main()
