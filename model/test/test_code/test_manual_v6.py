"""交互入口的输入契约检查，不加载大模型。"""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import unittest

from model.test.test_code.manual_v6 import interact, make_messages


class ManualV6Checks(unittest.TestCase):
    def run_cli(self, commands):
        captured = []
        commands = iter(commands)
        with redirect_stdout(StringIO()):
            interact(lambda messages: captured.append(messages) or "测试回复",
                     Namespace(mode="intent"), [], read=lambda _: next(commands))
        return captured

    def test_intent_uses_actual_rules_and_explicit_context(self):
        context = {"current_date": "2026-10-30", "pending_action": {
            "action_type": "cancel_leave", "draft": {"app_id": "A123"}}}
        messages = make_messages("intent", "放弃撤回，申请保留", context, [])
        self.assertEqual(len(messages), 1)
        text = messages[0]["content"]
        self.assertIn("TrainPilot 的意图解析模块", text)
        self.assertIn('"app_id": "A123"', text)
        self.assertTrue(text.endswith("用户输入：\n放弃撤回，申请保留"))

    def test_context_is_not_inferred_from_generated_reply(self):
        calls = self.run_cli(["你好", "确认", "/quit"])
        self.assertEqual(len(calls), 2)
        self.assertIn('"pending_action": null', calls[1][0]["content"])
        self.assertNotIn("测试回复", calls[1][0]["content"])

    def test_chat_history_and_reset(self):
        calls = self.run_cli(["/mode chat", "你好", "你是谁", "/reset", "重新开始", "/quit"])
        self.assertEqual([m["role"] for m in calls[1]], ["system", "user", "assistant", "user"])
        self.assertEqual([m["role"] for m in calls[2]], ["system", "user"])

    def test_context_command_and_invalid_update(self):
        calls = self.run_cli(['/context {"current_date":"2026-10-30","pending_action":'
                              '{"action_type":"cancel_leave","draft":{"app_id":"A123"}}}',
                              "/context []", "确认", "/quit"])
        self.assertIn('"app_id": "A123"', calls[0][0]["content"])


if __name__ == "__main__":
    unittest.main()
