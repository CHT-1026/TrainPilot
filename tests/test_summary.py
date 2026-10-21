"""会话摘要阈值与压缩节点单元测试。"""
import unittest

from app.workflows.chat_nodes.summarize import create_summarize_node, should_summarize


class FakeLLM:
    def invoke(self, prompt):
        return type("Response", (), {"content": "用户查询年假；已提交申请 leave_x。"})()


class SummaryTest(unittest.TestCase):
    def test_threshold(self):
        messages = [{"role": "user", "content": str(index)} for index in range(11)]
        self.assertFalse(should_summarize(messages, 0))
        messages.append({"role": "user", "content": "11"})
        self.assertTrue(should_summarize(messages, 0))
        self.assertFalse(should_summarize(messages, 4))

    def test_node_compresses_old_messages(self):
        state = {
            "messages": [
                {"role": "user" if index % 2 == 0 else "assistant", "content": f"消息{index}"}
                for index in range(14)
            ],
            "summarized_count": 0,
            "summary": "",
        }
        output = create_summarize_node(FakeLLM())["summarize"](state)
        self.assertIn("summary", output)
        self.assertEqual(output["summarized_count"], 6)
        self.assertTrue(output["summary"])

    def test_node_skips_when_below_threshold(self):
        state = {
            "messages": [{"role": "user", "content": "hi"}],
            "summarized_count": 0,
            "summary": "",
        }
        self.assertEqual(create_summarize_node(FakeLLM())["summarize"](state), {})


if __name__ == "__main__":
    unittest.main()
