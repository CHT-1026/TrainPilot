import json
import unittest

from model.test.test_code.qwen_tool_protocol import parse_tool_output, template_messages


class ToolProtocolChecks(unittest.TestCase):
    def test_multiple_calls_become_openai_objects(self):
        message = parse_tool_output('<tool_call>{"name":"search","arguments":{"query":"员工"}}</tool_call>\n'
                                    '<tool_call>{"name":"list","arguments":{}}</tool_call>')
        self.assertEqual([c["function"]["name"] for c in message["tool_calls"]], ["search", "list"])
        converted = template_messages([message])
        self.assertEqual(converted[0]["tool_calls"][0]["function"]["arguments"], {"query": "员工"})
        self.assertIsInstance(message["tool_calls"][0]["function"]["arguments"], str)
        self.assertEqual(converted[0]["content"], "")

    def test_incomplete_invalid_or_nonobject_calls_rejected(self):
        for value in ['<tool_call>{}', '<tool_call>{"name":"read","arguments":"path"}</tool_call>',
                      '<tool_call>{"name":"read","arguments":{},"extra":1}</tool_call>']:
            with self.assertRaises(ValueError):
                parse_tool_output(value)

    def test_text_does_not_become_a_tool_call(self):
        self.assertNotIn("tool_calls", parse_tool_output('我将搜索资料。'))

    def test_tool_looking_text_inside_thinking_is_not_executed(self):
        message = parse_tool_output('<think><tool_call>{"name":"list","arguments":{}}</tool_call></think>最终回答')
        self.assertNotIn("tool_calls", message)
        self.assertEqual(message["content"], "最终回答")
        self.assertIn("<tool_call>", message["reasoning_content"])

    def test_bare_compat_requires_exact_schema_and_declared_name(self):
        text = '{"name":"list","arguments":{}}'
        self.assertNotIn("tool_calls", parse_tool_output(text))
        self.assertIn("tool_calls", parse_tool_output(text, bare_names={"list"}))
        self.assertNotIn("tool_calls", parse_tool_output(text, bare_names={"read"}))
        self.assertNotIn("tool_calls", parse_tool_output('{"name":"list","arguments":{},"extra":1}', bare_names={"list"}))


if __name__ == "__main__":
    unittest.main()
