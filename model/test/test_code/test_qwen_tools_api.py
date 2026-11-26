from http.server import HTTPServer
import json
import threading
import unittest
from urllib.request import Request, build_opener, ProxyHandler

from model.test.test_code.qwen_agent_api import make_handler


class ToolAPIChecks(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.text = '<tool_call>{"name":"search","arguments":{"query":"员工"}}</tool_call>'
        self.reason = "eos"
        def infer(messages, limit, tools=None):
            self.calls.append((messages, limit, tools))
            return {"text": self.text, "raw_text": self.text, "input_tokens": 5,
                    "generated_tokens": 10, "stop_reason": self.reason}
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(infer, "trainpilot-lora", metadata={
            "adapter_sha256": "test"}, tool_inference={"trainpilot-lora": infer}))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = build_opener(ProxyHandler({}))

    def request(self, stream=False):
        body = {"model": "trainpilot-lora", "messages": [{"role": "user", "content": "员工上级"}],
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "max_tokens": 50, "stream": stream}
        with self.opener.open(Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})) as response:
            return response.read().decode()

    def test_complete_call_has_id_arguments_finish_and_raw_output(self):
        result = json.loads(self.request())
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"query": "员工"})
        self.assertEqual(result["local_trace"]["raw_output"], self.text)
        self.assertTrue(self.calls[0][2])

    def test_stream_call_has_index_and_done(self):
        lines = [s[6:] for s in self.request(True).splitlines() if s.startswith("data: ")]
        self.assertEqual(json.loads(lines[0])["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
        self.assertEqual(lines[-1], "[DONE]")

    def test_bare_compat_is_explicitly_marked_not_hidden(self):
        self.text = '{"name":"search","arguments":{"query":"员工"}}'
        result = json.loads(self.request())
        self.assertEqual(result["local_trace"]["tool_protocol_used"], "bare_json_compat")
        self.assertEqual(result["local_trace"]["raw_output"], self.text)

    def test_malformed_and_truncated_outputs_do_not_become_calls(self):
        self.text = '<tool_call>{"name":"search"}</tool_call>'
        result = json.loads(self.request())
        self.assertNotIn("tool_calls", result["choices"][0]["message"])
        self.assertTrue(result["local_trace"]["tool_parse_error"])
        self.reason = "max_new_tokens"
        self.text = '<tool_call>{"name":"search","arguments":{}}</tool_call>'
        result = json.loads(self.request())
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        self.assertNotIn("tool_calls", result["choices"][0]["message"])


if __name__ == "__main__":
    unittest.main()
