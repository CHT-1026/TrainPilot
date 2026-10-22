"""本机模型接口协议检查；使用固定返回，不加载真实模型。"""
from http.server import HTTPServer
import json
import threading
import unittest
from urllib.request import Request, urlopen

from test.test_code.qwen_agent_api import make_handler


class LocalModelAPIChecks(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def infer(messages, max_tokens):
            self.calls.append((messages, max_tokens))
            return {"text": "测试回复", "input_tokens": 5, "generated_tokens": 2,
                    "stop_reason": "max_new_tokens"}

        def lora_infer(messages, max_tokens):
            return {"text": "适配器回复", "input_tokens": 5, "generated_tokens": 2, "stop_reason": "eos"}

        self.server = HTTPServer(("127.0.0.1", 0), make_handler(
            infer, "trainpilot-sft", {"trainpilot-lora": lora_infer}))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, stream):
        messages = [{"role": "system", "content": "保持身份"}, {"role": "user", "content": "你好"}]
        request = Request(self.url + "/v1/chat/completions", data=json.dumps({
            "model": "trainpilot-sft", "messages": messages, "max_tokens": 24,
            "stream": stream}).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            return response.read().decode("utf-8"), messages

    def test_nonstream_preserves_messages_and_reports_truncation(self):
        body, messages = self.request(False)
        result = json.loads(body)
        self.assertEqual(self.calls, [(messages, 24)])
        self.assertEqual(result["choices"][0]["message"]["content"], "测试回复")
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        self.assertEqual(result["usage"]["total_tokens"], 7)

    def test_stream_has_content_finish_and_done_events(self):
        body, _ = self.request(True)
        lines = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
        self.assertEqual(json.loads(lines[0])["choices"][0]["delta"]["content"], "测试回复")
        self.assertEqual(json.loads(lines[1])["choices"][0]["finish_reason"], "length")
        self.assertEqual(lines[-1], "[DONE]")

    def test_model_alias_dispatch_and_unknown_model(self):
        request = Request(self.url + "/v1/chat/completions", data=json.dumps({
            "model": "trainpilot-lora", "messages": [{"role": "user", "content": "你好"}]
        }).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            result = json.loads(response.read())
        self.assertEqual(result["model"], "trainpilot-lora")
        self.assertEqual(result["choices"][0]["message"]["content"], "适配器回复")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.calls, [])
        with urlopen(self.url + "/v1/models") as response:
            self.assertEqual({v["id"] for v in json.loads(response.read())["data"]},
                             {"trainpilot-sft", "trainpilot-lora"})

    def test_business_grading_rejects_extra_fields_and_wrong_intent(self):
        from test.test_code.evaluate_agent_lora import score_business
        case = {"mode": "intent_parse", "expected": {"intent": "submit_leave", "extracted_info": {}}}
        self.assertTrue(score_business(case, '{"intent":"submit_leave","extracted_info":{}}')["passed"])
        extra = score_business(case, '{"intent":"submit_leave","extracted_info":{"reason":null}}')
        self.assertTrue(extra["intent_ok"])
        self.assertFalse(extra["passed"])
        self.assertEqual(extra["extra_field_keys"], ["reason"])
        self.assertFalse(score_business(case, '{"intent":"general_chat","extracted_info":{}}')["passed"])
        self.assertFalse(score_business(case, '```json\n{}\n```')["strict_json"])
        identity = {"mode": "assistant_reply", "category": "identity"}
        self.assertIsNone(score_business(identity, "如果原因缺失，我会向用户询问。")["passed"])


if __name__ == "__main__":
    unittest.main()
