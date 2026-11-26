import json
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from agent.app.capabilities.retrieval.corpus_tools import CorpusTools, build_snapshot
from agent.app.capabilities.retrieval.tool_chat import run_tool_chat
from agent.app.capabilities.retrieval.index_tools import IndexTools
from agent.app.domain.user.models import UserContext


def reply(content=None, calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": finish}]}


def call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class ToolChatChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "org.md").write_text("员工甲的上级是主管乙。\n主管乙担任研发主管。", encoding="utf-8")
        build_snapshot(corpus, root / "snapshot")
        self.backend = CorpusTools(root / "snapshot")

    def test_model_selects_search_then_read_and_receives_matching_tool_ids(self):
        stage = 0
        def complete(messages, tools, limit, timeout):
            nonlocal stage
            stage += 1
            if stage == 1:
                return reply(calls=[call("search", {"query": "员工甲"})], finish="tool_calls")
            self.assertEqual(messages[-1]["role"], "tool")
            if stage == 2:
                self.assertEqual(messages[-1]["tool_call_id"], "call_1")
                doc = json.loads(messages[-1]["content"])["hits"][0]["doc_id"]
                return reply(calls=[call("read", {"doc_id": doc}, "call_2")], finish="tool_calls")
            self.assertEqual(messages[-1]["tool_call_id"], "call_2")
            segment = json.loads(messages[-1]["content"])["segments"][0]
            return reply(json.dumps({"status": "answerable", "answer": "主管乙。", "citations": [{
                "doc_id": segment["doc_id"], "locator": segment["locator"], "quote": segment["text"]}]}, ensure_ascii=False))
        result = run_tool_chat("员工甲上级是谁", self.backend, complete)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["tool_calls"], 2)

    def test_fabricated_citation_is_not_accepted(self):
        final = {"status": "answerable", "answer": "伪造", "citations": [{"doc_id": "org", "locator": "line:1@0", "quote": "员工甲"}]}
        result = run_tool_chat("谁", self.backend, lambda *args: reply(json.dumps(final)))
        self.assertEqual(result["status"], "invalid_final_format_or_citation")
        self.assertIsNone(result["final"])

    def test_tools_cannot_write_and_argument_errors_return_to_model(self):
        for name, args in [("execute", {"command": "anything"}), ("read", {"doc_id": "org", "limit": 3000}),
                           ("search", {"query": "员工", "user_role": "admin"})]:
            self.assertIn("error", self.backend.run(name, args))
        result = run_tool_chat("谁", self.backend, lambda *args: reply(calls=[call("execute", {})]), max_rounds=2)
        self.assertEqual(result["status"], "round_budget_exhausted")
        self.assertTrue(all(r["tools"][0]["result"]["error"] == "unknown_tool" for r in result["trace"]))

    def test_truncated_call_is_never_executed_and_call_budget_is_enforced(self):
        complete = lambda *args: reply(calls=[call("search", {"query": "员工"})], finish="length")
        self.assertEqual(run_tool_chat("谁", self.backend, complete)["tool_calls"], 0)
        result = run_tool_chat("谁", self.backend, lambda *args: reply(calls=[call("list", {})]), max_calls=0)
        self.assertEqual(result["status"], "tool_budget_exhausted")

    def test_empty_search_can_be_followed_by_different_query(self):
        self.assertEqual(self.backend.run("search", {"query": "zzzzzz"})["hits"], [])
        self.assertTrue(self.backend.run("search", {"query": "员工甲"})["hits"])
        self.assertTrue(self.backend.run("list", {})["documents"])

    def test_production_read_rechecks_acl_and_hides_namespaced_documents(self):
        user = UserContext(user_id="employee", name="员工", role="employee")
        chunks = [{"chunk_id": "1", "doc_id": "public", "namespace": "lantu", "title": "公开", "text": "公开资料", "visibility": "all"},
                  {"chunk_id": "2", "doc_id": "secret", "namespace": "lantu", "title": "秘密", "text": "禁止读取", "visibility": "admin"}]
        index = type("Index", (), {"chunks": chunks})()
        oa = type("Index", (), {"chunks": []})()
        with patch("agent.app.capabilities.retrieval.index_tools.load_index", return_value=index), \
             patch("agent.app.capabilities.retrieval.index_tools.get_index", return_value=oa):
            tools = IndexTools(user)
            self.assertIn("error", tools.run("read", {"doc_id": "lantu:secret"}))
            first = tools.run("read", {"doc_id": "lantu:public"})["segments"][0]
            citation = {"doc_id": first["doc_id"], "locator": first["locator"], "quote": first["text"]}
            chunks[0]["visibility"] = "admin"
            with self.assertRaisesRegex(ValueError, "no_longer_accessible"):
                tools.validate_citations([citation])

    def test_business_knowledge_branch_uses_tools_without_fixed_retrieval(self):
        from agent.app.workflows.chat_nodes.policy_qa import create_policy_nodes
        user = UserContext(user_id="test", name="员工", role="employee")
        state = {"user_message": "澜图智能的业务是什么？", "intent": "policy_qa", "user": user.model_dump(),
                 "messages": [{"role": "user", "content": "澜图智能的业务是什么？"}]}
        settings = SimpleNamespace(rag_tool_calls=True, openai_api_base="http://127.0.0.1:8011/v1",
                                   openai_model="trainpilot-lora", openai_api_key="local", llm_timeout_seconds=30)
        outcome = {"status": "answered", "tool_calls": 1, "trace": [], "final": {
            "status": "answerable", "answer": "测试业务", "citations": [{"doc_id": "a", "locator": "b", "quote": "c"}]}}
        with patch("agent.app.workflows.chat_nodes.policy_qa.get_settings", return_value=settings), \
             patch("agent.enterprise_rag.retrieval.retrieve") as fixed, \
             patch("agent.app.capabilities.retrieval.tool_chat.run_tool_chat", return_value=outcome) as loop:
            nodes = create_policy_nodes(None)
            updated = {**state, **nodes["retrieve"](state)}
            self.assertEqual(updated["knowledge_namespace"], "tools")
            result = asyncio.run(nodes["answer"](updated))
            self.assertEqual(result["answer_method"], "model_tool_loop")
            self.assertEqual(result["knowledge_tool_trace"], outcome)
            self.assertEqual(result["citations"][0]["index"], 1)
            fixed.assert_not_called()
            loop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
