"""Knowledge/chat integration using production nodes, real MemorySaver, fake I/O."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from starlette.requests import Request

from app.capabilities.retrieval.answer import REFUSAL_TEXT
from app.domain.common.results import ToolResult
from app.domain.user.models import UserContext
from app.models.requests import ChatRequest
from app.routes import chat
from app.workflows.chat_agent import TASK_TO_NODE
from app.workflows.chat_nodes_factory import create_all_nodes
from app.workflows.chat_state import ChatState
from enterprise_rag import config as rag_config
from enterprise_rag.pipeline import query as enterprise_query
from enterprise_rag.query import rewrite_query


USER = UserContext(user_id="knowledge_tester", name="测试用户", role="employee", department="技术部")
QUESTION = "澜图公司共有几个部门？"
CHUNKS = [{"doc_id": f"department_{index}", "chunk_id": f"department_{index}#c0",
           "title": "部门组织清单", "section": name, "version": "2026", "doc_updated_at": "2026-10-21",
           "text": f"部门名称：{name}。", "score": 1.0}
          for index, name in enumerate(("研发部", "财务部", "销售部", "人事部"), 1)]


def retrieval_result(question=QUESTION, chunks=None, complete=True):
    chunks = CHUNKS if chunks is None else chunks
    return {"chunks": chunks, "debug": {"evidence_budget_complete": complete, "top_score": 1.0},
            "query": rewrite_query(question), "as_of": None, "terms": [], "candidates": len(chunks)}


class FakeLLM:
    def __init__(self, *, oa_failure=False):
        self.oa_failure = oa_failure
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(("invoke", prompt))
        if "意图解析器" in prompt:
            intent = "general_chat" if "用户消息：你好" in prompt else "policy_qa"
            if "用户消息：确认" in prompt:
                intent = "confirm_action"
            return SimpleNamespace(content=json.dumps({"intent": intent, "extracted_info": {}}))
        return SimpleNamespace(content="旧消息摘要。")

    async def ainvoke(self, prompt):
        self.calls.append(("ainvoke", prompt))
        if self.oa_failure:
            raise RuntimeError("DO_NOT_EXPOSE_PROVIDER_INTERNAL")
        return SimpleNamespace(content="年假为5天 [1]。")

    async def astream(self, prompt):
        self.calls.append(("astream", prompt))
        yield SimpleNamespace(content="你好")
        yield SimpleNamespace(content="。")


def build_graph(llm):
    """Mirror production wiring while replacing only its external checkpointer."""
    nodes = create_all_nodes(llm)
    builder = StateGraph(ChatState)
    for name, node in nodes.items():
        if name != "route_task_execution":
            builder.add_node(name, node)
    builder.set_entry_point("intent_recognition")
    builder.add_edge("intent_recognition", "task_router")
    builder.add_edge("task_router", "task_executor")
    builder.add_conditional_edges("task_executor", nodes["route_task_execution"], TASK_TO_NODE)
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "response_generator")
    for target in TASK_TO_NODE.values():
        if target != "retrieve":
            builder.add_edge(target, "response_generator")
    builder.add_edge("response_generator", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=MemorySaver())


class MemoryRedis:
    def __init__(self): self.rows = {}
    def hget(self, key, field): return self.rows.get(key, {}).get(field)
    def hset(self, key, field, value): self.rows.setdefault(key, {})[field] = value
    def expire(self, key, seconds): return True


async def run_route(graph, question, *, redis=None, thread=None):
    redis = MemoryRedis() if redis is None else redis
    request = Request({"type": "http", "headers": []})
    with patch("app.routes.chat.get_redis_client", return_value=redis), \
         patch("app.routes.chat.check_rate_limit", return_value=(True, 99, 0)), \
         patch("app.routes.chat.get_chat_graph", new=AsyncMock(return_value=graph)):
        response = await chat.chat_stream(ChatRequest(user_message=question, thread_id=thread), request, USER)
        blocks = [block async for block in response.body_iterator]
    events = []
    for block in blocks:
        kind = next(line[7:] for line in block.splitlines() if line.startswith("event: "))
        data = "\n".join(line[6:] for line in block.splitlines() if line.startswith("data: "))
        events.append((kind, json.loads(data)))
    return events, redis


def final_reply(events):
    return next(data for kind, data in events if kind == "complete")


class KnowledgeFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_enterprise_api_and_chat_share_real_reasoning_and_citations(self):
        llm = FakeLLM()
        graph = build_graph(llm)
        with patch.object(rag_config, "USE_LLM", False), \
             patch("enterprise_rag.retrieval.retrieve", return_value=retrieval_result()), \
             patch("enterprise_rag.pipeline.retrieve", return_value=retrieval_result()):
            api = enterprise_query(QUESTION, user=USER, use_llm=False)
            events, _ = await run_route(graph, QUESTION)
        complete = final_reply(events)
        self.assertEqual(api["method"], "deterministic_reasoning")
        self.assertIn("共 4 个部门", api["answer"])
        self.assertTrue(complete["response"].startswith(api["answer"] + "\n\n来源："))
        self.assertEqual(api["citations"], complete["citations"])
        self.assertEqual("".join(data["token"] for kind, data in events if kind == "token"), complete["response"])
        self.assertEqual(complete["streaming_mode"], "verified")
        self.assertFalse(complete["stream_reset"])
        # The intent parser still uses a fake model; enterprise generation did
        # not accidentally call the OA LLM supplied to the graph.
        self.assertEqual([name for name, _ in llm.calls], ["invoke"])

    async def test_oa_verified_tokens_include_final_source_summary(self):
        graph = build_graph(FakeLLM())
        chunks = [{"doc_id": "oa_leave", "chunk_id": "oa_leave#c0", "title": "员工手册",
                   "section": "年假", "version": "2026", "text": "年假为5天。", "score": 100.0}]
        with patch("app.workflows.chat_nodes.policy_qa.retrieve", return_value=chunks), \
             patch("enterprise_rag.retrieval.retrieve", side_effect=AssertionError("wrong namespace")):
            events, _ = await run_route(graph, "年假几天？")
        complete = final_reply(events)
        self.assertIn("来源：", complete["response"])
        self.assertEqual("".join(data["token"] for kind, data in events if kind == "token"), complete["response"])
        self.assertEqual(complete["streaming_mode"], "verified")
        self.assertEqual(complete["citations"][0]["doc_id"], "oa_leave")

    async def test_followup_context_uses_user_topic_but_never_assistant_claim_as_evidence(self):
        graph = build_graph(FakeLLM())
        config = {"configurable": {"thread_id": "followup"}}
        await graph.aupdate_state(config, {"messages": [
            {"role": "user", "content": "澜图公司组织架构是什么？"},
            {"role": "assistant", "content": "UNTRUSTED_ASSISTANT_CLAIM 公司有999个部门。"},
        ]})
        redis = MemoryRedis()
        redis.hset(chat._session_key(USER.user_id), "followup", json.dumps({"title": "topic"}))
        with patch.object(rag_config, "USE_LLM", False), \
             patch("enterprise_rag.retrieval.retrieve", return_value=retrieval_result()) as retrieval:
            events, _ = await run_route(graph, "那它共有几个部门？", thread="followup", redis=redis)
        actual_question = retrieval.call_args.args[0]
        self.assertIn("澜图公司组织架构是什么", actual_question)
        self.assertIn("追问：那它共有几个部门", actual_question)
        self.assertNotIn("UNTRUSTED_ASSISTANT_CLAIM", actual_question)
        self.assertNotIn("999", final_reply(events)["response"])

    async def test_empty_enterprise_retrieval_never_falls_back_to_oa(self):
        graph = build_graph(FakeLLM())
        with patch("enterprise_rag.retrieval.retrieve", return_value=retrieval_result(chunks=[])), \
             patch("app.workflows.chat_nodes.policy_qa.retrieve", side_effect=AssertionError("must not fall back")) as oa:
            events, _ = await run_route(graph, QUESTION)
        oa.assert_not_called()
        complete = final_reply(events)
        self.assertEqual(complete["response"], REFUSAL_TEXT)
        self.assertEqual(complete["citations"], [])
        self.assertEqual(complete["streaming_mode"], "verified")

    async def test_oa_provider_failure_does_not_stream_raw_exception(self):
        graph = build_graph(FakeLLM(oa_failure=True))
        with patch("app.workflows.chat_nodes.policy_qa.retrieve", return_value=[{"text": "年假5天", "score": 100.0}]):
            events, _ = await run_route(graph, "年假几天？")
        complete = final_reply(events)
        self.assertEqual(complete["response"], REFUSAL_TEXT)
        self.assertNotIn("DO_NOT_EXPOSE", json.dumps(events))
        self.assertEqual(complete["citations"], [])

    async def test_enterprise_model_failure_fallback_is_same_for_api_and_chat(self):
        graph = build_graph(FakeLLM())
        question = "澜图公司总部在哪里？"
        chunks = [{"doc_id": "headquarters", "chunk_id": "headquarters#c0", "title": "公司简介", "section": "总部",
                   "version": "2026", "text": "澜图公司总部位于苏州。", "score": 1.0}]
        result = retrieval_result(question, chunks)
        with patch.object(rag_config, "USE_LLM", True), \
             patch("enterprise_rag.retrieval.retrieve", return_value=result), \
             patch("enterprise_rag.pipeline.retrieve", return_value=result), \
             patch("enterprise_rag.answer._llm_answer", side_effect=RuntimeError("DO_NOT_EXPOSE_PROVIDER_INTERNAL")), \
             patch("enterprise_rag.answer._focus_supported", return_value=True):
            api = enterprise_query(question, user=USER, use_llm=True)
            events, _ = await run_route(graph, question)
        complete = final_reply(events)
        self.assertEqual(api["method"], "extractive_fallback")
        self.assertEqual(api["model_error"], "RuntimeError")
        self.assertTrue(complete["response"].startswith(api["answer"]))
        self.assertNotIn("DO_NOT_EXPOSE", json.dumps(events))
        self.assertIn("苏州", complete["response"])

    async def test_enterprise_generation_crash_returns_only_generic_sse_error(self):
        graph = build_graph(FakeLLM())
        with patch("enterprise_rag.retrieval.retrieve", return_value=retrieval_result()), \
             patch("enterprise_rag.answer.generate_answer", side_effect=RuntimeError("DO_NOT_EXPOSE_ENGINE_INTERNAL")):
            events, _ = await run_route(graph, QUESTION)
        self.assertIn("error", [kind for kind, _ in events])
        self.assertNotIn("complete", [kind for kind, _ in events])
        self.assertNotIn("DO_NOT_EXPOSE", json.dumps(events))
        self.assertEqual([data for kind, data in events if kind == "token"], [])

    async def test_knowledge_then_chat_clears_every_current_turn_knowledge_field(self):
        graph = build_graph(FakeLLM())
        with patch.object(rag_config, "USE_LLM", False), \
             patch("enterprise_rag.retrieval.retrieve", return_value=retrieval_result()):
            first, redis = await run_route(graph, QUESTION)
        thread = final_reply(first)["thread_id"]
        events, _ = await run_route(graph, "你好", redis=redis, thread=thread)
        complete = final_reply(events)
        state = await graph.aget_state({"configurable": {"thread_id": thread}})
        self.assertEqual(complete["citations"], [])
        self.assertEqual(state.values["retrieved_chunks"], [])
        self.assertFalse(state.values["grounded"])
        self.assertEqual(state.values["retrieval_question"], "")
        self.assertEqual(state.values["knowledge_namespace"], "")
        self.assertEqual(state.values["retrieval_meta"], {})
        self.assertEqual(state.values["answer_method"], "")
        self.assertEqual(complete["streaming_mode"], "model")

    async def test_pending_confirmation_survives_turn_reset_and_executes_with_original_action_id(self):
        graph = build_graph(FakeLLM())
        config = {"configurable": {"thread_id": "confirmation"}}
        pending = {"action_id": "action_stable_id", "action_type": "submit_leave", "draft": {"leave_type": "事假"}}
        await graph.aupdate_state(config, {"pending_action": pending, "last_result": {"old": True}})
        redis = MemoryRedis()
        redis.hset(chat._session_key(USER.user_id), "confirmation", json.dumps({"title": "confirmation"}))
        with patch("app.workflows.chat_nodes.execute.leave_service.submit", return_value=ToolResult.success(message="已提交")) as submit:
            events, _ = await run_route(graph, "确认", redis=redis, thread="confirmation")
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(submit.call_args.kwargs["idempotency_key"], "action_stable_id")
        complete = final_reply(events)
        self.assertEqual(complete["response"], "已提交")
        self.assertIsNone(complete["pending_action"])
        self.assertTrue(complete["last_result"]["ok"])


if __name__ == "__main__":
    unittest.main()
