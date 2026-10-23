"""Remaining workflow boundaries: isolated Redis/clock/LLM, no business writes."""
import asyncio
import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from starlette.requests import Request
from app.domain.user.models import UserContext
from app.models.requests import ChatRequest
from app.routes import chat as current_chat
from app.workflows import dialog_state as current_dialog, fact_memory as current_memory
from app.workflows.chat_nodes import response as current_response, routing as current_routing
from tests.workflow_fakes import TransactionRedis

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ["WORKFLOW_SOURCE_ROOT"]) if os.environ.get("WORKFLOW_SOURCE_ROOT") else None
def selected(module, path):
    if SOURCE is None: return module
    spec = importlib.util.spec_from_file_location("baseline_" + module.__name__.replace(".", "_"), SOURCE / path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result
chat = selected(current_chat, "app/routes/chat.py")
dialog = selected(current_dialog, "app/workflows/dialog_state.py")
memory = selected(current_memory, "app/workflows/fact_memory.py")
create_response_nodes = selected(current_response, "app/workflows/chat_nodes/response.py").create_response_nodes
create_routing_nodes = selected(current_routing, "app/workflows/chat_nodes/routing.py").create_routing_nodes
USER = UserContext(user_id="revision2_fake_user", name="隔离用户", role="employee")


class DialogTopicTest(unittest.TestCase):
    def test_generic_department_count_is_not_a_named_department(self):
        self.assertEqual(dialog.extract_entities("那它共有几个部门？"), [])

    def test_same_named_entity_followup_keeps_topic_time(self):
        state = dialog.update_dialog_state(None, "2024年LT-R20负载是多少？")
        state = dialog.update_dialog_state(state, "那LT-R20的价格呢？")
        _, as_of = dialog.apply_context("那它尺寸呢？", [], state)
        self.assertEqual(as_of, "2024-12-31")

    def test_new_model_clears_previous_anchor(self):
        state = dialog.update_dialog_state(None, "2024年LT-R20负载是多少？")
        state = dialog.update_dialog_state(state, "LT-R30负载是多少？")
        state = dialog.update_dialog_state(state, "那价格呢？")
        question, as_of = dialog.apply_context("那价格呢？", [], state)
        self.assertIn("LT-R30", question)
        self.assertNotIn("LT-R20", question)
        self.assertIsNone(as_of)

    def test_new_year_does_not_keep_old_year_in_retrieval_text(self):
        state = dialog.update_dialog_state(None, "2024年LT-R20负载是多少？")
        state = dialog.update_dialog_state(state, "那2025年呢？")
        question, _ = dialog.apply_context("那2025年呢？", [{"role": "user", "content": "2024年LT-R20负载是多少？"}], state)
        self.assertIn("2025年", question)
        self.assertNotIn("2024年", question)
        self.assertIn("LT-R20", question)

    def test_new_department_followup_does_not_resurrect_old_model(self):
        state = dialog.update_dialog_state(None, "LT-R20负载是多少？")
        state = dialog.update_dialog_state(state, "换个话题，研发部负责人是谁？")
        state = dialog.update_dialog_state(state, "那他的上级呢？")
        question, _ = dialog.apply_context("那他的上级呢？", [], state)
        self.assertNotIn("LT-R20", question)
        self.assertIn("研发部", question)


class FactTransactionTest(unittest.TestCase):
    def setUp(self):
        self.redis = TransactionRedis()
        self.patch = patch.object(memory, "_client", return_value=self.redis); self.patch.start()
        self.addCleanup(self.patch.stop)
    def test_failed_insert_keeps_all_old_items(self):
        for n in range(memory.MAX_FACTS): memory.remember_fact(USER, f"old {n}")
        before = self.redis.hgetall(memory._key(USER.user_id))
        self.redis.fail_hset = True
        with self.assertRaises(memory.MemoryUnavailable): memory.remember_fact(USER, "new")
        self.assertEqual(self.redis.hgetall(memory._key(USER.user_id)), before)

    def test_concurrent_remember_preserves_capacity_and_each_committed_fact(self):
        for n in range(memory.MAX_FACTS):
            old = memory.remember_fact(USER, f"old {n}")
            # 明确是历史记录，避免 Windows 时钟同一 tick 的随机ID排序。
            old["created_at"] = f"2020-01-01T00:00:{n:02d}+08:00"
            self.redis.hset(memory._key(USER.user_id), old["fact_id"], json.dumps(old))
        interleaved = []
        self.redis.after_read = lambda: interleaved.append(memory.remember_fact(USER, "concurrent"))
        latest = memory.remember_fact(USER, "newest")
        facts = memory.list_facts(USER)
        self.assertEqual(len(facts), memory.MAX_FACTS)
        self.assertIn(latest["fact_id"], {f["fact_id"] for f in facts})
        self.assertIn(interleaved[0]["fact_id"], {f["fact_id"] for f in facts})

    def test_delete_during_correction_cannot_resurrect_fact(self):
        fact = memory.remember_fact(USER, "before")
        self.redis.after_read = lambda: memory.forget_fact(USER, fact["fact_id"])
        corrected = memory.correct_fact(USER, fact["fact_id"], "after")
        self.assertIsNone(corrected)
        self.assertEqual(memory.list_facts(USER), [])

    def test_duplicate_explicit_remember_refreshes_ttl_without_duplicate(self):
        fact = memory.remember_fact(USER, "中文")
        self.redis.now = memory._TTL_SECONDS - 1
        same = memory.remember_fact(USER, "中文")
        self.redis.now += 2
        self.assertEqual(fact["fact_id"], same["fact_id"])
        self.assertEqual(len(memory.list_facts(USER)), 1)


class ChatLeaseTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.redis = TransactionRedis()
        self.patch = patch.object(chat, "get_redis_client", return_value=self.redis); self.patch.start()
        self.addCleanup(self.patch.stop)
        memory_patch = patch.object(current_memory, "_client", return_value=self.redis)
        memory_patch.start(); self.addCleanup(memory_patch.stop)
        chat._touch_session(USER.user_id, "owned")
        self.request = Request({"type": "http", "headers": []})
    async def test_storage_outage_is_503_not_local_admission(self):
        with patch.object(chat, "get_redis_client", side_effect=RuntimeError("internal storage secret")):
            with self.assertRaises(HTTPException) as exc: chat._acquire_thread_lock("owned")
        self.assertEqual(exc.exception.status_code, 503)
        self.assertNotIn("secret", exc.exception.detail)

    async def test_anyio_disconnect_scope_still_releases_owned_lease(self):
        import anyio
        async def invoke(*_):
            scope.cancel()
            await anyio.sleep(0)
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={})), ainvoke=invoke)
        with patch.object(chat, "get_chat_graph", new=AsyncMock(return_value=graph)), patch.object(chat, "check_rate_limit", return_value=(True, 1, 0)):
            with anyio.CancelScope() as scope:
                await chat.chat_once(ChatRequest(thread_id="owned", user_message="你好"), self.request, USER)
        self.assertIsNone(self.redis.get(chat._thread_lock_key("owned")))

    async def test_once_renews_lease_while_graph_is_waiting(self):
        started, finish = asyncio.Event(), asyncio.Event()
        async def invoke(*_):
            started.set(); await finish.wait()
            return {"response": "完成", "messages": []}
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={})), ainvoke=invoke)
        with patch.object(chat, "_thread_lock_ttl", return_value=.09), patch.object(chat, "get_chat_graph", new=AsyncMock(return_value=graph)), patch.object(chat, "check_rate_limit", return_value=(True, 1, 0)):
            pending = asyncio.create_task(chat.chat_once(ChatRequest(thread_id="owned", user_message="你好"), self.request, USER))
            await started.wait(); await asyncio.sleep(.055)
            self.assertGreater(self.redis.eval_calls, 0)
            self.assertFalse(chat._acquire_thread_lock("owned"))
            finish.set(); await pending

    async def test_lease_loss_cancels_graph_and_does_not_publish_success(self):
        cancelled = asyncio.Event()
        async def blocked(*_):
            self.redis.set(chat._thread_lock_key("owned"), "other_owner")
            try:
                await asyncio.sleep(.08)
                return {"response": "不应发布", "messages": []}
            finally: cancelled.set()
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={})), ainvoke=blocked)
        with patch.object(chat, "_thread_lock_ttl", return_value=.03), patch.object(chat, "get_chat_graph", new=AsyncMock(return_value=graph)), patch.object(chat, "check_rate_limit", return_value=(True, 1, 0)):
            with self.assertRaises(HTTPException) as exc:
                await chat.chat_once(ChatRequest(thread_id="owned", user_message="你好"), self.request, USER)
        self.assertEqual(exc.exception.status_code, 503)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.redis.get(chat._thread_lock_key("owned")), "other_owner")

    async def test_cancelled_once_cleans_async_graph_before_releasing_lock(self):
        started, cleaned = asyncio.Event(), asyncio.Event()
        async def invoke(*_):
            started.set()
            try: await asyncio.Event().wait()
            finally: cleaned.set()
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={})), ainvoke=invoke)
        with patch.object(chat, "get_chat_graph", new=AsyncMock(return_value=graph)), patch.object(chat, "check_rate_limit", return_value=(True, 1, 0)):
            pending = asyncio.create_task(chat.chat_once(ChatRequest(thread_id="owned", user_message="你好"), self.request, USER))
            await started.wait(); pending.cancel()
            with self.assertRaises(asyncio.CancelledError): await pending
        self.assertTrue(cleaned.is_set())
        self.assertIsNone(self.redis.get(chat._thread_lock_key("owned")))

    async def test_delete_renews_lease_and_keeps_busy_thread_protected(self):
        started, finish = asyncio.Event(), asyncio.Event()
        async def delete(_): started.set(); await finish.wait()
        graph = SimpleNamespace(checkpointer=SimpleNamespace(adelete_thread=delete))
        with patch.object(chat, "_thread_lock_ttl", return_value=.09), patch.object(chat, "get_chat_graph", new=AsyncMock(return_value=graph)):
            pending = asyncio.create_task(chat.delete_chat_session("owned", USER))
            await started.wait(); await asyncio.sleep(.055)
            self.assertGreater(self.redis.eval_calls, 0)
            self.assertFalse(chat._acquire_thread_lock("owned"))
            finish.set(); await pending
        self.assertFalse(chat._owns_thread(USER.user_id, "owned"))

    async def test_sse_lease_loss_has_safe_terminal_error_and_no_complete(self):
        stopped = asyncio.Event()
        async def stream(*_, **kwargs):
            yield "custom", {"type": "token", "delta": "临时正文"}
            self.redis.set(chat._thread_lock_key("owned"), "new_owner")
            try:
                await asyncio.sleep(.08)
                yield "values", {"response": "不应发布", "messages": []}
            finally: stopped.set()
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={})), astream=stream)
        with patch.object(chat, "_thread_lock_ttl", return_value=.03), patch.object(chat, "get_chat_graph", new=AsyncMock(return_value=graph)), patch.object(chat, "check_rate_limit", return_value=(True, 1, 0)):
            response = await chat.chat_stream(ChatRequest(thread_id="owned", user_message="你好"), self.request, USER)
            blocks = [block async for block in response.body_iterator]
        text = "".join(blocks)
        self.assertIn("event: error", text)
        self.assertIn("event: replace", text)
        self.assertNotIn("event: complete", text)
        self.assertNotIn("new_owner", text)
        self.assertTrue(stopped.is_set())
        self.assertEqual(self.redis.get(chat._thread_lock_key("owned")), "new_owner")

    async def test_deleted_ownership_between_check_and_lock_does_not_run_graph(self):
        original = chat._acquire_thread_lock
        def acquire(thread):
            self.redis.hdel(chat._session_key(USER.user_id), thread)
            return original(thread)
        fake_graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(values={})), ainvoke=AsyncMock(return_value={"response":"完成", "messages":[]}))
        with patch.object(chat, "_acquire_thread_lock", side_effect=acquire), patch.object(chat, "get_chat_graph", new=AsyncMock(return_value=fake_graph)) as graph, patch.object(chat, "check_rate_limit", return_value=(True, 1, 0)):
            with self.assertRaises(HTTPException) as error:
                await chat.chat_once(ChatRequest(thread_id="owned", user_message="你好"), self.request, USER)
        self.assertEqual(error.exception.status_code, 403)
        graph.assert_not_called()
        self.assertFalse(chat._owns_thread(USER.user_id, "owned"))


class FinalTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_general_task_takes_precedence_over_old_intent(self):
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="用户文本处理结果")))
        with patch("app.workflows.chat_nodes.response.get_stream_writer", side_effect=RuntimeError()):
            output = await create_response_nodes(model)["response_generator"]({
                "intent": "policy_qa", "current_task": "general_chat", "task_result": "旧任务结果"})
        self.assertEqual(output["response"], "用户文本处理结果")

    async def test_corrected_task_and_published_intent_agree(self):
        state = {"intent": "general_chat", "user_message": "澜图LT-R20的负载是多少？"}
        output = {**state, **create_routing_nodes()["task_router"](state)}
        self.assertEqual(output["current_task"], "policy_qa")
        self.assertEqual(output["intent"], "policy_qa")


if __name__ == "__main__": unittest.main()
