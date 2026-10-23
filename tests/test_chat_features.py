"""Feature regressions with a real in-memory graph; never contact external services."""
import json
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.capabilities.retrieval.answer import REFUSAL_TEXT, format_with_sources, verify_answer
from app.capabilities.retrieval.search import to_citations
from app.domain.leave import repository
from app.domain.leave.state_machine import check_approval_permission, check_view_permission
from app.domain.user.models import UserContext
from app.models.requests import ChatRequest
from app.routes import chat
from app.routes.auth import get_user_context
from app.workflows.chat_nodes.chat import create_chat_nodes
from app.workflows.chat_nodes.response import create_response_nodes
from app.workflows.chat_state import ChatState


USER = UserContext(user_id="audit_user", name="审计用户", role="employee", department="技术部")


def sse_events(text):
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        kind = next(line[7:] for line in block.splitlines() if line.startswith("event: "))
        data = "\n".join(line[6:] for line in block.splitlines() if line.startswith("data: "))
        events.append((kind, json.loads(data)))
    return events


class FakeLLM:
    def __init__(self, *, fail=False):
        self.fail = fail

    async def ainvoke(self, prompt):
        return SimpleNamespace(content="你好。")

    async def astream(self, prompt):
        yield SimpleNamespace(content="你好")
        if self.fail:
            raise RuntimeError("provider internal secret")
        yield SimpleNamespace(content="。")


class GraphFeatureTest(unittest.IsolatedAsyncioTestCase):
    def graph(self, llm):
        builder = StateGraph(ChatState)
        builder.add_node("chat", create_chat_nodes()["general_chat"])
        builder.add_node("response", create_response_nodes(llm)["response_generator"])
        builder.set_entry_point("chat")
        builder.add_edge("chat", "response")
        builder.add_edge("response", END)
        return builder.compile(checkpointer=MemorySaver())

    async def test_new_turn_preserves_history_but_clears_previous_evidence_and_receipt(self):
        graph = self.graph(FakeLLM())
        config = {"configurable": {"thread_id": "history"}}
        await graph.aupdate_state(config, {"messages": [{"role": "user", "content": "年假几天"},
                                                        {"role": "assistant", "content": "5天[1]。"}],
                                          "citations": [{"doc_id": "old"}], "grounded": True,
                                          "retrieved_chunks": [{"text": "old"}], "last_result": {"ok": True}})
        initial, config = await chat.build_chat_initial_state(graph, "history", "你好", USER)
        result = await graph.ainvoke(initial, config)
        self.assertEqual(len(result["messages"]), 4)
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertFalse(result["grounded"])
        self.assertIsNone(result["last_result"])

    async def test_normal_chat_emits_model_deltas_before_final_state(self):
        graph = self.graph(FakeLLM())
        events = [event async for event in graph.astream(
            {"user_message": "你好", "intent": "general_chat", "messages": []},
            {"configurable": {"thread_id": "stream"}}, stream_mode=["custom", "values"],
        )]
        tokens = [event[1]["delta"] for event in events if event[0] == "custom"]
        self.assertEqual(tokens, ["你好", "。"])
        final = [event[1] for event in events if event[0] == "values"][-1]
        self.assertEqual("".join(tokens), final["response"])

    async def test_provider_failure_yields_safe_final_reply(self):
        graph = self.graph(FakeLLM(fail=True))
        result = await graph.ainvoke({"user_message": "你好", "intent": "general_chat"},
                                    {"configurable": {"thread_id": "failure"}})
        # The graph supplies a custom writer even when its caller awaits ainvoke.
        # A provider failure after its first delta must leave a safe final reply.
        self.assertNotIn("secret", result["response"])
        self.assertIn("暂时不可用", result["response"])


from tests.workflow_fakes import TransactionRedis as MemoryRedis

class RouteGraph:
    def __init__(self, *, partial=False):
        self.values = {}
        self.partial = partial

    async def aget_state(self, config): return SimpleNamespace(values=self.values)

    async def astream(self, initial, config, stream_mode):
        if self.partial:
            yield "custom", {"type": "token", "delta": "未完成内容"}
        self.values = {**self.values, **initial,
                       "messages": self.values.get("messages", []) + initial["messages"]
                                   + [{"role": "assistant", "content": "安全回复。"}],
                       "response": "安全回复。", "intent": "general_chat"}
        yield "values", self.values


class RouteFeatureTest(unittest.TestCase):
    def run_route(self, *, partial=False):
        app = FastAPI()
        app.include_router(chat.router)
        app.dependency_overrides[get_user_context] = lambda: USER
        client = TestClient(app)
        redis = MemoryRedis()
        graph = RouteGraph(partial=partial)
        with patch("app.routes.chat.get_redis_client", return_value=redis), \
             patch("app.routes.chat.get_chat_graph", new=AsyncMock(return_value=graph)), \
             patch("app.routes.chat.check_rate_limit", return_value=(True, 59, 0)):
            thread = client.post("/api/chat/session").json()["data"]["thread_id"]
            before = json.loads(redis.hget(chat._session_key(USER.user_id), thread))
            response = client.post("/api/chat/stream", json={"thread_id": thread, "user_message": "你好"})
            after = json.loads(redis.hget(chat._session_key(USER.user_id), thread))
        return before, after, sse_events(response.text)

    def test_counts_actual_messages_and_buffered_reply_is_not_fake_character_stream(self):
        before, after, events = self.run_route()
        self.assertEqual(before["message_count"], 0)
        self.assertEqual(after["message_count"], 2)
        tokens = [data["token"] for kind, data in events if kind == "token"]
        complete = next(data for kind, data in events if kind == "complete")
        self.assertEqual(tokens, [complete["response"]])
        self.assertEqual(complete["streaming_mode"], "buffered")

    def test_partial_provider_output_is_replaced_before_complete(self):
        _, _, events = self.run_route(partial=True)
        visible = ""
        for kind, data in events:
            if kind == "token": visible += data["token"]
            elif kind == "replace": visible = data["response"]
        complete = next(data for kind, data in events if kind == "complete")
        self.assertEqual(visible, complete["response"])
        self.assertTrue(complete["stream_reset"])


class SessionArchiveTest(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(chat.router)
        self.app.dependency_overrides[get_user_context] = lambda: USER
        self.client = TestClient(self.app)
        self.redis = MemoryRedis()
        self.patch = patch("app.routes.chat.get_redis_client", return_value=self.redis)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def create(self):
        return self.client.post("/api/chat/session").json()["data"]["thread_id"]

    def test_archive_and_restore_keep_the_same_session_record(self):
        thread = self.create()
        archived = self.client.patch(f"/api/chat/session/{thread}", json={"archived": True})
        self.assertEqual(archived.status_code, 200)
        self.assertTrue(archived.json()["data"]["archived"])
        self.assertIn("archived_at", archived.json()["data"])
        listed = self.client.get("/api/chat/sessions").json()["data"]
        self.assertTrue(any(item["thread_id"] == thread and item["archived"] for item in listed))

        restored = self.client.patch(f"/api/chat/session/{thread}", json={"archived": False})
        self.assertEqual(restored.status_code, 200)
        self.assertFalse(restored.json()["data"]["archived"])
        self.assertNotIn("archived_at", restored.json()["data"])

    def test_new_message_automatically_restores_an_archived_session(self):
        thread = self.create()
        self.client.patch(f"/api/chat/session/{thread}", json={"archived": True})
        chat._touch_session(USER.user_id, thread, user_message="继续刚才的话题")
        meta = json.loads(self.redis.hget(chat._session_key(USER.user_id), thread))
        self.assertFalse(meta["archived"])
        self.assertNotIn("archived_at", meta)

    def test_other_user_cannot_archive_the_session(self):
        thread = self.create()
        other = UserContext(user_id="other_user", name="其他员工", role="employee")
        self.app.dependency_overrides[get_user_context] = lambda: other
        response = self.client.patch(f"/api/chat/session/{thread}", json={"archived": True})
        self.assertEqual(response.status_code, 403)


class CitationCoverageTest(unittest.TestCase):
    chunks = [{"text": "年假5天", "doc_id": "one", "title": "制度"},
              {"text": "密码12位", "doc_id": "two", "title": "安全"}]

    def test_mixed_invalid_number_is_rejected(self):
        answer, citations, grounded = verify_answer("年假5天 [1]。报销99元 [99]。", self.chunks)
        self.assertEqual(answer, REFUSAL_TEXT)
        self.assertFalse(grounded)
        self.assertEqual(citations, [])

    def test_uncited_second_fact_is_rejected(self):
        self.assertFalse(verify_answer("年假5天 [1]。另有20天病假。", self.chunks)[2])

    def test_original_source_numbers_survive_all_output_formats(self):
        answer, chunks, grounded = verify_answer("密码不少于12位 [2]。", self.chunks)
        self.assertTrue(grounded)
        self.assertEqual(to_citations(chunks)[0]["index"], 2)
        self.assertIn("- [2]", format_with_sources(answer, chunks))

    def test_source_marker_after_sentence_punctuation_is_supported(self):
        self.assertTrue(verify_answer("年假5天。[1] 密码12位。[2]", self.chunks)[2])

    def test_section_labels_and_bullets_do_not_require_extra_markers(self):
        self.assertTrue(verify_answer("结论：\n- 年假5天 [1]。\n- 密码12位 [2]。", self.chunks)[2])


class IdempotencyCursor:
    def __init__(self): self.rows = {}; self.row = None
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, sql, params):
        if "INSERT INTO idempotency_keys" in sql:
            key, user, endpoint, body, _ = params
            self.rows[key] = (user, endpoint, body)
        else:
            scoped, legacy, user, endpoint, _ = params
            self.row = next(({"response_json": self.rows[key][2]} for key in (scoped, legacy)
                             if key in self.rows and self.rows[key][:2] == (user, endpoint)), None)
    def fetchone(self): return self.row


class IdempotencyScopeTest(unittest.TestCase):
    def test_same_key_is_independent_for_users_and_operations(self):
        cursor = IdempotencyCursor()
        conn = SimpleNamespace(cursor=lambda: cursor)
        for user, endpoint in (("alice", "leave.submit"), ("bob", "leave.submit"), ("alice", "leave.cancel")):
            repository.save_idempotent(conn, key="same-key", user_id=user, endpoint=endpoint,
                                       response={"user": user, "operation": endpoint})
        for user, endpoint in (("alice", "leave.submit"), ("bob", "leave.submit"), ("alice", "leave.cancel")):
            self.assertEqual(repository.get_idempotent(conn, "same-key", user_id=user, endpoint=endpoint),
                             {"user": user, "operation": endpoint})
        self.assertIsNone(repository.get_idempotent(conn, "same-key", user_id="eve", endpoint="leave.submit"))

    def test_legacy_keys_are_only_visible_to_original_user_and_operation(self):
        cursor = IdempotencyCursor()
        conn = SimpleNamespace(cursor=lambda: cursor)
        cursor.rows["legacy"] = ("alice", "leave.submit", json.dumps({"legacy": True}))
        self.assertEqual(repository.get_idempotent(conn, "legacy", user_id="alice", endpoint="leave.submit"),
                         {"legacy": True})
        self.assertIsNone(repository.get_idempotent(conn, "legacy", user_id="bob", endpoint="leave.submit"))


class SQLiteCursor:
    def __init__(self, conn): self.cursor = conn.cursor()
    def __enter__(self): return self
    def __exit__(self, *args): self.cursor.close()
    def execute(self, sql, params): self.cursor.execute(sql.replace("%s", "?"), params)
    def fetchone(self):
        row = self.cursor.fetchone()
        return dict(row) if row else None
    def fetchall(self): return [dict(row) for row in self.cursor.fetchall()]


class ReportingLineTest(unittest.TestCase):
    def test_pending_list_and_count_follow_direct_manager_authorization(self):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE users(username TEXT, name TEXT)")
        db.execute("CREATE TABLE leave_applications(id TEXT, employee_id TEXT, manager_id TEXT, department TEXT, status TEXT, approval_step TEXT, created_at TEXT)")
        rows = [("cross", "e1", "manager", "技术部", "pending", "manager", "1"),
                ("other", "e2", "other-manager", "财务部", "pending", "manager", "2"),
                ("fallback", "e3", None, "财务部", "pending", "manager", "3"),
                ("self", "manager", None, "财务部", "pending", "manager", "4")]
        db.executemany("INSERT INTO leave_applications VALUES(?,?,?,?,?,?,?)", rows)
        conn = SimpleNamespace(cursor=lambda: SQLiteCursor(db))
        actor = UserContext(user_id="manager", name="直属主管", role="manager", department="财务部")
        pending = repository.list_pending(conn, actor)
        self.assertEqual([row["id"] for row in pending], ["cross", "fallback"])
        self.assertEqual(repository.count_pending(conn, actor), len(pending))
        self.assertTrue(all(check_approval_permission(actor, row).ok and check_view_permission(actor, row).ok for row in pending))
        foreign = dict(zip(("id", "employee_id", "manager_id", "department", "status", "approval_step", "created_at"), rows[1]))
        self.assertFalse(check_view_permission(actor, foreign).ok)


class MessageValidationTest(unittest.TestCase):
    def test_blank_message_rejected_and_valid_message_trimmed(self):
        with self.assertRaises(ValueError): ChatRequest(user_message="  \n\t")
        self.assertEqual(ChatRequest(user_message="  你好  ").user_message, "你好")


if __name__ == "__main__":
    unittest.main()
