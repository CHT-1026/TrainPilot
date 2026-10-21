"""Read-only feature probes. All model, Redis and SQL access is replaced with fakes."""
import asyncio
import json
import logging
import sys
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from app.capabilities.retrieval.answer import verify_answer, REFUSAL_TEXT
from app.domain.leave import repository
from app.domain.leave.state_machine import check_approval_permission, check_view_permission
from app.domain.user.models import UserContext
from app.models.requests import ChatRequest
from app.routes import chat
from app.routes.auth import get_user_context
from app.services.leave_service import leave_service
from app.workflows.chat_nodes.chat import create_chat_nodes
from app.workflows.chat_nodes.policy_qa import create_policy_nodes
from app.workflows.chat_nodes.response import create_response_nodes
from app.workflows.chat_state import ChatState

OUT = Path(__file__).resolve().parent
RESULTS = []


def record(case_id, feature, passed, actual, reason=""):
    RESULTS.append({"case_id": case_id, "feature": feature, "passed": bool(passed),
                    "actual": actual, "reason": reason})


class MemoryRedis:
    def __init__(self):
        self.rows = {}

    def hget(self, key, field):
        return self.rows.get(key, {}).get(field)

    def hset(self, key, field, value):
        self.rows.setdefault(key, {})[field] = value

    def hdel(self, key, field):
        self.rows.get(key, {}).pop(field, None)

    def hgetall(self, key):
        return dict(self.rows.get(key, {}))

    def expire(self, key, seconds):
        return True


class FakeLLM:
    def __init__(self, text="你好，可以为你做什么？"):
        self.text = text
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(("invoke", prompt))
        return SimpleNamespace(content=self.text)

    async def ainvoke(self, prompt):
        self.calls.append(("ainvoke", prompt))
        return SimpleNamespace(content=self.text)

    async def astream(self, prompt):
        self.calls.append(("astream", prompt))
        for part in (self.text[:5], self.text[5:]):
            yield SimpleNamespace(content=part)


USER = UserContext(user_id="audit_user", name="审计用户", role="employee", department="技术部")
CHUNKS = [{"chunk_id": "audit#c0", "doc_id": "audit", "title": "审计示例", "section": "假期",
           "text": "员工年假为5天。", "version": "2026", "score": 100.0}]


class FakeGraph:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = 0
        self.values = {}

    async def aget_state(self, config):
        return SimpleNamespace(values=self.values)

    async def astream(self, state, config, stream_mode):
        self.calls += 1
        if self.fail:
            raise RuntimeError("INTERNAL_TEST_SECRET_SHOULD_NOT_ESCAPE")
        self.values = {**self.values, **state, "response": "这是一次普通回复。", "intent": "general_chat"}
        yield "values", self.values


def parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        kind = next((line[7:] for line in block.splitlines() if line.startswith("event: ")), "message")
        data = "\n".join(line[6:] for line in block.splitlines() if line.startswith("data: "))
        events.append((kind, json.loads(data)))
    return events


async def probes():
    # Use the actual LangGraph reducer/checkpointer and production initialization/response nodes.
    fake_llm = FakeLLM()
    builder = StateGraph(ChatState)
    builder.add_node("general_chat", create_chat_nodes()["general_chat"])
    builder.add_node("response", create_response_nodes(fake_llm)["response_generator"])
    builder.set_entry_point("general_chat")
    builder.add_edge("general_chat", "response")
    builder.add_edge("response", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "audit_history"}}
    await graph.aupdate_state(config, {"messages": [{"role": "user", "content": "年假几天？"},
                                                   {"role": "assistant", "content": "5天 [1]。"}],
                                       "citations": [{"doc_id": "old_policy"}], "grounded": True,
                                       "retrieved_chunks": CHUNKS, "last_result": {"old": True}})
    initial, config = await chat.build_chat_initial_state(graph, "audit_history", "你好", USER)
    final = await graph.ainvoke(initial, config)
    record("F01", "多轮历史保留", len(final["messages"]) == 4,
           {"message_count": len(final["messages"]), "roles": [m["role"] for m in final["messages"]]})
    record("F02", "跨轮引用隔离", final.get("citations") == [],
           {"current_reply": final["response"], "citations": final.get("citations")},
           "build_chat_initial_state 未重置本轮引用，普通回复节点未覆盖 citations，检查点保留旧引用。")
    record("F03", "跨轮结构化结果隔离", final.get("last_result") is None,
           {"last_result": final.get("last_result"), "grounded": final.get("grounded")},
           "last_result 和 grounded 也保留旧轮结果，会进入 SSE complete。")
    record("F04", "普通对话真实模型流", any(kind == "astream" for kind, _ in fake_llm.calls),
           {"llm_methods": [kind for kind, _ in fake_llm.calls]},
           "普通对话调用同步 invoke，API 仅在整图结束后每3字符发送 token。")

    invalid_llm = FakeLLM("公司有99个部门 [9]。")
    emitted = []
    with patch("app.workflows.chat_nodes.policy_qa.get_stream_writer", return_value=emitted.append):
        output = await create_policy_nodes(invalid_llm)["answer"]({"retrieved_chunks": CHUNKS,
                                                               "user_message": "年假几天？"})
    streamed_text = "".join(item["delta"] for item in emitted)
    record("F05", "拒答前不输出未经校验事实", streamed_text == output["response"],
           {"streamed": streamed_text, "complete": output["response"], "grounded": output["grounded"]},
           "原始模型 token 先发出，verify_answer 随后才拒绝非法引用。客户端期间显示错误事实。")

    mixed, mixed_sources, mixed_ok = verify_answer("年假5天 [1]。报销999万元 [99]。", CHUNKS)
    record("F06", "混合合法非法引用拒绝", not mixed_ok,
           {"answer": mixed, "grounded": mixed_ok, "citation_count": len(mixed_sources)},
           "verify_answer 只要有任一合法编号即 grounded=True，忽略非法编号与无引用断言。")
    false_answer, _, false_ok = verify_answer("员工年假为999天 [1]。", CHUNKS)
    record("F07", "引用语义与数字一致性", not false_ok,
           {"source": CHUNKS[0]["text"], "answer": false_answer, "grounded": false_ok},
           "引用编号合法不代表命题受支持；当前没有事实/数值一致性校验。")

    app = FastAPI()
    app.include_router(chat.router)
    client = TestClient(app, raise_server_exceptions=False)
    missing = client.post("/api/chat/stream", json={"user_message": "你好"})
    record("F08", "未认证请求拒绝", missing.status_code == 401,
           {"status": missing.status_code, "body": missing.json()})
    app.dependency_overrides[get_user_context] = lambda: USER
    fake_redis = MemoryRedis()
    route_graph = FakeGraph()
    with patch("app.routes.chat.get_redis_client", return_value=fake_redis), \
         patch("app.routes.chat.get_chat_graph", new=AsyncMock(return_value=route_graph)), \
         patch("app.routes.chat.check_rate_limit", return_value=(True, 59, 0)):
        forbidden = client.get("/api/chat/history", params={"thread_id": "belongs_to_someone_else"})
        record("F09", "其他用户会话历史隔离", forbidden.status_code == 403,
               {"status": forbidden.status_code, "body": forbidden.json()})
        created = client.post("/api/chat/session").json()["data"]["thread_id"]
        start_count = json.loads(fake_redis.hget(chat._session_key(USER.user_id), created))["message_count"]
        response = client.post("/api/chat/stream", json={"thread_id": created, "user_message": "你好"})
        events = parse_sse(response.text)
        record("F10", "SSE事件协议", response.status_code == 200 and events[0][0] == "start"
               and events[-1][0] == "complete" and any(e[0] == "token" for e in events),
               {"status": response.status_code, "events": [e[0] for e in events]})
        after_count = json.loads(fake_redis.hget(chat._session_key(USER.user_id), created))["message_count"]
        record("F11", "会话消息计数", start_count == 0 and after_count == 2,
               {"after_create": start_count, "after_one_turn": after_count},
               "_touch_session 在创建、请求开始、请求完成时都加1；空会话已有1条，一轮后显示3条。")
        for _ in range(2):
            client.post("/api/chat/stream", json={"thread_id": created, "user_message": "你好", "client_request_id": "same-request"})
        record("F12", "重复客户端请求幂等", route_graph.calls == 2,
               {"expected_graph_calls": 2, "actual_graph_calls": route_graph.calls},
               "ChatRequest 声明 client_request_id，路由和状态构造均未使用；重复请求重复运行。")
        with patch("app.routes.chat.get_chat_graph", new=AsyncMock(return_value=FakeGraph(fail=True))):
            failed = client.post("/api/chat/stream", json={"thread_id": created, "user_message": "失败测试"})
            failed_events = parse_sse(failed.text)
        record("F13", "SSE异常脱敏", any(e[0] == "error" for e in failed_events)
               and "INTERNAL_TEST_SECRET" not in failed.text,
               {"status": failed.status_code, "events": failed_events})

    actor = UserContext(user_id="direct_manager", name="跨部门主管", role="manager", department="财务部")
    application = {"employee_id": "employee", "manager_id": actor.user_id, "department": "技术部",
                   "status": "pending", "approval_step": "manager"}
    approve = check_approval_permission(actor, application)
    view = check_view_permission(actor, application)
    record("F14", "跨部门直属主管权限一致性", approve.ok == view.ok,
           {"can_approve": approve.ok, "can_view": view.ok, "view_message": view.message},
           "approve 支持 manager_id 跨部门汇报线；view 和 pending SQL 仅按同部门判断，直属主管看不到应审批数据。")

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, sql, params): self.sql, self.params = sql, params
        def fetchall(self): return []
        def fetchone(self): return {"response_json": {"ok": True, "code": "OK", "data": {"owner": "other_user"}, "message": "cached"}}
    cursor = Cursor()
    conn = SimpleNamespace(cursor=lambda: cursor)
    repository.get_idempotent(conn, "known-key")
    record("F15", "幂等缓存按用户和操作隔离", "user_id" in cursor.sql and "endpoint" in cursor.sql,
           {"sql": " ".join(cursor.sql.split()), "params": cursor.params},
           "get_idempotent 仅按 key 查，服务层在权限检查前返回缓存；没有验证缓存 user_id 或 endpoint。")
    repository.list_pending(conn, actor)
    record("F16", "直属主管待办列表过滤", "manager_id" in cursor.sql,
           {"sql": " ".join(cursor.sql.split()), "params": cursor.params},
           "待办只按 department，不按直属主管；同部门非直属主管看到不应处理的请假原因，跨部门主管漏单。")

    try:
        ChatRequest(user_message="   ")
        whitespace_accepted = True
    except Exception:
        whitespace_accepted = False
    record("F17", "空白消息校验", not whitespace_accepted,
           {"accepted_whitespace_message": whitespace_accepted},
           "仅 min_length 校验，未 strip 后检查，纯空白会调用完整模型流程。")

    (OUT / "feature_probe_results.json").write_text(json.dumps({"method": "deterministic isolated probes; no live model, SQL or Redis",
          "cases": RESULTS, "passed": sum(row["passed"] for row in RESULTS), "total": len(RESULTS)}, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in RESULTS:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Passed {sum(row['passed'] for row in RESULTS)} / {len(RESULTS)}")


if __name__ == "__main__":
    asyncio.run(probes())
