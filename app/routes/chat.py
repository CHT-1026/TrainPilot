"""对话路由：会话归属校验、请求模型、SSE。"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from anyio import CancelScope

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.metrics import inc, observe
from app.core.rate_limit import check_rate_limit
from app.database.connection import get_redis_client
from app.domain.user.models import UserContext
from app.models.requests import ChatRequest
from app.routes.auth import get_user_context
from app.workflows.chat_agent import get_chat_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["对话交互"])


class SessionStateRequest(BaseModel):
    archived: bool


def _session_key(user_id: str) -> str:
    return f"session_metadata:{user_id}"


def _thread_lock_key(thread_id: str) -> str:
    return f"chat_thread_lock:{thread_id}"


def _thread_lock_ttl() -> int:
    return max(120, int(get_settings().llm_timeout_seconds * 3))


def _acquire_thread_lock(thread_id: str) -> str:
    """跨 worker 的会话互斥必须使用同一个权威存储。"""
    token = uuid.uuid4().hex
    ttl = _thread_lock_ttl()
    try:
        client = get_redis_client()
        acquired = client.set(_thread_lock_key(thread_id), token, nx=True, ex=ttl)
        return f"redis:{token}" if acquired else ""
    except Exception as exc:
        raise HTTPException(status_code=503, detail="会话存储暂不可用，请稍后重试") from exc


def _renew_thread_lock(thread_id: str, token: str) -> bool:
    """流式请求存活期间续租；崩溃的请求停止续租后由 TTL 释放。"""
    if not token:
        return False
    mode, _, raw_token = token.partition(":")
    if mode != "redis":
        return False
    try:
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
        )
        return bool(get_redis_client().eval(
            script, 1, _thread_lock_key(thread_id), raw_token, _thread_lock_ttl()
        ))
    except Exception:
        logger.debug("续租线程锁失败: %s", thread_id)
        return False


async def _keep_thread_lock(thread_id: str, token: str, stop: asyncio.Event,
                            lost: asyncio.Event | None = None) -> None:
    interval = max(0.01, _thread_lock_ttl() / 3)
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            if not await asyncio.to_thread(_renew_thread_lock, thread_id, token):
                if lost is not None:
                    lost.set()
                return


def _release_thread_lock(thread_id: str, token: str) -> None:
    if not token:
        return
    mode, _, raw_token = token.partition(":")
    if mode == "redis":
        try:
            client = get_redis_client()
            script = (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end"
            )
            client.eval(script, 1, _thread_lock_key(thread_id), raw_token)
        except Exception:
            logger.debug("释放线程锁失败: %s", thread_id)
        return

class ThreadLeaseLost(RuntimeError):
    pass


class _ThreadLease:
    """所有会话写路径共用续租；失去租约后停止等待且不得发布成功。"""
    def __init__(self, thread_id: str, token: str):
        self.thread_id, self.token = thread_id, token
        self.stop, self.lost = asyncio.Event(), asyncio.Event()
        self.task = None

    async def __aenter__(self):
        self.task = asyncio.create_task(_keep_thread_lock(self.thread_id, self.token, self.stop, self.lost))
        return self

    async def run(self, awaitable):
        operation = asyncio.ensure_future(awaitable)
        waiter = asyncio.create_task(self.lost.wait())
        try:
            await asyncio.wait((operation, waiter), return_when=asyncio.FIRST_COMPLETED)
            if self.lost.is_set():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                raise ThreadLeaseLost("会话锁已失效")
            return await operation
        finally:
            with CancelScope(shield=True):
                waiter.cancel()
                if not operation.done():
                    operation.cancel()
                    await asyncio.gather(operation, return_exceptions=True)
                await asyncio.gather(waiter, return_exceptions=True)

    async def check(self):
        if self.lost.is_set() or not await asyncio.to_thread(_renew_thread_lock, self.thread_id, self.token):
            self.lost.set()
            raise ThreadLeaseLost("会话锁已失效")

    async def __aexit__(self, *_):
        with CancelScope(shield=True):
            self.stop.set()
            if self.task is not None:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            await asyncio.to_thread(_release_thread_lock, self.thread_id, self.token)



def _owns_thread(user_id: str, thread_id: str) -> bool:
    redis_client = get_redis_client()
    return redis_client.hget(_session_key(user_id), thread_id) is not None


def _touch_session(
    user_id: str, thread_id: str, *, user_message: str = "", message_count: int | None = None
) -> None:
    redis_client = get_redis_client()
    key = _session_key(user_id)
    now = datetime.now().isoformat()
    existing = redis_client.hget(key, thread_id)
    if existing:
        meta = json.loads(existing)
        meta["updated_at"] = now
        # 用户重新在该会话发言时自动恢复归档，避免归档会话只能读不能继续。
        if user_message and meta.get("archived"):
            meta["archived"] = False
            meta.pop("archived_at", None)
        if meta.get("title") in ("", thread_id) and user_message:
            meta["title"] = user_message[:30]
    else:
        meta = {
            "thread_id": thread_id,
            "title": (user_message[:30] or thread_id),
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "archived": False,
        }
    if message_count is not None:
        meta["message_count"] = max(0, message_count)
    redis_client.hset(key, thread_id, json.dumps(meta, ensure_ascii=False))
    redis_client.expire(key, 60 * 60 * 24 * get_settings().session_ttl_days)


def _set_session_archived(user_id: str, thread_id: str, archived: bool) -> dict | None:
    """只改会话索引中的归档标记；聊天检查点仍由原来的 TTL 管理。"""
    redis_client = get_redis_client()
    key = _session_key(user_id)
    raw = redis_client.hget(key, thread_id)
    if not raw:
        return None
    meta = json.loads(raw)
    meta["archived"] = bool(archived)
    if archived:
        meta["archived_at"] = datetime.now().isoformat()
    else:
        meta.pop("archived_at", None)
    redis_client.hset(key, thread_id, json.dumps(meta, ensure_ascii=False))
    redis_client.expire(key, 60 * 60 * 24 * get_settings().session_ttl_days)
    return meta


@router.post("/chat/session")
async def create_chat_session(user: UserContext = Depends(get_user_context)):
    thread_id = f"chat_{uuid.uuid4().hex[:12]}"
    _touch_session(user.user_id, thread_id)
    logger.info("创建新会话: %s/%s", user.user_id, thread_id)
    return {"code": 200, "data": {"thread_id": thread_id}}


@router.patch("/chat/session/{thread_id}")
async def update_chat_session(
    thread_id: str,
    payload: SessionStateRequest,
    user: UserContext = Depends(get_user_context),
):
    """归档/恢复会话；归档不删除聊天记录，删除仍走 DELETE。"""
    if not _owns_thread(user.user_id, thread_id):
        raise HTTPException(status_code=403, detail="无权操作该会话")
    lock_token = _acquire_thread_lock(thread_id)
    if not lock_token:
        raise HTTPException(status_code=409, detail="会话正在处理消息，请稍后再操作")

    try:
        async with _ThreadLease(thread_id, lock_token) as lease:
            if not _owns_thread(user.user_id, thread_id):
                raise HTTPException(status_code=403, detail="无权操作该会话")
            updated = await asyncio.to_thread(
                _set_session_archived, user.user_id, thread_id, payload.archived
            )
            await lease.check()
            if updated is None:
                raise HTTPException(status_code=404, detail="会话不存在")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("更新会话归档状态失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="会话状态更新失败，请稍后重试")

    return {"code": 200, "data": updated}


@router.delete("/chat/session/{thread_id}")
async def delete_chat_session(thread_id: str, user: UserContext = Depends(get_user_context)):
    if not _owns_thread(user.user_id, thread_id):
        raise HTTPException(status_code=403, detail="无权操作该会话")
    lock_token = _acquire_thread_lock(thread_id)
    if not lock_token:
        raise HTTPException(status_code=409, detail="会话正在处理消息，请稍后再删除")

    try:
        async with _ThreadLease(thread_id, lock_token) as lease:
            if not _owns_thread(user.user_id, thread_id):
                raise HTTPException(status_code=403, detail="无权操作该会话")
            graph = await lease.run(get_chat_graph())
            if getattr(graph, "checkpointer", None) is not None:
                await lease.run(graph.checkpointer.adelete_thread(thread_id))
            await lease.check()
            get_redis_client().hdel(_session_key(user.user_id), thread_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("删除检查点失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="会话历史删除失败，请稍后重试")

    return {"code": 200, "message": "删除成功"}


@router.get("/chat/sessions")
async def list_chat_sessions(user: UserContext = Depends(get_user_context)):
    redis_client = get_redis_client()
    sessions_data = redis_client.hgetall(_session_key(user.user_id))
    sessions = []
    for _, meta_json in sessions_data.items():
        try:
            meta = json.loads(meta_json)
        except json.JSONDecodeError:
            continue
        meta.setdefault("archived", False)
        sessions.append(meta)
    sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return {"code": 200, "data": sessions}


@router.get("/chat/history")
async def get_chat_history(thread_id: str, user: UserContext = Depends(get_user_context)):
    if not _owns_thread(user.user_id, thread_id):
        raise HTTPException(status_code=403, detail="无权查看该会话")

    graph = await get_chat_graph()
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not state or not state.values:
        return {"code": 200, "data": {"thread_id": thread_id, "messages": []}}

    messages = state.values.get("messages", [])
    formatted = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            formatted.append(
                {
                    "role": "ai" if role == "assistant" else role,
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                }
            )
    return {"code": 200, "data": {"thread_id": thread_id, "messages": formatted}}


async def build_chat_initial_state(
    graph, thread_id: str, user_message: str, user: UserContext, trace_id: str = ""
):
    config = {"configurable": {"thread_id": thread_id}}
    existing = await graph.aget_state(config)
    existing_values = existing.values if existing and existing.values else {}

    context = dict(existing_values.get("conversation_context") or {})
    context.update(
        {
            "user_id": user.user_id,
            "user_name": user.name,
            "user_role": user.role,
            "user_department": user.department,
            "user_position": user.position,
            "manager_id": user.manager_id,
        }
    )
    try:
        from app.workflows.fact_memory import list_facts, render_memory_block

        context["fact_memory"] = render_memory_block(list_facts(user))
    except Exception:
        context["fact_memory"] = ""

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            }
        ],
        "user_message": user_message,
        "intent": "",
        "extracted_info": {},
        "current_task": "",
        "task_result": "",
        # LangGraph persists fields omitted by an update. These fields describe
        # this turn; retaining them would attach earlier sources/tool receipts.
        "retrieved_chunks": [],
        "retrieval_question": "",
        "knowledge_namespace": "",
        "retrieval_meta": {},
        "answer_method": "",
        "citations": [],
        "grounded": False,
        "response": "",
        "last_result": None,
        "user": user.model_dump(),
        "conversation_context": context,
        "trace_id": trace_id,
        "state_version": int(existing_values.get("state_version", 0) or 0) + 1,
        "draft_version": int(existing_values.get("draft_version", 0) or 0),
        "dialog_state": dict(existing_values.get("dialog_state") or {}),
        "source_kind": "",
    }
    return initial_state, config


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    http_request: Request,
    user: UserContext = Depends(get_user_context),
):
    thread_id = payload.thread_id or f"chat_{uuid.uuid4().hex[:12]}"
    if payload.thread_id and not _owns_thread(user.user_id, thread_id):
        raise HTTPException(status_code=403, detail="无权操作该会话")

    settings = get_settings()
    allowed, _, retry_after = check_rate_limit(
        f"chat:{user.user_id}",
        settings.chat_rate_limit,
        settings.chat_rate_window_seconds,
    )
    if not allowed:
        inc("chat_rate_limited_total")
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请 {retry_after} 秒后重试",
            headers={"Retry-After": str(retry_after)},
        )

    lock_token = _acquire_thread_lock(thread_id)
    if not lock_token:
        raise HTTPException(status_code=409, detail="同一会话正在处理另一条消息，请稍后再试")
    trace_id = getattr(http_request.state, "trace_id", "")

    async def event_generator():
        started = time.monotonic()
        seq = 0
        lease = _ThreadLease(thread_id, lock_token)
        await lease.__aenter__()
        iterator = None
        try:
            if payload.thread_id and not _owns_thread(user.user_id, thread_id):
                raise HTTPException(status_code=403, detail="无权操作该会话")
            _touch_session(user.user_id, thread_id, user_message=payload.user_message)
            graph = await lease.run(get_chat_graph())
            seq += 1
            yield f"event: start\ndata: {json.dumps({'thread_id': thread_id, 'seq': seq})}\n\n"
            seq += 1
            yield f"event: status\ndata: {json.dumps({'stage': 'understanding', 'phase': 'accepted', 'seq': seq})}\n\n"

            initial_state, config = await lease.run(build_chat_initial_state(
                graph, thread_id, payload.user_message, user, trace_id
            ))
            seq += 1
            yield f"event: status\ndata: {json.dumps({'stage': 'generating', 'phase': 'accepted', 'seq': seq})}\n\n"

            final_state = {}
            streamed_text = ""
            stream_reset = False
            streaming_mode = "buffered"
            iterator = graph.astream(initial_state, config, stream_mode=["custom", "values"]).__aiter__()
            while True:
                try:
                    mode, chunk = await lease.run(anext(iterator))
                except StopAsyncIteration:
                    break
                if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "status":
                    seq += 1
                    payload_status = {
                        "stage": chunk.get("stage", ""),
                        "phase": chunk.get("phase", "progress"),
                        "seq": seq,
                    }
                    yield f"event: status\ndata: {json.dumps(payload_status, ensure_ascii=False)}\n\n"
                elif mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "token":
                    delta = chunk.get("delta") or ""
                    if delta:
                        streamed_text += delta
                        streaming_mode = chunk.get("mode") or "model"
                        seq += 1
                        yield (
                            "event: token\ndata: "
                            + json.dumps({
                                "token": delta,
                                "seq": seq,
                                "mode": streaming_mode,
                                "verified": streaming_mode == "verified",
                                "final": False,
                            }, ensure_ascii=False) + "\n\n"
                        )
                elif mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "replace":
                    streamed_text = chunk.get("text") or ""
                    stream_reset = True
                    seq += 1
                    yield (
                        "event: replace\ndata: "
                        + json.dumps({"response": streamed_text, "seq": seq, "final": True},
                                     ensure_ascii=False) + "\n\n"
                    )
                elif mode == "values" and isinstance(chunk, dict):
                    final_state = chunk

            await lease.check()
            response_text = final_state.get("response", "") or ""
            if not streamed_text and response_text:
                seq += 1
                yield f"event: token\ndata: {json.dumps({'token': response_text, 'seq': seq, 'mode': 'final', 'verified': streaming_mode == 'verified', 'final': True}, ensure_ascii=False)}\n\n"
            elif response_text.startswith(streamed_text):
                remaining = response_text[len(streamed_text) :]
                if remaining:
                    seq += 1
                    yield f"event: token\ndata: {json.dumps({'token': remaining, 'seq': seq, 'mode': streaming_mode, 'verified': streaming_mode == 'verified', 'final': True}, ensure_ascii=False)}\n\n"
            elif streamed_text != response_text:
                # A provider may fail after partial output. Explicit replacement
                # keeps a provisional fragment from masquerading as the final reply.
                stream_reset = True
                seq += 1
                yield f"event: replace\ndata: {json.dumps({'response': response_text, 'seq': seq, 'final': True}, ensure_ascii=False)}\n\n"

            actual_messages = final_state.get("messages")
            _touch_session(
                user.user_id,
                thread_id,
                message_count=len(actual_messages) if isinstance(actual_messages, list) else None,
            )

            complete_payload = {
                "thread_id": thread_id,
                "response": response_text,
                "intent": final_state.get("intent", ""),
                "current_task": final_state.get("current_task", ""),
                "citations": final_state.get("citations", []),
                "pending_action": final_state.get("pending_action"),
                "last_result": final_state.get("last_result"),
                "streaming_mode": streaming_mode,
                "stream_reset": stream_reset,
                "knowledge_namespace": final_state.get("knowledge_namespace", ""),
                "answer_method": final_state.get("answer_method", ""),
                "grounded": final_state.get("grounded", False),
                "state_version": final_state.get("state_version", 0),
                "seq": seq + 1,
            }
            yield f"event: complete\ndata: {json.dumps(complete_payload, ensure_ascii=False, default=str)}\n\n"
            inc("chat_requests_total", {"intent": final_state.get("intent", "")})
            observe("chat_turn_seconds", time.monotonic() - started)
        except Exception as exc:
            inc("chat_errors_total")
            logger.error("对话处理异常: %s", exc, exc_info=True)
            seq += 1
            yield (
                "event: error\ndata: "
                + json.dumps({"error": "处理失败，请稍后重试", "seq": seq,
                              "final": True, "draft_invalidated": True},
                             ensure_ascii=False) + "\n\n"
            )
            yield (
                "event: replace\ndata: "
                + json.dumps({"response": "处理失败，请稍后重试。", "seq": seq + 1,
                              "final": True}, ensure_ascii=False) + "\n\n"
            )
        finally:
            try:
                if iterator is not None and hasattr(iterator, "aclose"):
                    await iterator.aclose()
            finally:
                await lease.__aexit__()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat_once(
    payload: ChatRequest,
    http_request: Request,
    user: UserContext = Depends(get_user_context),
):
    """非流式对话：与 SSE 共用同一图、同一证据边界与线程锁。"""
    thread_id = payload.thread_id or f"chat_{uuid.uuid4().hex[:12]}"
    if payload.thread_id and not _owns_thread(user.user_id, thread_id):
        raise HTTPException(status_code=403, detail="无权操作该会话")
    settings = get_settings()
    allowed, _, retry_after = check_rate_limit(
        f"chat:{user.user_id}", settings.chat_rate_limit, settings.chat_rate_window_seconds
    )
    if not allowed:
        inc("chat_rate_limited_total")
        raise HTTPException(status_code=429, detail=f"请求过于频繁，请 {retry_after} 秒后重试",
                            headers={"Retry-After": str(retry_after)})
    lock_token = _acquire_thread_lock(thread_id)
    if not lock_token:
        raise HTTPException(status_code=409, detail="同一会话正在处理另一条消息，请稍后再试")
    trace_id = getattr(http_request.state, "trace_id", "")
    started = time.monotonic()
    lease = _ThreadLease(thread_id, lock_token)
    await lease.__aenter__()
    try:
        if payload.thread_id and not _owns_thread(user.user_id, thread_id):
            raise HTTPException(status_code=403, detail="无权操作该会话")
        _touch_session(user.user_id, thread_id, user_message=payload.user_message)
        graph = await lease.run(get_chat_graph())
        initial_state, config = await lease.run(build_chat_initial_state(
            graph, thread_id, payload.user_message, user, trace_id
        ))
        final_state = await lease.run(graph.ainvoke(initial_state, config))
        await lease.check()
        actual_messages = final_state.get("messages")
        _touch_session(
            user.user_id, thread_id,
            message_count=len(actual_messages) if isinstance(actual_messages, list) else None,
        )
        inc("chat_requests_total", {"intent": final_state.get("intent", "")})
        observe("chat_turn_seconds", time.monotonic() - started)
        return {
            "code": 200,
            "data": {
                "thread_id": thread_id,
                "response": final_state.get("response", ""),
                "intent": final_state.get("intent", ""),
                "current_task": final_state.get("current_task", ""),
                "citations": final_state.get("citations", []),
                "pending_action": final_state.get("pending_action"),
                "last_result": final_state.get("last_result"),
                "knowledge_namespace": final_state.get("knowledge_namespace", ""),
                "answer_method": final_state.get("answer_method", ""),
                "grounded": final_state.get("grounded", False),
                "state_version": final_state.get("state_version", 0),
            },
        }
    except ThreadLeaseLost as exc:
        raise HTTPException(status_code=503, detail="会话处理已中断，请先查询记录再决定是否重试") from exc
    finally:
        await lease.__aexit__()
