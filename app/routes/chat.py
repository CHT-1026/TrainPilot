"""对话路由：会话归属校验、请求模型、SSE。"""
import json
import logging
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

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


def _session_key(user_id: str) -> str:
    return f"session_metadata:{user_id}"


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
        if meta.get("title") in ("", thread_id) and user_message:
            meta["title"] = user_message[:30]
    else:
        meta = {
            "thread_id": thread_id,
            "title": (user_message[:30] or thread_id),
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
    if message_count is not None:
        meta["message_count"] = max(0, message_count)
    redis_client.hset(key, thread_id, json.dumps(meta, ensure_ascii=False))
    redis_client.expire(key, 60 * 60 * 24 * get_settings().session_ttl_days)


@router.post("/chat/session")
async def create_chat_session(user: UserContext = Depends(get_user_context)):
    thread_id = f"chat_{uuid.uuid4().hex[:12]}"
    _touch_session(user.user_id, thread_id)
    logger.info("创建新会话: %s/%s", user.user_id, thread_id)
    return {"code": 200, "data": {"thread_id": thread_id}}


@router.delete("/chat/session/{thread_id}")
async def delete_chat_session(thread_id: str, user: UserContext = Depends(get_user_context)):
    if not _owns_thread(user.user_id, thread_id):
        raise HTTPException(status_code=403, detail="无权操作该会话")

    redis_client = get_redis_client()
    redis_client.hdel(_session_key(user.user_id), thread_id)

    try:
        graph = await get_chat_graph()
        if getattr(graph, "checkpointer", None) is not None:
            await graph.checkpointer.adelete_thread(thread_id)
    except Exception as exc:
        logger.error("删除检查点失败: %s", exc, exc_info=True)

    return {"code": 200, "message": "删除成功"}


@router.get("/chat/sessions")
async def list_chat_sessions(user: UserContext = Depends(get_user_context)):
    redis_client = get_redis_client()
    sessions_data = redis_client.hgetall(_session_key(user.user_id))
    sessions = []
    for _, meta_json in sessions_data.items():
        try:
            sessions.append(json.loads(meta_json))
        except json.JSONDecodeError:
            continue
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

    _touch_session(user.user_id, thread_id, user_message=payload.user_message)
    graph = await get_chat_graph()
    trace_id = getattr(http_request.state, "trace_id", "")

    async def event_generator():
        started = time.monotonic()
        try:
            yield f"event: start\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
            yield f"event: status\ndata: {json.dumps({'stage': 'understanding'})}\n\n"

            initial_state, config = await build_chat_initial_state(
                graph, thread_id, payload.user_message, user, trace_id
            )
            yield f"event: status\ndata: {json.dumps({'stage': 'generating'})}\n\n"

            final_state = {}
            streamed_text = ""
            stream_reset = False
            streaming_mode = "buffered"
            async for mode, chunk in graph.astream(
                initial_state, config, stream_mode=["custom", "values"]
            ):
                if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "token":
                    delta = chunk.get("delta") or ""
                    if delta:
                        streamed_text += delta
                        streaming_mode = chunk.get("mode") or "model"
                        yield f"event: token\ndata: {json.dumps({'token': delta}, ensure_ascii=False)}\n\n"
                elif mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "replace":
                    streamed_text = chunk.get("text") or ""
                    stream_reset = True
                    yield f"event: replace\ndata: {json.dumps({'response': streamed_text}, ensure_ascii=False)}\n\n"
                elif mode == "values" and isinstance(chunk, dict):
                    final_state = chunk

            response_text = final_state.get("response", "") or ""
            if not streamed_text and response_text:
                yield f"event: token\ndata: {json.dumps({'token': response_text}, ensure_ascii=False)}\n\n"
            elif response_text.startswith(streamed_text):
                remaining = response_text[len(streamed_text) :]
                if remaining:
                    yield f"event: token\ndata: {json.dumps({'token': remaining}, ensure_ascii=False)}\n\n"
            elif streamed_text != response_text:
                # A provider may fail after partial output. Explicit replacement
                # keeps a provisional fragment from masquerading as the final reply.
                stream_reset = True
                yield f"event: replace\ndata: {json.dumps({'response': response_text}, ensure_ascii=False)}\n\n"

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
                "citations": final_state.get("citations", []),
                "pending_action": final_state.get("pending_action"),
                "last_result": final_state.get("last_result"),
                "streaming_mode": streaming_mode,
                "stream_reset": stream_reset,
            }
            yield f"event: complete\ndata: {json.dumps(complete_payload, ensure_ascii=False, default=str)}\n\n"
            inc("chat_requests_total", {"intent": final_state.get("intent", "")})
            observe("chat_turn_seconds", time.monotonic() - started)
        except Exception as exc:
            inc("chat_errors_total")
            logger.error("对话处理异常: %s", exc, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': '处理失败，请稍后重试'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
