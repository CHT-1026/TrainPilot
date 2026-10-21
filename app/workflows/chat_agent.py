"""对话 Agent 工作流主图。"""
import asyncio
import logging

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.core.config import get_settings
from app.database.connection import get_async_redis_client
from app.workflows.chat_nodes_factory import create_all_nodes
from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)

_chat_graph = None
_graph_lock = asyncio.Lock()

TASK_TO_NODE = {
    "submit_application": "submit_leave",
    "approve_application": "approve",
    "query": "query",
    "query_pending": "query_pending",
    "policy_qa": "retrieve",
    "delete_application": "delete",
    "suggestion": "suggestion",
    "general_chat": "general_chat",
    "confirm_action": "execute_pending",
    "cancel_action": "clear_pending",
    "clarify": "clarify",
}


async def create_chat_graph():
    settings = get_settings()
    logger.info("创建对话工作流图，模型=%s", settings.openai_model)

    extra_body = None
    if settings.openai_enable_thinking.strip().lower() == "false":
        extra_body = {"enable_thinking": False}

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
        temperature=0.2,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        extra_body=extra_body,
    )

    nodes = create_all_nodes(llm)
    builder = StateGraph(ChatState)

    for node_name, node_func in nodes.items():
        if node_name != "route_task_execution":
            builder.add_node(node_name, node_func)

    builder.set_entry_point("intent_recognition")
    builder.add_edge("intent_recognition", "task_router")
    builder.add_edge("task_router", "task_executor")
    builder.add_conditional_edges(
        "task_executor",
        nodes["route_task_execution"],
        TASK_TO_NODE,
    )

    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "response_generator")
    for target in TASK_TO_NODE.values():
        if target != "retrieve":
            builder.add_edge(target, "response_generator")
    builder.add_edge("response_generator", "summarize")
    builder.add_edge("summarize", END)

    checkpointer = AsyncRedisSaver(redis_client=get_async_redis_client())
    await checkpointer.setup()

    return builder.compile(checkpointer=checkpointer)


async def get_chat_graph():
    global _chat_graph
    if _chat_graph is None:
        async with _graph_lock:
            if _chat_graph is None:
                _chat_graph = await create_chat_graph()
    return _chat_graph
