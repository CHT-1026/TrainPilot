"""Project tools are a chat graph task, not a second response-generation path."""
import re
import json

from langgraph.config import get_stream_writer

from app.capabilities.tools.actions import execute_confirmed
from app.capabilities.tools.agent import run_agent
from app.capabilities.tools.registry import build_registry
from app.workflows.chat_nodes.common import actor_from_state

_TOOL_REQUEST = re.compile(
    r"^/tool(?:\s|$)|\bMCP\b|(?:调用|使用).{0,6}(?:工具|tool)|"
    r"(?:读取|读一下|查看|写入|创建|修改).{0,40}(?:README|[\w./\\-]+\.(?:py|md|txt|json|toml|ya?ml|js|html)\b|(?:本地|工作区|项目)文件)|"
    r"(?:检索|搜索|查找).{0,8}(?:代码|源码)|(?:运行|执行).{0,8}(?:命令|git\b|python\b)|"
    r"(?:查询|查看).{0,8}项目配置", re.IGNORECASE)


def is_tool_request(message: str) -> bool:
    return bool(_TOOL_REQUEST.search(message))


def create_tool_nodes(llm):
    async def tool_agent_node(state):
        writer = get_stream_writer()
        writer({"type": "status", "stage": "tools", "phase": "start"})
        registry = await build_registry(actor_from_state(state))
        result = await run_agent(registry, llm, state["user_message"], state.get("draft_version", 0))
        writer({"type": "status", "stage": "tools", "phase": "complete"})
        return result

    async def execute_tool_node(state):
        registry = await build_registry(actor_from_state(state))
        pending = state["pending_action"]
        result = await execute_confirmed(registry, pending, state["draft_version"])
        return {"task_result": result.message + ("\n" + json.dumps(result.data, ensure_ascii=False, indent=2) if result.data else ""),
                "pending_action": None, "last_result": result.to_dict(),
                "tool_events": [{"tool": pending["draft"]["name"], "arguments": pending["draft"]["arguments"], "result": result.to_dict()}]}

    return {"tool_agent": tool_agent_node, "execute_tool": execute_tool_node}
