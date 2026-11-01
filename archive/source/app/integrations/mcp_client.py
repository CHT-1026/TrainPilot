"""MCP 2.3 client: stdio and Streamable HTTP, scoped connections, no app-level retry."""
import asyncio
import hashlib
import logging
import re
import sys
from contextlib import asynccontextmanager

import httpx2
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from app.capabilities.tools.config import MCPServerConfig, ROOT, resolve_env
from app.capabilities.tools.local import ToolError, process_env

logger = logging.getLogger(__name__)


@asynccontextmanager
async def http_transport(server: MCPServerConfig):
    headers = {key: resolve_env(value) for key, value in server.headers.items()}
    async with httpx2.AsyncClient(headers=headers, timeout=server.timeout_seconds, follow_redirects=False) as http:
        async with streamable_http_client(server.url, http_client=http) as streams:
            yield streams


@asynccontextmanager
async def connection(server: MCPServerConfig):
    if server.transport == "stdio":
        target = StdioServerParameters(command=server.command.replace("{python}", sys.executable),
            args=[arg.replace("{python}", sys.executable) for arg in server.args],
            cwd=(ROOT / server.cwd).resolve(),
            env={**process_env(), **{key: resolve_env(value) for key, value in server.env.items()}})
    else:
        target = http_transport(server)
    # No cross-request session globals or cached stale results.
    async with Client(target, read_timeout_seconds=server.timeout_seconds, cache=None) as client:
        yield client


def tool_alias(server: str, name: str) -> str:
    raw = f"mcp_{server}_{name}"
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if clean == raw and len(clean) <= 64:
        return clean
    return clean[:55] + "_" + hashlib.sha256(raw.encode()).hexdigest()[:8]


async def discover(server: MCPServerConfig) -> list:
    async with asyncio.timeout(server.timeout_seconds):
        async with connection(server) as client:
            tools = []
            cursor = None
            while True:
                page = await client.list_tools(cursor=cursor)
                tools.extend(tool for tool in page.tools if tool.name in server.tools)
                cursor = page.next_cursor
                if cursor is None:
                    return tools


async def call_tool(server: MCPServerConfig, name: str, arguments: dict) -> dict:
    try:
        async with asyncio.timeout(server.timeout_seconds):
            async with connection(server) as client:
                result = await client.call_tool(name, arguments, read_timeout_seconds=server.timeout_seconds)
        if result.is_error:
            raise ToolError("MCP_TOOL_ERROR", "外部工具返回执行错误，未将错误当作业务结果")
        return {"server": server.name, "tool": name,
                "structured_content": result.structured_content,
                "content": [item.model_dump(mode="json", by_alias=True) for item in result.content]}
    except ToolError:
        raise
    except TimeoutError:
        raise ToolError("MCP_TIMEOUT", "外部工具调用超时，尚无确定回执；请核对远端结果，未自动重试") from None
    except Exception as exc:
        logger.warning("MCP %s call failed: %s", server.name, type(exc).__name__)
        raise ToolError("MCP_UNAVAILABLE", "外部工具连接失败；可继续使用其他已注册工具") from None
