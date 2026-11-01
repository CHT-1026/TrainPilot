"""One tool registry and one validation/dispatch boundary for model and API calls."""
import asyncio
import json
import hashlib
import logging
from dataclasses import dataclass
from functools import partial
from typing import Awaitable, Callable

from jsonschema import Draft202012Validator, ValidationError as SchemaError, SchemaError as InvalidSchema
from referencing import Registry
from referencing.exceptions import Unresolvable
from pydantic import BaseModel, ValidationError

from app.capabilities.tools.config import ToolsConfig, load_config
from app.capabilities.tools.local import (
    LocalTools, ToolError, ListFiles, ReadFile, WriteFile, SearchCode, RunCommand, QueryConfig, MAX_OUTPUT,
)
from app.domain.common.results import ToolResult
from app.domain.user.models import UserContext
from app.integrations import mcp_client

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable[..., Awaitable[dict]]
    read_only: bool = True
    args_model: type[BaseModel] | None = None
    confirmation: Callable[[BaseModel], bool] | None = None
    origin: str = "local"

    def model_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.schema}}


@dataclass
class PreparedCall:
    tool: Tool
    arguments: BaseModel | dict
    needs_confirmation: bool

    def payload(self) -> dict:
        return self.arguments.model_dump() if isinstance(self.arguments, BaseModel) else self.arguments


class ToolRegistry:
    def __init__(self, actor: UserContext, config: ToolsConfig):
        self.actor, self.config = actor, config
        self.tools: dict[str, Tool] = {}
        self.unavailable: list[dict] = []
        self.local = LocalTools(config)
        if actor.role in config.local_roles:
            for name, description, model, read_only in (
                ("list_files", "列出工作区目录中的文件；不读取密钥或环境目录。", ListFiles, True),
                ("read_file", "按行读取工作区UTF-8文本，并返回sha256供后续安全覆盖。", ReadFile, True),
                ("write_file", "在允许目录创建或替换UTF-8文件；写入前需用户确认，覆盖需当前sha256。", WriteFile, False),
                ("search_code", "使用ripgrep按字面量检索代码，返回文件、行号和匹配文本。", SearchCode, True),
                ("run_command", "执行配置注册的命令ID；paths仅为相对路径。写命令须确认，不接受任意shell。", RunCommand, True),
                ("query_project_config", "查询model/rag/tools/mcp的公开配置，不返回API密钥、密码、认证头。", QueryConfig, True),
            ):
                schema = model.model_json_schema()
                if name == "write_file":
                    description += "允许写入目录：" + ", ".join(config.write_roots)
                if name == "run_command":
                    schema["properties"]["command"]["enum"] = list(config.commands)
                self.register(Tool(name, description, schema, getattr(self.local, name),
                                   read_only, model,
                                   (lambda args: not self.local.command(args).read_only) if name == "run_command" else None))

    def register(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        Draft202012Validator.check_schema(tool.schema)
        self.tools[tool.name] = tool

    def prepare(self, name: str, arguments: dict) -> PreparedCall:
        try:
            tool = self.tools[name]
        except KeyError:
            raise ToolError("TOOL_NOT_ALLOWED", "工具未注册或当前用户无权调用") from None
        try:
            if tool.args_model:
                validated = tool.args_model.model_validate(arguments)
            else:
                Draft202012Validator(tool.schema, registry=Registry()).validate(arguments)
                validated = arguments
        except (ValidationError, SchemaError, Unresolvable):
            raise ToolError("INVALID_TOOL_ARGS", "参数不符合该工具公布的schema") from None
        needs_confirmation = tool.confirmation(validated) if tool.confirmation else not tool.read_only
        return PreparedCall(tool, validated, needs_confirmation)

    async def execute(self, call: PreparedCall) -> ToolResult:
        try:
            data = await call.tool.handler(call.arguments)
            command_failed = call.tool.name == "run_command" and data["exit_code"] != 0
            encoded = json.dumps(data, ensure_ascii=False)
            if len(encoded) > MAX_OUTPUT:
                data = {**{key: data[key] for key in ("path", "sha256", "exit_code") if key in data},
                        "text": encoded[:MAX_OUTPUT], "truncated": True}
            if command_failed:
                return ToolResult.failure("COMMAND_FAILED", "命令执行失败，见退出码与输出", data=data)
            return ToolResult.success(data=data, message="工具调用完成")
        except ToolError as exc:
            return ToolResult.failure(exc.code, str(exc))
        except TimeoutError:
            return ToolResult.failure("TOOL_TIMEOUT", "工具执行超时，已停止本地进程或取消外部等待")
        except (OSError, UnicodeError) as exc:
            logger.warning("Tool %s failed: %s", call.tool.name, type(exc).__name__)
            return ToolResult.failure("TOOL_IO_ERROR", "工具访问文件或启动进程失败")

    def action_binding(self, call: PreparedCall) -> str:
        """Bind a confirmation to its workspace/command/server and input schema."""
        if call.tool.origin == "local":
            target = {"root": str(self.local.root)}
            if call.tool.name == "write_file":
                target["write_roots"] = self.config.write_roots
            elif call.tool.name == "run_command":
                target["command"] = self.local.command(call.arguments).model_dump()
        else:
            server = next(server for server in self.config.mcp_servers if call.tool.origin == f"mcp:{server.name}")
            target = server.model_dump(exclude={"headers", "env", "timeout_seconds", "allowed_roles", "tools"})
        data = {"name": call.tool.name, "schema": call.tool.schema, "target": target}
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    async def invoke(self, name: str, arguments: dict) -> ToolResult:
        """Read-only direct calls; mutations go through persisted conversation confirmation."""
        try:
            call = self.prepare(name, arguments)
            if call.needs_confirmation:
                return ToolResult.failure("CONFIRMATION_REQUIRED", "请在对话中提出该操作并确认后执行")
            return await self.execute(call)
        except ToolError as exc:
            return ToolResult.failure(exc.code, str(exc))

    def describe(self) -> dict:
        return {"tools": [{"name": tool.name, "description": tool.description, "parameters": tool.schema,
                           "read_only": tool.read_only, "origin": tool.origin} for tool in self.tools.values()],
                "unavailable": self.unavailable}


async def build_registry(actor: UserContext, config: ToolsConfig | None = None) -> ToolRegistry:
    registry = ToolRegistry(actor, config or load_config())
    servers = [server for server in registry.config.mcp_servers if actor.role in server.allowed_roles]
    # Each SDK context stays inside its own coroutine. One failed endpoint does
    # not serialize its timeout in front of every healthy endpoint.
    discoveries = await asyncio.gather(*(mcp_client.discover(server) for server in servers), return_exceptions=True)
    for server, discovered in zip(servers, discoveries):
        if isinstance(discovered, asyncio.CancelledError):
            raise discovered
        if isinstance(discovered, Exception):
            logger.warning("MCP %s discovery failed: %s", server.name, type(discovered).__name__)
            registry.unavailable.append({"server": server.name, "code": "MCP_UNAVAILABLE"})
            continue
        try:
            found = {tool.name for tool in discovered}
            for name in server.tools.keys() - found:
                registry.unavailable.append({"server": server.name, "tool": name, "code": "MCP_TOOL_NOT_FOUND"})
            for remote in discovered:
                alias = mcp_client.tool_alias(server.name, remote.name)
                registry.register(Tool(alias, remote.description or remote.name, remote.input_schema,
                    partial(mcp_client.call_tool, server, remote.name), server.tools[remote.name].read_only,
                    origin=f"mcp:{server.name}"))
        except (ValueError, TypeError, InvalidSchema) as exc:
            logger.warning("MCP %s discovery failed: %s", server.name, type(exc).__name__)
            registry.unavailable.append({"server": server.name, "code": "MCP_SCHEMA_INVALID"})
    return registry
