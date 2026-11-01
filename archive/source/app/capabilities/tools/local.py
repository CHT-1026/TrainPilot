"""Local tool implementations. Arguments are validated once by the registry."""
import asyncio
import fnmatch
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.capabilities.tools.config import ROOT, ToolsConfig, config_path

MAX_OUTPUT = 16000
BLOCKED_PARTS = {".git", ".venv", ".venv312", "__pycache__", "logs", "originals"}
BLOCKED_NAMES = (".env*", "github_token.txt", "access_token.*", "refresh_token.*", "credentials.*", "secrets.*", "*.token", "*.pem", "*.key", "*.pfx", "*.p12")


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ListFiles(Args):
    path: str = Field(default=".", description="工作区内相对目录")
    limit: int = Field(default=100, ge=1, le=200, description="最多返回条目数")


class ReadFile(Args):
    path: str = Field(description="工作区内UTF-8文本文件的相对路径")
    start_line: int = Field(default=1, ge=1, description="起始行，含此行，从1计数")
    line_count: int = Field(default=100, ge=1, le=200, description="读取行数")


class WriteFile(Args):
    path: str = Field(description="允许写入目录下的相对文件路径")
    content: str = Field(max_length=64000, description="完整UTF-8文件内容；保持用户指定的字符串值，不添加包裹引号、句号、换行或说明")
    expected_sha256: str | None = Field(default=None, description="覆盖已有文件时必填，使用read_file返回的sha256；新文件不填")


class SearchCode(Args):
    query: str = Field(min_length=1, max_length=500, description="按字面量搜索的代码/文本")
    path: str = Field(default=".", description="工作区内搜索目录或文件")
    limit: int = Field(default=30, ge=1, le=100, description="最多返回匹配行数")


class RunCommand(Args):
    command: str = Field(description="配置中注册的命令ID，如python_version、git_status、git_diff")
    paths: list[str] = Field(default_factory=list, max_length=20, description="仅接受路径的命令可填写；不是额外命令行选项")


class QueryConfig(Args):
    section: str = Field(pattern=r"^(model|rag|tools|mcp)$", description="model/rag/tools/mcp；只返回公开配置，不返回密钥")


def process_env() -> dict[str, str]:
    names = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "HOME", "USERPROFILE")
    return {**{name: os.environ[name] for name in names if name in os.environ}, "PYTHONIOENCODING": "utf-8"}


async def run_process(argv: list[str], cwd: Path, timeout: float) -> dict:
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True})
    process = await asyncio.create_subprocess_exec(*argv, cwd=cwd, env=process_env(),
        stdin=subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **options)
    output = bytearray()
    truncated = False
    try:
        async with asyncio.timeout(timeout):
            while chunk := await process.stdout.read(4096):
                remaining = MAX_OUTPUT - len(output)
                output.extend(chunk[:remaining])
                truncated |= len(chunk) > remaining
            await process.wait()
    finally:
        if process.returncode is None:
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F",
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                await killer.wait()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
    return {"exit_code": process.returncode, "output": output.decode("utf-8", errors="replace"), "truncated": truncated}


class LocalTools:
    def __init__(self, config: ToolsConfig):
        self.config = config
        self.root = config.root

    def path(self, relative: str, *, write: bool = False) -> Path:
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise ToolError("PATH_NOT_ALLOWED", "路径超出工作区")
        parts = candidate.relative_to(self.root).parts
        if (any(part.lower() in BLOCKED_PARTS or part.lower().startswith(".venv") for part in parts)
                or (candidate.name != ".env.example" and any(fnmatch.fnmatch(candidate.name.lower(), pattern) for pattern in BLOCKED_NAMES))
                or candidate == config_path()):
            raise ToolError("PATH_NOT_ALLOWED", "该路径包含受保护配置、密钥或运行环境")
        if write and not any(candidate.is_relative_to((self.root / folder).resolve()) for folder in self.config.write_roots):
            raise ToolError("PATH_NOT_ALLOWED", "文件不在已配置的可写目录内")
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    async def list_files(self, args: ListFiles) -> dict:
        directory = self.path(args.path)
        items = []
        for child in sorted(directory.iterdir(), key=lambda path: path.name.lower()):
            try:
                self.path(self.relative(child))
            except ToolError:
                continue
            items.append({"path": self.relative(child), "directory": child.is_dir()})
        return {"entries": items[:args.limit], "truncated": len(items) > args.limit}

    async def read_file(self, args: ReadFile) -> dict:
        path = self.path(args.path)
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ToolError("FILE_TOO_LARGE", "单次工具读取上限为4MiB")
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").splitlines()
        selected = lines[args.start_line - 1:args.start_line - 1 + args.line_count]
        return {"path": self.relative(path), "start_line": args.start_line, "total_lines": len(lines),
                "text": "\n".join(selected), "sha256": hashlib.sha256(raw).hexdigest()}

    async def write_file(self, args: WriteFile) -> dict:
        path = self.path(args.path, write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = args.content.encode("utf-8")
        if path.exists():
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ToolError("FILE_TOO_LARGE", "单次工具修改上限为4MiB")
            if hashlib.sha256(path.read_bytes()).hexdigest() != args.expected_sha256:
                raise ToolError("FILE_CHANGED", "已有文件必须提供当前sha256；内容变化后请重新读取")
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            try:
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            with path.open("xb") as stream:
                stream.write(data)
        return {"path": self.relative(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    async def search_code(self, args: SearchCode) -> dict:
        path = self.path(args.path)
        argv = ["rg", "--line-number", "--no-heading", "--fixed-strings", "--color", "never", "--max-count", str(args.limit)]
        for part in BLOCKED_PARTS:
            argv += ["--glob", f"!**/{part}/**"]
        for pattern in BLOCKED_NAMES:
            argv += ["--glob", "!" + pattern]
        if config_path().is_relative_to(self.root):
            argv += ["--glob", "!" + self.relative(config_path())]
        argv += ["--glob", "!**/.venv*/**", "--glob", "!config/tools*.json", "--", args.query, self.relative(path)]
        result = await run_process(argv, self.root, self.config.command_timeout_seconds)
        if result["exit_code"] not in (0, 1):
            raise ToolError("SEARCH_FAILED", "代码检索执行失败")
        matches = result["output"].splitlines()
        return {"matches": matches[:args.limit], "truncated": result["truncated"] or len(matches) > args.limit}

    def command(self, args: RunCommand):
        try:
            return self.config.commands[args.command]
        except KeyError:
            raise ToolError("COMMAND_NOT_REGISTERED", "命令不在配置的注册表中") from None

    async def run_command(self, args: RunCommand) -> dict:
        command = self.command(args)
        argv = [value.replace("{python}", sys.executable) for value in command.argv]
        if args.paths:
            if not command.accepts_paths:
                raise ToolError("INVALID_TOOL_ARGS", "该命令不接受路径参数")
            argv += ["--", *[self.relative(self.path(path)) for path in args.paths]]
        return await run_process(argv, self.root, self.config.command_timeout_seconds)

    async def query_project_config(self, args: QueryConfig) -> dict:
        from app.core.config import get_settings
        from enterprise_rag import config as rag

        if args.section == "model":
            settings = get_settings()
            return {"model": settings.openai_model, "thinking": settings.openai_enable_thinking,
                    "timeout_seconds": settings.llm_timeout_seconds, "key_configured": bool(settings.openai_api_key)}
        if args.section == "rag":
            return {"namespace": rag.RAG_NAMESPACE, "model": rag.LLM_MODEL, "embedding_model": rag.EMBEDDING_MODEL,
                    "embedding_dim": rag.EMBEDDING_DIM, "top_k": rag.TOP_K, "use_llm": rag.USE_LLM}
        if args.section == "tools":
            return {"write_roots": self.config.write_roots, "commands": {
                name: {"description": command.description, "read_only": command.read_only}
                for name, command in self.config.commands.items()}}
        return {"servers": [{"name": server.name, "transport": server.transport,
                             "tools": list(server.tools)} for server in self.config.mcp_servers]}
