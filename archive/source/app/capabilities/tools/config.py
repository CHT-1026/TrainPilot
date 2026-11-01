"""Operator configuration. Models cannot choose roots, credentials or executables."""
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[3]


class ToolConfigError(ValueError):
    pass


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Command(ConfigModel):
    argv: list[str] = Field(min_length=1)
    description: str
    read_only: bool = False
    accepts_paths: bool = False


class MCPToolPolicy(ConfigModel):
    read_only: bool = False


class MCPServerConfig(ConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,15}$")
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str = "."
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=15, gt=0, le=120)
    allowed_roles: list[str] = Field(default_factory=lambda: ["admin"])
    tools: dict[str, MCPToolPolicy] = Field(min_length=1)

    @model_validator(mode="after")
    def check_transport(self):
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio requires command")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("streamable_http requires url")
        return self


class ToolsConfig(ConfigModel):
    workspace_root: str = "."
    write_roots: list[str] = Field(default_factory=lambda: ["tool_workspace"])
    local_roles: list[str] = Field(default_factory=lambda: ["admin"])
    command_timeout_seconds: float = Field(default=30, gt=0, le=120)
    agent_timeout_seconds: float = Field(default=90, gt=0, le=300)
    max_rounds: int = Field(default=4, ge=1, le=8)
    max_calls: int = Field(default=8, ge=1, le=20)
    commands: dict[str, Command] = Field(default_factory=dict)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def unique_servers(self):
        names = [server.name for server in self.mcp_servers]
        if len(set(names)) != len(names):
            raise ValueError("MCP server names must be unique")
        return self

    @property
    def root(self) -> Path:
        return (ROOT / self.workspace_root).resolve()


def config_path() -> Path:
    return Path(os.getenv("TOOLS_CONFIG_PATH", ROOT / "config/tools.json")).resolve()


def load_config() -> ToolsConfig:
    # Invalid operator configuration is an error, never a silent empty registry.
    try:
        return ToolsConfig.model_validate_json(config_path().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError):
        raise ToolConfigError("工具配置不可用，请检查TOOLS_CONFIG_PATH指定的JSON配置") from None


def resolve_env(value: str) -> str:
    """Only an entire env reference is expanded; no shell evaluation."""
    if value.startswith("${") and value.endswith("}"):
        return os.environ[value[2:-1]]
    return value
