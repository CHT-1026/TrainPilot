"""结构化工具结果：工作流不再依赖字符串反解析。"""
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    ok: bool
    code: str = "OK"
    data: dict[str, Any] | None = None
    message: str = ""
    retryable: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(cls, data: dict[str, Any] | None = None, message: str = "") -> "ToolResult":
        return cls(ok=True, code="OK", data=data or {}, message=message)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        data: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(ok=False, code=code, data=data, message=message, retryable=retryable)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
