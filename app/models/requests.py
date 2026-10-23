"""API 请求模型：边界校验放在 Pydantic，服务层不再重复判空。"""
import re

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """对话请求：服务端校验长度与幂等标识。"""

    thread_id: str | None = Field(default=None, max_length=64)
    user_message: str = Field(..., min_length=1, max_length=2000)
    client_request_id: str | None = Field(default=None, max_length=64)

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("消息不能为空或仅包含空白字符")
        return value


class LoginRequest(BaseModel):
    """登录请求"""

    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, max_length=100, description="密码")

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9_]+", value):
            raise ValueError("用户名只能包含字母、数字和下划线")
        return value



