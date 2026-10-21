"""
Pydantic请求模型
提供API请求参数验证和类型安全
"""
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


# 登录请求模型
class LoginRequest(BaseModel):
    """登录请求"""
    # 用户名字段（3-50字符）
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    # 密码字段（6-100字符）
    password: str = Field(..., min_length=6, max_length=100, description="密码")
    
    # 验证用户名格式
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """验证用户名格式"""
        import re
        # 如果不包含字母、数字和下划线
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            # 抛出验证错误
            raise ValueError('用户名只能包含字母、数字和下划线')
        # 返回验证后的用户名
        return v



