"""服务端用户上下文：权限判断只信任这里，不信任模型输出。"""
from typing import Any

from pydantic import BaseModel, ConfigDict

VALID_ROLES = {"employee", "manager", "finance", "hr", "admin"}


class UserContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    name: str
    role: str
    department: str | None = None
    position: str | None = None
    manager_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserContext":
        return cls(
            user_id=data.get("username") or data.get("user_id") or "",
            name=data.get("name") or "",
            role=data.get("role") or "employee",
            department=data.get("department"),
            position=data.get("position"),
            manager_id=data.get("manager_id"),
        )

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_manager(self) -> bool:
        return self.role == "manager"

    @property
    def is_hr(self) -> bool:
        return self.role == "hr"

    @property
    def is_approver(self) -> bool:
        return self.role in {"manager", "hr", "admin"}
