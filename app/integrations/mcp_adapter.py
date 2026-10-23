"""MCP/外部系统适配器契约（S33）。

MCP 只作为受限外部系统适配器：服务器端注入 actor、只读优先、动作白名单、
超时与错误码固定。授权永远由本平台服务端决定，不把 MCP 服务当授权系统。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.user.models import UserContext


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    description: str
    read_only: bool = True
    actions: tuple[str, ...] = ()
    timeout_seconds: float = 5.0
    allowed_roles: tuple[str, ...] = ("admin", "hr", "manager", "finance", "employee")


SPECS = {
    "hr_directory": AdapterSpec(
        name="hr_directory",
        description="只读组织与员工目录",
        read_only=True,
        actions=("lookup_employee", "list_department"),
        allowed_roles=("admin", "hr", "manager"),
    ),
    "asset_registry": AdapterSpec(
        name="asset_registry",
        description="只读资产归属查询",
        read_only=True,
        actions=("lookup_asset",),
        allowed_roles=("admin", "hr", "manager", "finance"),
    ),
}

_MUTATING = {"create", "update", "delete", "approve", "submit", "cancel"}


@dataclass(frozen=True)
class AdapterDecision:
    allowed: bool
    code: str = "OK"
    message: str = ""
    actor_id: str = ""
    injected: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "message": self.message,
            "actor_id": self.actor_id,
            "injected": self.injected,
        }


def authorize(actor: UserContext, adapter_name: str, action: str) -> AdapterDecision:
    spec = SPECS.get(adapter_name)
    if spec is None:
        return AdapterDecision(False, "UNKNOWN_ADAPTER", "未注册的外部适配器")
    if actor.role not in spec.allowed_roles:
        return AdapterDecision(False, "FORBIDDEN", "当前角色无权使用该外部适配器",
                               actor.user_id)
    if action not in spec.actions:
        return AdapterDecision(False, "ACTION_NOT_ALLOWED", "动作不在适配器白名单内",
                               actor.user_id)
    if any(word in action.lower() for word in _MUTATING):
        return AdapterDecision(False, "WRITE_NOT_ALLOWED", "外部适配器默认只读", actor.user_id)
    return AdapterDecision(
        True,
        actor_id=actor.user_id,
        injected={"actor_id": actor.user_id, "role": actor.role,
                  "department": actor.department or "", "read_only": spec.read_only,
                  "timeout_seconds": spec.timeout_seconds},
    )


def describe_adapters() -> list[dict]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "read_only": spec.read_only,
            "actions": list(spec.actions),
            "timeout_seconds": spec.timeout_seconds,
            "allowed_roles": list(spec.allowed_roles),
        }
        for spec in SPECS.values()
    ]
