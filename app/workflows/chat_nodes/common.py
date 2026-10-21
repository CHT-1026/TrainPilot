"""节点公共工具：统一从状态里取服务端用户上下文。"""
from app.domain.user.models import UserContext
from app.workflows.chat_state import ChatState


def actor_from_state(state: ChatState) -> UserContext:
    user = state.get("user", {}) or {}
    context = state.get("conversation_context", {}) or {}
    return UserContext.from_dict(
        {
            "user_id": user.get("user_id") or context.get("user_id", ""),
            "name": user.get("name") or context.get("user_name") or context.get("user_id", ""),
            "role": user.get("role") or context.get("user_role", "employee"),
            "department": user.get("department") or context.get("user_department"),
            "position": user.get("position") or context.get("user_position"),
            "manager_id": user.get("manager_id") or context.get("manager_id"),
        }
    )


def pick(info: dict, *keys, default=""):
    for key in keys:
        if key in info and info[key] not in (None, ""):
            return info[key]
    return default
