"""节点公共工具：统一从状态里取服务端用户上下文。"""
from app.domain.user.models import UserContext
from app.workflows.chat_state import ChatState


def actor_from_state(state: ChatState) -> UserContext:
    return UserContext.model_validate(state["user"])
