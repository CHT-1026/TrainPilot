"""查询节点：结构化服务 + 模板文案，不做字符串反解析。"""
import logging

from app.services.leave_service import leave_service
from app.workflows.chat_nodes.common import actor_from_state
from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)


def create_query_nodes():
    nodes = {}

    def query_node(state: ChatState):
        result = leave_service.mine(actor_from_state(state), limit=20)
        return {"task_result": result.message, "last_result": result.to_dict(), "pending_action": None}

    nodes["query"] = query_node

    def query_pending_node(state: ChatState):
        result = leave_service.pending(actor_from_state(state), limit=20)
        return {"task_result": result.message, "last_result": result.to_dict(), "pending_action": None}

    nodes["query_pending"] = query_pending_node
    return nodes
