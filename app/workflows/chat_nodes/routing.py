"""任务路由：纯代码映射，不交给模型决定。"""
import logging

from app.workflows.chat_nodes.enterprise_scope import looks_enterprise, looks_oa_policy
from app.workflows.chat_state import ChatState
from app.workflows.source_policy import (
    SOURCE_BUSINESS, SOURCE_PUBLIC, classify_source, looks_text_task, suggested_task,
)

logger = logging.getLogger(__name__)

INTENT_TO_TASK = {
    "submit_leave": "submit_application",
    "delete_application": "delete_application",
    "approve": "approve_application",
    "query_status": "query",
    "query_pending": "query_pending",
    "policy_qa": "policy_qa",
    "get_suggestion": "suggestion",
    "confirm_action": "confirm_action",
    "cancel_action": "cancel_action",
    "clarify": "clarify",
}


def create_routing_nodes():
    nodes = {}

    def task_router_node(state: ChatState):
        intent = state.get("intent", "")
        message = state.get("user_message", "")
        current_task = INTENT_TO_TASK.get(intent, "general_chat")
        source = classify_source(message)
        # 模型把企业知识问题识别成闲聊时，仍交给企业知识检索链路。
        if (
            current_task == "general_chat"
            and looks_enterprise(message)
            and not looks_oa_policy(message)
            and not looks_text_task(message)
        ):
            current_task = "policy_qa"
        # 来源路由由代码决定：当前业务记录走业务查询，公开信息不转外网。
        if source["kind"] == SOURCE_PUBLIC:
            current_task = "clarify"
        elif source["kind"] == SOURCE_BUSINESS and current_task in {"policy_qa", "general_chat"}:
            current_task = suggested_task(message)
        final_intent = {
            "submit_application": "submit_leave", "approve_application": "approve",
            "query": "query_status", "suggestion": "get_suggestion",
        }.get(current_task, current_task)
        logger.info("任务路由: intent=%s task=%s source=%s", intent, current_task, source["kind"])
        if current_task != "submit_application":
            return {"current_task": current_task, "intent": final_intent, "iteration_count": 0,
                    "source_kind": source["kind"]}
        return {"current_task": current_task, "intent": final_intent, "source_kind": source["kind"]}

    nodes["task_router"] = task_router_node

    def task_executor_node(state: ChatState):
        return {"current_task": state.get("current_task", "")}

    nodes["task_executor"] = task_executor_node

    def route_task_execution(state: ChatState) -> str:
        return state.get("current_task", "general_chat")

    nodes["route_task_execution"] = route_task_execution

    return nodes
