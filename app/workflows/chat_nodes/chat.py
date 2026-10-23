"""日常对话与澄清节点。"""
import logging

from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)


def create_chat_nodes():
    nodes = {}

    def general_chat_node(state: ChatState):
        return {"task_result": "", "pending_action": None}

    nodes["general_chat"] = general_chat_node

    def clarify_node(state: ChatState):
        info = state.get("extracted_info", {}) or {}
        if info.get("guard_reason") == "prompt_injection":
            return {
                "task_result": "该请求包含无法处理的指令。请用正常业务语言描述您的需求。",
                "pending_action": None,
            }
        return {
            "task_result": "抱歉，我没太理解您的需求。您可以询问企业知识、查询制度、提交请假，还是审批申请。",
            "pending_action": None,
        }

    nodes["clarify"] = clarify_node

    def record_unsupported_node(state: ChatState):
        message = state.get("user_message", "") or ""
        if "考勤" in message:
            detail = "考勤打卡记录"
        elif "年假" in message or "休假" in message or "调休" in message:
            detail = "年假余额与休假记录"
        else:
            detail = "实时业务记录"
        return {
            "task_result": f"{detail}尚未接入实时服务，暂时无法查询；不会用请假申请列表或制度天数替代。",
            "pending_action": None,
        }

    nodes["record_unsupported"] = record_unsupported_node
    return nodes
