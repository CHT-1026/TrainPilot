"""意图识别节点：只做理解，不做检索，不执行写操作。"""
import json
import logging
import re
from datetime import datetime

from langchain_openai import ChatOpenAI

from agent.app.workflows.chat_state import ChatState
from agent.app.workflows.intent_prompt import build_intent_prompt

logger = logging.getLogger(__name__)

ALLOWED_INTENTS = {
    "submit_leave",
    "delete_application",
    "approve",
    "query_status",
    "query_pending",
    "policy_qa",
    "get_suggestion",
    "confirm_action",
    "cancel_action",
    "general_chat",
    "clarify",
}

INJECTION_PATTERNS = (
    "忽略以上",
    "忽略之前",
    "忽略先前",
    "无视以上",
    "无视之前",
    "ignore previous",
    "ignore all previous",
    "ignore the above",
    "system prompt",
    "系统提示词",
    "开发者模式",
    "developer mode",
    "批准所有",
    "通过所有",
    "绕过权限",
    "跳过权限",
    "你现在是",
    "drop table",
    "delete from",
    "truncate table",
)


def _looks_like_injection(text: str) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in INJECTION_PATTERNS)


def _parse_payload(content: str) -> dict:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    intent = payload.get("intent")
    if intent not in ALLOWED_INTENTS:
        return {}
    info = payload.get("extracted_info")
    if not isinstance(info, dict):
        payload["extracted_info"] = {}
    return payload


def create_recognition_nodes(llm: ChatOpenAI, *, prompt_builder=None):
    nodes = {}

    def intent_recognition_node(state: ChatState):
        user_message = state.get("user_message", "")

        if _looks_like_injection(user_message):
            logger.warning("疑似提示词注入，已拦截")
            return {
                "intent": "clarify",
                "extracted_info": {"guard_reason": "prompt_injection"},
                "turn": state.get("turn", 0) + 1,
            }

        prompt = (prompt_builder or build_intent_prompt)(state, datetime.now())
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            payload = _parse_payload(content)
            if not payload:
                logger.warning("意图解析失败，返回澄清: %s", content[:200])
                return {
                    "intent": "clarify",
                    "extracted_info": {},
                    "turn": state.get("turn", 0) + 1,
                }
            logger.info(
                "意图识别: %s | 槽位: %s",
                payload.get("intent"),
                payload.get("extracted_info"),
            )
            return {
                "intent": payload["intent"],
                "extracted_info": payload.get("extracted_info", {}),
                "turn": state.get("turn", 0) + 1,
            }
        except Exception as exc:
            logger.error("意图识别异常: %s", exc, exc_info=True)
            return {
                "intent": "clarify",
                "extracted_info": {},
                "turn": state.get("turn", 0) + 1,
            }

    nodes["intent_recognition"] = intent_recognition_node
    return nodes
