"""TrainPilot 意图协议：网页、终端与训练侧使用同一份规则文本。"""
import json
from pathlib import Path

from agent.app.workflows.chat_state import INTENT_RECOGNITION_HISTORY_COUNT


TEMPLATE_PATH = Path(__file__).with_suffix(".txt")
INTENT_TEMPLATE = TEMPLATE_PATH.read_text(encoding="utf-8")


def build_intent_prompt(state, now, *, template=None):
    pending = state.get("pending_action")
    context = {
        "current_date": now.strftime("%Y-%m-%d"),
        "history": [{"role": m["role"], "content": str(m["content"])[:120]}
                    for m in state.get("messages", [])[-INTENT_RECOGNITION_HISTORY_COUNT:]],
        "pending_action": {"action_type": pending["action_type"], "draft": pending["draft"]} if pending else None,
    }
    if state.get("summary"):
        context["summary"] = state["summary"]
    known = (state.get("conversation_context") or {}).get("accumulated_extracted_info")
    if known:
        context["accumulated_extracted_info"] = known
    rules = INTENT_TEMPLATE if template is None else template
    return rules + "\n给定上下文：\n" + json.dumps(context, ensure_ascii=False, default=str) + "\n用户输入：\n" + state["user_message"]
