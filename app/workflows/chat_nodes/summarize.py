"""会话摘要：只压缩旧消息，保留最近对话原文。"""
import logging

from langchain_openai import ChatOpenAI

from app.workflows.chat_state import ChatState

logger = logging.getLogger(__name__)

MIN_NEW_MESSAGES = 12
KEEP_RECENT = 8


def should_summarize(messages: list[dict], summarized_count: int) -> bool:
    if not messages:
        return False
    return len(messages) - max(0, summarized_count) >= MIN_NEW_MESSAGES


def create_summarize_node(llm: ChatOpenAI):
    def summarize_node(state: ChatState):
        messages = state.get("messages", []) or []
        summarized_count = int(state.get("summarized_count", 0) or 0)
        if not should_summarize(messages, summarized_count):
            return {}

        end = max(summarized_count, len(messages) - KEEP_RECENT)
        old_messages = messages[summarized_count:end]
        if not old_messages:
            return {}

        transcript = []
        for message in old_messages:
            role = "用户" if message.get("role") == "user" else "助手"
            transcript.append(f"{role}: {str(message.get('content', ''))[:200]}")

        previous = state.get("summary", "")
        prompt = f"""
你是 OA 会话摘要器。请把旧对话压缩成不超过 300 字的摘要。

要求：
1. 保留：用户目标、已确认的槽位、已完成动作与结果、未解决问题。
2. 不复述病假原因、家庭情况等敏感细节，只保留“已提交/已审批”等状态。
3. 不要编造未发生的事实。

已有摘要：
{previous or "无"}

需要压缩的旧对话：
{chr(10).join(transcript)}

输出摘要："""

        try:
            response = llm.invoke(prompt)
            summary = response.content if hasattr(response, "content") else str(response)
            summary = (summary or "").strip()[:800]
        except Exception as exc:
            logger.warning("会话摘要失败: %s", exc)
            return {}

        if not summary:
            return {}
        logger.info("会话摘要完成: %s 条消息 -> %s 字", len(old_messages), len(summary))
        return {"summary": summary, "summarized_count": end}

    return {"summarize": summarize_node}
