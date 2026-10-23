"""最终回复：确定性结果直接使用模板，只有普通对话才调用模型。"""
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from app.workflows.chat_state import DETERMINISTIC_INTENTS, ChatState

logger = logging.getLogger(__name__)


def create_response_nodes(llm: ChatOpenAI):
    nodes = {}

    async def response_generator_node(state: ChatState):
        intent = state.get("intent", "")
        task_result = state.get("task_result", "") or ""
        user_message = state.get("user_message", "")
        messages = state.get("messages", []) or []
        context = state.get("conversation_context", {}) or {}
        summary = state.get("summary", "")

        current_task = state.get("current_task", "")
        if (current_task and current_task != "general_chat") or (
            not current_task and intent in DETERMINISTIC_INTENTS
        ):
            response_text = task_result or "抱歉，本次操作没有产生结果。"
        else:
            history = ""
            for msg in messages[-4:]:
                role = "用户" if msg.get("role") == "user" else "助手"
                history += f"{role}: {str(msg.get('content', ''))[:150]}\n"
            prompt = f"""
你是澜图智能企业智能体平台的企业助手，负责：
1. 企业知识与制度问答：依据检索到的资料回答；
2. 请假流程：提交、查询、取消；
3. 审批：待审批列表、审批建议、通过/驳回。

限制：
- 只能依据系统提供的资料回答企业知识问题，不得编造。
- 资料未覆盖时明确说明，并建议联系对应业务部门确认。
- 不要声称已经执行任何写操作；写操作必须经过用户确认。
- 回答简洁、直接。

用户：{context.get('user_name') or context.get('user_id', '')}（{context.get('user_department') or '未填部门'}）
已记住的用户事实（不是企业证据，不能替代检索）：{context.get('fact_memory') or '无'}
会话摘要：{summary or "无"}
最近对话：
{history}
用户消息：{user_message}
系统已知结果：{task_result}
"""
            try:
                try:
                    writer = get_stream_writer()
                except RuntimeError:
                    writer = None
                if writer is None:
                    result = await llm.ainvoke(prompt)
                    response_text = result.content if hasattr(result, "content") else str(result)
                else:
                    parts = []
                    async for delta in llm.astream(prompt):
                        text = delta.content if hasattr(delta, "content") else str(delta)
                        if text:
                            parts.append(text)
                            writer({"type": "token", "delta": text})
                    response_text = "".join(parts)
            except Exception as exc:
                logger.error("生成普通回复失败: %s", exc, exc_info=True)
                response_text = task_result or "抱歉，服务暂时不可用，请稍后重试。"

        ai_message = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
        }

        if intent in {"confirm_action", "cancel_action", "policy_qa", "general_chat"}:
            accumulated = {}
        else:
            accumulated = state.get("extracted_info") or context.get("accumulated_extracted_info") or {}
        updated_context = {**context, "accumulated_extracted_info": accumulated}

        return {
            "response": response_text,
            "messages": [ai_message],
            "conversation_context": updated_context,
        }

    nodes["response_generator"] = response_generator_node
    return nodes
