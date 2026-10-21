"""意图识别节点：只做理解，不做检索，不执行写操作。"""
import json
import logging
import re
from datetime import datetime

from langchain_openai import ChatOpenAI

from app.workflows.chat_state import INTENT_RECOGNITION_HISTORY_COUNT, ChatState

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

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

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


def _pending_summary(pending_action: dict | None) -> str:
    if not pending_action:
        return "无"
    return json.dumps(
        {
            "action_type": pending_action.get("action_type"),
            "draft": pending_action.get("draft"),
        },
        ensure_ascii=False,
        default=str,
    )


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


def create_recognition_nodes(llm: ChatOpenAI):
    nodes = {}

    def intent_recognition_node(state: ChatState):
        user_message = state.get("user_message", "")
        messages = state.get("messages", [])
        pending_action = state.get("pending_action")

        if _looks_like_injection(user_message):
            logger.warning("疑似提示词注入，已拦截")
            return {
                "intent": "clarify",
                "extracted_info": {"guard_reason": "prompt_injection"},
                "turn": state.get("turn", 0) + 1,
            }

        history = ""
        if messages:
            history = "\n最近对话历史:\n"
            for msg in messages[-INTENT_RECOGNITION_HISTORY_COUNT:]:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = str(msg.get("content", ""))[:120]
                history += f"{role}: {content}\n"

        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_weekday = WEEKDAYS[now.weekday()]
        summary = state.get("summary", "")

        prompt = f"""
你是澜图智能企业智能体平台的意图解析器。你只输出 JSON，不执行任何操作。

当前日期：{current_date}（{current_weekday}）
当前待确认动作：{_pending_summary(pending_action)}
会话摘要：{summary or "无"}
{history}
用户消息：{user_message}

意图枚举（必须从中选择一个）：
- submit_leave：提交请假申请
- delete_application：取消本人的待审批申请
- approve：通过或驳回申请
- query_status：查询本人的申请
- query_pending：查看待审批列表
- policy_qa：询问公司制度、流程、福利、报销、安全、绩效、采购、项目、产品、客服、IT 运维、招聘等内部规定
- get_suggestion：获取某条申请的审批建议
- confirm_action：用户确认执行当前待确认动作
- cancel_action：用户放弃当前待确认动作
- general_chat：与 OA 业务无关的普通对话
- clarify：信息不足或无法判断

判断规则：
1. 如果「当前待确认动作」不是无，且用户回复“确认/确定/提交/可以/是/对/好/ok/yes”，选择 confirm_action。
2. 如果「当前待确认动作」不是无，且用户回复“取消/算了/不用了/放弃/不提交”，选择 cancel_action。
3. 用户明确要请假，选择 submit_leave。
4. 用户明确要“通过/批准/同意”或“驳回/拒绝”某个申请，选择 approve。
5. 用户问“我的申请/进度/状态”，选择 query_status。
6. 用户问“待审批/待办”，选择 query_pending。
7. 用户要“建议/怎么看/该不该批”，选择 get_suggestion。
8. 用户要“删除/取消某个申请”，选择 delete_application。
9. 用户询问公司制度、流程、标准、福利、报销、安全、绩效、晋升、采购、项目、产品、客服、IT 运维、招聘等内部规定，选择 policy_qa。
10. 只有与公司业务完全无关的闲聊才选择 general_chat；信息不足或歧义，选择 clarify。

不要猜测缺失的日期。日期统一转换为 YYYY-MM-DD；相对日期以当前日期为基准。
请假天数不要自行推断，由系统按工作日计算。

只返回 JSON：
{{
  "intent": "意图",
  "extracted_info": {{
    "leave_type": "年假/病假/事假/婚假/产假/丧假（可选）",
    "start_date": "YYYY-MM-DD（可选）",
    "end_date": "YYYY-MM-DD（可选）",
    "reason": "请假原因（可选）",
    "app_id": "leave_xxx（可选）",
    "decision": "通过/驳回（可选）",
    "comments": "审批意见（可选）",
    "employee_name": "员工姓名（可选）",
    "page": 1
  }}
}}
"""
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
