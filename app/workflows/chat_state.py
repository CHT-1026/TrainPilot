"""对话状态与常量。字段类型必须与实际使用一致，避免止于注释的类型保护。"""
from operator import add
from typing import Annotated, Any, TypedDict


class ChatState(TypedDict, total=False):
    # 对话消息历史（字典格式，保持与前端历史接口兼容）
    messages: Annotated[list[dict[str, Any]], add]
    # 当前轮用户消息
    user_message: str
    # 意图识别结果
    intent: str
    # 当前轮抽取的槽位
    extracted_info: dict[str, Any]
    # 路由后的任务类型
    current_task: str
    # 面向用户的确定性结果文本
    task_result: str
    # 服务端用户上下文：user_id / user_role / user_department / user_name
    user: dict[str, Any]
    # 对话上下文：用户信息 + 已累计槽位
    conversation_context: dict[str, Any]
    # 等待用户确认的动作（提交/审批/取消）
    pending_action: dict[str, Any] | None
    # 最近一次结构化工具结果
    last_result: dict[str, Any] | None
    # 制度问答检索到的文本块
    retrieved_chunks: list[dict[str, Any]]
    retrieval_question: str
    knowledge_namespace: str
    retrieval_meta: dict[str, Any]
    answer_method: str
    # 回答引用的来源
    citations: list[dict[str, Any]]
    # 回答是否有资料支撑/是否为正确拒答
    grounded: bool
    # 信息收集轮次，防止无限循环
    iteration_count: int
    # 当前会话轮次
    turn: int
    # 旧消息摘要（上下文压缩）
    summary: str
    # 已纳入摘要的消息数量
    summarized_count: int
    # 请求链路ID
    trace_id: str
    # 最终回复
    response: str


INTENT_RECOGNITION_HISTORY_COUNT = 6

# 需要直接使用确定性文案、不再让模型改写的意图
DETERMINISTIC_INTENTS = {
    "policy_qa",
    "submit_leave",
    "approve",
    "query",
    "query_status",
    "query_pending",
    "delete_application",
    "get_suggestion",
    "confirm_action",
    "cancel_action",
    "clarify",
}
