"""首轮可验证 RL 的业务协议奖励；只评原始输出，不评 Agent 的修正或兜底。

适用任务是已标注意图与完整字段的抽取，不给开放问答或 RAG 事实自动打分。
格式合法只是必要条件；意图、全部字段都正确才奖励 1，否则为 0。
"""
import json

INTENTS = {"submit_leave", "delete_application", "approve", "query_status", "query_pending",
           "policy_qa", "get_suggestion", "confirm_action", "cancel_action", "general_chat", "clarify"}
FIELDS = {"leave_type", "start_date", "end_date", "reason", "app_id", "decision", "comments"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _valid_payload(value):
    return (isinstance(value, dict) and set(value) == {"intent", "extracted_info"}
            and isinstance(value["intent"], str) and value["intent"] in INTENTS
            and isinstance(value["extracted_info"], dict)
            and set(value["extracted_info"]) <= FIELDS
            and all(isinstance(v, str) and bool(v.strip()) for v in value["extracted_info"].values()))


def intent_reward(completion: str, expected: dict) -> dict:
    """返回二值奖励及诊断。expected 必须是经过复核的金标准，不能从模型生成。"""
    if not _valid_payload(expected):
        raise ValueError("意图奖励的参考标签不符合业务协议")
    try:
        payload = json.loads(completion, object_pairs_hook=_unique_object)
        valid = _valid_payload(payload)
    except (ValueError, TypeError):
        payload, valid = {}, False
    intent_ok = valid and payload["intent"] == expected["intent"]
    fields_exact = valid and payload["extracted_info"] == expected["extracted_info"]
    return {"reward": float(intent_ok and fields_exact), "format_ok": bool(valid),
            "intent_ok": bool(intent_ok), "fields_exact": bool(fields_exact)}
