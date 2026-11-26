"""分层奖励候选；仅显式选择graded的对照实验使用，旧严格奖励保留。"""
import json

from model.trainer.agent_rewards import _unique_object, _valid_payload, intent_reward

FIELD_WEIGHTS = {"app_id": 4, "decision": 4, "start_date": 3, "end_date": 3,
                 "leave_type": 2, "reason": 1, "comments": 1}
CRITICAL_FIELDS = {"app_id", "decision", "start_date", "end_date", "leave_type"}
WRITE_INTENTS = {"submit_leave", "approve", "delete_application", "confirm_action", "cancel_action"}


def field_differences(predicted: dict, expected: dict) -> dict:
    """返回四类字段名列表；每类按字段名排序。

    correct：两边都有且值相同。
    missing：expected中存在，predicted中不存在。
    wrong：两边都有，但值不同。
    extra：predicted中存在，expected中不存在。

    None表示键存在而值为null，不能把它当作键不存在。
    字段类型是否合法留给协议检查，本函数只负责对比。
    """
    # 示例：逐个检查标准答案的键，避免访问预测中不存在的键。
    correct = sorted(
        key for key in expected
        if key in predicted and predicted[key] == expected[key]
    )

    missing = sorted(key for key in expected if key not in predicted)

    wrong = sorted(
        key for key in expected
        if key in predicted and predicted[key] != expected[key]
    )

    extra = sorted(key for key in predicted if key not in expected)

    return {"correct": correct, "missing": missing, "wrong": wrong, "extra": extra}


def graded_intent_reward(completion: str, expected: dict, *, ended=True) -> dict:
    """固定候选v2：关键错误分级，字段加权F1；不是RAG语义奖励。"""
    strict = intent_reward(completion, expected)
    result = {"strict_reward": strict["reward"] if ended else 0.0,
              **{k: v for k, v in strict.items() if k != "reward"},
              "eligible": False, "severe_error": False, "field_differences": None,
              "field_f1": 0.0, "critical_missing": [], "critical_wrong": []}
    def finish(reward, level, severe=False):
        return {**result, "reward": reward, "level": level, "severe_error": severe}
    if not ended:
        return finish(-1.0, "truncated")
    try:
        payload = json.loads(completion, object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        return finish(-1.0, "invalid_json")
    if not _valid_payload(payload):
        return finish(-0.8, "invalid_schema")
    if not strict["intent_ok"]:
        risky = payload["intent"] in WRITE_INTENTS
        return finish(-1.0 if risky else -0.6, "wrong_action" if risky else "wrong_intent", risky)
    differences = field_differences(payload["extracted_info"], expected["extracted_info"])
    result["field_differences"] = differences
    result["critical_missing"] = sorted(set(differences["missing"]) & CRITICAL_FIELDS)
    result["critical_wrong"] = sorted((set(differences["wrong"]) | set(differences["extra"])) & CRITICAL_FIELDS)
    if result["critical_wrong"]:
        return finish(-1.0, "critical_wrong", True)
    result["eligible"] = True
    weight = lambda keys: sum(FIELD_WEIGHTS[k] for k in keys)
    correct = weight(differences["correct"])
    errors = 2 * weight(differences["wrong"]) + weight(differences["missing"]) + weight(differences["extra"])
    result["field_f1"] = 2 * correct / (2 * correct + errors) if 2 * correct + errors else 1.0
    reward = 0.4 + 0.5 * result["field_f1"] + 0.1 * result["strict_reward"]
    if result["critical_missing"]:
        return finish(min(0.2, reward), "critical_missing")
    return finish(reward, "complete" if result["strict_reward"] else "partial")
