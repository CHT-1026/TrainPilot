"""按真实任务与复杂度选择模型与输出预算（S38）。

未配置路由时沿用 LLM_MODEL，行为与既有基线一致；只有显式配置模型别名才切换。
"""
from __future__ import annotations

from enterprise_rag import config


def choose_model(complexity: str = "medium", task: str = "answer") -> str:
    if task == "query_plan":
        return config.MODEL_ROUTE_SIMPLE or config.LLM_MODEL
    if complexity == "high":
        return config.MODEL_ROUTE_COMPLEX or config.LLM_MODEL
    if complexity == "low":
        return config.MODEL_ROUTE_SIMPLE or config.LLM_MODEL
    return config.LLM_MODEL


def max_tokens_for(complexity: str = "medium", task: str = "answer") -> int:
    if task == "query_plan":
        return 220
    # 复杂对比题需要多段 claim；预留足够输出，避免 JSON 被截断后整次调用作废。
    return 2600 if complexity == "high" else 1800


def route_for(plan: dict | None = None, task: str = "answer") -> dict:
    complexity = (plan or {}).get("complexity")
    if not complexity:
        complexity = "medium"
    return {
        "task": task,
        "complexity": complexity,
        "model": choose_model(complexity, task),
        "max_tokens": max_tokens_for(complexity, task),
    }
