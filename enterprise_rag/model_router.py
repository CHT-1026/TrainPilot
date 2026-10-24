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


def max_tokens_for(complexity: str = "medium", task: str = "answer",
                   synthesis: bool = False) -> int:
    if task == "query_plan":
        return 220
    # 综合题只保留有限 claim；过长输出既拖慢网关，也更容易在截断后整次作废。
    if synthesis:
        return config.MAX_TOKENS_SYNTHESIS
    if complexity == "high":
        return config.MAX_TOKENS_COMPLEX
    return config.MAX_TOKENS_MEDIUM


def route_for(plan: dict | None = None, task: str = "answer") -> dict:
    plan = plan or {}
    complexity = plan.get("complexity") or "medium"
    synthesis = bool(plan.get("synthesis"))
    return {
        "task": task,
        "complexity": complexity,
        "model": choose_model(complexity, task),
        "max_tokens": max_tokens_for(complexity, task, synthesis=synthesis),
    }
