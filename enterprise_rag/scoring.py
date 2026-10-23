"""检索分数语义声明（S14）。

不同来源的分数方向与量纲不同，必须先声明再融合：
- BM25：无上界，越大越相关；只在同一索引内比较。
- 向量：余弦相似度，[-1, 1]，越大越相似；不同模型/维度不可比较。
- RRF：排名融合分，越大越靠前；不携带校准概率含义。
- 规则重排 score：0~1 的启发式分数，不是概率；阈值只用于拒答门槛。
"""
from __future__ import annotations

from enterprise_rag import config

SCORE_SEMANTICS = {
    "bm25": {"direction": "higher_is_better", "range": [0, None],
             "comparable_across_queries": False,
             "meaning": "同一索引内的词项相关度，不是概率"},
    "vector": {"direction": "higher_is_better", "range": [-1, 1],
               "comparable_across_models": False,
               "meaning": "当前 embedding 模型下的余弦相似度"},
    "rrf": {"direction": "higher_is_better", "range": [0, None],
            "comparable_across_queries": True,
            "meaning": "多查询排名融合分，仅用于排序"},
    "final": {"direction": "higher_is_better", "range": [0, 1.5],
              "meaning": "启发式重排分，不是校准概率；MIN_SCORE 是拒答门槛"},
    "graph": {"direction": "higher_is_better", "range": [0, None],
              "meaning": "关系置信度与跳数衰减的加和"},
}


def clamp_vector_score(score: float) -> float:
    return max(-1.0, min(1.0, float(score or 0.0)))


def threshold_for(comparison: bool) -> float:
    return config.MIN_SCORE_COMPARISON if comparison else config.MIN_SCORE


def describe() -> dict:
    return {
        "semantics": SCORE_SEMANTICS,
        "thresholds": {
            "min_score": config.MIN_SCORE,
            "min_score_comparison": config.MIN_SCORE_COMPARISON,
            "meaning": "低于阈值只触发拒答/证据不足，不代表概率置信度",
        },
    }
