"""摘要窗口与 Checkpoint 保留分开治理（S22）。

摘要只压缩旧消息，不等于删除 Checkpoint；两者有独立的保留周期。
"""
from __future__ import annotations

from datetime import datetime, timedelta

MIN_NEW_MESSAGES = 12
KEEP_RECENT = 8
CHECKPOINT_RETENTION_DAYS = 30


def should_summarize(messages: list[dict], summarized_count: int) -> bool:
    if not messages:
        return False
    return len(messages) - max(0, summarized_count) >= MIN_NEW_MESSAGES


def retention_plan(messages: list[dict], summarized_count: int,
                   checkpoint_ttl_days: int = CHECKPOINT_RETENTION_DAYS) -> dict:
    total = len(messages or [])
    summarized = max(0, summarized_count)
    if not should_summarize(messages, summarized):
        return {
            "should_summarize": False,
            "summarize_range": [summarized, total],
            "keep_recent": min(total, KEEP_RECENT),
            "checkpoint_ttl_days": checkpoint_ttl_days,
            "note": "未达到摘要窗口；消息仍由 Checkpoint 保留",
        }
    end = max(summarized, total - KEEP_RECENT)
    return {
        "should_summarize": end > summarized,
        "summarize_range": [summarized, end],
        "keep_recent": total - end,
        "checkpoint_ttl_days": checkpoint_ttl_days,
        "note": "摘要范围与 Checkpoint 保留周期相互独立",
    }


def stale_before(days: int = CHECKPOINT_RETENTION_DAYS) -> datetime:
    return datetime.now() - timedelta(days=max(days, 1))


def checkpoint_pattern(thread_id: str = "*") -> str:
    return f"checkpoint:{thread_id}:*"


def checkpoint_patterns(thread_id: str = "*") -> list[str]:
    """Cover checkpoint / blob / write / latest key families used by the saver."""
    return [
        f"checkpoint:{thread_id}:*",
        f"checkpoint_blob:{thread_id}:*",
        f"checkpoint_write:{thread_id}:*",
        f"checkpoint_latest:{thread_id}:*",
    ]


def summarize_transcript(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = "用户" if message.get("role") == "user" else "助手"
        lines.append(f"{role}: {str(message.get('content', ''))[:200]}")
    return "\n".join(lines)
