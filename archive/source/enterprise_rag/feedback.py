"""审计后的反馈闭环（S41）。

只记录用户显式反馈与可复核标识，不自动把反馈写回提示词、图谱或语料。
需要人工审核后才能进入评测集或优化任务。
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.domain.user.models import UserContext

FEEDBACK_DIR = Path(__file__).resolve().parents[1] / "audit_artifacts" / "feedback"
_LOCK = threading.Lock()
_TZ = timezone(timedelta(hours=8))

RATINGS = {-1, 0, 1}


def record_feedback(
    user: UserContext,
    *,
    question: str,
    answer_excerpt: str = "",
    rating: int | None = None,
    correction: str = "",
    citation_ids: list[str] | None = None,
    trace_id: str = "",
    snapshot: str = "",
    failure_stage: str = "",
) -> dict:
    if rating is not None and rating not in RATINGS:
        raise ValueError("rating 只能是 -1/0/1")
    if not question.strip():
        raise ValueError("question 不能为空")
    record = {
        "feedback_id": f"fb_{uuid.uuid4().hex[:12]}",
        "created_at": datetime.now(_TZ).isoformat(),
        "user_id": user.user_id,
        "role": user.role,
        "department": user.department or "",
        "question": question.strip()[:500],
        "answer_excerpt": (answer_excerpt or "").strip()[:800],
        "rating": rating,
        "correction": (correction or "").strip()[:1200],
        "citation_ids": list(dict.fromkeys(citation_ids or []))[:40],
        "trace_id": trace_id[:64],
        "snapshot": snapshot[:64],
        "failure_stage": failure_stage[:32],
        "review_status": "pending",
    }
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    path = FEEDBACK_DIR / "feedback.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return record


def read_feedback(limit: int = 100) -> list[dict]:
    path = FEEDBACK_DIR / "feedback.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-max(1, min(limit, 500)):]


def feedback_summary() -> dict:
    records = read_feedback(500)
    by_rating = {"up": 0, "down": 0, "neutral": 0, "unrated": 0}
    for record in records:
        rating = record.get("rating")
        if rating == 1:
            by_rating["up"] += 1
        elif rating == -1:
            by_rating["down"] += 1
        elif rating == 0:
            by_rating["neutral"] += 1
        else:
            by_rating["unrated"] += 1
    return {"total": len(records), "ratings": by_rating, "pending_review": sum(
        1 for record in records if record.get("review_status") == "pending"
    )}
