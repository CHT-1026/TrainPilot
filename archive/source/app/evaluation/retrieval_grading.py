"""Deterministic typed assertions and quote provenance. Unknown semantics stay unreviewed."""
from __future__ import annotations

import re
from decimal import Decimal

JUDGE_VERSION = 'typed_evidence_v2'


def equal_value(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, dict):
        return isinstance(actual, dict) and expected.keys() == actual.keys() and all(equal_value(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(equal_value(e, a) for e, a in zip(expected, actual))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return Decimal(str(expected)) == Decimal(str(actual))
    return type(expected) is type(actual) and expected == actual


def compact(text: str) -> str:
    # Only whitespace differs across PDF extraction and model quotations. Keep signs, units and punctuation.
    return re.sub(r"\s+", "", text)


def visible_segments(trace: dict, after_turn: int = 0) -> list[dict]:
    rows = []
    for call in trace.get("tool_calls_detail", []):
        if call.get("turn", 0) < after_turn or not call.get("visible", True):
            continue
        result = call["result"]
        rows.extend(result.get("hits", []))
        rows.extend(result.get("segments", []))
    return rows


def grade(trace: dict, gold: dict) -> dict:
    rubric = gold.get("retrieval_rubric")
    answer = trace.get("answer")
    base = {"judge_version": JUDGE_VERSION, "correct": None, "evidence_supported": False,
            "success": False, "review_status": "pending_semantic_review", "reasons": []}
    if answer is None:
        return {**base, "correct": False, "review_status": "terminal_failure", "reasons": [trace.get("status", "missing_answer")]}
    if rubric is None:
        return {**base, "reasons": ["No task-specific assertions; lexical overlap is not semantic correctness"]}
    correct_answer = equal_value(rubric["expected"], answer["answer"])
    correct_status = answer["status"] in rubric['accepted_statuses']
    correct = correct_answer and correct_status
    rows = visible_segments(trace, rubric.get("evidence_after_turn", 0))
    citations = answer["citations"]
    valid, quoted = [], []
    for c in citations:
        matches = [s for s in rows if c["doc_id"] == s["doc_id"] and c["locator"] == s["locator"]
                   and bool(compact(c["quote"])) and compact(c["quote"]) in compact(s["text"])]
        valid.append(bool(matches))
        if matches:
            quoted.append(c)
    groups = rubric["evidence_groups"]
    covered = [any(c["doc_id"] == ref["doc_id"] and compact(ref["quote"]) in compact(c["quote"])
                   for c in quoted for ref in group) for group in groups]
    supported = bool(citations) and all(valid) and all(covered)
    reasons = []
    if not correct:
        reasons.append("typed_answer_or_status_mismatch")
    if not supported:
        reasons.append("missing_or_unseen_or_insufficient_quote_evidence")
    return {**base, "correct": correct, "correct_answer": correct_answer, "correct_status": correct_status,
            "evidence_supported": supported, "success": correct and supported,
            "review_status": "rule_graded", "evidence_groups": covered, "citations_valid": valid,
            "reasons": reasons}
