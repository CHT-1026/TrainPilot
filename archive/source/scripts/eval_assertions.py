"""Promptfoo 式分维度断言基线：答案、来源、引用、拒答、权限、成本与时延分别计分。"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from app.domain.user.models import UserContext  # noqa: E402
from enterprise_rag.pipeline import query  # noqa: E402

TESTSET = ROOT / "enterprise_rag" / "eval" / "testset.jsonl"
OUT = ROOT / "audit_artifacts" / "eval_assertions"
USER = UserContext(user_id="u_assert", name="断言用户", role="employee", department="产品部")


def normalize(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def evaluate_case(case: dict, result: dict, latency_ms: int) -> dict:
    answer = result.get("answer", "") or ""
    citations = result.get("citations") or []
    cited_docs = {item.get("doc_id") for item in citations}
    expected_docs = set(case.get("expected_doc_ids") or [])
    accepted_docs = expected_docs | set(case.get("accepted_doc_ids") or [])
    excluded_docs = set(case.get("excluded_doc_ids") or [])
    keywords = case.get("answer_keywords") or case.get("keywords") or []
    expect_refusal = case.get("category") == "refusal" or bool(case.get("expect_refusal"))
    budget = result.get("observability") or {}
    issues = []
    refs = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    try:
        indices = [int(item.get("index", 0)) for item in citations]
    except (TypeError, ValueError):
        indices = []
    retrieval_docs = {item.get("doc_id") for item in result.get("retrieval", {}).get("chunks", []) if item.get("doc_id")}
    citation_structure = (bool(citations) and len(indices) == len(citations)
                          and set(indices) == set(range(1, len(citations) + 1))
                          and refs == set(indices)
                          and (not retrieval_docs or cited_docs <= retrieval_docs))
    source_contract = bool(accepted_docs & cited_docs) if accepted_docs else bool(citations)
    source_contract = (source_contract and not bool(excluded_docs & cited_docs)
                       and set(case.get("required_doc_ids") or []) <= cited_docs)

    if expect_refusal:
        answer_ok = None
        source_ok = None
        citation_ok = None
        # A refusal normally has no factual citations. If it does cite sources,
        # the case must explicitly name the acceptable evidence; N/A must not
        # hide fabricated references merely because refused=true.
        if citations or refs:
            source_ok = bool(accepted_docs) and source_contract
            citation_ok = citation_structure
            if not accepted_docs:
                issues.append("refusal_citations_without_source_gold")
        refusal_ok = bool(result.get("refused"))
    else:
        answer_ok = bool(keywords) and (not result.get("refused")) and all(normalize(word) in normalize(answer) for word in keywords)
        if not keywords:
            issues.append("missing_answer_gold")
        source_ok = source_contract
        citation_ok = citation_structure
        refusal_ok = result.get("refused") is False
    permission_ok = None
    if USER.role == "employee" and citations:
        visibilities = [item.get("visibility") for item in citations]
        if all(value is not None for value in visibilities):
            permission_ok = all(value == "all" for value in visibilities)
    cost_ok = (
        int(budget.get("model_calls", 0)) <= int(budget.get("max_model_calls", 4))
        and int(budget.get("embedding_calls", 0)) <= int(budget.get("max_embedding_calls", 8))
    )
    latency_ok = latency_ms <= 30000
    unknown_dimensions = []
    factuality_ok = None
    if not expect_refusal and not result.get("refused"):
        fact_status = (result.get("fact_check") or {}).get("verified")
        if type(fact_status) is bool:
            factuality_ok = fact_status
        else:
            unknown_dimensions.append("factuality")
        if USER.role == "employee" and permission_ok is None:
            unknown_dimensions.append("permission")
    dimensions = {
        "answer": answer_ok, "source": source_ok, "citation": citation_ok,
        "factuality": factuality_ok,
        "refusal": refusal_ok, "permission": permission_ok, "cost": cost_ok, "latency": latency_ok,
    }
    applicable = [value for value in dimensions.values() if value is not None]
    return {
        "id": case.get("id"), "question": case["question"], "method": result.get("method"),
        "refused": bool(result.get("refused")), "answer": answer, "dimensions": dimensions,
        "refusal_reason": result.get("refusal_reason"),
        "citations": citations, "fact_check": result.get("fact_check"),
        "evidence_scope": result.get("evidence_scope"),
        "model_used": result.get("model_used"), "prompt_version": result.get("prompt_version"),
        "retrieval_docs": sorted(retrieval_docs),
        "passed": all(applicable) and not unknown_dimensions, "latency_ms": latency_ms,
        "unknown_dimensions": unknown_dimensions,
        "model_calls": budget.get("model_calls", 0),
        "embedding_calls": budget.get("embedding_calls", 0),
        "expected_docs": sorted(expected_docs), "cited_docs": sorted(doc for doc in cited_docs if doc),
        "assessment_issues": issues,
        "answer_check_kind": "refusal_contract" if expect_refusal else "gold_keyword_match",
    }


def run_case(case: dict, use_llm: bool) -> dict:
    started = time.monotonic()
    result = query(case["question"], user=USER, use_llm=use_llm, retrieval_mode="hybrid")
    return evaluate_case(case, result, int((time.monotonic() - started) * 1000))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="允许模型生成；默认走确定性/抽取式")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cases = [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    records = [run_case(case, args.llm) for case in cases]
    dimensions = ["answer", "source", "citation", "factuality", "refusal", "permission", "cost", "latency"]

    def dimension_stat(name: str) -> dict:
        applicable = [item for item in records if item["dimensions"][name] is not None]
        passed = sum(1 for item in applicable if item["dimensions"][name])
        unknown = sum(name in item["unknown_dimensions"] for item in records)
        return {
            "passed": passed,
            "applicable": len(applicable),
            "not_applicable": len(records) - len(applicable) - unknown,
            "unknown": unknown,
            "rate": round(passed / len(applicable), 4) if applicable else None,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assessment_contract": "v2.2-factuality-and-explicit-unknown",
        "model": "llm" if args.llm else "deterministic",
        "total": len(records),
        "overall_passed": sum(1 for item in records if item["passed"]),
        "dimensions": {name: dimension_stat(name) for name in dimensions},
        "records": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT / f"assertions_{stamp}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# 分维度断言基线（{summary['model']}）", "",
        f"- 用例：{summary['total']}；全部适用维度通过：{summary['overall_passed']}/{summary['total']}", "",
        "| 维度 | 通过/适用 | 不适用 | 未知 | 通过率 |", "| --- | --- | --- | --- | --- |",
    ]
    for name in dimensions:
        item = summary["dimensions"][name]
        rate = "-" if item["rate"] is None else f"{item['rate'] * 100:.2f}%"
        lines.append(f"| {name} | {item['passed']}/{item['applicable']} | {item['not_applicable']} | {item['unknown']} | {rate} |")
    lines += ["", "| 用例 | 答案 | 来源 | 引用 | 有限事实核验 | 拒答 | 权限 | 成本 | 时延 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in records:
        marks = []
        for name in dimensions:
            value = item["dimensions"][name]
            marks.append("UNKNOWN" if name in item["unknown_dimensions"] else "N/A" if value is None else ("PASS" if value else "FAIL"))
        lines.append(f"| {item['id']} | " + " | ".join(marks) + " |")
    md_path = OUT / f"assertions_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({**{key: summary[key] for key in ("total", "overall_passed")},
                      "dimensions": {name: summary["dimensions"][name]["rate"] for name in dimensions},
                      "report": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
