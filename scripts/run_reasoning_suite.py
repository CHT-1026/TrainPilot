"""Run the 100+ case reasoning suite over a live service and classify failures.

Usage:
    .\.venv312\Scripts\python.exe scripts\run_reasoning_suite.py --label qwen38-flash
    .\.venv312\Scripts\python.exe scripts\run_reasoning_suite.py --label qwen37-plus --limit 10
    .\.venv312\Scripts\python.exe scripts\run_reasoning_suite.py --label probe --category multi_hop_relation

Cases come from enterprise_rag/eval/testset.jsonl (74 general cases) plus
enterprise_rag/eval/reasoning_cases.jsonl (40 version/multi-hop cases).
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "enterprise_rag" / "eval"
OUTPUT_DIR = ROOT / "audit_artifacts" / "reasoning_suite"


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("％", "%").replace("—", "-").replace("－", "-").replace("至", "-")
    return re.sub(r"[\s,，、]+", "", text)


def keyword_hit(keyword: str, answer_norm: str) -> bool:
    """A keyword may list alternatives as 5/10/15 天; every part must appear."""
    parts = [normalize(part) for part in keyword.split("/") if part.strip()]
    if len(parts) > 1:
        return all(part in answer_norm for part in parts)
    return normalize(keyword) in answer_norm


def load_cases(paths: list[Path]) -> list[dict]:
    cases, seen = [], set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            if case["id"] in seen:
                continue
            seen.add(case["id"])
            cases.append(case)
    return cases


def http_json(base: str, path: str, payload: dict | None = None, token: str | None = None,
              timeout: float = 180) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(base + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def classify(case: dict, record: dict) -> tuple[bool, str]:
    expected_docs = set(case.get("expected_doc_ids") or [])
    cited_docs = set(record.get("cited_doc_ids") or [])
    missing_docs = sorted(expected_docs - cited_docs)
    missing_keywords = record.get("missing_keywords") or []
    refused = record.get("refused")
    answer_norm = normalize(record.get("answer", ""))
    forbidden = [word for word in case.get("forbidden_keywords") or []
                 if normalize(word) in answer_norm]

    if case.get("expect_refusal"):
        refusal_text = any(marker in answer_norm for marker in
                           ("未覆盖", "资料不足", "无法回答", "没有找到", "未提及", "建议联系"))
        passed = bool(refused) or refusal_text
        return passed, "" if passed else "refusal_miss"

    if forbidden:
        return False, "forbidden_content"
    if refused:
        return False, "unexpected_refusal"
    strict_docs = record.get("category") in {"multi_hop_relation", "version_comparison", "version_conflict"}
    if strict_docs and missing_docs:
        return False, "retrieval_miss"
    if not cited_docs:
        return False, "citation_missing"
    if missing_keywords:
        return False, "wrong_answer"
    return True, ""


def run(args) -> dict:
    base = args.base_url.rstrip("/")
    login = http_json(base, "/api/auth/login",
                      {"username": args.username, "password": args.password},
                      timeout=30)["data"]
    token = login["token"]
    cases = load_cases([EVAL_DIR / "testset.jsonl", EVAL_DIR / "reasoning_cases.jsonl"])
    if args.category:
        wanted = {item.strip() for item in args.category.split(",") if item.strip()}
        cases = [case for case in cases if case.get("category") in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("没有匹配的测试用例")

    started_at = datetime.now(timezone(timedelta(hours=8)))
    records = []
    for position, case in enumerate(cases, 1):
        payload = {
            "question": case["question"],
            "top_k": args.top_k,
            "use_llm": not args.no_llm,
        }
        if case.get("as_of"):
            payload["as_of"] = case["as_of"]
        started = time.monotonic()
        error = ""
        data = {}
        try:
            data = http_json(base, "/api/rag/query", payload, token, timeout=args.timeout)["data"]
            latency = int((time.monotonic() - started) * 1000)
        except urllib.error.HTTPError as exc:
            latency = int((time.monotonic() - started) * 1000)
            error = f"HTTP {exc.code}"
        except Exception as exc:  # timeout, connection, invalid JSON
            latency = int((time.monotonic() - started) * 1000)
            error = type(exc).__name__

        answer = data.get("answer", "")
        citations = data.get("citations", [])
        cited_docs = sorted({citation.get("doc_id") for citation in citations if citation.get("doc_id")})
        answer_norm = normalize(answer)
        missing_keywords = [
            keyword for keyword in case.get("answer_keywords", [])
            if not keyword_hit(keyword, answer_norm)
        ]
        record = {
            "id": case["id"],
            "category": case.get("category", "unknown"),
            "question": case["question"],
            "as_of": case.get("as_of"),
            "expected_doc_ids": sorted(case.get("expected_doc_ids") or []),
            "answer": answer,
            "refused": bool(data.get("refused")),
            "grounded": bool(data.get("grounded")),
            "method": data.get("method", ""),
            "cited_doc_ids": cited_docs,
            "missing_expected_doc_ids": sorted(set(case.get("expected_doc_ids") or []) - set(cited_docs)),
            "expected_docs_cited": not (set(case.get("expected_doc_ids") or []) - set(cited_docs)),
            "missing_keywords": missing_keywords,
            "top_score": (data.get("retrieval") or {}).get("debug", {}).get("top_score"),
            "comparison_mode": (data.get("retrieval") or {}).get("debug", {}).get("comparison_mode"),
            "latency_ms": latency,
            "error": error,
        }
        if error:
            passed, failure_type = False, "http_error"
        else:
            passed, failure_type = classify(case, record)
        record["passed"] = passed
        record["failure_type"] = failure_type
        records.append(record)
        status = "PASS" if passed else f"FAIL[{failure_type}]"
        print(f"[{position:03d}/{len(cases):03d}] {case['id']} {case.get('category')} {status} "
              f"{latency}ms {case['question'][:40]}", flush=True)

    elapsed = round((datetime.now(timezone(timedelta(hours=8))) - started_at).total_seconds(), 1)
    summary = summarize(records, args.label, elapsed, args)
    write_reports(summary, records, args.label)
    return summary


def summarize(records: list[dict], label: str, elapsed: float, args) -> dict:
    by_category = defaultdict(lambda: {"total": 0, "passed": 0})
    type_counts = Counter()
    for record in records:
        bucket = by_category[record["category"]]
        bucket["total"] += 1
        bucket["passed"] += int(record["passed"])
        if not record["passed"]:
            type_counts[record["failure_type"]] += 1
    categories = {
        name: {**values, "pass_rate": round(values["passed"] / values["total"], 4) if values["total"] else 0.0}
        for name, values in sorted(by_category.items())
    }
    passed = sum(record["passed"] for record in records)
    return {
        "label": label,
        "model_hint": args.model_hint,
        "base_url": args.base_url,
        "use_llm": not args.no_llm,
        "top_k": args.top_k,
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "elapsed_seconds": elapsed,
        "total": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "pass_rate": round(passed / len(records), 4) if records else 0.0,
        "categories": categories,
        "failure_types": dict(sorted(type_counts.items())),
        "avg_latency_ms": round(sum(record["latency_ms"] for record in records) / len(records), 1)
        if records else 0,
    }


def write_reports(summary: dict, records: list[dict], label: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    stem = f"{label}_{stamp}"
    (OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# 推理与版本对比规模测试报告（{label}）",
        "",
        f"- 模型标识：`{summary['model_hint']}`",
        f"- 服务地址：`{summary['base_url']}`",
        f"- 用例总数：**{summary['total']}**，通过 **{summary['passed']}**，失败 **{summary['failed']}**，"
        f"通过率 **{summary['pass_rate'] * 100:.2f}%**",
        f"- 平均耗时：{summary['avg_latency_ms']} ms，总耗时 {summary['elapsed_seconds']} s",
        "",
        "## 分类通过率",
        "",
        "| 类别 | 总数 | 通过 | 通过率 |",
        "| --- | --- | --- | --- |",
    ]
    for name, values in summary["categories"].items():
        lines.append(f"| {name} | {values['total']} | {values['passed']} | {values['pass_rate'] * 100:.2f}% |")
    lines += ["", "## 失败分类", "", "| 类型 | 数量 |", "| --- | --- |"]
    for name, count in summary["failure_types"].items():
        lines.append(f"| {name} | {count} |")
    failures = [record for record in records if not record["passed"]]
    lines += ["", "## 失败明细", "",
              "| ID | 类别 | 失败类型 | 问题 | 缺失关键词 | 缺失文档 | 回答摘录 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for record in failures:
        answer = record["answer"].replace("\n", " ").replace("|", "/")[:160]
        lines.append(
            f"| {record['id']} | {record['category']} | {record['failure_type']} | "
            f"{record['question']} | {'、'.join(record['missing_keywords']) or '—'} | "
            f"{'、'.join(record['missing_expected_doc_ids']) or '—'} | {answer} |"
        )
    (OUTPUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT {OUTPUT_DIR / (stem + '.md')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run the reasoning suite over HTTP")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="admin123")
    parser.add_argument("--label", default="run")
    parser.add_argument("--model-hint", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", default="")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
