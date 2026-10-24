"""冻结未见集评测：运行时从当前语料/图谱生成题目，记录配置快照与执行支路。

`--dry-run` 只输出数量，不输出题目；题目只在真实运行后写入报告。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from generate_reasoning_cases import generate_cases  # noqa: E402
import dynamic_document_acceptance as dda  # noqa: E402

OUT = ROOT / "audit_artifacts" / "frozen_eval"
TEST_SET = ROOT / "enterprise_rag" / "eval" / "testset.jsonl"
REASONING_SET = ROOT / "enterprise_rag" / "eval" / "reasoning_cases.jsonl"
SPEC = ROOT / "enterprise_rag" / "eval" / "unseen_spec.json"


def _load_expected_docs() -> set[str]:
    docs = set()
    for path in (TEST_SET, REASONING_SET):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            docs.update(case.get("expected_doc_ids") or [])
    return docs


def freeze_spec() -> dict:
    path = SPEC
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    spec = {
        "version": "2026.10-frozen-v1",
        "seed": 20261022,
        "sample_size": 60,
        "generated_multihop": 20,
        "generated_version": 10,
        "generated_counterexample": 10,
        "excluded_doc_ids": sorted(_load_expected_docs()),
        "rule": "题目由当前 manifest/索引/图谱在运行时生成，题目文本不进入冻结规格",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec


def config_snapshot(mode: str) -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        head = ""
    from enterprise_rag import config
    from enterprise_rag.graph.retrieve import load_graph
    from enterprise_rag.prompts import PROMPT_VERSIONS

    graph = load_graph()
    return {
        "git_head": head,
        "model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "retrieval_mode": mode,
        "graph_version": graph.version,
        "graph_entities": len(graph.entities),
        "graph_relations": len(graph.relations),
        "prompt_version": PROMPT_VERSIONS["answer"],
        "recorded_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    }


def _http_json(base: str, path: str, payload: dict | None = None, token: str | None = None,
               timeout: float = 180) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(base + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _case_from_candidate(item: dict) -> dict:
    return {
        "id": "frozen_" + hashlib.sha256(item["question"].encode("utf-8")).hexdigest()[:10],
        "category": item.get("category", "document"),
        "question": item["question"],
        "expected_doc_ids": [item.get("doc_id")] if item.get("doc_id") else [],
        "answer_keywords": [],
        "expect_refusal": False,
        "source_item": item,
    }


def build_cases(spec: dict) -> list[dict]:
    candidates, meta = dda.build_candidates()
    excluded = set(spec["excluded_doc_ids"])
    candidates = [item for item in candidates if item.get("doc_id") not in excluded]
    selected = dda.select(candidates, spec["sample_size"], spec["seed"])
    cases = [_case_from_candidate(item) for item in selected]
    # 复用现有的动态运算题（基于当前语料解析，不是固定题库）。
    for index, item in enumerate(dda.multi_hop_candidates(), 1):
        cases.append({
            "id": f"frozen_math_{index:02d}",
            "category": "multi_hop_numeric",
            "question": item["question"],
            "expected_doc_ids": [item.get("doc_id")] if item.get("doc_id") else [],
            "answer_keywords": [],
            "expect_refusal": False,
            "source_item": item,
        })
    generated = generate_cases(seed=spec["seed"],
                               version_limit=spec["generated_version"],
                               hop_limit=spec["generated_multihop"],
                               counter_limit=spec["generated_counterexample"])
    cases.extend(generated)
    # 所有类型组装后统一执行排除，避免生成/多跳样本绕过冻结规格。
    return [case for case in cases
            if not (set(case.get("expected_doc_ids") or []) & excluded)]


def evaluate(case: dict, data: dict) -> dict:
    if case.get("source_item"):
        item = case["source_item"]
        result = dda.evaluate(item, data)
        detail = dict(result)
        detail["source_ok"] = not result.get("failure_reasons", []) or all(
            reason != "无有效引用" for reason in result.get("failure_reasons", [])
        )
        detail["answer_ok"] = bool(result.get("value_ok")) and bool(result.get("anchor_ok"))
        return {"passed": bool(result.get("passed")), "detail": detail}
    answer = data.get("answer", "") or ""
    citations = data.get("citations") or []
    cited_docs = {citation.get("doc_id") for citation in citations if citation.get("doc_id")}
    if case.get("expect_refusal"):
        refused = bool(data.get("refused")) or any(
            marker in answer for marker in ("未覆盖", "资料不足", "无法回答", "没有找到", "未提及", "建议联系")
        )
        return {"passed": refused, "detail": {"refused": refused, "source_ok": True, "answer_ok": refused}}
    missing_docs = sorted(set(case.get("expected_doc_ids") or []) - cited_docs)
    keywords = case.get("answer_keywords") or []
    missing_keywords = [keyword for keyword in keywords
                        if dda.normalize(keyword) not in dda.normalize(answer)]
    source_ok = not missing_docs and bool(citations)
    # 没有可核对答案事实的用例不能只凭引用计为通过。
    answer_ok = bool(keywords) and not missing_keywords and not data.get("refused")
    return {"passed": source_ok and answer_ok,
            "detail": {"missing_docs": missing_docs, "missing_keywords": missing_keywords,
                       "source_ok": source_ok, "answer_ok": answer_ok}}


def run(args) -> None:
    spec = freeze_spec()
    cases = build_cases(spec)
    if args.limit:
        cases = cases[:args.limit]
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "count": len(cases),
            "by_category": dict(Counter(case["category"] for case in cases)),
            "spec": spec["version"], "seed": spec["seed"],
        }, ensure_ascii=False, indent=2))
        return

    base = args.base_url.rstrip("/")
    login = _http_json(base, "/api/auth/login",
                       {"username": args.username, "password": args.password}, timeout=30)["data"]
    token = login["token"]
    started = time.monotonic()
    records = []
    for position, case in enumerate(cases, 1):
        payload = {"question": case["question"], "top_k": args.top_k,
                   "use_llm": not args.no_llm, "retrieval_mode": args.mode}
        if case.get("as_of"):
            payload["as_of"] = case["as_of"]
        started_at = time.monotonic()
        error = ""
        data = {}
        try:
            data = _http_json(base, "/api/rag/query", payload, token, timeout=args.timeout)["data"]
            latency = int((time.monotonic() - started_at) * 1000)
        except urllib.error.HTTPError as exc:
            latency = int((time.monotonic() - started_at) * 1000)
            error = f"HTTP {exc.code}"
        except Exception as exc:
            latency = int((time.monotonic() - started_at) * 1000)
            error = type(exc).__name__
        score = {"passed": False, "detail": {}} if error else evaluate(case, data)
        records.append({
            **case,
            "answer": data.get("answer", ""),
            "refused": bool(data.get("refused")),
            "cited_doc_ids": sorted({citation.get("doc_id") for citation in data.get("citations") or []
                                     if citation.get("doc_id")}),
            "execution_path": (data.get("retrieval") or {}).get("debug", {}).get("retrieval_mode", args.mode),
            "fact_check": data.get("fact_check"),
            "evidence_scope": data.get("evidence_scope"),
            "latency_ms": latency,
            "error": error,
            "passed": score["passed"],
            "score_detail": score["detail"],
        })
        print(f"[{position:03d}/{len(cases):03d}] {case['id']} {case['category']} "
              f"{'PASS' if score['passed'] else 'FAIL'} {latency}ms", flush=True)

    passed = sum(record["passed"] for record in records)
    by_category = defaultdict(lambda: {"total": 0, "passed": 0})
    for record in records:
        bucket = by_category[record["category"]]
        bucket["total"] += 1
        bucket["passed"] += int(record["passed"])
    summary = {
        "spec": spec["version"], "seed": spec["seed"], "label": args.label,
        "config": config_snapshot(args.mode),
        "total": len(records), "passed": passed, "failed": len(records) - passed,
        "pass_rate": round(passed / len(records), 4) if records else 0.0,
        "categories": {name: {**value, "pass_rate": round(value["passed"] / value["total"], 4)}
                       for name, value in sorted(by_category.items())},
        "avg_latency_ms": round(sum(record["latency_ms"] for record in records) / (len(records) or 1), 1),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "fact_support_rate": round(sum(1 for record in records
                                       if (record.get("fact_check") or {}).get("verified"))
                                   / (len(records) or 1), 4),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.label}_{args.mode}_{stamp}"
    (OUT / f"{stem}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    lines = [
        f"# 冻结未见集评测（{args.label} / {args.mode}）", "",
        f"- 规格：`{spec['version']}`，seed `{spec['seed']}`",
        f"- 通过：**{passed}/{len(records)}**（{summary['pass_rate'] * 100:.2f}%）",
        f"- 平均耗时：{summary['avg_latency_ms']} ms",
        f"- 配置快照：`{json.dumps(summary['config'], ensure_ascii=False)}`", "",
        "## 分类", "", "| 类别 | 总数 | 通过 | 通过率 |", "| --- | --- | --- | --- |",
    ]
    for name, value in summary["categories"].items():
        lines.append(f"| {name} | {value['total']} | {value['passed']} | {value['pass_rate'] * 100:.2f}% |")
    lines += ["", "## 题目与结果（运行后生成）", "",
              "| ID | 类别 | 结果 | 问题 | 失败原因 |", "| --- | --- | --- | --- | --- |"]
    for record in records:
        detail = json.dumps(record.get("score_detail"), ensure_ascii=False).replace("|", "/")[:120]
        lines.append(f"| {record['id']} | {record['category']} | "
                     f"{'PASS' if record['passed'] else 'FAIL'} | {record['question']} | {detail} |")
    (OUT / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"REPORT {OUT / (stem + '.md')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="admin123")
    parser.add_argument("--label", default="frozen")
    parser.add_argument("--mode", default="hybrid", choices=("vector", "hybrid", "graph"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
