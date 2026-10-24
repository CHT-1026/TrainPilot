"""Fixed synthetic reasoning audit through the production retrieval/answer pipeline.

Only index loading is substituted: no retriever, answer, prompt, or model stub.
Fixtures do not enter the persisted knowledge base. This evaluates correctness,
full supporting-evidence recall, citations, refusal, and authorization separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from app.domain.user.models import UserContext
from enterprise_rag import config
from enterprise_rag.embedding import embed_texts
from enterprise_rag.pipeline import query
from enterprise_rag.store import RagIndex

OUT = ROOT / "audit_artifacts" / "reasoning"
FIXTURES = OUT / "cases.jsonl"


def source_hashes():
    files = ["enterprise_rag/answer.py", "enterprise_rag/retrieval.py", "enterprise_rag/store.py", "enterprise_rag/query.py", "enterprise_rag/citations.py", "enterprise_rag/pipeline.py"]
    files += [name for name in ["enterprise_rag/reasoning.py", "enterprise_rag/config.py"] if (ROOT / name).exists()]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


_EMBED_CACHE: dict[str, list[float]] = {}


def _embeddings(texts: list[str]) -> list[list[float]]:
    missing = [text for text in dict.fromkeys(texts) if text not in _EMBED_CACHE]
    if missing:
        vectors = embed_texts(missing)
        _EMBED_CACHE.update(zip(missing, vectors))
    return [_EMBED_CACHE[text] for text in texts]


def rows_for(case):
    rows = []
    for i, fact in enumerate(case["documents"], 1):
        rows.append({
            "chunk_id": fact["id"] + "#0", "doc_id": fact["id"],
            "title": fact.get("title", "合成审计资料"), "section": fact.get("section", "事实记录"),
            "chunk_type": "text", "ord": 0, "text": fact["text"],
            "version": fact.get("version", "v1"), "doc_family": fact.get("family", fact["id"]),
            "effective_date": fact.get("effective", "2024-01-01"),
            "doc_updated_at": fact.get("updated", "2024-01-01T00:00:00"),
            "visibility": fact.get("visibility", "all"), "department": fact.get("department", "审计部"),
            "namespace": "synthetic_reasoning_audit", "format": "md",
            "source_path": "synthetic://" + case["id"] + "/" + fact["id"],
        })
    vectors = _embeddings([row["text"] for row in rows])
    for row, vector in zip(rows, vectors):
        row["embedding"] = vector
        row["embedding_dim"] = config.EMBEDDING_DIM
        row["embedding_model"] = config.EMBEDDING_MODEL
        row["embedding_version"] = config.EMBEDDING_VERSION
    return rows


def evaluate(case, use_llm):
    started = time.monotonic()
    user = UserContext(user_id=case.get("user_id", "audit-reader"), name="合成评测用户", role=case.get("role", "employee"), department="审计部")
    index = RagIndex(rows_for(case))
    result = None
    error = None
    logs = []
    class CaptureWarnings(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                logs.append(record.getMessage())
    handler = CaptureWarnings()
    logging.getLogger("enterprise_rag").addHandler(handler)
    try:
        with patch("enterprise_rag.retrieval.load_index", return_value=index), patch("enterprise_rag.store.load_index", return_value=index):
            result = query(case["question"], user=user, as_of=case.get("as_of"), use_llm=use_llm)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        logging.getLogger("enterprise_rag").removeHandler(handler)
    result = result or {"answer": "", "refused": False, "citations": [], "retrieval": {"chunks": []}, "method": "error"}
    answer = result["answer"]
    expected_docs = set(case.get("expected_docs", []))
    recalled = {c["doc_id"] for c in result["retrieval"]["chunks"]}
    cited = {c["doc_id"] for c in result["citations"]}
    forbidden = set(case.get("forbidden_docs", []))
    pattern_hits = {p: bool(re.search(p, answer, re.I)) for p in case.get("answer_patterns", [])}
    negative_hits = {p: bool(re.search(p, answer, re.I)) for p in case.get("forbidden_answer_patterns", [])}
    all_evidence = expected_docs <= recalled
    citation_complete = expected_docs <= cited
    refused_correct = result["refused"] == case.get("expect_refusal", False)
    no_leak = not (forbidden & (recalled | cited)) and not any(negative_hits.values())
    answer_correct = all(pattern_hits.values()) and refused_correct and no_leak
    if not case.get("expect_refusal"):
        answer_correct = answer_correct and bool(pattern_hits) and bool(cited)
    passed = bool(not error and answer_correct and all_evidence and (case.get("expect_refusal") or citation_complete))
    reasons = []
    if error:
        reasons.append("执行异常: " + error)
    if not all_evidence:
        reasons.append("检索未召回全部证据: " + ", ".join(sorted(expected_docs - recalled)))
    if not all(pattern_hits.values()):
        reasons.append("生成未给出预期推断结论")
    if not refused_correct:
        reasons.append("应拒答而未拒答" if case.get("expect_refusal") else "有足够依据但误拒答")
    if not no_leak:
        reasons.append("访问控制或未来版本泄露")
    if not case.get("expect_refusal") and not citation_complete:
        reasons.append("引用未覆盖推断所需全部来源")
    return {
        "id": case["id"], "category": case["category"], "question": case["question"],
        "expected": case["expected"], "answer": answer,
        "strict_passed": passed, "answer_correct": answer_correct,
        "all_evidence_recalled": all_evidence, "all_evidence_cited": citation_complete,
        "expected_docs": sorted(expected_docs), "retrieved_docs": sorted(recalled), "cited_docs": sorted(cited),
        "pattern_hits": pattern_hits, "negative_pattern_hits": negative_hits,
        "refused": result["refused"], "method": result["method"], "llm_requested": use_llm,
        "failure_reasons": reasons, "error": error, "latency_ms": round((time.monotonic() - started) * 1000),
        "runtime_warnings": logs,
        "result": result,
    }


def markdown(report):
    m = report["metrics"]
    lines = ["# 合成多文档推理审计 - " + report["phase"], "", f"执行时间：{report['generated_at']}。固定用例：{m['total']}；严格通过：{m['strict_passed']}；全部证据召回：{m['all_evidence_recalled']}。", "", "测试边界：仅替换内存索引加载，真实执行 BM25、向量召回、权限/版本过滤、重排、上下文与答案生成。无模型替身；LLM 测试使用配置的真实接口。用例语料与企业库隔离，不写入业务数据库。", "", f"请求生成方式：{'允许真实 LLM（以逐条实际method为准）' if report['use_llm'] else '显式use_llm=False离线生成'}；配置模型：{report['config']['llm_model']}；向量提供方：{report['config']['embedding_provider']}。", "", "严格通过要求：结论正确、全部必要证据召回并引用、拒答/权限符合要求。答案出现源文档中的数字不等于完成计算。", ""]
    for item in report["cases"]:
        lines += ["## " + item["id"] + " - " + ("通过" if item["strict_passed"] else "失败"), "", "问题：" + item["question"], "", "期望：" + item["expected"], "", "实际回答：" + item["answer"], "", f"生成方式：{item['method']}；全部证据召回：{item['all_evidence_recalled']}；全部证据引用：{item['all_evidence_cited']}；耗时：{item['latency_ms']} ms。", "", "原因：" + ("；".join(item["failure_reasons"]) or "满足严格断言"), ""]
        if item.get("runtime_warnings"):
            lines += ["运行警告：" + "；".join(item["runtime_warnings"]), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "regression"], required=True)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--ids", default="", help="Comma-separated fixed case IDs")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES, help="Reviewed JSONL fixture file; supports an independent holdout set")
    parser.add_argument("--output-name", default="", help="Optional report basename, to preserve baseline and separate holdouts")
    parser.add_argument("--gate", action="store_true", help="Nonzero exit unless every case strictly passes")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.fixtures.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        ids = set(args.ids.split(","))
        cases = [case for case in cases if case["id"] in ids]
    before = source_hashes()
    results = []
    for case in cases:
        result = evaluate(case, args.use_llm)
        results.append(result)
        print(json.dumps({key: result[key] for key in ["id", "strict_passed", "all_evidence_recalled", "method", "latency_ms", "failure_reasons"]}, ensure_ascii=False), flush=True)
    report = {
        "phase": args.phase, "generated_at": datetime.now().astimezone().isoformat(), "use_llm": args.use_llm,
        "fixtures_path": str(args.fixtures), "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(), "source_hashes_at_start": before,
        "source_hashes_at_end": source_hashes(),
        "config": {"llm_model": config.LLM_MODEL, "embedding_provider": config.EMBEDDING_PROVIDER, "top_k": config.TOP_K},
        "metrics": {"total": len(results), "strict_passed": sum(r["strict_passed"] for r in results), "answer_correct": sum(r["answer_correct"] for r in results), "all_evidence_recalled": sum(r["all_evidence_recalled"] for r in results)},
        "cases": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_llm" if args.use_llm else ""
    target = OUT / (args.output_name or (args.phase + suffix))
    target.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    target.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(target.with_suffix('.md')), "metrics": report["metrics"]}, ensure_ascii=False), flush=True)
    if args.gate and not all(item["strict_passed"] for item in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
