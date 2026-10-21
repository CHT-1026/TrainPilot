"""企业 RAG 固定题集回归：检索命中、关键词、拒答和引用结构。

关键词与引用编号只能作为代理指标，不能证明语义正确、推理正确或引用忠实。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe -m enterprise_rag.eval.run_eval
    .venv312\\Scripts\\python.exe -m enterprise_rag.eval.run_eval --gate
"""
import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env.local", override=False)
load_dotenv(BACKEND_ROOT / ".env", override=False)

from app.domain.user.models import UserContext  # noqa: E402
from enterprise_rag import config  # noqa: E402
from enterprise_rag.ingest import load_manifest  # noqa: E402
from enterprise_rag.pipeline import query  # noqa: E402
from enterprise_rag.store import index_stats  # noqa: E402

TESTSET = config.EVAL_DIR / "testset.jsonl"
EVAL_USER = UserContext(user_id="rag-eval", name="评测用户", role="admin", department="管理部")
SCORING_LIMITATIONS = [
    "keyword_accuracy（兼容旧键 answer_accuracy）仅检查标准化关键词是否全部出现，不是语义准确率，不能排除否定、矛盾或错误推理。",
    "any_expected_doc_hit@5/@10（兼容旧键 recall@5/@10）检查前 N 个检索块是否含任一期望文档，不是全部证据召回率；文档编号在这 N 块内去重。",
    "all_expected_docs_recalled@10 与 all_expected_citation_coverage 只针对题集标注的期望文档；标注完整性和逐句证据支持仍需人工审查。",
    "引用编号校验只验证答案标记、返回引用和召回文档之间的结构一致性，不是语义蕴含或引用忠实度校验。",
    "本评测用于原固定题集回归；计数、数值/时间运算、多跳依赖及证据缺失应查看独立 reasoning 评测，不能由本报告的关键词分数推定通过。",
]


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("％", "%").replace("—", "-").replace("－", "-")
    text = re.sub(r"[\s,，、]+", "", text)
    return text


def load_cases(limit: int | None = None) -> list[dict]:
    cases = []
    with TESTSET.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def doc_ranking(chunks: list[dict], top_n: int) -> list[str]:
    """Legacy ranking unit is retrieved chunks; deduplicate document IDs inside that window."""
    seen = []
    for chunk in chunks[:top_n]:
        doc_id = chunk["doc_id"]
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def ndcg(ranked: list[str], expected: set[str]) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(ranked[:10], 1):
        if doc_id in expected:
            dcg += 1 / math.log2(index + 1)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(expected), 10) + 1))
    return dcg / ideal if ideal else 0.0


def audit_citations(answer: str, citations: list[dict], chunks: list[dict], expected: set[str]) -> dict:
    """Check marker/source structure without claiming that a source entails an answer."""
    markers = sorted({int(marker) for marker in re.findall(r"\[(\d+)\]", answer)})
    retrieved_docs = {chunk.get("doc_id") for chunk in chunks}
    by_index = {}
    for citation in citations:
        try:
            index = int(citation.get("index", 0))
        except (TypeError, ValueError, OverflowError):
            continue
        # A duplicate index is ambiguous, even when one of its sources is relevant.
        by_index[index] = citation if index not in by_index else None
    valid = [
        index for index in markers
        if index > 0 and by_index.get(index) and by_index[index].get("doc_id")
        and by_index[index]["doc_id"] in retrieved_docs
    ]
    cited = {by_index[index]["doc_id"] for index in valid}
    cited_expected = cited & expected
    return {
        "valid_citation_markers": valid,
        "invalid_citation_markers": sorted(set(markers) - set(valid)),
        "citation_markers_valid": bool(markers) and len(valid) == len(markers),
        "cited_doc_ids": sorted(cited),
        "cited_expected_doc_ids": sorted(cited_expected),
        "missing_expected_citation_doc_ids": sorted(expected - cited_expected),
        "expected_citation_sources_complete": bool(expected) and expected <= cited,
        "expected_citation_source_fraction": round(len(cited_expected) / len(expected), 4) if expected else 0.0,
    }


def evaluate_case(case: dict, use_llm: bool | None = None) -> dict:
    started = time.monotonic()
    requested_top_k = max(10, config.TOP_K)
    result = query(case["question"], user=EVAL_USER, top_k=requested_top_k, use_llm=use_llm)
    latency = int((time.monotonic() - started) * 1000)

    chunks = result["retrieval"]["chunks"]
    top5 = doc_ranking(chunks, 5)
    top10 = doc_ranking(chunks, 10)
    expected = set(case.get("expected_doc_ids") or [])

    rank = next(
        (index for index, doc_id in enumerate(top10, 1) if doc_id in expected),
        None,
    ) if expected else None
    answer = result["answer"]
    answer_norm = normalize(answer)
    keyword_hits = [keyword for keyword in case.get("answer_keywords", []) if normalize(keyword) in answer_norm]
    answer_correct = bool(case.get("answer_keywords")) and len(keyword_hits) == len(case["answer_keywords"])

    citation_docs = [citation["doc_id"] for citation in result["citations"]]
    citation_audit = audit_citations(answer, result["citations"], chunks, expected)
    citation_hit = bool(citation_audit["cited_expected_doc_ids"])
    expect_refusal = bool(case.get("expect_refusal"))
    refusal_correct = result["refused"] == expect_refusal

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_doc_ids": sorted(expected),
        "top5": top5,
        "top10": top10,
        "first_relevant_rank": rank,
        "ndcg@10": round(ndcg(top10, expected), 4),
        "answer": answer,
        "refused": result["refused"],
        "grounded": result["grounded"],
        "answer_correct": answer_correct,
        "keyword_correct": answer_correct,
        "answer_method": result.get("method", "unknown"),
        "requested_top_k": requested_top_k,
        "retrieved_chunk_count": len(chunks),
        "retrieved_distinct_document_count": len({chunk["doc_id"] for chunk in chunks}),
        "keyword_hits": keyword_hits,
        "missing_keywords": [
            keyword for keyword in case.get("answer_keywords", []) if normalize(keyword) not in answer_norm
        ],
        "citation_docs": citation_docs,
        "citation_hit": citation_hit,
        **citation_audit,
        "time_filter": result["retrieval"]["as_of"],
        "query_rewrite": result["query_rewrite"],
        "top_score": result["retrieval"]["top_score"],
        "latency_ms": latency,
        "expect_refusal": expect_refusal,
        "refusal_correct": refusal_correct,
    }


def aggregate(results: list[dict]) -> dict:
    non_refusal = [item for item in results if not item["expect_refusal"]]
    refusal_cases = [item for item in results if item["expect_refusal"]]
    total = len(non_refusal) or 1

    recall5 = sum(1 for item in non_refusal if set(item["top5"]) & set(item["expected_doc_ids"])) / total
    recall10 = sum(1 for item in non_refusal if set(item["top10"]) & set(item["expected_doc_ids"])) / total
    full_recall10 = sum(
        1 for item in non_refusal
        if item["expected_doc_ids"] and set(item["expected_doc_ids"]) <= set(item["top10"])
    ) / total
    mrr = sum(
        (1 / item["first_relevant_rank"]) if item["first_relevant_rank"] else 0 for item in non_refusal
    ) / total
    ndcg_value = sum(item["ndcg@10"] for item in non_refusal) / total
    accuracy = sum(1 for item in non_refusal if item["answer_correct"]) / total
    citation_coverage = sum(1 for item in non_refusal if item["citation_hit"]) / total
    complete_citation_coverage = sum(
        1 for item in non_refusal if item.get("expected_citation_sources_complete", False)
    ) / total
    valid_marker_rate = sum(1 for item in non_refusal if item.get("citation_markers_valid", False)) / total
    grounded_rate = sum(1 for item in non_refusal if item["grounded"]) / total
    false_refusal = sum(1 for item in non_refusal if item["refused"]) / total
    refusal_accuracy = (
        sum(1 for item in refusal_cases if item["refused"]) / len(refusal_cases)
        if refusal_cases
        else 0.0
    )

    return {
        "total_cases": len(results),
        "retrieval_cases": len(non_refusal),
        "recall@5": round(recall5, 4),
        "recall@10": round(recall10, 4),
        "any_expected_doc_hit@5": round(recall5, 4),
        "any_expected_doc_hit@10": round(recall10, 4),
        "all_expected_docs_in_top10": round(full_recall10, 4),
        "all_expected_docs_recalled@10": round(full_recall10, 4),
        "mrr": round(mrr, 4),
        "ndcg@10": round(ndcg_value, 4),
        "answer_accuracy": round(accuracy, 4),
        "keyword_accuracy": round(accuracy, 4),
        "citation_coverage": round(citation_coverage, 4),
        "all_expected_citation_coverage": round(complete_citation_coverage, 4),
        "valid_citation_marker_rate": round(valid_marker_rate, 4),
        "grounded_rate": round(grounded_rate, 4),
        "false_refusal_rate": round(false_refusal, 4),
        "refusal_accuracy": round(refusal_accuracy, 4),
        "avg_latency_ms": round(sum(item["latency_ms"] for item in results) / (len(results) or 1), 1),
    }


def by_category(results: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        groups[item["category"]].append(item)
    return {category: aggregate(items) for category, items in groups.items()}


def write_report(report: dict, results: list[dict]) -> None:
    report.setdefault("scoring_limitations", SCORING_LIMITATIONS)
    report.setdefault("evaluation_scope", "legacy_fixed_corpus_keyword_regression")
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (config.REPORT_DIR / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = report["metrics"]
    lines = [
        "# 澜图智能企业 RAG 评测报告",
        "",
        f"- 语料命名空间：`{report['corpus']['namespace']}`",
        f"- 文档数：{report['corpus']['documents']}，文本块数：{report['corpus']['chunks']}",
        f"- 评测用例：{metrics['total_cases']} 条（检索类 {metrics['retrieval_cases']} 条，拒答类 {metrics['total_cases'] - metrics['retrieval_cases']} 条）",
        f"- 生成方式：{report['config']['answer_method']}，向量模型：{report['config']['embedding_model']}",
        f"- 召回请求块数：{report['config']['top_k']}（下列 TopN 按前 N 个检索块计算）",
        "- 本报告是固定题集的检索/关键词/引用结构回归；独立 reasoning 评测用于核对多跳、计算和时间推断。",
        "",
        "## 评分边界",
        "",
        *[f"- {limitation}" for limitation in report["scoring_limitations"]],
        "",
        "## 一、总体指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 前5块含任一期望文档（旧 Recall@5） | {metrics['any_expected_doc_hit@5']:.2%} |",
        f"| 前10块含任一期望文档（旧 Recall@10） | {metrics['any_expected_doc_hit@10']:.2%} |",
        f"| 前10块包含全部期望文档 | {metrics['all_expected_docs_recalled@10']:.2%} |",
        f"| MRR | {metrics['mrr']:.4f} |",
        f"| nDCG@10 | {metrics['ndcg@10']:.4f} |",
        f"| 全部关键词命中率（旧 answer_accuracy） | {metrics['keyword_accuracy']:.2%} |",
        f"| 至少一个期望来源被有效标记引用 | {metrics['citation_coverage']:.2%} |",
        f"| 全部期望来源被有效标记引用 | {metrics['all_expected_citation_coverage']:.2%} |",
        f"| 引用编号结构合法率 | {metrics['valid_citation_marker_rate']:.2%} |",
        f"| 系统 grounded 标志率（非独立语义审查） | {metrics['grounded_rate']:.2%} |",
        f"| 拒答准确率 | {metrics['refusal_accuracy']:.2%} |",
        f"| 非拒答问题误拒率 | {metrics['false_refusal_rate']:.2%} |",
        f"| 平均延迟（毫秒） | {metrics['avg_latency_ms']} |",
        "",
        "## 二、分类指标",
        "",
        "| 类别 | 用例数 | 任一期望文档@5 | 任一期望文档@10 | MRR | 全关键词命中 | 至少一期望来源引用 | 拒答准确率 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for category, item in report["by_category"].items():
        if category == "refusal":
            lines.append(
                f"| {category} | {item['total_cases']} | — | — | — | — | — | {item['refusal_accuracy']:.2%} |"
            )
        else:
            lines.append(
                f"| {category} | {item['total_cases']} | {item['recall@5']:.2%} | {item['recall@10']:.2%} "
                f"| {item['mrr']:.4f} | {item['answer_accuracy']:.2%} | {item['citation_coverage']:.2%} "
                f"| {item['refusal_accuracy']:.2%} |"
            )

    lines.extend(["", "## 三、失败与拒答明细", ""])
    failures = [
        item
        for item in results
        if (not item["expect_refusal"] and (not item["answer_correct"] or not item["refused"] is False))
        or (item["expect_refusal"] and not item["refused"])
    ]
    if failures:
        lines.append("| 用例 | 类别 | 问题 | 命中文档 | 缺失关键词 | 是否拒答 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in failures:
            lines.append(
                f"| {item['id']} | {item['category']} | {item['question']} | "
                f"{', '.join(item['top5'][:2])} | {'、'.join(item['missing_keywords']) or '—'} | "
                f"{'是' if item['refused'] else '否'} |"
            )
    else:
        lines.append("无失败用例。")

    lines.extend(
        [
            "",
            "## 四、逐条结果",
            "",
            "| 用例 | 类别 | 问题 | Top1 文档 | 首个相关排名 | 关键词/拒答检查通过 | 拒答 | 耗时(ms) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in results:
        top1 = item["top5"][0] if item["top5"] else "—"
        correct = "是" if (item["refusal_correct"] if item["expect_refusal"] else item["answer_correct"]) else "否"
        lines.append(
            f"| {item['id']} | {item['category']} | {item['question']} | {top1} | "
            f"{item['first_relevant_rank'] or '—'} | {correct} | {'是' if item['refused'] else '否'} | {item['latency_ms']} |"
        )

    lines.extend(
        [
            "",
            "完整机器可读逐条结果见 `per_case.jsonl`，包含检索 Top10、实际答案/生成方法、有效与无效引用编号、全部期望来源完整度、缺失关键词与改写信息。",
            "",
            "## 五、结论与分析",
            "",
            "- 本次向量模型、实际答案生成方法和检索参数以上方配置及逐条结果为准；hash 向量仅表示词项/字符特征相似，不等于语义模型。",
            "- 全关键词命中可能与错误结论同时出现；必须结合实际回答与独立 reasoning 评测判断推理能力。",
            "- 全部期望文档召回与引用完整度揭示多来源缺失；合法引用编号仍不能证明每个事实被资料支持。",
            "- 原 gate 阈值保留用于固定题集回归，不能用通过 gate 替代语义审查或多跳推理验收。",
        ]
    )
    (config.REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="企业 RAG 离线评测")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--llm", action="store_true", help="允许调用 LLM；实际生成方法记录在逐条结果中")
    args = parser.parse_args()

    cases = load_cases(args.limit or None)
    manifest = load_manifest()
    known_ids = {record["doc_id"] for record in manifest}
    missing = sorted({doc_id for case in cases for doc_id in case.get("expected_doc_ids", []) if doc_id not in known_ids})
    if missing:
        raise SystemExit(f"测试集引用了不存在的文档：{missing}")

    results = [evaluate_case(case, use_llm=True if args.llm else False) for case in cases]
    metrics = aggregate(results)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_scope": "legacy_fixed_corpus_keyword_regression",
        "scoring_limitations": SCORING_LIMITATIONS,
        "corpus": index_stats(),
        "config": {
            "embedding_model": config.EMBEDDING_MODEL,
            "answer_method": dict(Counter(item["answer_method"] for item in results)),
            "llm_requested": args.llm,
            "top_k": max(10, config.TOP_K),
            "candidate_k": config.CANDIDATE_K,
            "rrf_k": config.RRF_K,
            "min_score": config.MIN_SCORE,
        },
        "metrics": metrics,
        "by_category": by_category(results),
    }

    gate_failures = []
    if args.gate:
        thresholds = {
            "recall@5": 0.85,
            "answer_accuracy": 0.75,
            "citation_coverage": 0.85,
            "refusal_accuracy": 0.85,
        }
        for key, threshold in thresholds.items():
            if metrics[key] < threshold:
                gate_failures.append(f"{key} {metrics[key]} < {threshold}")
        conflict = report["by_category"].get("conflict", {}).get("answer_accuracy", 0)
        if conflict < 0.85:
            gate_failures.append(f"conflict accuracy {conflict} < 0.85")
    report["gate"] = {"passed": not gate_failures, "failures": gate_failures, "thresholds": {k: v for k, v in [("recall@5", 0.85), ("answer_accuracy", 0.75), ("citation_coverage", 0.85), ("refusal_accuracy", 0.85)]}}

    write_report(report, results)
    print(json.dumps({"metrics": metrics, "gate": report["gate"], "report": str(config.REPORT_DIR / "report.md")}, ensure_ascii=False, indent=2))
    if args.gate and gate_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
