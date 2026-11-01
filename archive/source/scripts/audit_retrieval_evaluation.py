"""Read-only audit of historical evaluation claims and source provenance."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.retrieval_corpus import read_jsonl, write_json
from app.evaluation.retrieval_grading import equal_value

OLD = ROOT / "research/retrieval_evaluation_20261101"
OUT = ROOT / "research/retrieval_repair_20261101"


def main():
    source = OUT / "before/scripts/retrieval_eval.py"
    ns = {"__file__": str(ROOT / "scripts/retrieval_eval.py"), "__name__": "historical_source_audit"}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), ns)
    samples = [
        ("unit", "预算为52亿元。", "预算为52万元。", {"yuan_yi": 52}, {"yuan_yi": 0.0052}),
        ("negation", "允许使用该设备。", "不允许使用该设备。", {"allowed": True}, {"allowed": False}),
        ("relation_swap", "甲部门有30人，乙部门有90人。", "甲部门有90人，乙部门有30人；30，90。", {"甲": 30, "乙": 90}, {"甲": 90, "乙": 30}),
        ("precision", "重复精度为0.02毫米，采用标准控制系统，支持视觉配套与传送带跟踪。", "重复精度为0.09毫米，采用标准控制系统，支持视觉配套与传送带跟踪。", {"precision_mm": 0.02}, {"precision_mm": 0.09}),
    ]
    counterexamples = []
    for name, expected, answer, new_expected, new_actual in samples:
        trace = {"answer": {"answer": answer, "citations": ["a"]},
                 "tool_calls_detail": [{"result": {"doc_id": "a", "text": "无关内容"}}]}
        result = ns["grade"](trace, {"answer": expected, "expected_doc_ids": ["a", "b"]})
        counterexamples.append({"case": name, "gold": expected, "actual_answer": answer,
            "old_verdict": result, "typed_repair_accepts": equal_value(new_expected, new_actual)})
    write_json(OUT / "counterexamples.json", counterexamples)
    qs, golds = ns["load_bank"]()
    core = json.loads((OLD / "bank/core400.json").read_text(encoding="utf-8"))
    extra = json.loads((OLD / "bank/q4_extra400.json").read_text(encoding="utf-8"))
    dim_rows = read_jsonl(ROOT / "eval_bench/versions/20261030_v2/question_bank_dimensioned.jsonl")
    docs = {d["doc_id"]: d for d in read_jsonl(OLD / "workspace/docs.jsonl")}
    located_text = {}
    for segment in read_jsonl(OUT / "public/segments.jsonl"):
        located_text.setdefault(segment["doc_id"], []).append(segment["text"])
    quote_issues, source_issues = [], []
    families = Counter()
    for qid, gold in golds.items():
        families[hashlib.sha256((str(gold["answer"]) + json.dumps(gold.get("expected_doc_ids", []))).encode()).hexdigest()[:12]] += 1
        for evidence in gold.get("evidence", []):
            doc = docs.get(evidence.get("doc_id"))
            if doc is None:
                if qs[qid].get("evaluation_track") != "context_reasoning":
                    source_issues.append({"qid": qid, "doc": evidence.get("doc_id"), "issue": "missing_public_doc"})
                continue
            actual_source = ROOT / "enterprise_rag/corpus/docs" / doc["rel"]
            if evidence.get("source_sha256") and hashlib.sha256(actual_source.read_bytes()).hexdigest() != evidence["source_sha256"]:
                source_issues.append({"qid": qid, "doc": doc["doc_id"], "issue": "source_sha_mismatch"})
            # Compare against source-located extraction: XLSX quotes carry cell coordinates.
            compact_source = re.sub(r"\s+", "", "\n".join(located_text[doc["doc_id"]]))
            segs = evidence.get("segments", []) or ([{"text": evidence["text"]}] if evidence.get("text") else [])
            for segment in segs:
                if re.sub(r"\s+", "", segment["text"]) not in compact_source:
                    quote_issues.append({"qid": qid, "doc": doc["doc_id"], "locator": segment.get("locator"), "issue": "quote_requires_manual_source_review"})
    audit = {"unique_questions": len(qs), "unique_golds": len(golds), "dimensioned_lines": len(dim_rows),
             "dimensioned_duplicate_qids": len(dim_rows) - len({r['qid'] for r in dim_rows}),
             "core_count": len(core["qids"]), "q4_additional_count": len(extra["qids"]),
             "q4_total_prepared": len(set(core["qids"]) | set(extra["qids"])),
             "q4_additional_shortfall": 400 - len(extra["qids"]),
             "embedded_context_questions": sum(q.get("evaluation_track") == "context_reasoning" for q in qs.values()),
             "text_embedded_dialogue_questions": sum(qid.startswith('M15') for qid in qs),
             "source_characters": {"max": max(len(d["text"]) for d in docs.values()),
                                   "median": sorted(len(d["text"]) for d in docs.values())[len(docs)//2]},
             "same_answer_and_sources_clusters": sum(count > 1 for count in families.values()),
             "source_issues": source_issues, "quote_issues": quote_issues,
             "semantic_answer_accuracy": "not_established_by_quote_matching_or_lexical_overlap",
             "core_embedded_context_count": sum(qs[q].get('evaluation_track') == 'context_reasoning' for q in core['qids'])}
    write_json(OUT / "bank_audit.json", audit)
    # Explicit historical run selection; do not discard the all-error D run.
    gold_out = OUT / "private/historical_gold.jsonl"
    gold_out.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golds.values()), encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k not in {'source_issues','quote_issues'}}, ensure_ascii=False))
    print('source_issues', len(source_issues), 'quote_issues', len(quote_issues))


if __name__ == '__main__':
    main()
