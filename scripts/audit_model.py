"""Real-model generation-stage comparison; no mocked responses or local operators."""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env.local", override=False)
from enterprise_rag.answer import _llm_answer
from enterprise_rag import config


def main():
    ids = {"R01_four_departments", "R03_foundation_cn", "R06_conflicting_year_anchor", "R08_sum_money_units"}
    cases = [json.loads(line) for line in (ROOT / "audit_artifacts/reasoning/cases.jsonl").read_text(encoding="utf-8").splitlines()]
    results = []
    for case in cases:
        if case["id"] not in ids:
            continue
        chunks = [{"doc_id": doc["id"], "chunk_id": doc["id"] + "#0", "title": doc.get("title", ""),
                   "text": doc["text"], "score": 1.0} for doc in case["documents"]]
        started = time.monotonic()
        try:
            answer, indices, _ = _llm_answer(case["question"], chunks)
            refusal = any(s in answer for s in ("未覆盖", "资料不足", "无法确定", "不足以确定", "无法推算", "无法得出", "不能确定"))
            correct = refusal if case.get("expect_refusal") else all(re.search(p, answer) for p in case["answer_patterns"])
            cited = {chunks[i-1]["doc_id"] for i in indices}
            complete = set(case["expected_docs"]) <= cited
            results.append({"id": case["id"], "answer": answer, "method": "real_llm_generation_stage",
                            "expected": case["expected"], "cited_docs": sorted(cited), "passed": bool(correct and (complete or case.get("expect_refusal"))),
                            "seconds": round(time.monotonic()-started, 3)})
        except Exception as exc:
            results.append({"id": case["id"], "passed": False, "error": type(exc).__name__,
                            "answer": getattr(exc, "answer", ""),
                            "validation_error": str(exc) if hasattr(exc, "answer") else "",
                            "method": "real_llm_request_failed", "seconds": round(time.monotonic()-started, 3)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    out = ROOT / "audit_artifacts/reasoning/generation_model_regression.json"
    out.write_text(json.dumps({"scope": "Generation stage only; retrieval tested separately", "model": config.LLM_MODEL,
        "enable_thinking": False, "max_tokens": 1400, "max_retries": config.LLM_MAX_RETRIES,
        "passed": sum(x["passed"] for x in results), "total": len(results), "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
