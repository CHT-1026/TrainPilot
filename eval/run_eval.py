"""制度问答离线评估：Recall@5、MRR、关键词覆盖，可选抽样回答评估。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe eval\\run_eval.py
    .venv312\\Scripts\\python.exe eval\\run_eval.py --answers 5
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from app.capabilities.retrieval.search import retrieve  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.domain.user.models import UserContext  # noqa: E402

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
REPORT_DIR = ROOT / "eval" / "reports"


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def load_golden() -> list[dict]:
    items = []
    with GOLDEN.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate_retrieval(items: list[dict], user: UserContext) -> dict:
    recall_hits = 0
    top1_hits = 0
    mrr_total = 0.0
    keyword_found = 0
    keyword_total = 0
    failures = []

    for item in items:
        chunks = retrieve(item["question"], user, top_k=5)
        titles = [normalize(chunk.get("title", "")) for chunk in chunks]
        merged = normalize("\n".join(chunk.get("text", "") for chunk in chunks))
        expected = normalize(item["expected_doc"])

        doc_hit = any(expected in title for title in titles) or expected in merged
        if doc_hit:
            recall_hits += 1
        if titles and expected in titles[0]:
            top1_hits += 1

        rank = next(
            (index for index, title in enumerate(titles, 1) if expected in title),
            None,
        )
        if rank:
            mrr_total += 1 / rank

        keywords = item.get("keywords", [])
        found = sum(1 for keyword in keywords if normalize(keyword) in merged)
        keyword_found += found
        keyword_total += len(keywords)

        if not doc_hit or found < len(keywords):
            failures.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_doc": item["expected_doc"],
                    "doc_hit": doc_hit,
                    "found_keywords": found,
                    "total_keywords": len(keywords),
                    "top_docs": [chunk.get("title") for chunk in chunks[:3]],
                }
            )

    total = len(items) or 1
    return {
        "total": len(items),
        "recall_at_5": round(recall_hits / total, 4),
        "top1_doc_hit": round(top1_hits / total, 4),
        "mrr": round(mrr_total / total, 4),
        "keyword_coverage": round(keyword_found / (keyword_total or 1), 4),
        "failures": failures,
    }


async def evaluate_answers(items: list[dict], user: UserContext, limit: int) -> dict:
    from langchain_openai import ChatOpenAI

    from app.workflows.chat_nodes.policy_qa import create_policy_nodes

    settings = get_settings()
    extra_body = (
        {"enable_thinking": False}
        if settings.openai_enable_thinking.strip().lower() == "false"
        else None
    )
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
        temperature=0.2,
        timeout=settings.llm_timeout_seconds,
        extra_body=extra_body,
    )
    nodes = create_policy_nodes(llm)
    results = []
    for item in items[:limit]:
        state = {
            "user_message": item["question"],
            "conversation_context": {
                "user_id": user.user_id,
                "user_name": user.name,
                "user_role": user.role,
                "user_department": user.department,
            },
        }
        state.update(nodes["retrieve"](state))
        output = await nodes["answer"](state)
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "grounded": bool(output.get("grounded")),
                "citations": len(output.get("citations", [])),
                "answer_preview": (output.get("task_result") or "")[:160],
            }
        )
    covered = sum(1 for result in results if result["grounded"] and result["citations"] > 0)
    return {
        "total": len(results),
        "citation_coverage": round(covered / (len(results) or 1), 4),
        "items": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=int, default=0, help="抽样回答评估的条数")
    parser.add_argument("--gate", action="store_true", help="按验收门槛返回非零退出码")
    parser.add_argument("--min-recall", type=float, default=0.85)
    parser.add_argument("--min-citation", type=float, default=0.90)
    args = parser.parse_args()

    items = load_golden()
    user = UserContext(user_id="admin", name="管理员", role="admin", department="管理部")
    report = {
        "golden_total": len(items),
        "retrieval": evaluate_retrieval(items, user),
    }
    if args.answers:
        report["answers"] = asyncio.run(evaluate_answers(items, user, args.answers))

    gate_failures = []
    if report["retrieval"]["recall_at_5"] < args.min_recall:
        gate_failures.append(
            f"recall_at_5 {report['retrieval']['recall_at_5']} < {args.min_recall}"
        )
    if args.answers and report["answers"]["citation_coverage"] < args.min_citation:
        gate_failures.append(
            f"citation_coverage {report['answers']['citation_coverage']} < {args.min_citation}"
        )
    report["gate"] = {"passed": not gate_failures, "failures": gate_failures}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.gate and gate_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
