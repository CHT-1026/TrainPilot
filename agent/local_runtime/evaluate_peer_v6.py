"""冻结同学题库，使用当前 RAG 与 V6；只在 runs 中写评测产物。"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT.parent / "runs/peer_rag_v6_20261101"


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def report(out, questions, answers, dimensions, core):
    data = rows(out / "results.jsonl") if (out / "results.jsonl").exists() else []
    from agent.enterprise_rag.eval.run_eval import normalize
    for row in data:
        gold = answers[row["qid"]]
        expected = set(gold["expected_doc_ids"])
        result = row.get("result", {})
        retrieved = result.get("retrieval", {}).get("chunks", [])
        docs = {r["doc_id"] for r in retrieved[:10]}
        text = normalize(result.get("answer", ""))
        keywords = gold.get("answer_keywords") or []
        row["grade"] = {
            "any_doc_hit10": bool(expected & docs) if expected else None,
            "all_docs_hit10": expected <= docs if expected else None,
            "keywords_all": all(normalize(k) in text for k in keywords) if keywords else None,
            "keyword_hits": [k for k in keywords if normalize(k) in text],
            "missing_keywords": [k for k in keywords if normalize(k) not in text],
            "answerability": dimensions[row["qid"]].get("answerability"),
            "refused": result.get("refused"),
            "api_error": any(c["status"] != 200 for c in row.get("calls", [])),
            "truncated": any(c.get("finish_reason") == "length" for c in row.get("calls", [])),
        }
    def group(items):
        scores = [r["grade"] for r in items]
        metrics = {}
        for key in ("any_doc_hit10", "all_docs_hit10", "keywords_all", "refused", "api_error", "truncated"):
            eligible = [g[key] for g in scores if g[key] is not None]
            metrics[key] = {"count": sum(eligible), "total": len(eligible),
                            "rate": sum(eligible) / len(eligible) if eligible else None}
        return {"cases": len(items), "exceptions": sum("error" in r for r in items), **metrics}
    summary = {"completed": len(data), "total": len(questions), "complete": len(data) == len(questions),
               "all": group(data), "core400": group([r for r in data if r["qid"] in core]),
               "dimensions": {d: group([r for r in data if dimensions[r["qid"]]["dimension"] == d])
                              for d in sorted({d["dimension"] for d in dimensions.values()})},
               "methods": dict(Counter(r.get("result", {}).get("method", "error") for r in data))}
    save(out / "summary.json", summary)
    with (out / "graded.jsonl").open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    lines = ["# V6＋当前 RAG：同学题库评测（2026-11-01）", "",
             f"进度：{len(data)}/{len(questions)}。模型为冻结 V6；未训练，未使用外部模型。",
             "语料为同学题库配套快照；运行当前本地混合检索、图谱及证据校验，索引存储隔离，不修改生产数据库。",
             "题目直接进入企业 RAG；不包含意图路由、网页交互和业务工具执行成功率。",
             "关键词全命中仅是字面覆盖率，不是语义准确率；引用与事实检查也不能替代人工核验。", "",
             "| 范围 | 题数 | 任一期望文档命中@10 | 全部期望文档命中@10 | 关键词全命中 | 拒答 | 异常 |", "|---|---:|---:|---:|---:|---:|---:|"]
    def cell(m):
        return f"{m['count']}/{m['total']}" + (f"（{m['rate']:.1%}）" if m["rate"] is not None else "")
    for label, g in [("全部已完成", summary["all"]), ("原共同400题", summary["core400"]), *summary["dimensions"].items()]:
        lines.append(f"| {label} | {g['cases']} | {cell(g['any_doc_hit10'])} | {cell(g['all_docs_hit10'])} | {cell(g['keywords_all'])} | {cell(g['refused'])} | {g['exceptions']} |")
    lines += ["", "## 回答样例", ""]
    for row in data[:6]:
        lines += [f"### {row['qid']}", questions[row["qid"]]["question"], "", row.get("result", {}).get("answer", row.get("error", "")), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    out = args.output
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env.local", override=True)
    os.environ.update(OPENAI_API_BASE="http://127.0.0.1:8011/v1", OPENAI_API_KEY="local-evaluation",
                      OPENAI_MODEL="trainpilot-lora", OPENAI_ENABLE_THINKING="false",
                      RAG_LLM_MODEL="trainpilot-lora", RAG_MODEL_SIMPLE="trainpilot-lora",
                      RAG_MODEL_COMPLEX="trainpilot-lora", RAG_USE_LLM="true", LANGSMITH_TRACING="false")
    from agent.local_runtime.start_model import model_info
    info = model_info()
    if not info:
        raise RuntimeError("请先启动已冻结的本机 V6 模型接口")
    from agent.local_runtime import business_eval_support as support
    from agent.app.domain.user.models import UserContext
    from agent.enterprise_rag.pipeline import query
    import httpx
    import openai
    qs = rows(out / "snapshot/archive/questions/question_bank.jsonl")
    questions = {q["qid"]: q for q in qs}
    answers = {r["qid"]: r for r in rows(out / "snapshot/archive/answers/answer_bank.jsonl")}
    dimensions = {r["qid"]: r for r in rows(out / "snapshot/archive/questions/question_bank_dimensioned.jsonl")}
    core = set(json.loads((out / "snapshot/archive/questions/frozen/core400.json").read_text(encoding="utf-8"))["qids"])
    signature = {"source": json.loads((out / "source_manifest.json").read_text(encoding="utf-8")), "model": info,
                 "code": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for folder in (ROOT / "enterprise_rag", ROOT / "app/capabilities/retrieval")
                          for f in folder.rglob("*.py")},
                 "evaluator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "settings": {k: v for k, v in os.environ.items() if k.startswith("RAG_") and "KEY" not in k},
                 "max_completion_tokens": 512, "temperature": 0, "actor": "admin"}
    sig = out / "run_signature.json"
    if sig.exists():
        assert json.loads(sig.read_text(encoding="utf-8")) == signature, "评测输入、代码或模型已变化，不能续写同一结果目录"
    else:
        save(sig, signature)
    if not (out / "chunks.json").exists():
        with patch.object(support, "ROOT", out / "snapshot"):
            support.prepare_chunks(out)
    done = rows(out / "results.jsonl") if (out / "results.jsonl").exists() else []
    assert [r["qid"] for r in done] == [q["qid"] for q in qs[:len(done)]]
    calls = []
    def capture(response):
        response.read()
        item = {"status": response.status_code, "request": json.loads(response.request.content)}
        try:
            body = response.json()
            choice = (body.get("choices") or [{}])[0]
            item.update(text=choice.get("message", {}).get("content"), finish_reason=choice.get("finish_reason"), usage=body.get("usage"))
        except ValueError:
            item["response"] = response.text
        calls.append(item)
    original = openai.OpenAI
    with httpx.Client(trust_env=False, event_hooks={"response": [capture]}) as client, \
         patch("openai.OpenAI", side_effect=lambda *a, **kw: original(*a, **{**kw, "http_client": client})), \
         support.file_indexes(out, build=not (out / "index_summary.json").exists()):
        actor = UserContext(user_id="peer-evaluation", name="隔离评测", role="admin")
        for i, q in enumerate(qs[len(done):], len(done) + 1):
            if args.limit and i > args.limit:
                break
            calls.clear()
            start = time.monotonic()
            row = {"qid": q["qid"]}
            try:
                row["result"] = query(q["question"], user=actor, use_llm=True)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row.update(calls=list(calls), seconds=time.monotonic() - start)
            with (out / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(f"{i}/{len(qs)} {q['qid']} method={row.get('result',{}).get('method')} seconds={row['seconds']:.1f} calls={len(calls)}", flush=True)
            if i % 25 == 0:
                report(out, questions, answers, dimensions, core)
    summary = report(out, questions, answers, dimensions, core)
    print("PEER_EVAL_DONE", summary["completed"], summary["total"], summary["complete"], flush=True)


if __name__ == "__main__":
    main()
