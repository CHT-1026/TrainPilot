"""同一 V5 的框架修复回归：冻结旧题，真实本地推理，隔离存储。"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil

from agent.local_runtime.business_eval_support import read, save, sha
from agent.local_runtime.trainpilot_cli import configure


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("chunks.json", "cases.json"):
        shutil.copy2(args.baseline / name, args.output / name)
    for name in ("graph", "releases"):
        shutil.copytree(args.baseline / name, args.output / name)
    old = read(args.baseline / "rag_results.json")
    # 权重和给定证据没变；18条原始生成保留作诊断，检索及业务图重新跑。
    save(args.output / "rag_results.json", {"oracle": old["oracle"], "retrieval": [], "graph": []})
    root = Path(__file__).resolve().parents[1]
    sources = ("enterprise_rag/facts.py", "enterprise_rag/answer.py", "enterprise_rag/retrieval.py",
               "app/workflows/dialog_state.py", "app/workflows/chat_nodes/routing.py",
               "app/workflows/chat_nodes/policy_qa.py")
    save(args.output / "plan.json", {
        "baseline": str(args.baseline.resolve()), "cases_sha256": sha(args.output / "cases.json"),
        "chunks_sha256": sha(args.output / "chunks.json"), "model": "V5 epoch 2 unchanged",
        "rerun": {"rag_graph_turns": 6, "tool_turns": 25, "tool_trajectories": 15,
                  "retrieval_diagnostics": len(old["retrieval"])},
        "reused_only": {"oracle_generations": len(old["oracle"])},
        "source_sha256": {name: sha(root / name) for name in sources},
        "success_rule": "事实及对应引用正确；抽取回退不计模型成功；工具字段与落库结果逐项比对",
    })


async def run(args):
    from agent.local_runtime.evaluate_trainpilot_business import rag, tools
    assert sha(args.output / "cases.json") == read(args.output / "plan.json")["cases_sha256"]
    await rag(args)
    await tools(args)
    print("REPAIR_WORKFLOWS_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--intent-prompt-file", type=Path, required=True)
    args = parser.parse_args()
    args.model = "trainpilot-lora"
    args.env_file = Path(__file__).resolve().parents[1] / ".env.eval-not-used"
    configure(args)
    os.environ.update(RAG_EMBEDDING_PROVIDER="hash", RAG_EMBEDDING_DIM="384", RAG_REQUEST_TIMEOUT="180")
    prepare(args) if args.mode == "prepare" else asyncio.run(run(args))


if __name__ == "__main__":
    main()
