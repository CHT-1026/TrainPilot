"""独立只读工具体验入口；沿用远程语料工具，记录模型原始输出和真实执行结果。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "enterprise_rag/corpus/docs")
    parser.add_argument("--workspace", type=Path, default=ROOT / "local_runtime/logs/tool_corpus")
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model", default="trainpilot-lora")
    parser.add_argument("--question")
    parser.add_argument("--output", type=Path, help="仅显式指定时保存逐轮记录")
    parser.add_argument("--max-rounds", type=int, default=8)
    args = parser.parse_args()
    import httpx
    from agent.app.capabilities.retrieval.corpus_tools import CorpusTools, build_snapshot
    from agent.app.capabilities.retrieval.tool_chat import LocalCompletion, run_tool_chat
    with httpx.Client(trust_env=False, timeout=5) as client:
        info = client.get(args.base_url.removesuffix("/v1") + "/local/model-info")
        info.raise_for_status()
        metadata = info.json()
    from agent.local_runtime.start_model import EXPECTED_V6
    if metadata.get("adapter_sha256") != EXPECTED_V6 or not metadata.get("tool_calls"):
        raise RuntimeError("需要冻结V6的新版工具API；未改用其他模型。")
    if not args.workspace.exists():
        print("建立隔离语料快照：", build_snapshot(args.corpus, args.workspace))
    backend = CorpusTools(args.workspace)
    complete = LocalCompletion(args.base_url, args.model)
    print("TrainPilot V6；只读工具 search/read/list；每次独立问答；/quit退出。")
    while True:
        question = args.question if args.question is not None else input("你> ").strip()
        if not question or question == "/quit":
            break
        outcome = run_tool_chat(question, backend, complete, max_rounds=args.max_rounds, timeout=300)
        for event in outcome["trace"]:
            reply = event.get("response", {})
            print("模型原始输出>", reply.get("local_trace", {}).get("raw_output") or
                  (reply.get("choices") or [{}])[0].get("message", {}))
            for tool in event.get("tools", []):
                print("执行工具>", tool["name"], json.dumps(tool["arguments"], ensure_ascii=False))
                print("工具结果>", json.dumps(tool["result"], ensure_ascii=False)[:1800])
        print("状态>", outcome["status"], "调用数>", outcome["tool_calls"])
        if outcome["final"]:
            print("回答>", outcome["final"]["answer"])
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"question": question, "model_info": metadata, **outcome}, ensure_ascii=False) + "\n")
        if args.question is not None:
            break


if __name__ == "__main__":
    main()
