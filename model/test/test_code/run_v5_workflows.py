"""等待本机V5评测API，然后复用原Agent真实检索和隔离业务测试；不连接外部服务。"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
    args = parser.parse_args()
    if not args.base_url.startswith("http://127.0.0.1:"):
        raise ValueError("只允许本机评测API")
    args.model = "trainpilot-lora"
    args.intent_prompt_file = Path(__file__).resolve().parents[3] / "model/dataset/datasets/trainpilot_agent/intent_prompt_v4.txt"
    args.env_file = args.agent_project / ".env.eval-not-used"
    sys.path.insert(0, str(args.agent_project.resolve()))
    from local_runtime.trainpilot_cli import configure
    configure(args)
    os.environ.update(RAG_EMBEDDING_PROVIDER="hash", RAG_EMBEDDING_DIM="384", RAG_REQUEST_TIMEOUT="180")
    from local_runtime.evaluate_trainpilot_business import rag, tools
    started = time.monotonic()
    ready = False
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() - started < 7200:
        try:
            with opener.open(args.base_url + "/models", timeout=3) as response:
                ready = response.status == 200
        except OSError:
            ready = False
        if ready:
            break
        time.sleep(3)
    if not ready:
        raise TimeoutError("本机模型评测API尚未就绪")
    print("Local V5 API ready; run frozen RAG and business workflows", flush=True)
    asyncio.run(rag(args))
    asyncio.run(tools(args))
    print("WORKFLOWS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
