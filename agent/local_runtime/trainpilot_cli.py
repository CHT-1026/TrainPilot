"""本地 TrainPilot 接入：复用生产业务图，保留真实检索、业务服务及证据校验。

默认只检查依赖；--run 才进入交互。不会创建数据库、入库或生成语料。
"""
import argparse
import asyncio
from datetime import datetime, timezone, timedelta
from getpass import getpass
import hashlib
import json
import os
from pathlib import Path
import socket
from urllib.parse import urlparse
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def local_endpoint(value):
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("此入口用于本地权重评测，base-url 必须是回环地址")
    return value.rstrip("/")


def v4_prompt_builder(path):
    from agent.app.workflows.intent_prompt import build_intent_prompt
    template = Path(path).read_text(encoding="utf-8")

    def build(state, now):
        return build_intent_prompt(state, now, template=template)
    return build


def configure(args):
    from dotenv import load_dotenv
    if args.env_file.exists():
        load_dotenv(args.env_file, override=True)
    # 意图、问答与RAG复杂度路由均指向同一份本地权重，不回退其他模型API。
    os.environ.update(OPENAI_API_BASE=local_endpoint(args.base_url), OPENAI_API_KEY="local-evaluation",
                      OPENAI_MODEL=args.model, OPENAI_ENABLE_THINKING="",
                      RAG_LLM_MODEL=args.model, RAG_MODEL_SIMPLE=args.model, RAG_MODEL_COMPLEX=args.model,
                      RAG_USE_LLM="true", LANGSMITH_TRACING="false")
    from agent.app.core.config import get_settings
    get_settings.cache_clear()


def reachable(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def inspect_setup(args):
    from agent.app.core.config import get_settings
    from agent.enterprise_rag import config
    from agent.app.workflows.chat_agent import TASK_TO_NODE
    settings = get_settings()
    redis = urlparse(settings.redis_url)
    api = urlparse(args.base_url)
    sources = ["app/workflows/chat_agent.py", "app/workflows/chat_nodes/recognition.py",
               "app/workflows/chat_nodes/policy_qa.py", "enterprise_rag/prompts.py",
               "enterprise_rag/answer.py", "app/services/leave_service.py"]
    from agent.app.workflows.intent_prompt import TEMPLATE_PATH
    prompt = args.intent_prompt_file or TEMPLATE_PATH
    return {
        "model": args.model, "base_url": args.base_url,
        "intent_prompt": str(prompt.resolve()) if prompt else "remote_default",
        "intent_prompt_exists": prompt.exists() if prompt else True,
        "intent_prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest() if prompt and prompt.exists() else None,
        "enterprise_answer_schema": {"refused": "bool", "claims": [{"text": "str", "sources": ["int"]}]},
        "model_port_reachable": reachable(api.hostname, api.port or (443 if api.scheme == "https" else 80)),
        "mysql_port_reachable": reachable(settings.mysql_host, settings.mysql_port),
        "redis_port_reachable": reachable(redis.hostname, redis.port or 6379),
        "enterprise_corpus": str(config.CORPUS_DIR), "enterprise_corpus_exists": config.CORPUS_DIR.exists(),
        "manifest_exists": config.MANIFEST_PATH.exists(), "oa_corpus_exists": (ROOT / "data/oa_docs").exists(),
        "embedding_provider": config.EMBEDDING_PROVIDER, "embedding_model": config.EMBEDDING_MODEL,
        "nodes": TASK_TO_NODE,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "scope": "仅检查配置、文件和端口；不代表模型/数据库可用或业务测评通过。不自动导入旧语料或配置。",
    }


def append_trace(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


async def run_turn(graph, thread_id, message, user, emit=print):
    from agent.app.routes.chat import build_chat_initial_state
    state, config = await build_chat_initial_state(graph, thread_id, message, user)
    async for update in graph.astream(state, config, stream_mode="updates"):
        for node, fields in update.items():
            emit({"event": "node", "node": node, "fields": fields or {}})
    return (await graph.aget_state(config)).values


async def interact(args, setup):
    import httpx
    import openai
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from agent.app.domain.user.models import UserContext
    from agent.app.services.user_service import verify_login
    from agent.app.workflows.chat_agent import build_chat_graph

    user_data = verify_login(args.username, getpass("测试库账号密码（不记录）："))
    if user_data is None:
        raise ValueError("测试库账号或密码不正确")
    user = UserContext.from_dict(user_data)
    session = "trainpilot_" + uuid.uuid4().hex[:12]

    def record(event):
        append_trace(args.trace, {"session": session, **event})

    def capture(response):
        # 本地评测使用非流式完整响应；日志不包含API密钥或请求头。
        response.read()
        record({"event": "model_http", "request": json.loads(response.request.content),
                "status": response.status_code, "raw_response": response.text})

    async def capture_async(response):
        await response.aread()
        record({"event": "model_http", "request": json.loads(response.request.content),
                "status": response.status_code, "raw_response": response.text})

    with httpx.Client(event_hooks={"response": [capture]}, trust_env=False) as client:
        async with httpx.AsyncClient(event_hooks={"response": [capture_async]}, trust_env=False) as async_client:
            llm = ChatOpenAI(model=args.model, api_key="local-evaluation", base_url=args.base_url,
                             temperature=0, timeout=180, max_retries=0, disable_streaming=True,
                             http_client=client, http_async_client=async_client)
            original_openai = openai.OpenAI

            def recorded_openai(*a, **kw):
                kw["http_client"] = client
                return original_openai(*a, **kw)

            builder = v4_prompt_builder(args.intent_prompt_file) if args.intent_prompt_file else None
            graph = build_chat_graph(llm, InMemorySaver(), recognition_prompt_builder=builder)
            record({"event": "setup", **setup, "actor": user.model_dump(),
                    "checkpointer": "memory", "business_services": "original_mysql_services",
                    "time": datetime.now(timezone(timedelta(hours=8))).isoformat()})
            # 仅观察RAG内部独立创建的OpenAI客户端；不替换模型回答、检索或业务结果。
            with patch("openai.OpenAI", new=recorded_openai):
                print("已接入原业务图。数据库为指定测试库；输入 /quit 退出，/new 清空当前会话。")
                while True:
                    message = input("你> ").strip()
                    if message == "/quit":
                        break
                    if message == "/new":
                        session = "trainpilot_" + uuid.uuid4().hex[:12]
                        continue
                    if not message:
                        continue
                    record({"event": "user", "message": message})
                    state = await run_turn(graph, session, message, user, record)
                    print("TrainPilot>", state.get("response", ""))
                    print("[路径]", state.get("current_task"), "[回答来源]", state.get("answer_method") or "业务节点/对话")
                    if state.get("last_result"):
                        print("[工具结果]", json.dumps(state["last_result"], ensure_ascii=False, default=str))
                    record({"event": "final", "state": state})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1", type=local_endpoint)
    parser.add_argument("--model", default="trainpilot-lora")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    parser.add_argument("--intent-prompt-file", type=Path, help="可选覆盖默认TrainPilot意图提示词；通常无需指定")
    parser.add_argument("--run", action="store_true", help="登录配置的测试库并进入真实业务图")
    parser.add_argument("--username")
    parser.add_argument("--trace", type=Path, default=ROOT / "local_runtime/logs/trainpilot_cli.jsonl")
    args = parser.parse_args()
    configure(args)
    setup = inspect_setup(args)
    print(json.dumps(setup, ensure_ascii=False, indent=2))
    if args.run:
        if not args.username:
            parser.error("--run 需要 --username；身份和权限从测试库验证，不能由模型指定")
        if not setup["intent_prompt_exists"]:
            parser.error("指定的意图提示词文件不存在")
        asyncio.run(interact(args, setup))


if __name__ == "__main__":
    main()
