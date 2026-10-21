"""Redacted live SDK transport diagnostic; never prints secrets or raw headers."""
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env.local", override=False)
from app.core.config import get_settings
from openai import OpenAI
import httpx
import openai

s = get_settings()


def endpoint(value):
    try:
        u = urlsplit(value)
        return {"scheme": u.scheme, "host": u.hostname, "port": u.port, "path": u.path}
    except Exception:
        return {"set": bool(value)}


def safe(message):
    text = str(message)
    if s.openai_api_key:
        text = text.replace(s.openai_api_key, "[REDACTED]")
    text = re.sub(r"(sk-)[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", text)
    return text[:700]


report = {"endpoint": endpoint(s.openai_api_base), "model": s.openai_model, "key_present": bool(s.openai_api_key),
          "sdk": openai.__version__, "httpx": httpx.__version__,
          "proxy": {key: endpoint(os.getenv(key, "")) if key != "NO_PROXY" else {"set": bool(os.getenv(key))} for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]}, "attempts": []}
for mode in ["default", "trust_env_false", "original_thinking_setting"]:
    started = time.monotonic()
    kwargs = {"api_key": s.openai_api_key, "base_url": s.openai_api_base, "timeout": 12, "max_retries": 0}
    if mode == "trust_env_false":
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=12)
    try:
        client = OpenAI(**kwargs)
        payload = {"model": s.openai_model, "messages": [{"role": "user", "content": "只依据这两份资料回答并附来源编号。资料[1]：槐序公司今年是2026年。资料[2]：槐序公司于七十年前创立。问题：槐序公司成立于哪一年？"}], "temperature": 0.2}
        if mode != "original_thinking_setting":
            payload.update(max_tokens=512, extra_body={"enable_thinking": False})
        result = client.chat.completions.create(**payload)
        item = {"mode": mode, "succeeded": True, "answer": result.choices[0].message.content, "model": result.model}
    except Exception as exc:
        chain = []
        while exc and len(chain) < 6:
            chain.append({"type": type(exc).__name__, "message": safe(exc)})
            exc = exc.__cause__ or exc.__context__
        item = {"mode": mode, "succeeded": False, "exception_chain": chain}
    item["latency_ms"] = round((time.monotonic() - started) * 1000)
    report["attempts"].append(item)
    print(json.dumps(item, ensure_ascii=False), flush=True)
path = Path(__file__).with_name("transport_diagnostic.json")
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "attempts"}, ensure_ascii=False), flush=True)
