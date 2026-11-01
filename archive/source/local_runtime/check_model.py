"""Check the locally configured model without printing its API key."""
import argparse
import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone

from dotenv import dotenv_values

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.config import thinking_extra_body
REPORT = ROOT / "local_runtime" / "model_check.json"


def fingerprint(values):
    text = "\n".join(values.get(name, "") or "" for name in
                     ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_MODEL", "OPENAI_ENABLE_THINKING"))
    return hashlib.sha256(text.encode()).hexdigest()


def check(if_changed=False):
    values = dotenv_values(ROOT / ".env.local")
    identity = fingerprint(values)
    if if_changed and REPORT.exists():
        old = json.loads(REPORT.read_text(encoding="utf-8"))
        if old.get("fingerprint") == identity:
            return old
    result = {"fingerprint": identity, "ok": False, "model": values.get("OPENAI_MODEL", ""),
              "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}
    if not values.get("OPENAI_API_KEY"):
        result.update(status=0, error_code="missing_key", message="请在 .env.local 填写有效的模型密钥。")
    else:
        body = {"model": values["OPENAI_MODEL"], "max_tokens": 128, "temperature": 0,
                "messages": [{"role": "user", "content": "请只回复OK。"}]}
        extra = thinking_extra_body(values["OPENAI_MODEL"], value=values.get("OPENAI_ENABLE_THINKING") or "")
        if extra:
            body.update(extra)
        request = urllib.request.Request(
            values["OPENAI_API_BASE"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
            "Authorization": "Bearer " + values["OPENAI_API_KEY"]})
        try:
            no_proxy = values.get("NO_PROXY")
            bypass = bool(no_proxy and urllib.request.proxy_bypass_environment(
                urllib.parse.urlsplit(values["OPENAI_API_BASE"]).hostname or "", {"no": no_proxy}))
            open_url = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                        if bypass else urllib.request.urlopen)
            with open_url(request, timeout=30) as response:
                data = json.load(response)
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            result.update(ok=bool(text), status=200, message="模型请求成功。" if text else "接口返回了空回答。")
        except urllib.error.HTTPError as exc:
            try:
                code = json.loads(exc.read()).get("error", {}).get("code", "provider_error")
            except Exception:
                code = "provider_error"
            message = "模型密钥无效；请修改 .env.local 后重启服务。" if exc.code == 401 else "模型服务拒绝请求，请核对模型名称、地域、权限和额度。"
            result.update(status=exc.code, error_code=code, message=message)
        except Exception as exc:
            result.update(status=0, error_code=type(exc).__name__, message="模型连接失败，请检查接口地址与网络。")
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-changed", action="store_true")
    args = parser.parse_args()
    result = check(args.if_changed)
    print(json.dumps({key: value for key, value in result.items() if key != "fingerprint"}, ensure_ascii=False))
    sys.exit(0 if result.get("ok") else 2)
