"""A localhost-only operation page mounted on the original FastAPI application."""
import os
import pathlib
import sys

from dotenv import dotenv_values
from fastapi.responses import HTMLResponse

from local_runtime.check_model import REPORT, fingerprint
from main import app

ROOT = pathlib.Path(__file__).resolve().parents[1]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def local_operation_page():
    return HTMLResponse((ROOT / "local_runtime" / "operation.html").read_text(encoding="utf-8"))


@app.get("/local/status", include_in_schema=False)
def local_status():
    import json
    values = dotenv_values(ROOT / ".env.local")
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    current = report.get("fingerprint") == fingerprint(values)
    loaded = {name: os.getenv(name, "") for name in ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_MODEL", "OPENAI_ENABLE_THINKING")}
    restart = fingerprint(loaded) != fingerprint(values)
    return {"service": "lantu-agent-platform", "python": sys.version.split()[0],
            "model": os.getenv("OPENAI_MODEL", ""), "model_check_current": current,
            "model_ready": bool(current and report.get("ok") and not restart),
            "restart_required": restart,
            "model_message": "配置已修改，请重启本地服务。" if restart else report.get("message", "模型尚未验证。"),
            "checked_at": report.get("checked_at", "")}
