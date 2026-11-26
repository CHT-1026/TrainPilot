"""One local entry point: WSL databases, pinned model, LangGraph and browser UI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

from agent.local_runtime.demo_config import AGENT_ROOT, PROJECT_ROOT, load_config, project_path

STATE = AGENT_ROOT / "local_runtime/processes.json"
LOGS = AGENT_ROOT / "local_runtime/logs"
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
MARKER = "trainpilot-demo-" + hashlib.sha256(str(PROJECT_ROOT).encode()).hexdigest()[:12]


def owns_process(command, role):
    if role == "backend":
        return ("agent.local_runtime.web_app:app" in command
                and "--env-file" in command
                and command[command.index("--env-file")+1:][:1] == [str(AGENT_ROOT / ".env.local")])
    if role == "model":
        config = load_config()
        return ("model.serving.tools_api" in command and config["adapter_sha256"] in command
                and str(project_path(config["adapter_dir"])) in command)
    return role == "wsl" and any(MARKER in value for value in command)


def status(port):
    try:
        with OPENER.open(f"http://127.0.0.1:{port}/local/status", timeout=2) as response:
            data = json.load(response)
        return data if data.get("project_root") == str(PROJECT_ROOT) else None
    except (OSError, ValueError):
        return None


def require_ready(port):
    with OPENER.open(f"http://127.0.0.1:{port}/api/ready", timeout=30) as response:
        checks = json.load(response)
    if checks.get("status") != "ready":
        raise RuntimeError("数据库或知识索引尚未就绪，请查看后端日志。")


def stop_owned():
    """Stop only recorded processes whose command line still identifies this project."""
    if not STATE.exists():
        print("本项目没有记录中的后台服务。")
        return
    import psutil
    data = json.loads(STATE.read_text(encoding="utf-8"))
    for role in ("backend", "model", "wsl"):
        pid = data.get(role + "_pid")
        if not pid:
            continue
        try:
            parent = psutil.Process(pid)
            processes = parent.children(recursive=True) + [parent]
            for process in processes:
                if owns_process(process.cmdline(), role):
                    process.terminate()
        except psutil.NoSuchProcess:
            pass
    STATE.unlink(missing_ok=True)
    print("已停止本项目后台进程；MySQL、Redis 数据及其他项目进程保留。")


def start_databases():
    config = load_config()
    log_path = LOGS / "wsl.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen([
            "wsl.exe", "-d", config["wsl_distribution"], "-u", "root", "--exec", "bash", "-lc",
            "systemctl start mysql redis-server && exec -a " + MARKER + " sleep infinity",
        ], stdout=log, stderr=log, creationflags=HIDDEN)
    for _ in range(45):
        if process.poll() is not None:
            raise RuntimeError("WSL 数据库启动失败，请查看 agent/local_runtime/logs/wsl.log。")
        try:
            for port in (3306, 6379):
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    pass
            return process
        except OSError:
            time.sleep(1)
    process.terminate()
    raise RuntimeError("MySQL/Redis 未就绪，请检查 Ubuntu-24.04 中的服务。")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()
    python = project_path(config["agent_python"])
    if not python.exists():
        raise RuntimeError("请先安装 .venv-agent 环境，依赖见 agent/requirements-lock.txt。")
    if Path(sys.executable).resolve() != python.resolve():
        return subprocess.call([str(python), "-u", "-m", "agent", *sys.argv[1:]], cwd=PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="TrainPilot 本地完整 Agent，固定 V7")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--port", type=int, default=config["web_port"])
    args = parser.parse_args()
    if args.stop:
        stop_owned()
        return 0
    if args.restart:
        stop_owned()
        time.sleep(2)
    existing = status(args.port)
    url = f"http://127.0.0.1:{args.port}"
    if existing:
        from agent.local_runtime.start_model import model_info
        if not model_info():
            raise RuntimeError("网页仍在运行，但固定 V7 不在线，请使用 --restart。")
        require_ready(args.port)
        print("TrainPilot 已运行：" + url)
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    from dotenv import load_dotenv
    load_dotenv(AGENT_ROOT / ".env.local", override=True)
    os.environ.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    LOGS.mkdir(parents=True, exist_ok=True)
    backend = model = keeper = None
    try:
        print("[1/4] 启动本机 MySQL / Redis……", flush=True)
        keeper = start_databases()
        print("[2/4] 准备独立演示数据库与现有企业资料……", flush=True)
        for module in ("agent.local_runtime.setup_database", "agent.sql.apply_migrations",
                       "agent.enterprise_rag.ensure_ingested"):
            subprocess.run([str(python), "-u", "-m", module], cwd=PROJECT_ROOT, check=True)
        print("[3/4] 校验并加载固定 V7……", flush=True)
        from agent.local_runtime.start_model import ensure_model
        model = ensure_model()
        from agent.local_runtime.check_model import check
        checked = check()
        if not checked.get("ok"):
            raise RuntimeError(checked["message"])
        print("[4/4] 启动 LangGraph 与本地页面……", flush=True)
        with (LOGS / "backend.log").open("ab") as log:
            backend = subprocess.Popen([str(python), "-u", "-m", "uvicorn",
                "agent.local_runtime.web_app:app", "--env-file", str(AGENT_ROOT / ".env.local"),
                "--host", "127.0.0.1", "--port", str(args.port)],
                cwd=PROJECT_ROOT, stdout=log, stderr=log, creationflags=HIDDEN)
        STATE.write_text(json.dumps({"backend_pid": backend.pid,
            "model_pid": model.pid if model else None, "wsl_pid": keeper.pid,
            "port": args.port, "url": url}, indent=2), encoding="utf-8")
        for _ in range(90):
            if backend.poll() is not None:
                raise RuntimeError("后端启动失败，请查看 agent/local_runtime/logs/backend.log。")
            if status(args.port):
                break
            time.sleep(1)
        else:
            raise RuntimeError("后端启动超时，请查看后端日志。")
        require_ready(args.port)
        print("\nTrainPilot 已启动：" + url, flush=True)
        print("模型：固定 V7；知识工具：search / read / list；本机独立数据库。", flush=True)
        print("演示账号：admin / employee_zhao / manager_zhang，密码均为 admin123。", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        if args.background:
            return 0
        print("保持窗口运行，Ctrl+C 停止；后台服务可双击“停止项目.cmd”。", flush=True)
        backend.wait()
    except KeyboardInterrupt:
        pass
    except BaseException:
        for process in (backend, model, keeper):
            if process is not None and process.poll() is None:
                process.terminate()
        raise
    stop_owned()
    return 0
