"""Run from PyCharm or double-click 启动服务.cmd; the Python 3.12 environment is selected automatically."""
import argparse
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent
PYTHON = ROOT / ".venv312" / "Scripts" / "python.exe"
RUNTIME = ROOT / "local_runtime"
STATE = RUNTIME / "processes.json"
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
KEEPER_MARKER = "oa-local-" + hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:12]


def status(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/local/status", timeout=2) as response:
            data = json.load(response)
        return data if data.get("service") == "lantu-agent-platform" else None
    except Exception:
        return None


def stop_owned():
    data = {}
    if STATE.exists():
        try:
            data = json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print("进程记录不可用，正在按本项目的完整后端命令定位。")
    try:
        backend_pid = max(0, int(data.get("backend_pid", 0)))
    except (TypeError, ValueError):
        backend_pid = 0

    # On Windows a venv redirector can spawn the base Python interpreter. Its
    # child can outlive the recorded Popen PID, so also locate exact commands.
    # An absolute --env-file token identifies ROOT without matching siblings
    # such as ROOT-copy, and the executable Name prevents matching this shell.
    env_path = str(ROOT / ".env.local").replace("'", "''")
    script = (
        "$oaEnvPath = [regex]::Escape('" + env_path + "'); "
        "$oaEnvPattern = '(?i)(?:^|\\s)--env-file(?:=|\\s+)(?:\"' + $oaEnvPath + '\"|' + $oaEnvPath + ')(?=\\s|$)'; "
        "$oaCommandPattern = '(?i)^\\s*(?:\"[^\"]+\"|\\S+)\\s+(?:-u\\s+)?-m\\s+uvicorn\\s+local_runtime\\.web_app:app(?=\\s|$)'; "
        "function Test-OaOwnedBackend($oaCandidate) { return ($oaCandidate -and "
        "$oaCandidate.Name -in @('python.exe', 'pythonw.exe') -and "
        "$oaCandidate.CommandLine -match $oaCommandPattern -and "
        "$oaCandidate.CommandLine -match $oaEnvPattern) }; "
        "$oaCandidates = @(); "
        "if (" + str(backend_pid) + " -gt 0) { "
        "$oaCandidates += @(Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(backend_pid) + "') }; "
        "$oaCandidates += @(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\"); "
        "$oaCandidateIds = @($oaCandidates | Where-Object { Test-OaOwnedBackend $_ } | "
        "Select-Object -ExpandProperty ProcessId -Unique); "
        "foreach ($oaCandidateId in $oaCandidateIds) { "
        "$oaCurrent = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $oaCandidateId); "
        "if (Test-OaOwnedBackend $oaCurrent) { "
        "Stop-Process -Id $oaCandidateId -ErrorAction SilentlyContinue } }"
    )
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], creationflags=HIDDEN)

    # The WSL keeper remains PID-bound and must carry this project's unique
    # marker; never scan for or stop another project's WSL/SQL/Redis process.
    try:
        wsl_pid = max(0, int(data.get("wsl_pid", 0)))
    except (TypeError, ValueError):
        wsl_pid = 0
    marker = data.get("wsl_marker", KEEPER_MARKER)
    if wsl_pid and marker == KEEPER_MARKER:
        script = (
            "$oaProcess = Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(wsl_pid) + "'; "
            "if ($oaProcess -and $oaProcess.CommandLine -like '*" + KEEPER_MARKER + "*') "
            "{ Stop-Process -Id " + str(wsl_pid) + " -ErrorAction SilentlyContinue }"
        )
        subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], creationflags=HIDDEN)
    STATE.unlink(missing_ok=True)
    print("已停止本项目的后端进程；数据库资料保留。")


def wait_database():
    for _ in range(45):
        ready = True
        for port in (3306, 6379):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    pass
            except OSError:
                ready = False
        if ready:
            return
        time.sleep(1)
    raise RuntimeError("本地 MySQL 或 Redis 未就绪，请查看 local_runtime/logs/wsl.log。")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if not PYTHON.exists():
        raise RuntimeError("未找到 .venv312，请先完成本机环境安装。")
    if pathlib.Path(sys.executable).resolve() != PYTHON.resolve():
        return subprocess.call([str(PYTHON), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser(description="启动澜图智能企业智能体平台本地服务")
    parser.add_argument("--background", action="store_true", help="后台运行并返回")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--stop", action="store_true", help="停止本项目记录的后台进程")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.stop:
        stop_owned()
        return 0
    if not (1 <= args.port <= 65535):
        raise RuntimeError("端口必须在 1 到 65535 之间。")
    url = f"http://127.0.0.1:{args.port}"
    existing = status(args.port)
    if existing and not existing.get("restart_required"):
        print("本地服务已经运行：" + url)
        print(existing.get("model_message", ""))
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    if existing:
        print("检测到本地配置已修改，正在重启本项目后端。")
        stop_owned()
        time.sleep(1)
    elif STATE.exists():
        stop_owned()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env.local", override=True)
    if os.getenv("MYSQL_HOST") not in ("127.0.0.1", "localhost"):
        raise RuntimeError("本地启动器只连接本机数据库。请核对 .env.local。")
    logs = RUNTIME / "logs"
    logs.mkdir(exist_ok=True)
    print("[1/4] 启动本机数据库服务……", flush=True)
    with (logs / "wsl.log").open("ab") as log:
        keeper = subprocess.Popen(
            ["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--exec", "bash", "-lc",
             "systemctl start mysql redis-server && exec -a " + KEEPER_MARKER + " sleep infinity"],
            stdout=log, stderr=log, creationflags=HIDDEN,
        )
    backend = None
    try:
        wait_database()
        print("[2/4] 检查本地数据表（保留已有资料）……", flush=True)
        subprocess.run([str(PYTHON), str(RUNTIME / "setup_database.py")], cwd=ROOT, check=True)
        subprocess.run([str(PYTHON), str(ROOT / "sql" / "apply_migrations.py")], cwd=ROOT, check=True)
        subprocess.run([str(PYTHON), "-m", "enterprise_rag.ensure_ingested"], cwd=ROOT, check=True)
        print("[3/4] 检查模型配置……", flush=True)
        from local_runtime.check_model import check
        model = check(if_changed=True)
        print(model["message"], flush=True)
        print("[4/4] 启动后端……", flush=True)
        with (logs / "backend.log").open("ab") as log:
            backend = subprocess.Popen(
                [str(PYTHON), "-u", "-m", "uvicorn", "local_runtime.web_app:app",
                 "--env-file", str(ROOT / ".env.local"), "--host", "127.0.0.1", "--port", str(args.port)],
                cwd=ROOT, stdout=log, stderr=log, creationflags=HIDDEN,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            )
        STATE.write_text(json.dumps({"backend_pid": backend.pid, "wsl_pid": keeper.pid, "wsl_marker": KEEPER_MARKER,
                                     "port": args.port, "url": url}, indent=2), encoding="utf-8")
        for _ in range(60):
            if backend.poll() is not None:
                raise RuntimeError("后端启动失败，请查看 local_runtime/logs/backend.log。")
            if status(args.port):
                break
            time.sleep(1)
        else:
            raise RuntimeError("后端启动超时，请查看 local_runtime/logs/backend.log。")
        print("\n服务已运行：" + url)
        print("演示账号：admin / employee_zhao / manager_zhang；密码均为 admin123。")
        print("可以直接询问“公司叫什么名字？”，也可以提交请假或审批。")
        print("模型状态：" + model["message"])
        if not args.no_browser:
            webbrowser.open(url)
        if args.background:
            return 0
        print("保持此窗口运行；按 Ctrl+C 停止后端。")
        try:
            backend.wait()
        except KeyboardInterrupt:
            pass
    except Exception:
        if backend and backend.poll() is None:
            backend.terminate()
        keeper.terminate()
        raise
    if backend and backend.poll() is None:
        backend.terminate()
    keeper.terminate()
    STATE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("启动失败：" + str(exc), file=sys.stderr)
        sys.exit(1)
