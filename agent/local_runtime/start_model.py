"""Load the pinned V7 adapter; never replace an unrelated model server."""
import json
import subprocess
import time
import urllib.error
import urllib.request

from agent.local_runtime.demo_config import AGENT_ROOT, PROJECT_ROOT, load_config, project_path

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def model_info():
    config = load_config()
    try:
        with OPENER.open(f"http://127.0.0.1:{config['model_port']}/local/model-info", timeout=2) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError("模型端口已被其他接口占用，未覆盖该进程。") from error
    except (urllib.error.URLError, TimeoutError):
        return None
    if data.get("adapter_sha256") != config["adapter_sha256"] or not data.get("tool_calls"):
        raise RuntimeError("模型端口上的权重或工具协议与固定 V7 不一致，未切换其他模型。")
    return data


def ensure_model():
    config = load_config()
    if model_info():
        print("固定 V7 模型接口已就绪。", flush=True)
        return None
    log_path = AGENT_ROOT / "local_runtime/logs/model.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(project_path(config["model_python"])), "-u", "-m", "model.serving.tools_api",
        "--checkpoint", str(project_path(config["checkpoint"])),
        "--config-dir", str(project_path(config["config_dir"])),
        "--adapter-dir", str(project_path(config["adapter_dir"])),
        "--expected-adapter-sha256", config["adapter_sha256"],
        "--port", str(config["model_port"]), "--max-new-tokens", str(config["max_new_tokens"]),
        "--output-dir", str(PROJECT_ROOT / "runs/local-agent-v7")]
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print("正在加载固定 V7，日志：" + str(log_path), flush=True)
    try:
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError("V7 加载失败，请查看模型日志。")
            if model_info():
                print("固定 V7 模型接口已就绪。", flush=True)
                return process
            time.sleep(1)
        raise RuntimeError("V7 加载超时，请查看模型日志。")
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise
