# -*- coding: utf-8 -*-
"""本地 Qwen3.8-27B 推理服务管理（Q6_K / Q4_K_M 逐个加载）。

用 LM Studio 自带的 llama.cpp CUDA 后端直接起 OpenAI 兼容服务，
不依赖 LM Studio 桌面端（桌面端未安装，lms server 无后端可唤起）。

约束：
- 一次只加载一个量化版本（方案第 5 节：保持一次只加载一个本地模型）。
- Q6_K 权重超过单卡显存；Q4_K_M 接近上限，另需KV与计算缓冲，均采用部分CPU卸载。
- 端口与 .env.local 的 BLIND_EVAL_LOCAL_Q6/Q4_BASE_URL 对齐（1235）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BACKEND = Path(r"C:/Users/34157/.lmstudio/extensions/backends/"
               r"llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.67.0")
VENDOR = Path(r"C:/Users/34157/.lmstudio/extensions/backends/vendor/win-llama-cuda12-vendor-v2")
MODELS = Path(r"C:/Users/34157/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF")
RUNTIME = Path(__file__).resolve().parents[1] / 'research/retrieval_repair_20261101/runtime'

QUANTS = {
    "Q4_K_M": MODELS / "Qwen3.8-27B-Q4_K_M.gguf",
    "Q6_K": MODELS / "Qwen3.8-27B-Q6_K.gguf",
}

# Q4_K_M 约16032MiB，接近本机16303MiB显存，尚需KV与计算缓冲；Q6_K权重更大。
# 单 slot（-np 1）后 KV cache 只占一份，显存能给更多层；
# Explicit partial offload: asking for 99 layers can spill into Windows shared memory.
# These are initial settings; actual GPU use and throughput must be measured.
GPU_LAYERS = {"Q4_K_M": "55", "Q6_K": "40"}
CTX = 16384
PORT = 1235


def check() -> int:
    ok = True
    for path, label in [(BACKEND / "llama-server.exe", "llama-server"),
                        (VENDOR / "cublas64_12.dll", "cublas"),
                        (QUANTS["Q4_K_M"], "Q4_K_M gguf"),
                        (QUANTS["Q6_K"], "Q6_K gguf")]:
        exists = path.exists()
        print(f"{'OK ' if exists else 'MISS'}  {label}: {path}")
        ok = ok and exists
    return 0 if ok else 1


def wait_ready(proc, quant: str, timeout: int = 900) -> bool:
    """轮询 /v1/models 直到服务就绪。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                data = json.load(r)
                names = [m["id"] for m in data["data"]]
                if f"qwen3.8-27b-{quant}" not in names:
                    raise RuntimeError(f"Port {PORT} serves a different model: {names}")
                print(f"[ready] models={names}")
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    return False


def start(quant: str, gpu_layers: int | None = None) -> int:
    """启动服务并阻塞在前台等待（由调用方以后台任务方式托管）。

    不使用 DETACHED：父进程退出时 Windows 会连带回收该进程，
    实测服务在 start 返回后即被清理（/v1/models 随后拒绝连接）。
    """
    model = QUANTS[quant]
    if not model.exists():
        print(f"缺少模型文件：{model}")
        return 1
    RUNTIME.mkdir(parents=True, exist_ok=True)
    attempt = str(time.time_ns())
    log_path = RUNTIME / f"llama_{quant}_{attempt}.log"
    env = os.environ.copy()
    env["PATH"] = f"{BACKEND};{VENDOR};" + env.get("PATH", "")
    cmd = [
        str(BACKEND / "llama-server.exe"),
        "-m", str(model),
        "--alias", f"qwen3.8-27b-{quant}",
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "-c", str(CTX),
        "-np", "1",           # 单 slot：4 slot 会各自占一份 16K KV cache，
                             # 把显存挤满导致大量层卸载到 CPU（实测 prefill 仅 34 tok/s）
        "-ngl", str(gpu_layers) if gpu_layers is not None else GPU_LAYERS[quant],
        "--no-context-shift", "--flash-attn", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--threads", "12",
        "--no-webui",
    ]
    print("启动：", " ".join(cmd), flush=True)
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(BACKEND), env=env, stdout=handle,
                            stderr=subprocess.STDOUT)
    launch = {
        "pid": proc.pid, "command": cmd, "model_path": str(model),
        "model_bytes": model.stat().st_size, "quant": quant, "log": str(log_path),
        "started_at": time.time()}
    for name in [f'llama_{quant}_launch.json', f'llama_{quant}_{attempt}_launch.json']:
        (RUNTIME / name).write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding='utf-8')
    (RUNTIME / f"llama_{quant}.pid").write_text(str(proc.pid), encoding="utf-8")
    print(f"pid={proc.pid} log={log_path}", flush=True)
    try:
        if not wait_ready(proc, quant, 900):
            print("[timeout] 服务未在 900s 内就绪，详见日志", flush=True)
            return 1
        print("[ready] 服务常驻中，按 Ctrl+C 结束", flush=True)
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        handle.close()
    return proc.returncode


def stop() -> int:
    for quant in QUANTS:
        receipt = RUNTIME / f"llama_{quant}_launch.json"
        if not receipt.exists():
            continue
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        pid = int(saved['pid'])
        # A stored PID can be recycled. Verify executable and model before stopping it.
        query = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            f'Get-CimInstance Win32_Process -Filter "ProcessId={pid}" | Select-Object ExecutablePath,CommandLine | ConvertTo-Json -Compress'],
            capture_output=True, text=True, check=True)
        if not query.stdout.strip():
            print(f"already exited pid={pid} ({quant})")
            continue
        actual = json.loads(query.stdout)
        if (Path(actual['ExecutablePath']).resolve() != (BACKEND / 'llama-server.exe').resolve()
                or str(QUANTS[quant]) not in actual['CommandLine']):
            raise RuntimeError(f"PID {pid} no longer belongs to this {quant} runtime; left untouched")
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=True)
        print(f"stopped pid={pid} ({quant})")
    return 0


def status() -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
            print("服务在线：", json.dumps(json.load(r), ensure_ascii=False))
            return 0
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"服务离线：{exc}")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["check", "start", "stop", "status"])
    ap.add_argument("--quant", default="Q4_K_M", choices=list(QUANTS))
    ap.add_argument("--gpu-layers", type=int)
    args = ap.parse_args()
    if args.action == "check":
        return check()
    if args.action == "start":
        return start(args.quant, args.gpu_layers)
    if args.action == "stop":
        return stop()
    return status()


if __name__ == "__main__":
    sys.exit(main())
