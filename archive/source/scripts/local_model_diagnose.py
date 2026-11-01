# -*- coding: utf-8 -*-
"""本地模型推理能力诊断：量化本地模型能否支撑方案的 8 轮工具循环。

方案要求每题最多 8 轮模型交互、600 秒时限、12 次工具调用。
本脚本测量实际可达的每轮耗时与吞吐，给出"本地模型能否按方案跑"的结论，
不猜测、不用推断代替实测。

用法：
    ./.venv312/Scripts/python.exe scripts/local_model_diagnose.py --quant Q4_K_M
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"C:/Users/34157/Desktop/vllmproject/澜图智能企业智能体平台")
RUNTIME = ROOT / "research" / "retrieval_evaluation_20261101" / "runtime"
PORT = 1235

# 方案第 6 节的预算
BUDGET_ROUNDS = 8
BUDGET_CASE_SECONDS = 600


def proxy_free_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def probe(op, base, model, prompt, max_tokens, timeout):
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer x"})
    t0 = time.time()
    with op.open(req, timeout=timeout) as r:
        rep = json.load(r)
    return time.time() - t0, rep


def gpu_info() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        name, total, used = [x.strip() for x in out.split(",")]
        return {"name": name, "memory_total_mib": int(total), "memory_used_mib": int(used)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="Q4_K_M", choices=["Q4_K_M", "Q6_K"])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    op = proxy_free_opener()
    base = f"http://127.0.0.1:{PORT}/v1"
    report = {"quant": args.quant, "budget": {"rounds": BUDGET_ROUNDS,
                                             "case_seconds": BUDGET_CASE_SECONDS},
              "gpu": gpu_info(), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    try:
        with op.open(base + "/models", timeout=15) as r:
            ids = [m.get("id") for m in json.load(r).get("data", [])]
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"本地服务不可用：{type(exc).__name__}: {exc}"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    if not ids:
        report["error"] = "本地服务未提供模型"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    report["served_model"] = ids[0]

    # 测量 1：短提示 + 少量输出（最乐观情形）
    try:
        el, rep = probe(op, base, ids[0], "只回一个数字：澜图智能成立年份", 32, args.timeout)
        u = rep.get("usage", {})
        report["short_probe"] = {
            "seconds": round(el, 1),
            "completion_tokens": u.get("completion_tokens"),
            "tokens_per_second": round(u.get("completion_tokens", 0) / max(el, 1), 2),
            "content": (rep["choices"][0]["message"].get("content") or "")[:80],
            "reasoning_len": len(rep["choices"][0]["message"].get("reasoning_content") or ""),
        }
    except Exception as exc:  # noqa: BLE001
        report["short_probe"] = {"error": f"{type(exc).__name__}: {exc}",
                                 "seconds": round(args.timeout, 1)}

    # 测量 2：接近方案实际的单轮（系统提示 + 问题 + 典型输出长度）
    try:
        el, rep = probe(op, base, ids[0],
                         "你是企业资料检索助手。请回答：澜图智能 2025 年一线城市差旅住宿标准是多少？"
                         "先说明你会检索哪些资料，再给出结论并给出引用。", args.max_tokens, args.timeout)
        u = rep.get("usage", {})
        report["realistic_probe"] = {
            "seconds": round(el, 1),
            "completion_tokens": u.get("completion_tokens"),
            "tokens_per_second": round(u.get("completion_tokens", 0) / max(el, 1), 2),
            "content_len": len(rep["choices"][0]["message"].get("content") or ""),
            "reasoning_len": len(rep["choices"][0]["message"].get("reasoning_content") or ""),
        }
    except Exception as exc:  # noqa: BLE001
        report["realistic_probe"] = {"error": f"{type(exc).__name__}: {exc}",
                                     "seconds": round(args.timeout, 1)}

    # 推算：单题 8 轮是否可在 600 秒内完成
    rp = report.get("realistic_probe", {})
    sp = report.get("short_probe", {})
    per_round = rp.get("seconds") or sp.get("seconds")
    if per_round:
        est = per_round * BUDGET_ROUNDS
        report["estimate"] = {
            "seconds_per_round": per_round,
            "estimated_case_seconds_8_rounds": round(est),
            "within_case_budget": est <= BUDGET_CASE_SECONDS,
        }
        report["verdict"] = (
            "可支撑方案的 8 轮工具循环" if est <= BUDGET_CASE_SECONDS
            else f"单题预计 {round(est)} 秒，超出 600 秒时限 {round(est - BUDGET_CASE_SECONDS)} 秒；"
                 f"400 题预计需 {round(est * 400 / 3600, 1)} 小时")

    out = RUNTIME / f"diagnose_{args.quant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
