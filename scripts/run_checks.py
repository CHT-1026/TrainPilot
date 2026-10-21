"""一键回归门禁：单元测试 + 检索/引用评估，可选 HTTP 端到端。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe scripts\\run_checks.py
    .venv312\\Scripts\\python.exe scripts\\run_checks.py --http
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], name: str, env: dict | None = None) -> bool:
    print(f"\n=== {name} ===", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env)
    ok = result.returncode == 0
    print(f"{name}: {'PASS' if ok else 'FAIL'} (exit {result.returncode})", flush=True)
    return ok


def server_ready(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/api/health", timeout=3):
            return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="运行需要后端在线的端到端验收")
    parser.add_argument("--answers", type=int, default=5, help="回答抽样条数")
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results = []
    results.append(
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-q"], "unit tests")
    )
    results.append(
        run(
            [sys.executable, "eval/run_eval.py", "--answers", str(args.answers), "--gate"],
            "retrieval + answer eval",
        )
    )
    results.append(
        run(
            [sys.executable, "-m", "enterprise_rag.eval.run_eval", "--gate"],
            "enterprise RAG eval",
        )
    )

    if args.http:
        if not server_ready(args.base):
            print(f"HTTP checks: SKIP (server not reachable at {args.base})")
        else:
            import os

            env = {**os.environ, "OA_BASE": args.base}
            results.append(run([sys.executable, "local_runtime/verify_kb.py"], "KB E2E", env))
            results.append(run([sys.executable, "local_runtime/verify_flow.py"], "leave E2E", env))
            for script, name in (
                ("tests/smoke_hr.py", "two-level approval"),
                ("tests/smoke_reporting_line.py", "reporting line"),
                ("tests/smoke_api_phase2.py", "approval API"),
                ("tests/smoke_suggestion_chat.py", "suggestion chat"),
                ("tests/smoke_manager_snapshot.py", "manager snapshot"),
                ("tests/smoke_user_admin.py", "user admin"),
                ("tests/smoke_kb_admin.py", "knowledge base admin"),
                ("tests/smoke_rag.py", "enterprise RAG API"),
                ("tests/smoke_guard.py", "prompt injection guard"),
            ):
                results.append(run([sys.executable, script], name, env))

    passed = all(results)
    print("\n=== SUMMARY ===")
    print("ALL_CHECKS_PASS" if passed else "CHECKS_FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
