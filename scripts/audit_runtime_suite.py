"""Bounded sequential local integration audit with full per-command logs."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_artifacts/runtime"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commands = [
        "tests/smoke_service.py", "tests/smoke_hr.py", "tests/smoke_reporting_line.py",
        "tests/smoke_api_phase2.py", "tests/smoke_suggestion_chat.py", "tests/smoke_manager_snapshot.py",
        "tests/smoke_user_admin.py", "tests/smoke_kb_admin.py", "tests/smoke_rag.py",
        "tests/smoke_guard.py", "local_runtime/verify_kb.py", "local_runtime/verify_flow.py",
    ]
    records = []
    for script in commands:
        started = time.monotonic()
        try:
            result = subprocess.run([sys.executable, "-X", "utf8", script], cwd=ROOT,
                env={**os.environ, "OA_BASE": "http://127.0.0.1:8080", "PYTHONUTF8": "1"},
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
            output = result.stdout + "\n" + result.stderr
            record = {"script": script, "passed": result.returncode == 0, "exit_code": result.returncode}
        except subprocess.TimeoutExpired as exc:
            output = str(exc.stdout or "") + "\n" + str(exc.stderr or "")
            record = {"script": script, "passed": False, "error": "timeout"}
        log = OUT / (Path(script).stem + ".txt")
        log.write_text(output, encoding="utf-8")
        record.update(seconds=round(time.monotonic()-started, 3), log=str(log.relative_to(ROOT)))
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (OUT / "suite.json").write_text(json.dumps({"cases": records,
            "passed": sum(r["passed"] for r in records), "total": len(records)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
