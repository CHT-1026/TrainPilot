"""Run final release checks without lowering gates or substituting model answers."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_artifacts"


def main():
    checks = [
        ("unit", ["-m", "unittest", "discover", "-s", "tests", "-t", ".", "-q"], "final_unit_tests.txt"),
        ("dependencies", ["-m", "pip", "check"], "dependency_check.txt"),
        ("reasoning", ["scripts/evaluate_reasoning.py", "--phase", "regression", "--gate"], "reasoning/final_gate.txt"),
        ("holdout", ["scripts/evaluate_reasoning.py", "--phase", "regression", "--fixtures", "audit_artifacts/reasoning/holdout_cases.jsonl", "--output-name", "holdout_regression", "--gate"], "reasoning/holdout_gate.txt"),
        ("legacy", ["-m", "enterprise_rag.eval.run_eval", "--gate"], "legacy_regression/final_gate.txt"),
        ("http", ["scripts/audit_http.py", "--phase", "regression"], "http/final_run.txt"),
    ]
    records = []
    for name, args, log in checks:
        result = subprocess.run([sys.executable, "-X", "utf8", *args], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUTF8": "1",
                "RAG_REPORT_DIR": str(ROOT / "enterprise_rag/eval/reports")}, timeout=240)
        (OUT / log).parent.mkdir(parents=True, exist_ok=True)
        (OUT / log).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        record = {"name": name, "passed": result.returncode == 0, "exit_code": result.returncode, "log": log}
        if name == "unit":
            match = re.search(r"Ran (\d+) tests", result.stderr)
            record["test_count"] = int(match.group(1)) if match else None
        if name == "http" and result.returncode == 0:
            data = json.loads((OUT / "http/regression.json").read_text(encoding="utf-8"))
            record["passed"] = all(c["passed"] for c in data["cases"])
            record["stream_consistency"] = all(c.get("token_matches_complete", True) for c in data["cases"])
            record["passed"] &= record["stream_consistency"]
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    for name in ("report.md", "latest.json", "per_case.jsonl"):
        shutil.copy2(ROOT / "enterprise_rag/eval/reports" / name, OUT / "legacy_regression" / name)
    (OUT / "final_checks.json").write_text(json.dumps({"checks": records, "passed": all(c["passed"] for c in records)},ensure_ascii=False,indent=2), encoding="utf-8")
    return 0 if all(c["passed"] for c in records) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
