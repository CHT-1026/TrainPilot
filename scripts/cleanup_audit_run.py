"""Clean only exact synthetic IDs from this run's verification report."""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env.local", override=False)
from app.infra.mysql import connection


def main():
    report = json.loads((ROOT / "local_runtime/verification.json").read_text(encoding="utf-8"))
    ids = {s["application_id"] for s in report.get("steps", []) if s.get("application_id")}
    if report.get("application_id"):
        ids.add(report["application_id"])
    deleted = 0
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT idempotency_key,user_id,endpoint,response_json FROM idempotency_keys WHERE user_id IN (%s,%s)", ("employee_qian", "manager_zhang"))
            for row in cursor.fetchall():
                def owns(value):
                    if isinstance(value, dict):
                        return any(owns(v) for v in value.values())
                    if isinstance(value, list):
                        return any(owns(v) for v in value)
                    return isinstance(value, str) and value in ids
                payload = row["response_json"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if owns(payload):
                    cursor.execute("DELETE FROM idempotency_keys WHERE idempotency_key=%s AND user_id=%s AND endpoint=%s", (row["idempotency_key"], row["user_id"], row["endpoint"]))
                    deleted += cursor.rowcount
        conn.commit()
    cleaned = []
    for field, username in (("employee_thread", "employee_qian"), ("manager_thread", "manager_zhang")):
        thread = report.get(field)
        if not thread:
            continue
        req = urllib.request.Request("http://127.0.0.1:8080/api/auth/login", data=json.dumps({"username": username,"password":"admin123"}).encode(), headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            token = json.load(response)["data"]["token"]
        req = urllib.request.Request("http://127.0.0.1:8080/api/chat/session/" + thread, method="DELETE", headers={"Authorization":"Bearer " + token})
        with urllib.request.urlopen(req, timeout=15) as response:
            json.load(response)
        cleaned.append(thread)
    out = ROOT / "audit_artifacts/runtime/initial_run_cleanup.json"
    out.write_text(json.dumps({"application_ids":sorted(ids),"deleted_cache_rows":deleted,"deleted_sessions":cleaned},ensure_ascii=False,indent=2),encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
