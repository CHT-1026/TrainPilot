"""Exercise the real localhost UI/API with frozen V7; keep failures as observed."""
import argparse
import json
from pathlib import Path
import time
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with httpx.Client(base_url=args.url, trust_env=False, timeout=300) as client:
        status = client.get("/local/status"); status.raise_for_status()
        rows.append({"check": "local_status", "response": status.json()})
        for username in ["employee_zhao", "manager_zhang", "admin"]:
            response = client.post("/api/auth/login", json={"username": username, "password": "admin123"})
            response.raise_for_status()
            body = response.json()
            # Do not save authentication tokens in experiment records.
            rows.append({"check": "login", "username": username, "code": body["code"]})
            if username == "employee_zhao":
                employee_token = body["data"]["token"]
        client.headers["Authorization"] = "Bearer " + employee_token
        for question in ["你好，你是谁？", "根据企业资料，澜图智能公司的成立日期是什么？",
                         "我想请一天事假，先帮我收集需要的信息，不要提交。"]:
            started = time.monotonic()
            response = client.post("/api/chat", json={"user_message": question})
            response.raise_for_status()
            data = response.json()["data"]
            rows.append({"check": "chat", "question": question, "seconds": time.monotonic()-started,
                         "response": data})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            tools = data.get("tool_execution") or {}
            print(json.dumps({"question": question, "intent": data.get("intent"),
                "answer": data.get("response"), "tool_status": tools.get("status"),
                "tool_calls": tools.get("tool_calls",0)}, ensure_ascii=False), flush=True)
    print("LOCAL_DEMO_CHECK_SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
