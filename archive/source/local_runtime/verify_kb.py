"""制度问答端到端验收：带引用回答 + 无依据拒答 + SSE token 流。"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.getenv("OA_BASE", "http://127.0.0.1:8080")
REPORT = os.path.join(ROOT, "local_runtime", "verification_kb.json")


def request(path, token=None, payload=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return urllib.request.Request(BASE + path, data=data, headers=headers, method=method)


def api(path, token=None, payload=None, method=None):
    with urllib.request.urlopen(request(path, token, payload, method), timeout=30) as response:
        return json.load(response)


def login(username):
    return api("/api/auth/login", payload={"username": username, "password": "admin123"})["data"]["token"]


def chat(token, thread, text):
    start = time.monotonic()
    events, name, lines, final, tokens = [], "", [], None, 0
    with urllib.request.urlopen(
        request("/api/chat/stream", token, {"thread_id": thread, "user_message": text}), timeout=180
    ) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].lstrip())
            elif not line and lines:
                data = json.loads("\n".join(lines))
                lines = []
                events.append(name)
                if name == "token":
                    tokens += 1
                if name == "error":
                    raise RuntimeError(data)
                if name == "complete":
                    final = data
    if not final:
        raise RuntimeError("Missing SSE complete event")
    final["elapsed_seconds"] = round(time.monotonic() - start, 2)
    final["event_types"] = sorted(set(events))
    final["token_events"] = tokens
    return final


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "kind": "policy_qa_with_citations",
        "steps": [],
    }

    def step(name, **extra):
        record = {"name": name, "result": "pass", **extra}
        report["steps"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    try:
        assert api("/local/status")["model_ready"]
        token = login("employee_qian")
        thread = api("/api/chat/session", token, method="POST")["data"]["thread_id"]
        try:
            answer = chat(token, thread, "年假有几天？")
            assert answer["intent"] == "policy_qa", answer["intent"]
            assert answer.get("citations"), answer
            assert "员工手册" in answer["response"], answer["response"]
            # Knowledge answers are buffered until validation; a single verified
            # event is intentional and must still carry the complete answer.
            assert answer["token_events"] >= 1, answer["token_events"]
            assert answer.get("streaming_mode") == "verified", answer
            step(
                "policy_answer_with_citations",
                citations=len(answer["citations"]),
                token_events=answer["token_events"],
                elapsed_seconds=answer["elapsed_seconds"],
            )

            refusal = chat(token, thread, "公司食堂几点开门？")
            assert refusal["intent"] == "policy_qa", refusal["intent"]
            assert "未覆盖" in refusal["response"] or "资料不足" in refusal["response"], refusal["response"]
            assert not refusal.get("citations"), refusal.get("citations")
            step("unanswerable_refused", elapsed_seconds=refusal["elapsed_seconds"])

            history = api("/api/chat/history?thread_id=" + thread, token)["data"]["messages"]
            assert len(history) >= 4, len(history)
            step("history_roundtrip", message_count=len(history))
            report["result"] = "pass"
        finally:
            api("/api/chat/session/" + thread, token, method="DELETE")
    except Exception as exc:
        report.update(result="failed", error_type=type(exc).__name__, error=str(exc)[:300])
        raise
    finally:
        with open(REPORT, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print("KB_E2E_PASS", flush=True)


if __name__ == "__main__":
    main()
