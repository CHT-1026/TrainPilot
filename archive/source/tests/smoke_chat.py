"""HTTP 对话冒烟：登录 -> 建会话 -> 发一条查询 -> 打印 SSE 结果。"""
import json
import os
import sys
import urllib.request

BASE = os.getenv("OA_BASE", "http://127.0.0.1:8081")


def api(path, token=None, payload=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def chat(token, thread, text):
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
    data = json.dumps({"thread_id": thread, "user_message": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(BASE + "/api/chat/stream", data=data, headers=headers, method="POST")
    events, name, lines, final = [], "", [], None
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].lstrip())
            elif not line and lines:
                payload = json.loads("\n".join(lines))
                lines = []
                events.append(name)
                if name == "error":
                    raise RuntimeError(payload)
                if name == "complete":
                    final = payload
    return final, events


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    login = api("/api/auth/login", payload={"username": "employee_qian", "password": "admin123"})
    token = login["data"]["token"]
    thread = api("/api/chat/session", token, method="POST")["data"]["thread_id"]
    question = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OA_QUESTION", "查询我的请假申请。")
    try:
        final, events = chat(token, thread, question)
        print(json.dumps({"intent": final.get("intent"), "response": final.get("response"),
                          "citations": final.get("citations"),
                          "events": sorted(set(events)), "pending": final.get("pending_action")},
                         ensure_ascii=False, indent=2))
    finally:
        api("/api/chat/session/" + thread, token, method="DELETE")


if __name__ == "__main__":
    main()
