"""企业 RAG HTTP 冒烟：鉴权、统计、查询、版本冲突、拒答与引用。"""
import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env.local"), override=True)

BASE = os.getenv("OA_BASE", "http://127.0.0.1:8080")


def _request(path, token=None, payload=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return urllib.request.Request(BASE + path, data=data, headers=headers, method=method)


def api(path, token=None, payload=None, method=None):
    with urllib.request.urlopen(_request(path, token, payload, method), timeout=60) as response:
        return json.load(response)


def chat(token, thread_id, text):
    events, name, lines, final = [], "", [], None
    with urllib.request.urlopen(
        _request(
            "/api/chat/stream",
            token,
            {"thread_id": thread_id, "user_message": text},
        ),
        timeout=180,
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
                if name == "error":
                    raise RuntimeError(data)
                if name == "complete":
                    final = data
    if not final:
        raise RuntimeError("Missing SSE complete event")
    return final


def login(username):
    return api("/api/auth/login", payload={"username": username, "password": "admin123"})["data"]["token"]


def ask(token, question):
    return api("/api/rag/query", token, {"question": question, "top_k": 8})["data"]


def main():
    admin = login("admin")
    employee = login("employee_qian")

    with urllib.request.urlopen(BASE + "/", timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    assert "澜图智能企业智能体平台" in html and "小滴" not in html, "页面品牌未更新"
    print("branding ok")

    try:
        api("/api/rag/query", None, {"question": "测试"})
        raise AssertionError("unauthenticated query should fail")
    except urllib.error.HTTPError as exc:
        assert exc.code in (401, 403), exc.code
        print("unauthenticated blocked")

    stats = api("/api/rag/stats", employee)["data"]
    assert stats["namespace"] == "lantu" and stats["documents"] >= 100, stats
    assert stats["chunks"] > 0 and stats["versioned_families"] >= 5, stats
    print("stats ok", stats["documents"], stats["chunks"])

    latest = ask(employee, "一线城市出差住宿标准是多少？")
    assert latest["grounded"] and "600" in latest["answer"], latest["answer"]
    assert latest["citations"] and latest["citations"][0]["doc_id"] == "lantu_policy_travel_2025h2"
    print("latest conflict resolved")

    older = ask(employee, "2024年一线城市出差住宿标准是多少？")
    assert "450" in older["answer"] and older["retrieval"]["as_of"] == "2024-12-31", older
    print("time range resolved")

    refused = ask(employee, "公司食堂今天中午的菜单是什么？")
    assert refused["refused"] is True and not refused["citations"], refused["answer"]
    print("refusal ok")

    rewritten = ask(employee, "买机器人最多能打几折，找谁批？")
    assert "25%" in rewritten["answer"] and "董事会" in rewritten["answer"], rewritten["answer"]
    print("fuzzy query ok")

    thread = api("/api/chat/session", employee, method="POST")["data"]["thread_id"]
    try:
        chat_answer = chat(employee, thread, "公司叫什么名字？")
        assert "澜图智能装备股份有限公司" in chat_answer["response"], chat_answer["response"]
        assert "未覆盖" not in chat_answer["response"], chat_answer["response"]
        assert chat_answer.get("citations"), chat_answer
        print("chat company identity ok")
    finally:
        api("/api/chat/session/" + thread, employee, method="DELETE")

    try:
        api("/api/rag/documents", employee)
        raise AssertionError("employee should not list all documents")
    except urllib.error.HTTPError as exc:
        assert exc.code == 403, exc.code
        print("documents admin only")

    documents = api("/api/rag/documents", admin)["data"]
    assert documents["total"] >= 100, documents["total"]
    print("documents list ok", documents["total"])

    reindexed = api("/api/rag/reindex", admin, method="POST")["data"]
    assert reindexed["documents"] >= 100, reindexed
    print("reindex ok")

    print("RAG_SMOKE_PASS")


if __name__ == "__main__":
    main()
