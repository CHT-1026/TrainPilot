"""公司架构时间/版本黑盒回归：逐条调用生产接口，题目不写入任何固定题库。"""
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8080"

QUESTIONS = [
    ("meta", "现在的公司架构是什么时候更新的", ["2025-04-10"], []),
    ("meta", "公司架构是什么时候更新的", ["2025-04-10"], []),
    ("meta", "当前组织架构的更新时间是什么", ["2025-04-10"], []),
    ("meta", "最新的公司组织架构是哪一版", ["v2.0"], []),
    ("meta", "公司架构更新到哪一版了", ["v2.0"], []),
    ("meta", "组织架构最近一次改版是什么时候", ["2025-04-10"], []),
    ("meta", "2025版组织架构是哪天上线的", ["2025-04-10"], []),
    ("meta", "公司架构上次调整日期是哪天", ["2025-04-10"], []),
    ("meta", "现在执行的是哪一版组织架构", ["v2.0"], []),
    ("meta", "公司架构版本号和更新时间分别是什么", ["v2.0", "2025-04-10"], []),
    ("disambig", "公司技术架构是什么时候更新的", ["v3.3", "2026-10-22"], []),
    ("disambig", "控制系统架构的版本和更新时间是什么", ["v3.3", "2026-10-22"], []),
    ("disambig", "公司架构与技术架构分别是哪一版", ["v2.0", "v3.3"], []),
    ("disambig", "旧版组织架构是哪一年的", ["2024"], []),
    ("disambig", "2024版和2025版组织架构各有多少个一级部门", ["11", "12"], []),
    ("premise", "公司架构是2026年更新的吗", ["2025-04-10"], ["是2026年更新"]),
    ("premise", "组织架构2026版有哪些部门", ["2025", "没有2026"], []),
    ("premise", "公司架构更新于2023年吗", ["2025-04-10"], []),
    ("premise", "现在的组织架构还是2024版吗", ["2025", "不是"], []),
    ("premise", "公司架构最近一次更新是上月吗", ["2025-04-10"], []),
    ("upgrade", "公司架构有哪些升级", ["12", "客户成功部"], []),
    ("upgrade", "2024版到2025版组织架构新增了哪个一级部门", ["客户成功部"], []),
    ("upgrade", "售后服务部在2025版是什么级别", ["一级部门"], []),
    ("upgrade", "战略投资部在2025版还是一级部门吗", ["不是"], ["仍为一级部门", "仍是"]),
    ("upgrade", "2025版组织架构有多少个一级部门", ["12"], []),
    ("meta", "组织架构的发布日期是哪天", ["2025-04-10"], []),
    ("meta", "组织架构的生效日期是哪天", ["2025-04-01"], []),
    ("meta", "公司架构更新时间是2025年4月10日吗", ["2025-04-10"], []),
    ("disambig", "组织架构的旧版和新版分别是哪一版", ["v1.0", "v2.0"], []),
    ("premise", "组织架构2025版是2024年发布的吗", ["不是"], ["是2024年发布"]),
    ("premise", "公司架构2022版有多少个一级部门", ["没有2022"], []),
    ("premise", "公司架构是2024年1月更新的吗", ["2025-04-10"], ["是2024"]),
    ("upgrade", "2025版组织架构比2024版多了几个一级部门", ["1"], []),
    ("upgrade", "2024版组织架构比2025版少几个一级部门", ["1"], []),
    ("upgrade", "2025版组织架构比2024版少了哪个部门", ["战略投资部"], []),
    ("upgrade", "客户成功部在2024版组织架构中存在吗", ["不在"], []),
    ("upgrade", "售后服务部在2024版是一级部门吗", ["不是"], []),
    ("upgrade", "客户成功部在2025版是一级部门吗", ["是"], []),
    ("regression", "公司叫什么名字", ["澜图智能装备股份有限公司"], []),
    ("regression", "公司2025年营业收入是多少", ["42.63"], []),
    ("regression", "一线城市差旅住宿标准是多少", ["600"], []),
    ("regression", "P1故障的响应时间是多少", ["30"], []),
    ("regression", "2025版研发中心下设几个研究所", ["5"], []),
    ("regression", "公司总部在哪里", ["苏州"], []),
    ("regression", "2025年末员工总数是多少", ["3860"], []),
    ("regression", "销售折扣超过25%由谁审批", ["董事会"], []),
    ("regression", "采购标准供应商付款账期是多少天", ["90"], []),
    ("regression", "公司有哪些机器人产品", ["六轴", "SCARA", "AMR"], []),
    ("regression", "公司哪一年成立", ["2011"], []),
    ("regression", "公司董事长是谁", ["顾承澜"], []),
    ("regression", "2025年归母净利润是多少", ["4.12"], []),
    ("regression", "研发中心有多少人", ["1286"], []),
    ("regression", "员工每月可以申请远程办公几天", ["4"], []),
    ("regression", "年假按工龄分别是多少天", ["5", "10", "15"], []),
    ("regression", "2024年公司营业收入是多少", ["35.80"], []),
]


def call(path, payload=None, token=None, timeout=300):
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    login = call("/api/auth/login", {"username": "admin", "password": "admin123"})["data"]
    token = login["token"]
    records = []
    for category, question, must, forbidden in QUESTIONS:
        started = time.monotonic()
        error = ""
        data = {}
        try:
            data = call("/api/rag/query", {"question": question, "top_k": 10,
                                           "use_llm": True, "retrieval_mode": "hybrid"}, token)["data"]
        except Exception as exc:
            error = type(exc).__name__
        answer = (data.get("answer") or "")
        normalized = answer.replace(" ", "").replace(",", "").lower()
        missing = [item for item in must if item.lower().replace(" ", "") not in normalized]
        forbidden_hits = [item for item in forbidden if item.lower().replace(" ", "") in normalized]
        passed = not error and not data.get("refused") and not missing and not forbidden_hits
        record = {
            "category": category, "question": question, "must_contain": must,
            "forbidden": forbidden, "passed": passed, "missing": missing,
            "forbidden_hits": forbidden_hits, "refused": bool(data.get("refused")),
            "method": data.get("method"), "answer": answer,
            "citations": [{"index": c.get("index"), "doc_id": c.get("doc_id"),
                           "version": c.get("version")} for c in data.get("citations") or []],
            "latency_ms": int((time.monotonic() - started) * 1000), "error": error,
        }
        records.append(record)
        print(json.dumps({"n": len(records), "cat": category, "pass": passed,
                          "q": question, "missing": missing, "refused": record["refused"],
                          "ms": record["latency_ms"], "answer": answer[:160]}, ensure_ascii=False), flush=True)

    passed = sum(1 for record in records if record["passed"])
    report = {"total": len(records), "passed": passed,
              "pass_rate": round(passed / len(records), 4), "records": records}
    stamp = datetime.now().strftime("%Y%m%d")
    out_dir = ROOT / "audit_artifacts"
    json_path = out_dir / f"architecture_blackbox_{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 公司架构时间/版本黑盒测试（{stamp}）", "",
             f"- 题目：{len(records)} 条（运行时逐条调用生产接口，未预置题库）",
             f"- 通过：**{passed}/{len(records)}（{report['pass_rate']*100:.2f}%）**", "",
             "| # | 类别 | 问题 | 结果 | 方法 | 耗时ms | 答案摘要 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for index, record in enumerate(records, 1):
        summary = record["answer"].replace("|", "/").replace("\n", " ")[:120]
        lines.append(f"| {index} | {record['category']} | {record['question']} | "
                     f"{'PASS' if record['passed'] else 'FAIL'} | {record['method']} | "
                     f"{record['latency_ms']} | {summary} |")
    md_path = out_dir / f"architecture_blackbox_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"total": len(records), "passed": passed, "json": str(json_path),
                      "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
