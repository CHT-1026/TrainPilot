"""从当前企业文档动态生成问题并逐题验收。

设计约束：
1. 只读取 enterprise_rag/corpus 下的 134 篇企业文档，不使用固定题库或既有 testset；
2. 先解析文档，再从句子/表格归纳事实并生成问题；
3. 预期答案由同一条源事实独立推导（数值/单位/主体锚点），不由问答系统提供；
4. 通过生产 HTTP /api/rag/query 逐题询问真实服务，记录逐题答案、引用、耗时与失败原因。

用法：
    python scripts/dynamic_document_acceptance.py --dry-run
    python scripts/dynamic_document_acceptance.py --limit 60
    python scripts/dynamic_document_acceptance.py --limit 60 --direct
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from enterprise_rag import config  # noqa: E402
from enterprise_rag.loaders import load_document  # noqa: E402

MANIFEST = config.MANIFEST_PATH
OUT_DIR = ROOT / "audit_artifacts" / "dynamic_acceptance"
BASE = "http://127.0.0.1:8080"

_SENTENCE = re.compile(r"(?<=[。！？\n])")
_NUM = re.compile(
    r"(?<![\d.-])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(亿元|万元|元/晚|元/天|台/年|片/小时|个工作日|个自然日|kg|mm|m/s|%|元|台|人|天|小时|分钟|秒|个月|年|月|日)"
)
_GENERIC_KEYS = {
    "公司", "企业", "项目", "产品", "方案", "服务", "该", "其", "其中", "同时", "此外",
    "目前", "当前", "总", "合计", "其他", "相关", "以上", "以下", "备注", "说明",
}
_LIST_MARKER = re.compile(r"^(?:第[一二三四五六七八九十百零]+[章节条]|[一二三四五六七八九十百零]+、|\d+[.、)]|\-|•)\s*")


def normalize(text: str) -> str:
    text = (text or "").lower().replace("％", "%")
    return re.sub(r"[\s,，。；;：:、（）()“”\"'`]+", "", text)


def split_sentences(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE.split(text.replace("\r", "\n")) if item.strip()]


def clean_key(raw: str, section: str) -> str:
    raw = re.sub(r"（[^）]*）|\([^)]*\)", "", raw)
    raw = re.split(r"[、，,。；;：:]", raw)[-1]
    raw = _LIST_MARKER.sub("", raw.strip(" -—·:：±"))
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"(?:为|是|达到|达|有|不超过|不低于|至|约|在|于)$", "", raw).strip()
    if len(raw) < 2 or len(raw) > 48:
        return ""
    if raw in _GENERIC_KEYS:
        return ""
    # 纯日期碎片（“2025 年 4 月”“8 日”）不作为独立业务事实。
    if re.fullmatch(r"(?:19|20)\d{2}\s*年(?:\s*\d{1,2}\s*月)?", raw) or re.fullmatch(r"\d{1,2}\s*[月日]", raw):
        return ""
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", raw):
        return ""
    return raw


def anchor_terms_of(key: str) -> list[str]:
    normalized = normalize(key)
    if len(normalized) <= 4:
        terms = [normalized] + [normalized[index : index + 2] for index in range(len(normalized) - 1)]
        return [term for term in dict.fromkeys(terms) if len(term) >= 2]
    terms = [normalized[-4:]]
    for width in (2, 3):
        terms.extend(normalized[index : index + width] for index in range(0, len(normalized) - width + 1))
    return [term for term in dict.fromkeys(terms) if len(term) >= 2 and term not in _GENERIC_KEYS]


def anchor_of(key: str) -> str:
    terms = anchor_terms_of(key)
    return terms[0] if terms else normalize(key)


def parse_rows(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return [], []
    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]
    header = cells(lines[0])
    body = [cells(line) for line in lines[1:] if not re.match(r"^\|?\s*-{3,}", line.replace(" ", ""))]
    return header, [row for row in body if len(row) == len(header)]


def candidates_from_record(record: dict, loaded: dict) -> list[dict]:
    candidates = []
    title = record["title"]
    category = record.get("category", "")
    fmt = record.get("format", "")
    doc_id = record["doc_id"]
    as_of = record.get("effective_date") if record.get("doc_family") else None

    for block in loaded["blocks"]:
        section = block.get("section") or "正文"
        if block.get("type") == "table":
            header, rows = parse_rows(block["text"])
            if not header:
                continue
            table_section = section if not section.startswith("lantu_") else "表格"
            for row in rows:
                key = row[0].strip()
                if not key or key in _GENERIC_KEYS or len(key) > 32:
                    continue
                for index, cell in enumerate(row[1:], 1):
                    if index >= len(header):
                        continue
                    value_match = re.fullmatch(r"([\d,]+(?:\.\d+)?)\s*(%|亿元|万元|元|台|人|kg|mm|m/s|小时|天)?", cell.strip())
                    if not value_match:
                        continue
                    name = header[index]
                    if not name or any(word in name for word in ("序号", "备注", "年份", "状态")):
                        continue
                    question = f"《{title}》的“{table_section}”表中，{key}的{name}是多少？"
                    candidates.append({
                        "question": question,
                        "key": key,
                        "anchor": anchor_of(key),
                        "value": value_match.group(1).replace(",", ""),
                        "unit": value_match.group(2) or "",
                        "source": f"{doc_id} · {section} · 表格行 {key}",
                        "doc_id": doc_id,
                        "title": title,
                        "category": category,
                        "format": fmt,
                        "kind": "table",
                        "as_of": as_of,
                    })
            continue

        for sentence in split_sentences(block["text"]):
            for match in _NUM.finditer(sentence):
                key = clean_key(sentence[: match.start()], section)
                if not key or len(key) < 3:
                    continue
                value = match.group(1).replace(",", "")
                unit = match.group(2)
                # 跳过年份/日期类长句中的泛化数字，避免把“2025 年”当作业务数值。
                if unit == "年" and re.fullmatch(r"(?:19|20)\d{2}", value):
                    if not re.search(r"成立|创立|创办|开始|始于", key):
                        continue
                if unit in {"日", "月"}:
                    continue
                question = f"《{title}》中，“{section}”部分提到的{key}是多少？"
                candidates.append({
                    "question": question,
                    "key": key,
                    "anchor": anchor_of(key),
                    "value": value,
                    "unit": unit,
                    "source": f"{doc_id} · {section} · {sentence[:180]}",
                    "doc_id": doc_id,
                    "title": title,
                    "category": category,
                    "format": fmt,
                    "kind": "text",
                    "as_of": as_of,
                })
    return candidates


def build_candidates() -> tuple[list[dict], dict]:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_candidates = []
    loaded_counts = defaultdict(int)
    for record in records:
        path = config.PACKAGE_ROOT / "corpus" / record["path"]
        if not path.exists():
            continue
        loaded = load_document(path)
        loaded_counts[record["format"]] += 1
        all_candidates.extend(candidates_from_record(record, loaded))
    # 去重：同一问题/同一数值只保留一条
    seen, unique = set(), []
    for item in all_candidates:
        signature = (normalize(item["question"]), item["value"], normalize(item["anchor"]))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique, {"documents": len(records), "formats": dict(loaded_counts), "raw_candidates": len(all_candidates)}


def select(candidates: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for item in candidates:
        by_category[item["category"]].append(item)
    for items in by_category.values():
        rng.shuffle(items)
    selected = []
    # 每类最多取 6 条，保证跨部门/跨格式覆盖
    for category in sorted(by_category):
        selected.extend(by_category[category][:6])
    rng.shuffle(selected)
    selected = selected[: max(limit, 50)]
    # 补充自然问法（不带文档标题），验证同义改写与检索泛化
    natural = []
    for item in selected:
        if len(natural) >= 12:
            break
        key = item["key"]
        if len(normalize(key)) < 4 or normalize(key) in _GENERIC_KEYS:
            continue
        natural.append({
            **item,
            "question": f"{key}是多少？",
            "kind": item["kind"] + "_natural",
        })
    merged = selected + natural
    return merged[: limit] if limit else merged


def multi_hop_candidates() -> list[dict]:
    """从文档已解析的表格/年报数值中现场构造少量运算题。"""
    results = []
    try:
        from enterprise_rag.loaders import load_document
        # 三个制造基地员工合计
        path = config.PACKAGE_ROOT / "corpus" / "docs" / "org" / "lantu_org_headcount.xlsx"
        if path.exists():
            blocks = load_document(path)["blocks"]
            for block in blocks:
                if block["section"] == "基地分布":
                    header, rows = parse_rows(block["text"])
                    values = {}
                    for row in rows:
                        if row and row[0] in {"苏州总部", "东莞华南", "成都西部"} and len(row) > 1:
                            try:
                                values[row[0]] = float(row[1].replace(",", ""))
                            except ValueError:
                                pass
                    if len(values) == 3:
                        total = sum(values.values())
                        results.append({
                            "question": "苏州、东莞、成都三个制造基地的员工合计是多少人？",
                            "key": "三个基地员工合计",
                            "anchor": "人",
                            "value": f"{total:.0f}",
                            "unit": "人",
                            "source": "lantu_org_headcount · 基地分布",
                            "doc_id": "lantu_org_headcount",
                            "title": "澜图智能员工分布统计（2025）",
                            "category": "org",
                            "format": "xlsx",
                            "kind": "multi_hop",
                            "calculation": f"{values} 求和",
                        })
    except Exception:
        pass

    def find_value(doc_id: str, anchor: str) -> float | None:
        record = next((item for item in records if item["doc_id"] == doc_id), None)
        if not record:
            return None
        loaded = load_document(config.PACKAGE_ROOT / "corpus" / record["path"])
        for block in loaded["blocks"]:
            for sentence in split_sentences(block["text"]):
                if anchor in sentence:
                    match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|元|人|台|kg)", sentence)
                    if match:
                        return float(match.group(1).replace(",", ""))
        return None

    try:
        revenue_2025 = find_value("lantu_finance_annual_2025", "营业收入")
        revenue_2024 = find_value("lantu_finance_annual_2024", "营业收入")
        if revenue_2025 and revenue_2024:
            results.append({
                "question": "2025年营业收入比2024年多多少亿元？",
                "key": "两年营业收入差额", "anchor": "亿元", "value": f"{revenue_2025 - revenue_2024:.2f}", "unit": "亿元",
                "source": "lantu_finance_annual_2025 / lantu_finance_annual_2024",
                "doc_id": "lantu_finance_annual_2025", "title": "2025年度报告摘要", "category": "finance",
                "format": "pdf", "kind": "multi_hop", "calculation": f"{revenue_2025} - {revenue_2024}",
            })
        net_2025 = find_value("lantu_finance_annual_2025", "归母净利润")
        net_2024 = find_value("lantu_finance_annual_2024", "归母净利润")
        if net_2025 and net_2024:
            results.append({
                "question": "2025年归母净利润比2024年多多少亿元？",
                "key": "两年净利润差额", "anchor": "亿元", "value": f"{net_2025 - net_2024:.2f}", "unit": "亿元",
                "source": "lantu_finance_annual_2025 / lantu_finance_annual_2024",
                "doc_id": "lantu_finance_annual_2025", "title": "2025年度报告摘要", "category": "finance",
                "format": "pdf", "kind": "multi_hop", "calculation": f"{net_2025} - {net_2024}",
            })
    except Exception:
        pass
    return results


def evaluate(item: dict, result: dict) -> dict:
    answer = result.get("answer", "") or ""
    answer_norm = normalize(answer)
    expected = normalize(item["value"])
    anchor_terms = anchor_terms_of(item["key"])
    value_ok = expected in answer_norm
    anchor_ok = not anchor_terms or any(term in answer_norm for term in anchor_terms)
    citation_ok = bool(result.get("citations"))
    grounded = bool(result.get("grounded"))
    refused = bool(result.get("refused"))
    passed = (not refused) and grounded and citation_ok and value_ok and anchor_ok
    reasons = []
    if refused:
        reasons.append("模型拒答")
    if not grounded:
        reasons.append("未标记为有依据回答")
    if not citation_ok:
        reasons.append("无有效引用")
    if not value_ok:
        reasons.append(f"答案缺少预期值 {item['value']}{item['unit']}")
    if not anchor_ok:
        reasons.append(f"答案缺少主体锚点 {item['anchor']}")
    return {
        **item,
        "answer": answer,
        "expected": f"{item['value']}{item['unit']}",
        "passed": passed,
        "value_ok": value_ok,
        "anchor_ok": anchor_ok,
        "refused": refused,
        "grounded": grounded,
        "citation_count": len(result.get("citations", [])),
        "method": result.get("method"),
        "retrieved_docs": [c.get("doc_id") for c in result.get("retrieval", {}).get("chunks", [])],
        "latency_ms": result.get("latency_ms", 0),
        "failure_reasons": reasons,
    }


def login() -> str:
    body = json.dumps({"username": "employee_qian", "password": "admin123"}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BASE + "/api/auth/login", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["data"]["token"]


def ask_http(token: str, question: str, as_of: str | None = None, use_llm: bool = True) -> dict:
    payload = {"question": question, "top_k": 8, "use_llm": use_llm}
    if as_of:
        payload["as_of"] = as_of
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BASE + "/api/rag/query",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["data"]


def ask_direct(question: str, as_of: str | None = None) -> dict:
    from app.domain.user.models import UserContext
    from enterprise_rag.pipeline import query

    return query(
        question,
        user=UserContext(user_id="dynamic-audit", name="动态验收", role="admin", department="管理部"),
        as_of=as_of,
    )


def write_report(results: list[dict], meta: dict) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"dynamic_acceptance_{stamp}.json"
    md_path = OUT_DIR / f"dynamic_acceptance_{stamp}.md"
    passed = sum(1 for item in results if item["passed"])
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "meta": meta,
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / (len(results) or 1), 4),
        "results": results,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 企业文档动态验收报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 事实来源：`enterprise_rag/corpus`（{meta['documents']} 篇企业文档，仅此来源）",
        f"- 问题生成：从文档句子/表格现场归纳，共生成 {meta['raw_candidates']} 条候选，抽取 {len(results)} 条问题；未使用固定题库或既有 testset",
        f"- 通过：**{passed}/{len(results)}（{report['pass_rate']:.2%}）**",
        f"- 执行方式：{'生产 HTTP /api/rag/query' if meta['transport'] == 'http' else '直接调用生产 pipeline'}；模型：{config.LLM_MODEL}",
        f"- 随机种子：{meta['seed']}",
        "",
        "## 一、通过题目",
        "",
        "| 编号 | 类别 | 问题 | 预期 | 实际答案（摘要） | 引用 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(results, 1):
        if not item["passed"]:
            continue
        lines.append(
            f"| D{index:03d} | {item['category']} | {item['question']} | {item['expected']} | "
            f"{item['answer'][:120].replace(chr(10), ' ')} | {item['citation_count']} |"
        )
    lines += ["", "## 二、失败题目明细", "", "| 编号 | 类别 | 问题 | 预期 | 实际答案 | 失败原因 |", "| --- | --- | --- | --- | --- | --- |"]
    for index, item in enumerate(results, 1):
        if item["passed"]:
            continue
        lines.append(
            f"| D{index:03d} | {item['category']} | {item['question']} | {item['expected']} | "
            f"{item['answer'][:180].replace(chr(10), ' ')} | {'；'.join(item['failure_reasons'])} |"
        )
    lines += ["", "## 三、逐题来源与执行信息", "", "完整字段（来源文档、源句、检索文档、耗时、method）见同名 JSON。", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct", action="store_true", help="不经过 HTTP，直接调用生产 pipeline")
    args = parser.parse_args()

    global records
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates, stats = build_candidates()
    selected = select(candidates, args.limit, args.seed)
    selected += multi_hop_candidates()
    selected = selected[: args.limit + 5]
    print(json.dumps({"documents": stats["documents"], "formats": stats["formats"],
                      "raw_candidates": stats["raw_candidates"], "selected": len(selected),
                      "sample": selected[:5]}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    token = None if args.direct else login()
    results = []
    for index, item in enumerate(selected, 1):
        started = time.monotonic()
        try:
            result = ask_direct(item["question"], item.get("as_of")) if args.direct else ask_http(token, item["question"], item.get("as_of"))
            evaluated = evaluate(item, result)
        except Exception as exc:
            evaluated = {**item, "answer": "", "expected": f"{item['value']}{item['unit']}",
                         "passed": False, "value_ok": False, "anchor_ok": False, "refused": False,
                         "grounded": False, "citation_count": 0, "method": "error",
                         "retrieved_docs": [], "latency_ms": int((time.monotonic() - started) * 1000),
                         "failure_reasons": [f"执行异常: {type(exc).__name__}: {exc}"]}
        results.append(evaluated)
        print(json.dumps({"id": f"D{index:03d}", "passed": evaluated["passed"],
                          "question": item["question"], "expected": evaluated["expected"],
                          "refused": evaluated["refused"], "method": evaluated["method"],
                          "reasons": evaluated["failure_reasons"]}, ensure_ascii=False), flush=True)

    meta = {"documents": stats["documents"], "formats": stats["formats"], "raw_candidates": stats["raw_candidates"],
            "seed": args.seed, "transport": "direct" if args.direct else "http"}
    json_path, md_path = write_report(results, meta)
    passed = sum(1 for item in results if item["passed"])
    print(json.dumps({"total": len(results), "passed": passed, "pass_rate": round(passed / (len(results) or 1), 4),
                      "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
