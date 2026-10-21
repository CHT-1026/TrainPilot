"""证据约束：数字/型号校验、版本对比对齐、枚举完整性提示。"""
from __future__ import annotations

import re

from enterprise_rag.citations import chunk_body

UNIT_RE = re.compile(
    r"\d[\d,\.]*\s*(?:ms|kHz|Hz|位|轴|小时|℃|级|mm|kg|%|元|万元|亿元|台|人|天|分钟|秒|MPa|N·m)"
)
MODEL_RE = re.compile(r"\b(?:LT|Lantu)[-A-Za-z0-9]*[A-Za-z0-9]+(?:[-A-Za-z0-9]+)*\b")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；\n])")

DIMENSIONS = (
    ("总线", ("总线", "EtherCAT", "CANopen")),
    ("控制器", ("控制器", "LT-C")),
    ("联动轴数", ("联动", "轴数", "支持 8 轴", "支持 4 轴")),
    ("插补周期", ("插补",)),
    ("伺服采样", ("伺服", "采样")),
    ("编码器", ("编码器",)),
    ("软件栈", ("软件栈", "实时内核", "PREEMPT_RT", "RTX")),
    ("可靠性", ("MTBF", "可靠性")),
    ("工作温度", ("工作温度", "℃")),
    ("EMC", ("EMC",)),
)


def normalize(text: str) -> str:
    return re.sub(r"[\s,，、]", "", (text or "")).lower()


def _sentences(answer: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(answer or "") if part.strip()]


def unsupported_numeric_tokens(answer: str, chunks: list[dict]) -> list[dict]:
    """Return claims whose numeric/model token is absent from the cited evidence.

    A combined claim may cite only the second source while using a model name
    stated in another cited sentence. Tokens are therefore checked against all
    chunks cited anywhere in the answer, not only the sentence-local refs.
    """
    problems = []
    bodies = [normalize(chunk_body(chunk)) for chunk in chunks]
    cited_indices = {
        int(value) for value in re.findall(r"\[(\d+)\]", answer or "")
        if 1 <= int(value) <= len(chunks)
    }
    cited_bodies = [bodies[index - 1] for index in cited_indices]
    for sentence in _sentences(answer):
        refs = [int(value) for value in re.findall(r"\[(\d+)\]", sentence)]
        if not refs:
            continue
        text = re.sub(r"\[\d+\]", "", sentence)
        # A derived value is not expected to appear verbatim in the source.
        # Its operands are validated by the sentences that state them.
        if re.search(r"[=＝]|计算式|算式", text):
            continue
        tokens = UNIT_RE.findall(text) + MODEL_RE.findall(text)
        for token in tokens:
            normalized = normalize(token)
            if not any(normalized in body for body in cited_bodies):
                problems.append({"token": token, "sentence": text.strip()[:160]})
    return problems


def model_scope_mismatch(question: str, chunks: list[dict]) -> list[dict]:
    """A question about LT-R20 must not cite LT-R20 Pro / Plus / Max evidence."""
    models = re.findall(r"\b(?:LT|Lantu)[-A-Za-z0-9]*[A-Za-z0-9]+(?:[-A-Za-z0-9]+)*\b", question)
    if not models:
        return []
    problems = []
    for chunk in chunks:
        body = normalize(chunk_body(chunk))
        for model in models:
            suffix = r"\s*(?:Pro|Plus|Max|II|2\.0)"
            if re.search(re.escape(normalize(model)) + suffix, body) and not re.search(
                re.escape(normalize(model)) + suffix, normalize(question)
            ):
                problems.append({"model": model, "doc_id": chunk.get("doc_id"),
                                 "section": chunk.get("section")})
    return problems


def build_comparison_table(question: str, chunks: list[dict]) -> str:
    """Align same-family versions by explicit dimensions for the prompt."""
    families: dict[str, list[dict]] = {}
    for chunk in chunks:
        family = chunk.get("doc_family")
        if family:
            families.setdefault(family, []).append(chunk)
    pair = next((docs for docs in families.values()
                 if len({doc.get("doc_id") for doc in docs}) >= 2), None)
    if not pair:
        return ""
    docs: dict[str, dict] = {}
    for chunk in pair:
        doc_id = chunk.get("doc_id")
        docs.setdefault(doc_id, chunk)
    if len(docs) < 2:
        return ""
    ordered = sorted(
        docs.values(),
        key=lambda chunk: (str(chunk.get("effective_date") or "9999"), str(chunk.get("version") or "")),
    )
    older, newer = ordered[0], ordered[-1]
    rows = []
    for label, keywords in DIMENSIONS:
        values = []
        for doc in (older, newer):
            sentence = ""
            for chunk in pair:
                if chunk.get("doc_id") != doc.get("doc_id"):
                    continue
                for candidate in _sentences(chunk_body(chunk)):
                    if any(keyword in candidate for keyword in keywords):
                        sentence = candidate.strip()
                        break
                if sentence:
                    break
            values.append(sentence)
        if values[0] and values[1] and values[0] != values[1]:
            rows.append((label, values[0][:180], values[1][:180]))
    if not rows:
        return ""
    header = (f"| 维度 | 旧版 {older.get('version', '')} | 新版 {newer.get('version', '')} |\n"
              "| --- | --- | --- |")
    body = "\n".join(f"| {label} | {old} | {new} |" for label, old, new in rows)
    return header + "\n" + body


def enumeration_hint(question: str, chunks: list[dict]) -> str:
    if not re.search(r"几个|哪些|分别|包括|清单|全部|有哪", question):
        return ""
    lists = []
    for chunk in chunks:
        for sentence in _sentences(chunk_body(chunk)):
            if re.search(r"包括|设有|分为|下设|共有|组成", sentence) and len(sentence) <= 260:
                if sentence not in lists:
                    lists.append(sentence.strip())
    if not lists:
        return ""
    return ("资料中的清单条目必须逐项列出，不得只给数量；以下原文列出了条目，"
            "答案必须覆盖其中的全部项目：\n" + "\n".join(f"- {item}" for item in lists[:4]))
