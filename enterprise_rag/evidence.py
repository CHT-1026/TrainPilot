"""证据约束：数字/型号校验、版本对比对齐、枚举完整性提示。"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from enterprise_rag.citations import chunk_body

UNIT_RE = re.compile(
    r"(?<![\d.])[-−]?\d[\d,\.]*\s*(?:ms|kHz|Hz|位|轴|小时|℃|级|mm|kg|%|亿元|万元|元|台|人|天|分钟|秒|MPa|N·m|个)"
)
MODEL_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:LT|Lantu)[-A-Za-z0-9]*[A-Za-z0-9]+"
    r"(?:\.\d+)*(?:\s+(?:Pro|Plus|Max|II))?(?:\s+\d+(?:\.\d+)+)?(?![A-Za-z0-9_.-])"
)
SENTENCE_SPLIT = re.compile(r"(?<=[。！？；\n])")
_TRAILING_REF = re.compile(r"([。！？；;])\s*((?:\[\d+\]\s*)+)")
_MONEY_SCALE = {"亿元": Decimal(100_000_000), "万元": Decimal(10_000), "元": Decimal(1)}
_MONEY_TOKEN = re.compile(r"(?<![\d.])([-−]?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|元)")
_NUMBER = re.compile(r"([-−]?\d[\d,]*(?:\.\d+)?)\s*(.*)")
_HEADER_UNITS = {
    "人": ("员工数", "人数", "人员"),
    "元": ("金额", "预算", "费用", "收入"),
    "天": ("天数", "假期"),
    "台": ("台数", "数量"),
}


def bind_trailing_refs(answer: str) -> str:
    """把紧随句尾的引用归位到该事实句，避免事实与引用被切到两段。"""
    return _TRAILING_REF.sub(lambda match: match.group(2).strip() + match.group(1), answer or "")

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
    return [part.strip() for part in SENTENCE_SPLIT.split(bind_trailing_refs(answer)) if part.strip()]


_EXPR = re.compile(
    r"(?<![\d.])(\d[\d,]*(?:\.\d+)?(?:\s*[+\-*/×÷x]\s*\d[\d,]*(?:\.\d+)?)+)"
    r"\s*[=＝]\s*(\d[\d,]*(?:\.\d+)?)"
)


def _derived_numbers(answer: str) -> set[float]:
    """Return verified arithmetic results explicitly shown in the answer."""
    values: set[float] = set()
    for match in _EXPR.finditer(answer or ""):
        expression = match.group(1).replace(",", "").replace("×", "*").replace("÷", "/")
        expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression)
        if not re.fullmatch(r"[\d+\-*/().\s]+", expression):
            continue
        try:
            value = eval(expression, {"__builtins__": {}}, {})
            expected = float(match.group(2).replace(",", ""))
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
            continue
        if abs(float(value) - expected) <= 1e-6:
            values.add(expected)
    # “相差 6.83 亿元”这类差值题不一定写出等式；同单位两个数之差可视为派生值。
    grouped: dict[str, list[float]] = {}
    for number, unit in _MONEY_TOKEN.findall(answer or ""):
        try:
            grouped.setdefault(unit, []).append(float(number.replace(",", "")))
        except ValueError:
            continue
    for numbers in grouped.values():
        for left in numbers:
            for right in numbers:
                if left != right:
                    values.add(round(abs(left - right), 6))
    return values


def quantity_key(token: str) -> tuple[Decimal, str] | None:
    match = _NUMBER.fullmatch(token.strip())
    if not match:
        return None
    value, unit = match.groups()
    try:
        value = Decimal(value.replace(",", "").replace("−", "-"))
    except InvalidOperation:
        return None
    return (value * _MONEY_SCALE[unit], "元") if unit in _MONEY_SCALE else (value, unit.lower())


def planned_quantities(question: str, chunks: list[dict]) -> set[tuple[Decimal, str]]:
    """Only a named, evidence-bound operator may supply a business derived value."""
    if not question:
        return set()
    from enterprise_rag.reasoning import infer
    result = infer(question, chunks)
    if not result or result["refused"] or result["operation"] not in {
        "sum", "subtract", "ratio", "growth_rate", "date_difference", "year_difference", "year_subtract"
    }:
        return set()
    return {key for token in UNIT_RE.findall(result["answer"]) if (key := quantity_key(token)) is not None}


def supported_derived_quantities(answer: str, chunks: list[dict]) -> set[tuple[Decimal, str]]:
    """Explicit same-unit arithmetic only; no arbitrary pairwise ratio whitelist.

    Business ratios, unit conversion and calendar arithmetic are verified against
    the bounded operator plan, rather than guessing operands from a pool of numbers.
    """
    bodies = [normalize(chunk_body(chunk)) for chunk in chunks]

    def supported(token: str) -> bool:
        return any(token_supported(token, chunk, body) for chunk, body in zip(chunks, bodies))

    values: set[tuple[Decimal, str]] = set()
    for match in _EXPR.finditer(answer or ""):
        quantity = UNIT_RE.match(answer, match.start(2))
        if quantity is None:
            continue
        unit = _NUMBER.match(quantity.group(0)).group(2).strip()
        if unit == "%":
            continue
        expression = match.group(1).replace(",", "").replace("×", "*").replace("÷", "/")
        expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression)
        if "*" in expression or "/" in expression:
            # Same-unit operands alone cannot prove the dimension of a product.
            continue
        if not re.fullmatch(r"[\d+\-*/().\s]+", expression):
            continue
        operands = re.findall(r"\d[\d,]*(?:\.\d+)?", expression)
        if not operands or not all(supported(operand + unit) for operand in operands):
            continue
        try:
            value = eval(expression, {"__builtins__": {}}, {})
            expected = float(match.group(2).replace(",", ""))
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
            continue
        if abs(float(value) - expected) <= 1e-6:
            key = quantity_key(quantity.group(0))
            if key is not None:
                values.add(key)
    return values


def supported_derived_numbers(answer: str, chunks: list[dict]) -> set[float]:
    """Compatibility view; validation uses the unit-preserving quantity keys."""
    return {float(value) for value, _ in supported_derived_quantities(answer, chunks)}


def answer_derived_quantities(answer: str, chunks: list[dict]) -> dict[tuple[int, ...], set[tuple[Decimal, str]]]:
    """Reuse a proved equation only for the identical cited source set and unit."""
    proofs: dict[tuple[int, ...], set[tuple[Decimal, str]]] = {}
    for sentence in _sentences(answer):
        refs = tuple(sorted({int(value) for value in re.findall(r"\[(\d+)\]", sentence)
                             if 1 <= int(value) <= len(chunks)}))
        if refs:
            proofs.setdefault(refs, set()).update(supported_derived_quantities(
                re.sub(r"\[\d+\]", "", sentence), [chunks[index - 1] for index in refs]))
    return proofs


def _numeric_value(token: str) -> float | None:
    match = re.search(r"[-−]?\d[\d,]*(?:\.\d+)?", token or "")
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "").replace("−", "-"))
    except ValueError:
        return None


def _to_money(number: str, unit: str) -> Decimal | None:
    scale = _MONEY_SCALE.get(unit)
    if scale is None:
        return None
    try:
        return Decimal(number.replace(",", "").replace("−", "-")) * scale
    except InvalidOperation:
        return None


def _table_header(chunk: dict) -> str:
    lines = [line for line in chunk_body(chunk).splitlines() if line.strip()]
    return lines[0] if lines else ""


def _header_supports(header: str, unit: str) -> bool:
    if normalize(unit) in normalize(header):
        return True
    return any(alias in header for alias in _HEADER_UNITS.get(unit, ()))


def token_supported(token: str, chunk: dict, body: str | None = None) -> bool:
    """Check an exact number/unit. A table cell may omit the unit in its header."""
    raw_body = chunk_body(chunk)
    body = normalize(raw_body) if body is None else body
    # A dash after an explicit quantity unit separates range endpoints; it is
    # not the sign of the second endpoint (13%-14% vs a standalone -14%).
    body = re.sub(r"(\d(?:%|℃|kg|mm|ms|万元|亿元|元|台|人|天))[-–—](?=\d)", r"\1 ", body)
    normalized = normalize(token)
    if normalized and not _NUMBER.match(token):
        # The document title can name the product family while the application
        # paragraph uses pronouns. Only identity tokens inherit the title;
        # quantities/parameters below still require body/header evidence.
        identity_body = normalize(str(chunk.get("title", "")) + " " + raw_body)
        return bool(re.search(r"(?<![a-z0-9_.-])" + re.escape(normalized) + r"(?![a-z0-9_.-])", identity_body))
    match = _NUMBER.match(token or "")
    if not match:
        return False
    number, unit = match.group(1), match.group(2).strip()
    raw_norm = normalize(number)
    if not raw_norm:
        return False
    target = _to_money(number, unit)
    if target is not None:
        for hit in _MONEY_TOKEN.finditer(body):
            if _to_money(hit.group(1), hit.group(2)) == target:
                return True
        if chunk.get("chunk_type") == "table":
            header = _table_header(chunk)
            header_unit = next((item for item in ("亿元", "万元", "元") if item in header), "")
            scale = _MONEY_SCALE.get(header_unit)
            if scale is not None:
                for hit in re.finditer(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)(?![\d.])", body):
                    try:
                        if Decimal(hit.group(1).replace(",", "")) * scale == target:
                            return True
                    except InvalidOperation:
                        continue
        return False
    target_key = quantity_key(token)
    if target_key is not None and any(quantity_key(candidate) == target_key for candidate in UNIT_RE.findall(body)):
        return True
    if not re.search(rf"(?<![\d.\-−]){re.escape(raw_norm)}(?![\d.])", body):
        return False
    if not unit:
        return True
    if re.search(rf"(?<![\d.\-−]){re.escape(raw_norm)}\s*{re.escape(normalize(unit))}(?![a-zA-Z])", body):
        return True
    return chunk.get("chunk_type") == "table" and _header_supports(_table_header(chunk), unit)


def unsupported_numeric_tokens(answer: str, chunks: list[dict], question: str = "") -> list[dict]:
    """Return claims whose numeric/model token is absent from that sentence's references."""
    problems = []
    bodies = [normalize(chunk_body(chunk)) for chunk in chunks]
    proved = answer_derived_quantities(answer, chunks)
    for sentence in _sentences(answer):
        refs = sorted({int(value) for value in re.findall(r"\[(\d+)\]", sentence)
                       if 1 <= int(value) <= len(chunks)})
        if not refs:
            continue
        text = re.sub(r"\[\d+\]", "", sentence)
        cited_chunks = [chunks[index - 1] for index in refs]
        cited_bodies = [bodies[index - 1] for index in refs]
        derived = proved.get(tuple(refs), set()) | planned_quantities(question, cited_chunks)
        for token in UNIT_RE.findall(text) + MODEL_RE.findall(text):
            if any(token_supported(token, chunk, body)
                   for chunk, body in zip(cited_chunks, cited_bodies)):
                continue
            if quantity_key(token) in derived:
                continue
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


_CJK = re.compile(r"[\u4e00-\u9fff]")


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[\s·、，,。.:：;；()（）\-—_《》\"]", "", text or "")
    compact = re.sub(r"澜图智能|澜图|20\d{2}\s*版?|v\d+(?:\.\d+)?", "", compact)
    return {compact[i:i + 2] for i in range(len(compact) - 1)} - {"公司", "文档", "文件", "报告", "版本"}


def _topic_terms(text: str) -> set[str]:
    return {term for term in _bigrams(text) if all(_CJK.match(char) for char in term)}


def version_pair(chunks: list[dict], question: str = "") -> list[dict] | None:
    """选出与问题主题一致的旧/新文档族，避免拿无关家族的维度表作答。"""
    families: dict[str, list[dict]] = {}
    for chunk in chunks:
        family = chunk.get("doc_family")
        if family:
            families.setdefault(family, []).append(chunk)
    question_terms = _topic_terms(question)
    candidates = []
    for items in families.values():
        if len({item.get("doc_id") for item in items}) < 2:
            continue
        top = max(items, key=lambda item: float(item.get("score") or 0.0))
        title_terms = _topic_terms(str(top.get("title") or ""))
        score = float(top.get("score") or 0.0)
        required = 1 if len(title_terms) <= 4 else 2
        matched = question_terms and len(title_terms & question_terms) >= required
        candidates.append((bool(matched), score, title_terms, items))
    if not candidates:
        return None
    matched = [item for item in candidates if item[0]]
    if matched:
        return max(matched, key=lambda item: item[1])[3]
    # 标题都是“文档 d1”这类无语义占位时，只能按检索分兜底。
    if not any(item[2] for item in candidates):
        return max(candidates, key=lambda item: item[1])[3]
    return None


def build_comparison_table(question: str, chunks: list[dict]) -> str:
    """Align same-family versions by explicit dimensions for the prompt."""
    pair = version_pair(chunks, question)
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
    cue = re.search(r"几个|哪些|分别|包括|清单|全部|有哪|列出|列举|逐项", question)
    if not cue:
        return ""
    # Scope an enumeration to its requested topic. “有哪些联系” must not pull
    # an unrelated institute list from a retrieved research paragraph.
    focus = question[max(0, cue.start() - 12):cue.end() + 16]
    focus_terms = _topic_terms(focus)
    lists = []
    for chunk in chunks:
        for sentence in _sentences(chunk_body(chunk)):
            marker = re.search(r"包括|设有|分为|下设|共有|组成", sentence)
            if marker and len(sentence) <= 260:
                heading = sentence[:marker.end() + 12]
                overlap = focus_terms & _topic_terms(heading)
                if len(overlap) < min(2, len(focus_terms)) or not overlap:
                    continue
                if sentence not in lists:
                    lists.append(sentence.strip())
    if not lists:
        return ""
    return ("以下清单与问题的枚举主题相关，只在本次可见资料范围内核对所问条目；"
            "不要扩展为未被问到的清单，也不要声称资料外的全公司穷尽性：\n"
            + "\n".join(f"- {item}" for item in lists[:4]))
