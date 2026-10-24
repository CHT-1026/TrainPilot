"""句级事实核验（S26）：来源可定位 ≠ 答案事实已核验。

本模块对每个带引用的句子分别检查：
- 数字/单位/型号是否出现在该句引用的证据中；
- 句子主体是否能在引用证据中找到；
- 明确的对立模态是否与证据冲突（仅可证冲突硬拒答）；
- 句中年份是否逐个有证据支撑。

引用紧随句尾时（“事实。 [1]”）会先归位到该事实再切分，避免把事实与引用拆开。
"""
from __future__ import annotations

import re

from enterprise_rag.citations import chunk_body
from enterprise_rag.evidence import (
    MODEL_RE,
    UNIT_RE,
    answer_derived_quantities,
    _numeric_value,
    bind_trailing_refs,
    normalize,
    supported_derived_quantities,
    planned_quantities,
    quantity_key,
    token_supported,
)
from enterprise_rag.facts import evidence_periods, factual_clauses, named_subjects, join_quantity_wraps

_SENTENCE = re.compile(r"(?<=[。！？；;\n])")
_REF = re.compile(r"\[(\d+)\]")
_NEGATIVE = re.compile(r"不得|禁止|不支持|不能|无法|无需|尚未|没有")
_MODAL_PAIRS = (("支持", "不支持"), ("允许", "禁止"), ("可以", "不能"),
                ("启用", "禁用"), ("必须", "不得"))
_STOP = {
    "公司", "企业", "资料", "文档", "知识库", "根据", "现有", "相关", "内容", "情况", "标准",
    "流程", "规定", "制度", "以上", "其中", "因此", "同时", "此外", "可以", "需要",
    "应当", "必须", "如果", "没有", "不是", "不得", "禁止", "以及", "或者", "并且",
}
_SUBSTANTIVE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_NAMED_SUBJECT = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{1,20}?(?:股份有限公司|有限公司|公司|集团|项目))"
)

# A deliberately finite quantitative guard, not a general semantic entailment
# classifier. A current-year value alone cannot establish a comparative claim.
_TREND_METRICS = {
    "订单": r"(?:新签|新增)?订单(?:金额|数量|额|量)?",
    "出货量": r"(?:机器人)?出货(?:数量|台数|量)?",
    "营业收入": r"营业收入|销售收入|营收",
    "归母净利润": r"归母净利润|归属于母公司股东的净利润",
    "净利润": r"(?<!归母)净利润",
    "毛利率": r"毛利率",
    "研发费用率": r"研发费用率|研发投入占比",
    "研发投入": r"研发投入(?!占比)|研发费用(?!率)",
}
_TREND_METRIC_RE = re.compile("|".join(f"(?P<m{i}>{pattern})" for i, pattern in enumerate(_TREND_METRICS.values())))
_TREND_KEYS = tuple(_TREND_METRICS)
_TREND_DIRECTION = re.compile(r"增长|增加|增多|上升|提升|提高|下降|下滑|减少|降低")
_TREND_UP = {"增长", "增加", "增多", "上升", "提升", "提高"}
_TREND_UNKNOWN = re.compile(r"无法(?:判断|确认|推断|证明)|不能(?:判断|确认|推断|证明)|未能(?:确认|证明)|尚无|没有证据|缺乏证据|未提供|未说明|未知|不确定|不意味着|是否|能否")
_TREND_NEGATED = re.compile(r"(?:没有|尚未|并未|未|不|难以|并非)\s*$")
_TREND_FORECAST = re.compile(r"预计|预期|计划|目标|有望|将(?:会)?|未来")
_TREND_PERIOD = re.compile(r"上半年|半年度|下半年|全年|年度|第一季度|第二季度|第三季度|第四季度|[Qq][1-4]")
_TREND_PERIOD_KEYS = {"上半年":"H1", "半年度":"H1", "下半年":"H2", "全年":"FY", "年度":"FY",
                      "第一季度":"Q1", "第二季度":"Q2", "第三季度":"Q3", "第四季度":"Q4"}


def _trend_period(text: str, position: int, chunk: dict | None = None) -> tuple[int | None, str | None]:
    """Read statistical period from the claim/source, never update timestamps."""
    prefix = text[:position]
    years = list(re.finditer(r"(?<!\d)((?:19|20)\d{2})\s*年?", prefix))
    context = " ".join(str((chunk or {}).get(key) or "") for key in ("fact_period", "title", "section"))
    if years:
        year = int(years[-1].group(1))
        period_text = prefix[years[-1].start():]
    else:
        matches = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", context)
        year = int(matches[-1]) if matches else None
        period_text = prefix
    kinds = _TREND_PERIOD.findall(period_text) or _TREND_PERIOD.findall(context)
    kind = (_TREND_PERIOD_KEYS.get(kinds[-1], kinds[-1].upper()) if kinds else "FY" if year else None)
    return year, kind


def _quantitative_trends(text: str, chunk: dict | None = None) -> list[dict]:
    """Bind each direction to its immediately preceding finite metric."""
    metrics = list(_TREND_METRIC_RE.finditer(text))
    trends = []
    for index, metric in enumerate(metrics):
        end = metrics[index+1].start() if index+1 < len(metrics) else len(text)
        stop = re.search(r"[。！？；;\n]", text[metric.end():end])
        if stop:
            end = metric.end() + stop.start()
        after = text[metric.end():end]
        if re.match(r"(?:的)?(?:增长率|增速|增幅|下降率|降幅)", after):
            # A movement in the rate is a different metric from a movement
            # in its level. This finite guard does not infer one from the other.
            continue
        direction = _TREND_DIRECTION.search(after)
        if direction is None:
            continue
        # Keep a single bounded predicate; don't reinterpret an unrelated
        # capability object as a numerical movement ("研发投入提升技术能力").
        following = after[direction.end():]
        if direction.group() in {"提升", "提高", "降低"} and re.match(r"(?:技术|研发|产品|服务)?(?:能力|水平|质量|效率|风险|成本压力)", following):
            continue
        local_start = max(text.rfind(mark, 0, metric.start()) for mark in ("。", "；", ";", "\n", "，", ",")) + 1
        lead = text[local_start:metric.start()] + after[:direction.start()]
        if (_TREND_UNKNOWN.search(lead) or _TREND_NEGATED.search(lead)
                or re.search(r"是否|能否", after[:direction.start()])
                or re.match(r"(?:趋势)?(?:尚未|未能|无法)(?:确认|证实|证明|判断)|(?:趋势)?尚不明确", following)):
            continue
        basis = ("qoq" if "环比" in lead else "yoy" if re.search(r"同比|(?:较|比)(?:上年|去年|上年度)", lead) else "")
        trends.append({"metric":_TREND_KEYS[int(metric.lastgroup[1:])],
                       "direction":"up" if direction.group() in _TREND_UP else "down",
                       "period":_trend_period(text, metric.end(), chunk),
                       "basis":basis,
                       "forecast":bool(_TREND_FORECAST.search(lead)),
                       "start":metric.start(), "end":metric.end()+direction.end()})
    return trends


def _trend_values(chunk: dict, metric_key: str, names: set[str]) -> list[dict]:
    body = join_quantity_wraps(chunk_body(chunk))
    metrics = list(_TREND_METRIC_RE.finditer(body))
    values = []
    for index, metric in enumerate(metrics):
        if _TREND_KEYS[int(metric.lastgroup[1:])] != metric_key:
            continue
        start = max(body.rfind(mark, 0, metric.start()) for mark in ("。", "；", ";", "\n")) + 1
        end = metrics[index+1].start() if index+1 < len(metrics) else len(body)
        stop = re.search(r"[。！？；;\n]", body[metric.end():end])
        if stop:
            end = metric.end() + stop.start()
        part = body[start:end]
        if names and not names <= named_subjects(part + " " + str(chunk.get("title", ""))):
            continue
        token = UNIT_RE.search(body[metric.end():end])
        if token is None:
            continue
        quantity = quantity_key(token.group())
        if quantity is None or (quantity[1] == "%" and metric_key not in {"毛利率", "研发费用率"}):
            continue
        values.append({"value":quantity, "period":_trend_period(body, metric.end(), chunk),
                       "forecast":bool(_TREND_FORECAST.search(part[:part.find(metric.group())]))})
    return values


def unsupported_quantitative_trends(answer: str, chunks: list[dict]) -> list[dict]:
    """Return unsupported *finite quantitative* trends using each claim's refs.

    Explicit matching source predicates or two comparable dated quantities can
    prove direction. Half-year/quarter and annual observations never substitute
    for each other. Inference labels do not excuse an invented factual premise.
    Unknown/negated trends and qualitative capability improvements are outside
    this positive-comparative gate. This does not claim general NLI coverage.
    """
    missing = []
    bound = bind_trailing_refs(answer)
    for sentence_match in re.finditer(r"[^。！？；;\n]+[。！？；;\n]?", bound):
        sentence = sentence_match.group()
        refs = sorted({int(n) for n in _REF.findall(sentence) if 1 <= int(n) <= len(chunks)})
        text = _REF.sub("", sentence)
        names = named_subjects(text)
        for claim in _quantitative_trends(text):
            # Inherit explicit "2025年" from earlier answer sentences, e.g.
            # a final "同年...增长" paragraph referring to that year.
            claim["period"] = _trend_period(bound, sentence_match.start()+claim["start"])
            cited = [chunks[i-1] for i in refs]
            evidence = []
            for chunk in cited:
                body = join_quantity_wraps(chunk_body(chunk))
                for supported in _quantitative_trends(body, chunk):
                    if supported["metric"] != claim["metric"]:
                        continue
                    if names and not names <= named_subjects(body + " " + str(chunk.get("title", ""))):
                        continue
                    year, kind = claim["period"]
                    sy, sk = supported["period"]
                    if ((year is None or sy == year) and (kind is None or sk == kind)
                            and supported["forecast"] == claim["forecast"]
                            and (not claim["basis"] or supported["basis"] == claim["basis"])):
                        evidence.append(supported["direction"])
            supported = bool(evidence) and set(evidence) == {claim["direction"]}
            if not supported:
                observations = [value for chunk in cited for value in _trend_values(chunk, claim["metric"], names)
                                if value["forecast"] == claim["forecast"]]
                year, kind = claim["period"]
                observations = [v for v in observations if v["period"][0] is not None
                                and (kind is None or v["period"][1] == kind)]
                unique = {(v["period"], v["value"]) for v in observations}
                if len(unique) == 2:
                    older, newer = sorted(unique, key=lambda v:v[0][0])
                    same_basis = (older[0][0] < newer[0][0] and older[0][1] == newer[0][1]
                                  and older[1][1] == newer[1][1] and (year is None or newer[0][0] == year))
                    if "同比" in text:
                        same_basis = same_basis and newer[0][0] == older[0][0]+1
                    if claim["basis"] == "qoq":
                        # This bounded arithmetic branch handles year pairs only.
                        same_basis = False
                    delta = newer[1][0] - older[1][0]
                    supported = same_basis and (delta > 0 if claim["direction"] == "up" else delta < 0)
            if not supported:
                missing.append({"metric":claim["metric"], "direction":claim["direction"],
                                "period":list(claim["period"]), "sentence":text.strip()[:240],
                                "refs":refs, "reason":"unsupported_quantitative_trend"})
    return missing


def _named_subjects(text: str) -> set[str]:
    return named_subjects(text)


def _relevant_cited(text: str, cited: list[str]) -> str:
    nouns = _nouns(text)
    if not nouns:
        return "\n".join(cited)
    anchor = nouns[0]
    parts = [
        sentence
        for body in cited
        for sentence in re.split(r"(?<=[。！？；;\n])", body)
        if anchor in sentence
    ]
    return "".join(parts) or "\n".join(cited)


def _sentences(answer: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(bind_trailing_refs(answer)) if part.strip()]


def _nouns(text: str) -> list[str]:
    try:
        import jieba.posseg as pseg

        words = []
        for word, flag in pseg.cut(text):
            word = word.strip()
            if len(word) < 2 or word in _STOP:
                continue
            if flag.startswith("n") or flag in {"eng", "nz", "vn"}:
                words.append(word)
        return list(dict.fromkeys(words))
    except Exception:
        return [word for word in re.findall(r"[\u4e00-\u9fff]{2,6}", text) if word not in _STOP][:8]


def _modal_conflict(text: str, cited_text: str) -> bool:
    """Only explicit opposite polarity for the same action is a hard conflict."""
    for positive, negative in _MODAL_PAIRS:
        claim_negative = negative in text
        marker = negative if claim_negative else positive
        if marker not in text:
            continue
        action = text.split(marker, 1)[1].strip(" 。；;！？")
        if not action:
            continue
        expected_opposite = (positive if claim_negative else negative) + action
        # “支持 X” is also a substring of “不支持 X”; remove the negative
        # form before checking whether positive evidence really exists.
        positive_evidence = cited_text.replace(negative + action, "")
        has_expected = (positive + action in positive_evidence) if claim_negative else (expected_opposite in cited_text)
        has_claim = (negative + action in cited_text) if claim_negative else (positive + action in positive_evidence)
        if has_expected and not has_claim:
            return True
    return False


_RANGE_UNIT = r"ms|kHz|Hz|位|轴|小时|℃|级|mm|kg|%|亿元|万元|元|台|人|天|分钟|秒|MPa|N·m|个"
_QUANTITY_RANGE = re.compile(
    rf"(?<![A-Za-z0-9_.-])(?P<low>[-−]?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{_RANGE_UNIT})?"
    rf"\s*[-–—]\s*(?P<high>[-−]?\d[\d,]*(?:\.\d+)?)\s*(?P<endunit>{_RANGE_UNIT})"
)


def _quantity_ranges(text: str) -> str:
    """Separate range punctuation from signs, retaining each actual endpoint."""
    return _QUANTITY_RANGE.sub(
        lambda match: (match["low"] + (match["unit"] or match["endunit"])
                       + "至" + match["high"] + match["endunit"]), text)


def _fact_bindings(text: str, initial_names: set[str] | None = None) -> list[tuple[str, set[str]]]:
    """Carry an explicit subject into its continuation, not across a new name."""
    names = initial_names or set()
    bindings = []
    for part in factual_clauses(text):
        explicit = _named_subjects(part)
        if explicit:
            names = explicit
        bindings.append((part, names))
    return bindings


def _token_in_fact(token: str, chunk: dict, part: str) -> bool:
    # normalize() removes enumeration punctuation, which would turn
    # "LT-R20、2台" into "LT-R202台". Keep real token boundaries here.
    part = _quantity_ranges(part)
    if MODEL_RE.fullmatch(token):
        identity = re.sub(r"\s+", "", str(chunk.get("title", "")) + " " + part).lower()
        return bool(re.search(r"(?<![a-z0-9_.-])" + re.escape(normalize(token))
                              + r"(?![a-z0-9_.-])", identity))
    part = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", part)
    body = normalize(re.sub(r"[,，、]", "；", part))
    return token_supported(token, chunk, body)


_PRODUCTION_METRIC = re.compile(r"出货(?:量|数量|台数)?|交付(?:量|数量|台数)?|下线(?:量|数量|台数)?|生产(?!线|基地|部门)(?:数量|台数|量)?|产量")
_COUNT_LINK_BREAK = re.compile(r"[，,、。！？；;\n]|以及|同时|并且|及|和|与|并")


def _production_quantity_bindings(text: str) -> dict[int, set[str]]:
    """Bind finite count metrics to their adjacent quantity, in either order.

    A metric anywhere in a fragment cannot support another metric's count.
    Metric and quantity spans retain that association across connector changes.
    """
    metrics = list(_PRODUCTION_METRIC.finditer(text))
    quantities = [hit for hit in UNIT_RE.finditer(text)
                  if (key := quantity_key(hit.group())) is not None and key[1] == "台"]
    bindings: dict[int, set[str]] = {}
    for index, metric in enumerate(metrics):
        left_bound = metrics[index - 1].end() if index else 0
        right_bound = metrics[index + 1].start() if index + 1 < len(metrics) else len(text)
        left = [hit for hit in quantities if left_bound <= hit.start() and hit.end() <= metric.start()]
        right = [hit for hit in quantities if metric.end() <= hit.start() and hit.end() <= right_bound]
        candidates = []
        for hit, gap in ((left[-1], text[left[-1].end():metric.start()]) if left else (None, ""),
                         (right[0], text[metric.end():right[0].start()]) if right else (None, "")):
            if hit is not None and not _COUNT_LINK_BREAK.search(gap):
                candidates.append((len(gap.strip()), hit))
        if not candidates:
            continue
        _, quantity = min(candidates, key=lambda item: item[0])
        name = "生产" if metric.group().startswith(("生产", "产量")) else metric.group()[:2]
        bindings.setdefault(quantity.start(), set()).add(name)
    for previous, current in zip(quantities, quantities[1:]):
        if text[previous.end():current.start()].strip() == "至" and previous.start() in bindings:
            bindings.setdefault(current.start(), set()).update(bindings[previous.start()])
    return bindings


def _production_quantity_supported(token: str, metrics: set[str], chunk: dict, part: str) -> bool:
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", _quantity_ranges(part))
    bindings = _production_quantity_bindings(text)
    target = quantity_key(token)
    if any(quantity_key(hit.group()) == target and metrics <= bindings.get(hit.start(), set())
           for hit in UNIT_RE.finditer(text)):
        return True
    if chunk.get("chunk_type") != "table":
        return False
    # Bare cells inherit their own column's metric and unit, not the whole table.
    lines = [line for line in chunk_body(chunk).splitlines() if line.strip().startswith("|")]
    if not lines:
        return False
    headers = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
    for line in part.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        for header, cell in zip(headers, cells):
            metric = _PRODUCTION_METRIC.search(header)
            unit = "万台" if "万台" in header else "台" if "台" in header else ""
            if not metric or not unit or not re.fullmatch(r"[-−]?\d[\d,]*(?:\.\d+)?", cell):
                continue
            name = "生产" if metric.group().startswith(("生产", "产量")) else metric.group()[:2]
            if metrics <= {name} and quantity_key(cell + unit) == target:
                return True
    return False


def verify_answer_facts(question: str, answer: str, chunks: list[dict]) -> dict:
    # Reproduce a bounded operator from these exact legal sources, not from a
    # free-standing model equation. Exact rendered equality excludes extra claims.
    from enterprise_rag.reasoning import infer
    proof = infer(question, chunks)
    def claim_text(value):
        return normalize(_REF.sub("", value or ""))
    if proof and not proof["refused"] and claim_text(proof["answer"]) == claim_text(answer):
        parts = [part for part in _sentences(answer) if _SUBSTANTIVE.search(_REF.sub("", part))]
        return {"checked_sentences": len(parts), "unbound_sentences": 0, "verified": True, "coverage": 1.0,
                "numeric_missing": [], "subject_missing": [], "conflicts": [], "negation_suspects": [],
                "period_missing": [], "unbound_facts": [], "trend_missing": [],
                "sentences": [{"sentence": _REF.sub("", part), "refs": [int(n) for n in _REF.findall(part)],
                               "numeric_ok": True, "subject_ok": True, "period_ok": True, "negation_ok": True} for part in parts],
                "verification_kind": "deterministic_derivation", "operation": proof["operation"],
                "source_kind": "question" if proof["operation"] == "year_arithmetic" else "retrieved_evidence",
                "note": "计算由同一有界算子和实际引用证据重新求值，未把模型自洽算式视为证明。"}
    bodies = [normalize(chunk_body(chunk)) for chunk in chunks]
    sentences = []
    numeric_missing: list[dict] = []
    subject_missing: list[dict] = []
    conflicts: list[dict] = []
    period_missing: list[dict] = []
    unbound: list[dict] = []
    suspects: list[dict] = []
    checked = 0
    proved = answer_derived_quantities(answer, chunks)
    for sentence in _sentences(answer):
        refs = sorted({int(value) for value in _REF.findall(sentence) if 1 <= int(value) <= len(chunks)})
        text = _REF.sub("", sentence).strip()
        if not _SUBSTANTIVE.search(text):
            continue
        if text.endswith((":", "：")) and len(text) <= 30:
            continue
        if not refs:
            unbound.append({"sentence": text[:160]})
            continue
        checked += 1
        cited_chunks = [chunks[index - 1] for index in refs]
        cited = [bodies[index - 1] for index in refs]
        cited_text = "\n".join(cited)
        # A single quoted sentence may contain independent founding/listing
        # years or an undated headcount before a later dated R&D fact. Bind
        # each atomic fact rather than demanding all years on every value.
        source_parts = [(chunk, part, names) for chunk in cited_chunks
                        for part, names in _fact_bindings(chunk_body(chunk), _named_subjects(chunk.get("title", "")))]
        scoped_parts = []
        names_ok = True
        missing_names = set()
        missing_years = set()
        derived = proved.get(tuple(refs), set()) | planned_quantities(question, cited_chunks)
        numeric_ok = True
        for claim, names in _fact_bindings(text):
            relevant = [(chunk, part) for chunk, part, source_names in source_parts
                        if not names or names <= source_names]
            scoped_parts.extend(relevant)
            if names and not relevant:
                names_ok = False
                missing_names.update(names)
            years = evidence_periods(claim)
            period_parts = [(chunk, part) for chunk, part in relevant
                            if not years or years <= evidence_periods(part, chunk)]
            quantities = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", _quantity_ranges(claim))
            count_bindings = _production_quantity_bindings(quantities)
            for match in list(UNIT_RE.finditer(quantities)) + list(MODEL_RE.finditer(quantities)):
                token = match.group()
                key = quantity_key(token)
                separators = ("，", ",", "；", ";", "、", "及")
                start = max(quantities.rfind(mark, 0, match.start()) for mark in separators) + 1
                ends = [position for mark in separators if (position := quantities.find(mark, match.end())) >= 0]
                atom = quantities[start:min(ends) if ends else len(quantities)]
                production_metric = count_bindings.get(match.start(), set()) if key and key[1] == "台" else set()
                unbound_metric = key and key[1] == "台" and _PRODUCTION_METRIC.search(atom) and not production_metric
                if unbound_metric:
                    numeric_ok = False
                    numeric_missing.append({"token": token, "sentence": text[:160], "refs": refs})
                    continue
                token_parts = period_parts
                if production_metric:
                    if any(_production_quantity_supported(token, production_metric, chunk, part)
                           for chunk, part in period_parts):
                        continue
                    token_parts = []
                if any(_token_in_fact(token, chunk, part) for chunk, part in token_parts):
                    continue
                if key in derived and not production_metric:
                    continue
                numeric_ok = False
                numeric_missing.append({"token": token, "sentence": text[:160], "refs": refs})
            missing_years.update(year for year in years if not any(
                year in evidence_periods(part, chunk) for chunk, part in relevant))
        nouns = _nouns(text)
        missing_subjects = [word for word in nouns if not any(word.lower() in body for body in cited)]
        if not names_ok:
            missing_subjects = list(dict.fromkeys(missing_subjects + sorted(missing_names)))
        if nouns and len(missing_subjects) == len(nouns):
            subject_missing.append({"subjects": missing_subjects[:4], "sentence": text[:160], "refs": refs})
        elif not names_ok:
            subject_missing.append({"subjects": sorted(missing_names)[:4], "sentence": text[:160], "refs": refs})
        negation_ok: bool | None = True
        negation_text = _relevant_cited(text, [normalize(part) for _, part in scoped_parts])
        if _modal_conflict(normalize(text), negation_text):
            conflicts.append({"type": "modal_conflict", "sentence": text[:160], "refs": refs})
            negation_ok = False
        elif _NEGATIVE.search(text) and not _NEGATIVE.search(negation_text):
            suspects.append({"type": "negation_suspect", "sentence": text[:160], "refs": refs})
            negation_ok = None
        if missing_years:
            period_missing.append({"years": sorted(missing_years), "sentence": text[:160], "refs": refs})
        sentences.append({
            "sentence": text[:240],
            "refs": refs,
            "numeric_ok": numeric_ok,
            "subject_ok": (not (nouns and len(missing_subjects) == len(nouns))) and names_ok,
            "period_ok": not missing_years,
            "negation_ok": negation_ok,
        })
    subject_total = sum(1 for item in sentences if item["subject_ok"])
    trend_missing = unsupported_quantitative_trends(answer, chunks)
    verified = bool(checked) and not numeric_missing and not subject_missing and not conflicts \
        and not period_missing and not unbound and not trend_missing
    denominator = checked + len(unbound)
    return {
        "checked_sentences": checked,
        "unbound_sentences": len(unbound),
        "verified": verified,
        "coverage": round(subject_total / denominator, 4) if denominator else 0.0,
        "numeric_missing": numeric_missing,
        "subject_missing": subject_missing,
        "conflicts": conflicts,
        "negation_suspects": suspects,
        "period_missing": period_missing,
        "trend_missing": trend_missing,
        "unbound_facts": unbound,
        "sentences": sentences,
        "note": "结构引用正确不等于语义蕴含；本报告只覆盖可机械核对的数字、主体、模态、年份与有限数量指标趋势。",
    }
