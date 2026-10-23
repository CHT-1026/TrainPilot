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
from enterprise_rag.facts import evidence_periods, factual_clauses, named_subjects

_SENTENCE = re.compile(r"(?<=[。！？；;\n])")
_REF = re.compile(r"\[(\d+)\]")
_NEGATIVE = re.compile(r"不得|禁止|不支持|不能|无法|无需|尚未|没有")
_MODAL_PAIRS = (("支持", "不支持"), ("允许", "禁止"), ("可以", "不能"),
                ("启用", "禁用"), ("必须", "不得"))
_STOP = {
    "公司", "企业", "资料", "文档", "根据", "现有", "相关", "内容", "情况", "标准",
    "流程", "规定", "制度", "以上", "其中", "因此", "同时", "此外", "可以", "需要",
    "应当", "必须", "如果", "没有", "不是", "不得", "禁止", "以及", "或者", "并且",
}
_SUBSTANTIVE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_NAMED_SUBJECT = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{1,20}?(?:股份有限公司|有限公司|公司|集团|项目))"
)


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
                "period_missing": [], "unbound_facts": [],
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
        if not refs:
            unbound.append({"sentence": text[:160]})
            continue
        checked += 1
        cited_chunks = [chunks[index - 1] for index in refs]
        cited = [bodies[index - 1] for index in refs]
        cited_text = "\n".join(cited)
        years = sorted(evidence_periods(text))
        names = _named_subjects(text)
        scoped_parts = [(chunk, part) for chunk in cited_chunks for part in factual_clauses(chunk_body(chunk))
                        if not names or names <= _named_subjects(part + " " + str(chunk.get("title", "")))]
        period_parts = [(chunk, normalize(part)) for chunk, part in scoped_parts
                        if not years or set(years) <= evidence_periods(part, chunk)]
        derived = proved.get(tuple(refs), set()) | planned_quantities(question, cited_chunks)
        numeric_ok = True
        for token in UNIT_RE.findall(text) + MODEL_RE.findall(text):
            supported = any(token_supported(token, chunk, part) for chunk, part in period_parts)
            if supported:
                continue
            if quantity_key(token) in derived:
                continue
            numeric_ok = False
            numeric_missing.append({"token": token, "sentence": text[:160], "refs": refs})
        nouns = _nouns(text)
        missing_subjects = [word for word in nouns if not any(word.lower() in body for body in cited)]
        names_ok = not names or bool(scoped_parts)
        if not names_ok:
            missing_subjects = list(dict.fromkeys(missing_subjects + sorted(names)))
        if nouns and len(missing_subjects) == len(nouns):
            subject_missing.append({"subjects": missing_subjects[:4], "sentence": text[:160], "refs": refs})
        elif not names_ok:
            subject_missing.append({"subjects": sorted(names)[:4], "sentence": text[:160], "refs": refs})
        negation_ok: bool | None = True
        negation_text = _relevant_cited(text, [normalize(part) for _, part in scoped_parts])
        if _modal_conflict(normalize(text), negation_text):
            conflicts.append({"type": "modal_conflict", "sentence": text[:160], "refs": refs})
            negation_ok = False
        elif _NEGATIVE.search(text) and not _NEGATIVE.search(negation_text):
            suspects.append({"type": "negation_suspect", "sentence": text[:160], "refs": refs})
            negation_ok = None
        missing_years = [year for year in years if not any(year in evidence_periods(part, chunk) for chunk, part in scoped_parts)]
        if missing_years:
            period_missing.append({"years": missing_years, "sentence": text[:160], "refs": refs})
        sentences.append({
            "sentence": text[:240],
            "refs": refs,
            "numeric_ok": numeric_ok,
            "subject_ok": (not (nouns and len(missing_subjects) == len(nouns))) and names_ok,
            "period_ok": not missing_years,
            "negation_ok": negation_ok,
        })
    subject_total = sum(1 for item in sentences if item["subject_ok"])
    verified = bool(checked) and not numeric_missing and not subject_missing and not conflicts \
        and not period_missing and not unbound
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
        "unbound_facts": unbound,
        "sentences": sentences,
        "note": "结构引用正确不等于语义蕴含；本报告只覆盖可机械核对的数字、主体、模态与年份。",
    }
