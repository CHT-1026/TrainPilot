"""有出处的类型化事实（S17）。

从合法证据块抽取“主体-指标-值-单位-时点”并保留 chunk/句偏移，
用于数值/时间推断和答案事实核验。这里只做可复核的浅层抽取，
不把模型生成的解释当成事实。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from enterprise_rag.citations import chunk_body

NUMBER_UNIT = re.compile(
    r"(?<![\d.-])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(亿元|万元|元/晚|元/天|个工作日|个自然日|ms|kHz|Hz|MPa|N·m|kg|mm|m/s|%|元|台|人|天|小时|分钟|秒|个月|年|月|日|℃)"
)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)\s*年?")
CLAUSE = re.compile(r"(?<=[。！？；;\n])")
_CELL_SPLIT = re.compile(r"\s*\|\s*")


def evidence_periods(text: str, chunk: dict | None = None) -> set[str]:
    """Statistical period belongs to the fact/title, never the update timestamp."""
    years = set(re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)(?=\s*(?:年|版|[-/]\d))", text or ""))
    if not years and chunk:
        context = " ".join(str(chunk.get(key) or "") for key in ("fact_period", "title", "section"))
        years = set(re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", context))
    return years


def named_subjects(text: str) -> set[str]:
    names = re.findall(r"([\u4e00-\u9fffA-Za-z0-9]{1,24}?(?:股份有限公司|有限公司|公司|集团|项目))", text or "")
    return {re.sub(r"^(?:(?:19|20)\d{2}\s*年|根据|其中|因此|以及|和|与)+", "", name)
            for name in names} - {"公司", "集团", "项目", "本公司", "该公司", "本项目", "该项目",
                                     "说明公司", "介绍公司", "了解公司", "分析公司", "我们公司", "你们公司",
                                     "这家公司", "那家公司", "什么公司", "哪家公司"}


def factual_clauses(text: str) -> list[str]:
    """Keep comma-separated thousands, split when a new named fact/period starts."""
    return [part.strip() for part in re.split(
        r"(?<=[。！？；;\n])|[，,](?=\s*(?:(?:19|20)\d{2}\s*年|[^，。；]{1,30}(?:公司|项目)))",
        text or "") if part.strip()]


@dataclass(frozen=True)
class TypedFact:
    subject: str
    predicate: str
    value: str
    unit: str
    period: str
    doc_id: str
    title: str
    chunk_id: str
    chunk_type: str
    sentence_index: int
    offset: int
    sentence: str
    citation_index: int = 0
    extra: dict = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "unit": self.unit,
            "period": self.period,
            "doc_id": self.doc_id,
            "title": self.title,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "sentence_index": self.sentence_index,
            "offset": self.offset,
            "sentence": self.sentence[:240],
            "citation_index": self.citation_index,
        }


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in CLAUSE.split(text or "") if part.strip()]


def _subject_of(sentence: str, start: int) -> str:
    prefix = sentence[:start]
    prefix = re.split(r"[，,：:；;。]", prefix)[-1]
    prefix = re.sub(r"^(?:其中|同时|此外|另外|目前|当前)\s*", "", prefix.strip())
    prefix = re.sub(r"(?:为|是|达到|达|有|共|约|不超过|不低于|在|于|占|较)$", "", prefix).strip()
    if len(prefix) > 30:
        prefix = prefix[-30:]
    return prefix


def _predicate_of(sentence: str, start: int) -> str:
    prefix = sentence[:start]
    prefix = re.split(r"[，,：:；;。]", prefix)[-1]
    return prefix.strip()[-20:] or "数值"


def _period_of(sentence: str) -> str:
    years = YEAR.findall(sentence)
    return years[-1].replace(" ", "") if years else ""


def _table_facts(chunk: dict, citation_index: int) -> list[TypedFact]:
    body = chunk_body(chunk)
    lines = [line.strip() for line in body.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [cell.strip() for cell in _CELL_SPLIT.split(lines[0].strip().strip("|"))]
    facts: list[TypedFact] = []
    for row_index, line in enumerate(lines[1:], 1):
        if re.match(r"^\|\s*-{3,}", line.replace(" ", "")):
            continue
        cells = [cell.strip() for cell in _CELL_SPLIT.split(line.strip().strip("|"))]
        if len(cells) < 2:
            continue
        subject = cells[0]
        if not subject:
            continue
        for column, cell in enumerate(cells[1:], 1):
            match = re.fullmatch(
                r"(\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|元/晚|元/天|个工作日|个自然日|ms|kHz|Hz|MPa|N·m|kg|mm|m/s|%|元|台|人|天|小时|分钟|秒|个月|年|月|日|℃)?",
                cell,
            )
            if not match:
                continue
            predicate = header[column] if column < len(header) else f"列{column}"
            facts.append(TypedFact(
                subject=subject, predicate=predicate, value=match.group(1),
                unit=match.group(2) or "", period=_period_of(cell) or _period_of(chunk.get("title", "")),
                doc_id=chunk.get("doc_id", ""), title=chunk.get("title", ""),
                chunk_id=chunk.get("chunk_id", ""), chunk_type="table",
                sentence_index=row_index, offset=0, sentence=line,
                citation_index=citation_index,
            ))
    return facts


def _text_facts(chunk: dict, citation_index: int) -> list[TypedFact]:
    facts: list[TypedFact] = []
    body = chunk_body(chunk)
    for sentence_index, sentence in enumerate(_sentences(body)):
        if re.search(r"[=＝]|计算式|算式", sentence):
            continue
        for match in NUMBER_UNIT.finditer(sentence):
            facts.append(TypedFact(
                subject=_subject_of(sentence, match.start()),
                predicate=_predicate_of(sentence, match.start()),
                value=match.group(1).replace(",", ""),
                unit=match.group(2),
                period=_period_of(sentence),
                doc_id=chunk.get("doc_id", ""),
                title=chunk.get("title", ""),
                chunk_id=chunk.get("chunk_id", ""),
                chunk_type=chunk.get("chunk_type", "text"),
                sentence_index=sentence_index,
                offset=match.start(),
                sentence=sentence,
                citation_index=citation_index,
            ))
    return facts


def extract_facts(chunks: list[dict], limit: int = 400) -> list[TypedFact]:
    facts: list[TypedFact] = []
    for citation_index, chunk in enumerate(chunks, 1):
        if chunk.get("chunk_type") == "table":
            facts.extend(_table_facts(chunk, citation_index))
        else:
            facts.extend(_text_facts(chunk, citation_index))
        if len(facts) >= limit:
            break
    return facts[:limit]


def facts_by_subject(facts: list[TypedFact]) -> dict[str, list[TypedFact]]:
    grouped: dict[str, list[TypedFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.subject, []).append(fact)
    return grouped


def find_fact(facts: list[TypedFact], *, subject: str = "", predicate: str = "",
              period: str = "") -> list[TypedFact]:
    matched = facts
    if subject:
        matched = [fact for fact in matched if subject in fact.subject or fact.subject in subject]
    if predicate:
        matched = [fact for fact in matched if predicate in fact.predicate or predicate in fact.sentence]
    if period:
        matched = [fact for fact in matched if not fact.period or fact.period == period]
    return matched
