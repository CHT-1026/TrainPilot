"""答案生成：默认抽取式（离线可复现），可选 OpenAI 兼容 LLM，并做引用校验与拒答。"""
import logging
import re
import json
import time
from dataclasses import dataclass, field

from app.capabilities.retrieval.tokenizer import tokenize
from enterprise_rag import config
from enterprise_rag.budget import RequestBudget, new_budget, model_usage_available
from enterprise_rag.citations import build_context, chunk_body, select_context
from enterprise_rag.evidence import (
    MODEL_RE,
    UNIT_RE,
    build_comparison_table,
    enumeration_hint,
    model_scope_mismatch,
    normalize,
    token_supported,
    unsupported_numeric_tokens,
    version_pair,
)
from enterprise_rag.facts import named_subjects
from enterprise_rag.facts import evidence_periods, factual_clauses
from enterprise_rag.model_router import route_for
from enterprise_rag.org_facts import answer_org_question
from enterprise_rag.prompts import PROMPT_VERSIONS, answer_messages
from enterprise_rag.query import query_terms, rewrite_query, reasoning_plan, resolve_relative_years
from enterprise_rag.reasoning import infer
from enterprise_rag.skills import match_skills, render_skill_block
from enterprise_rag.verification import verify_answer_facts, unsupported_quantitative_trends

logger = logging.getLogger(__name__)


class ModelAnswerValidationError(ValueError):
    def __init__(self, reason, answer):
        super().__init__(reason)
        self.answer = answer


@dataclass
class ModelAnswer:
    answer: str
    indices: list[int]
    confidence: float
    model: str
    skill_packs: list[str]
    rejected_claims: list[dict] = field(default_factory=list)

REFUSAL_TEXT = "现有企业资料未覆盖该问题，建议联系对应业务部门确认。"
REFUSAL_MARKERS = ("未覆盖", "资料不足", "无法回答", "没有找到", "未提及", "建议联系")

_SENTENCE = re.compile(r"(?<=[。！？\n])")
_META_PREFIXES = ("所属部门：", "作者：", "创建时间：", "更新时间：", "版本号：", "文档类型：", "文档族：", "生效日期：")
_TABLE_ROW = re.compile(r"^\|.*\|$")


def _sentences(text: str, split_clauses: bool = False) -> list[str]:
    parts: list[str] = []
    buffer = ""
    table_header: list[str] = []
    boundary = r"(?<=[。！？；;\n])" if split_clauses else _SENTENCE
    for raw in re.split(boundary, text.replace("\r", "\n")):
        line = raw.strip()
        if not line:
            continue
        if any(line.startswith(prefix) for prefix in _META_PREFIXES):
            continue
        if re.match(r"^\|\s*-{3,}", line.replace(" ", "")):
            continue
        if line.startswith("|") and line.endswith("|"):
            if buffer:
                parts.append(buffer)
                buffer = ""
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            if len(cells) >= 2:
                if not table_header:
                    table_header = cells
                elif len(cells) == len(table_header):
                    # Preserve metric names and per-column units when cells only
                    # contain numbers. Bare row values cannot answer a metric.
                    values = [cells[0]]
                    for header, cell in zip(table_header[1:], cells[1:]):
                        match = re.fullmatch(r"(.+?)[（(]([^）)]+)[）)]", header)
                        label, unit = match.groups() if match else (header, "")
                        values.append(f"{label}{cell}{unit}" if unit and re.fullmatch(r"[-−]?\d[\d,.]*", cell)
                                      else f"{label}{cell}")
                    parts.append("，".join(values))
            continue
        table_header = []
        buffer = f"{buffer}{line}" if buffer else line
        if buffer.endswith(("。", "！", "？", "；", ";")):
            if len(buffer) >= 8:
                parts.append(buffer)
            buffer = ""
    if buffer and len(buffer) >= 8:
        parts.append(buffer)
    return parts


def _financial_slots(question: str) -> list[tuple[str, str]]:
    """Explicit requested metrics, rather than one revenue gate for all facts."""
    amount = r"\s*(?:目标|预算|预计|计划)?\s*(?:为|达|达到|实现|约|[:：])?\s*\d[\d,.]*\s*(?:亿元|万元|元)"
    slots = []
    for label, trigger, metric in (
        ("营业收入", r"营业收入|营收|海外收入|赚", r"(?:营业收入|营收|收入)"),
        ("净利润", r"净利润|归母|赚", r"(?:归母净利润|净利润)"),
        ("研发投入", r"研发投入|研发费用|技术投入", r"(?:研发投入|研发费用|技术投入)"),
        ("资本开支", r"资本开支|资本支出", r"(?:资本开支|资本支出)"),
        ("订单", r"订单", r"(?:新签订单|订单金额|订单)"),
    ):
        if re.search(trigger, question):
            if label == "营业收入" and "海外" in question:
                metric = r"海外(?:营业收入|营收|收入)"
            slots.append((label, metric + amount))
    for label, trigger, metric in (
        ("研发人员", r"研发人员|研发团队", r"研发(?:人员|团队)"),
        ("员工人数", r"员工人数|员工总数", r"员工(?:人数|总数)?"),
    ):
        if re.search(trigger, question):
            slots.append((label, metric + r"\s*(?:为|有|共|约|[:：])?\s*\d[\d,]*\s*人"))
    return slots


def _same_reporting_period(question: str, sentence: str, chunk: dict) -> bool:
    """Match statistical granularity; publication/as-of dates are not months."""
    month = re.search(r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月", question)
    context = " ".join(str(chunk.get(k) or "") for k in ("fact_period", "section", "title"))
    if month:
        wanted = rf"{month.group(1)}\s*年\s*0?{int(month.group(2))}\s*月"
        # “截至12月31日，全年...” describes a snapshot/annual total.
        return bool(re.search(wanted, sentence) and not re.search(r"截至|全年|年度|累计", sentence))
    half = re.search(r"(上|下)半年|([12])H|H([12])", question, re.I)
    if half:
        wanted = half.group(1) or ("上" if (half.group(2) or half.group(3)) == "1" else "下")
        actual = re.search(r"(上|下)半年|([12])H|H([12])", sentence, re.I)
        actual = actual or re.search(r"(上|下)半年|([12])H|H([12])", context, re.I)
        if not actual:
            return False
        value = actual.group(1) or ("上" if (actual.group(2) or actual.group(3)) == "1" else "下")
        return value == wanted
    quarter = re.search(r"(?:第)?([一二三四1-4])\s*季度|Q\s*([1-4])", question, re.I)
    if quarter:
        raw = quarter.group(1) or quarter.group(2)
        number = {"一":1, "二":2, "三":3, "四":4}.get(raw, int(raw) if raw.isdigit() else 0)
        cumulative = bool(re.search(r"前[一二三四1-4]季度|累计", question))
        actual_cumulative = bool(re.search(r"前[一二三四1-4]季度|累计", sentence + " " + str(chunk.get("section", ""))))
        if cumulative != actual_cumulative:
            return False
        actual = re.search(r"(?:第)?([一二三四1-4])\s*季度|Q\s*([1-4])", sentence, re.I)
        actual = actual or re.search(r"(?:第)?([一二三四1-4])\s*季度|Q\s*([1-4])", context, re.I)
        if not actual:
            return False
        value = actual.group(1) or actual.group(2)
        return number == {"一":1, "二":2, "三":3, "四":4}.get(value, int(value) if value.isdigit() else 0)
    if re.search(r"(?:19|20)\d{2}", question) and not re.search(r"半年|截至", question):
        if re.search(r"半年|季度|单季|Q[1-4]", sentence + " " + context, re.I):
            return False
        # A report may open with “截至12月31日” before explicitly naming an
        # annual metric. Remove only that snapshot date for annual matching.
        statistical = re.sub(r"截至\s*(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", "", sentence)
        return not re.search(r"\d\s*月", statistical + " " + context)
    return True


def _coverage(tokens: set[str], terms: list[str]) -> float:
    total = sum(1.4 if len(term) >= 4 else 1.0 for term in terms) or 1.0
    matched = sum(1.4 if len(term) >= 4 else 1.0 for term in terms if term in tokens)
    return matched / total


def _sentence_score(sentence: str, terms: list[str], question: str, coverage: float, chunk_score: float) -> float:
    tokens = set(tokenize(sentence))
    bonus = 0.0
    if re.search(r"\d", sentence):
        if any(hint in question for hint in ("多少", "几", "标准", "比例", "时间", "多久", "金额", "人数", "价格", "参数")):
            bonus += 0.10
        if sentence.startswith("|") or " | " in sentence:
            bonus += 0.05
    if any(unit in sentence for unit in ("元", "万元", "亿元", "%", "小时", "天", "台", "人")):
        bonus += 0.03
    entity_terms = [
        term
        for term in terms
        if any(char.isalpha() for char in term)
        and any(char.isdigit() for char in term)
        and not term.startswith("20")
    ]
    if entity_terms and any(term in tokens for term in entity_terms):
        bonus += 0.18
    if "台" in question and re.search(r"\d+\s*台", sentence):
        bonus += 0.10
    if any(word in question for word in ("成效", "效果", "收益", "提升", "改善")) and re.search(
        r"(提升|降低|缩短|减少|下降|回收期|良率|效率)", sentence
    ):
        bonus += 0.16
    if ("时间" in question or "多久" in question) and re.search(r"\d+\s*(小时|分钟|工作日|天)", sentence):
        bonus += 0.08
    # 问题限定了年份时，排除其他年份的陈述，避免回答串期。
    question_years = set(re.findall(r"(20\d{2})", question))
    sentence_years = set(re.findall(r"(20\d{2})", sentence))
    if question_years and sentence_years and not (question_years & sentence_years):
        return 0.0
    if question_years and not sentence_years and any(word in sentence for word in ("上年", "去年", "明年")):
        return 0.0
    # 未限定季度/半年时，优先年度口径，降低阶段性数据干扰。
    if not any(word in question for word in ("季度", "Q1", "Q2", "Q3", "Q4", "半年", "上半年", "前三季度")):
        if any(word in sentence for word in ("前三季度", "上半年", "第三季度", "单季")):
            coverage *= 0.72
    return coverage + bonus + 0.35 * chunk_score


def _extractive_answer(question: str, chunks: list[dict]) -> tuple[str, list[int], float]:
    rewritten = rewrite_query(question)
    terms = query_terms(rewritten["rewritten"], question)
    requested_years = set(re.findall(r"(?:19|20)\d{2}", question))
    financial_value = bool(re.search(r"营业收入|营收|海外收入|净利润|归母|研发投入|研发费用|技术投入|资本开支|资本支出|订单|赚", question))
    annual_financial = (bool(requested_years) and financial_value
                        and not re.search(r"季度|半年|\d\s*月|截至|Q[1-4]", question, re.I))
    financial_slots = ([] if not financial_value or re.search(r"占|比例|比重|增长率|增幅|费用率|相差|差额", question)
                       else _financial_slots(question))
    place_match = re.search(r"([\u4e00-\u9fffA-Za-z]{2,16}(?:中心|基地|工厂|总部|办事处|分公司)).{0,5}(?:设在哪里|在哪里|位于哪里|地点|地址)", question)
    place_entity = re.sub(r"^.*[，,、]", "", place_match.group(1)) if place_match else ""
    candidates = []
    for index, chunk in enumerate(chunks, 1):
        context_tokens = set(tokenize(f"{chunk.get('title', '')} {chunk.get('section', '')}"))
        sentences = _sentences(chunk_body(chunk), split_clauses=bool(financial_slots))
        if financial_slots:
            # A later dated metric cannot date an earlier undated headcount.
            sentences = [part for sentence in sentences for part in factual_clauses(sentence)]
        for position, sentence in enumerate(sentences):
            matched_slots = [label for label, pattern in financial_slots if re.search(pattern, sentence)]
            place_hit = bool(place_entity and place_entity in sentence
                             and re.search(r"位于|设立|设在|坐落|地址", sentence))
            if financial_slots and not (matched_slots or place_hit):
                continue
            periods = evidence_periods(sentence, chunk)
            if financial_value and not place_hit and requested_years and periods and not requested_years.intersection(periods):
                continue
            if matched_slots and not _same_reporting_period(question, sentence, chunk):
                continue
            if annual_financial and not place_hit and re.search(r"上半年|下半年|半年度|季度|单季|Q[1-4]",
                                               sentence + " " + chunk.get("title", "") + " " + chunk.get("section", ""), re.I):
                continue
            tokens = set(tokenize(sentence))
            # 句子覆盖率纳入所属文档/章节上下文，避免“成效数据”这类无主语句子被误过滤。
            coverage = _coverage(tokens | context_tokens, terms)
            score = _sentence_score(sentence, terms, question, coverage, chunk.get("score", 0.0))
            if score < 0.28 or coverage < 0.10:
                continue
            candidates.append(
                {
                    "chunk_index": index,
                    "doc_id": chunk["doc_id"],
                    "title": chunk.get("title", ""),
                    "section": chunk.get("section", ""),
                    "sentence": sentence,
                    "score": score,
                    "coverage": coverage,
                    "position": position,
                    "slots": matched_slots + (["地点"] if place_hit else []),
                    "fact_periods": periods,
                    "planning_context": (chunk.get("title", "") + " · " + chunk.get("section", ""))
                        if re.search(r"目标|预算|计划|预测|预计", chunk.get("title", "") + chunk.get("section", "")) else "",
                }
            )
    if not candidates:
        return "", [], 0.0

    entity_terms = [
        term
        for term in terms
        if any(char.isalpha() for char in term)
        and any(char.isdigit() for char in term)
        and not term.startswith("20")
    ]
    if entity_terms:
        specific = [
            candidate
            for candidate in candidates
            if any(
                term in set(tokenize(candidate["sentence"])) or term in candidate["title"].lower()
                for term in entity_terms
            )
        ]
        if specific:
            candidates = specific

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best_score = candidates[0]["score"]
    selected = []
    per_doc: dict[str, int] = {}
    number_signatures: list[set[str]] = []
    for candidate in candidates:
        if candidate["score"] < best_score * 0.55:
            break
        if per_doc.get(candidate["doc_id"], 0) >= 2:
            continue
        if candidate["sentence"] in [item["sentence"] for item in selected]:
            continue
        signature = set(re.findall(r"\d[\d,\.]*", candidate["sentence"]))
        if signature and any(len(signature & existing) >= 2 for existing in number_signatures):
            continue
        selected.append(candidate)
        per_doc[candidate["doc_id"]] = per_doc.get(candidate["doc_id"], 0) + 1
        number_signatures.append(signature)
        if len(selected) >= 3:
            break

    # 意图槽位：数量、成效、时长、等级、地点、比例、参数等关键事实各命中一条。
    required: list[dict] = []

    def add_slot(pool: list[dict], chooser=None):
        if not pool:
            return
        chosen = chooser(pool) if chooser else max(pool, key=lambda item: item["score"])
        if chosen and chosen["sentence"] not in [item["sentence"] for item in required]:
            required.append(chosen)

    for label in [name for name, _ in financial_slots] + (["地点"] if place_entity else []):
        add_slot([c for c in candidates if label in c["slots"]],
                 chooser=lambda pool: max(pool, key=lambda c: (bool(requested_years & c["fact_periods"]), c["score"])))

    if "台" in question or (
        "机器人" in question and any(word in question for word in ("使用", "用了", "部署", "几台", "多少台"))
    ):
        add_slot([c for c in candidates if re.search(r"\d[\d,\.]*\s*台", c["sentence"])])
    if any(word in question for word in ("成效", "效果", "收益", "提升", "改善")):
        metric_pool = [
            c
            for c in candidates
            if re.search(r"(良率|回收期|毛利率|净利率)", c["sentence"]) and re.search(r"\d", c["sentence"])
        ]
        if not metric_pool:
            metric_pool = [
                c
                for c in candidates
                if re.search(r"(节拍|效率|产能)", c["sentence"]) and re.search(r"\d", c["sentence"])
            ]
        add_slot(metric_pool)
    if any(word in question for word in ("几天", "多久", "时限", "期限", "周期")):
        day_pool = [c for c in candidates if re.search(r"第\s*(\d+)\s*天", c["sentence"])]
        if day_pool:
            add_slot(
                day_pool,
                chooser=lambda pool: max(
                    pool,
                    key=lambda item: int(re.search(r"第\s*(\d+)\s*天", item["sentence"]).group(1)),
                ),
            )
        else:
            duration_pool = [
                c
                for c in candidates
                if re.search(r"\d+\s*(天|小时|分钟|个工作日|个自然日|自然日)", c["sentence"])
            ]
            if "报销" in question and "时限" in question:
                preferred = [c for c in duration_pool if "自然日" in c["sentence"]]
                duration_pool = preferred or duration_pool
            add_slot(duration_pool)
    if any(word in question for word in ("几级", "几个等级", "等级")):
        add_slot(
            [
                c
                for c in candidates
                if re.search(r"(四级|三级|两级|分为.{0,8}级)", c["sentence"])
            ]
        )
    if not place_entity and any(word in question for word in ("哪里", "地址", "设在", "位于", "地点")):
        place_pool = [c for c in candidates if re.search(r"(位于|设立|设在|坐落|地址[:：])", c["sentence"])]
        if "总部" in question:
            add_slot(
                place_pool,
                chooser=lambda pool: max(
                    pool,
                    key=lambda item: ("总部" in item["sentence"], item["score"]),
                ),
            )
        else:
            add_slot(place_pool)
    if any(word in question for word in ("叫什么", "名字", "名称", "全称", "哪家公司", "什么公司", "简介")):
        add_slot(
            [
                c
                for c in candidates
                if "澜图智能装备股份有限公司" in c["sentence"]
                or "成立于" in c["sentence"]
                or "公司简介" in c["title"]
            ]
        )
    if "使命" in question:
        add_slot([c for c in candidates if "使命" in c["section"]])
    if "愿景" in question:
        add_slot([c for c in candidates if "愿景" in c["section"]])
    if any(word in question for word in ("良率", "完成率", "占比", "比例", "毛利率", "净利率")):
        add_slot([c for c in candidates if re.search(r"\d+(?:\.\d+)?%", c["sentence"])])
    if any(word in question for word in ("负载", "参数", "指标", "节拍", "精度")):
        add_slot(
            [
                c
                for c in candidates
                if re.search(r"\d+\s*(kg|mm|N|m/s|秒)", c["sentence"])
            ]
        )
    if "额定" in question:
        add_slot([c for c in candidates if "额定负载" in c["sentence"]])
    month_match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", question)
    if month_match:
        exact_date = f"{month_match.group(1)}年{int(month_match.group(2))}月"
        add_slot(
            [
                c
                for c in candidates
                if c["sentence"].replace(" ", "").startswith(exact_date)
            ]
        )
    if entity_terms:
        model_pool = [
            c for c in candidates if any(term in set(tokenize(c["sentence"])) for term in entity_terms)
        ]
        if model_pool:
            add_slot(model_pool)
    if "方案" in question and "良率" in question:
        def best_yield(pool: list[dict]):
            def value(item):
                match = re.search(r"良率[^0-9]{0,8}(\d+(?:\.\d+)?)%", item["sentence"])
                return float(match.group(1)) if match else 0.0
            return max(pool, key=lambda item: (value(item), item["score"]))

        add_slot(
            [
                c
                for c in candidates
                if "解决方案" in c["title"] and "良率" in c["sentence"]
            ],
            chooser=best_yield,
        )
    if any(word in question for word in ("哪些产品", "哪些型号", "有哪些")):
        add_slot(
            [
                c
                for c in candidates
                if len(re.findall(r"[A-Za-z]{2,5}-[A-Za-z0-9]+", c["sentence"])) >= 2
            ]
        )
    if "苏州" in question and "人数" in question:
        add_slot(
            [
                c
                for c in candidates
                if "苏州" in c["sentence"]
                and ("人数" in c["sentence"] or "员工" in c["sentence"] or "苏州总部" in c["sentence"])
                and re.search(r"\d[\d,\.]{2,}", c["sentence"])
                and "%" not in c["sentence"]
                and "利用率" not in c["sentence"]
            ]
        )
    if "董事会" in question and any(word in question for word in ("利润分配", "分红")):
        add_slot(
            [
                c
                for c in candidates
                if "董事会" in c["title"] and re.search(r"每\s*10\s*股", c["sentence"])
            ]
        )
    if "销售合同" in question and "预付款" in question:
        add_slot(
            [
                c
                for c in candidates
                if "销售合同" in c["title"] and "预付款" in c["sentence"]
            ]
        )
    if "出口管制" in question:
        add_slot([c for c in candidates if re.search(r"(两用物项|ECCN)", c["sentence"])])
    if "折扣" in question and any(word in question for word in ("审批", "批", "权限")):
        add_slot(
            [
                c
                for c in candidates
                if "%" in c["sentence"] and re.search(r"(审批|由.*批)", c["sentence"])
            ]
        )

    for item in required:
        if item["sentence"] not in [selected_item["sentence"] for selected_item in selected]:
            selected.append(item)
    # 保留全部意图槽位，再用高分句补足到 4 句。
    selected = list({item["sentence"]: item for item in required}.values()) + selected
    unique = []
    for item in selected:
        if item["sentence"] not in [existing["sentence"] for existing in unique]:
            unique.append(item)
    selected = unique[:4]
    if financial_slots:
        # Preserve distinct slot values and periods, without repeating the
        # identical requested metric from several summaries of one fact.
        deduped, seen_facts = [], set()
        for item in selected:
            identity = tuple(sorted(named_subjects(item["sentence"] + " " + item["title"])))
            signatures = {(label, re.sub(r"[\s,]", "", hit.group()), tuple(sorted(item["fact_periods"])), identity,
                           bool(item["planning_context"]))
                          for label, pattern in financial_slots for hit in re.finditer(pattern, item["sentence"])}
            if signatures and all(any(sig[:2] == old[:2] and sig[3:] == old[3:]
                                      and (sig[2] == old[2] or not sig[2]) for old in seen_facts)
                                  for sig in signatures):
                continue
            deduped.append(item)
            seen_facts.update(signatures)
        selected = deduped
    required_sentences = {item["sentence"] for item in required}
    selected.sort(key=lambda item: (item["sentence"] not in required_sentences, -item["score"]))

    best_coverage = max(item["score"] for item in candidates)
    parts = []
    for candidate in selected:
        sentence = candidate["sentence"]
        # Source headings carry modality. Keep it visible when a plan contains
        # an otherwise unqualified amount; never turn a budget into an actual.
        if candidate["planning_context"] and candidate["slots"] and not re.search(r"目标|预算|计划|预测|预计", sentence):
            sentence = candidate["planning_context"] + "：" + sentence
        for clause in re.split(r"(?<=[。！？；;\n])", sentence):
            clause = clause.strip().rstrip("。！？；; ")
            if clause:
                parts.append(f"{clause} [{candidate['chunk_index']}]")
    answer = "根据企业知识库资料：" + "；".join(parts) + "。"
    return answer, [item["chunk_index"] for item in selected], best_coverage


_METADATA_QUESTION = re.compile(
    r"什么时候更新|何时更新|更新时间|更新日期|哪天更新|哪一版|哪个版本|版本号|版本是|"
    r"生效时间|生效日期|什么时候生效|发布时间|发布日期|最近.{0,6}(更新|改版|修订|调整|上线)|"
    r"上一次(更新|改版|修订|调整)|上次(更新|改版|修订|调整|上线)|哪天(更新|上线|改版|修订)|什么时候(改版|修订|上线)|"
    r"哪一年更新|哪年更新|更新于|是.{0,8}年.{0,4}(发布|生效|更新)的吗"
)


def _metadata_answer(question: str, chunks: list[dict]) -> tuple[str, list[int], float] | None:
    """Answer document version/update-time questions from chunk metadata.

    `doc_updated_at` and `effective_date` are citation metadata, not body text;
    the model and sentence extractor cannot see them reliably.
    """
    # ponytail:shortcut 文档级元数据直接取 chunk 字段，不调用模型。
    if not _METADATA_QUESTION.search(question) or not chunks:
        return None
    candidates = chunks[:10]
    org_preferred = [c for c in candidates
                     if "org_chart" in str(c.get("doc_id", ""))
                     or "组织架构" in str(c.get("title", ""))]
    tech_preferred = [c for c in candidates
                      if "control_architecture" in str(c.get("doc_id", ""))
                      or "控制系统架构" in str(c.get("title", ""))]

    def describe(chunk: dict) -> tuple[str, int] | None:
        version = str(chunk.get("version") or "").strip()
        updated = str(chunk.get("doc_updated_at") or "").strip()
        effective = str(chunk.get("effective_date") or "").strip()
        if not (version or updated or effective):
            return None
        target_id = chunk.get("chunk_id")
        index = next((position for position, item in enumerate(chunks, 1)
                      if item.get("chunk_id") == target_id), 1)
        title = chunk.get("title") or chunk.get("doc_id") or "该文档"
        facts = []
        if version:
            facts.append(f"当前版本为 {version}")
        if updated:
            facts.append(f"更新时间为 {updated[:10]}")
        if effective:
            facts.append(f"生效日期为 {effective[:10]}")
        return f"《{title}》" + "，".join(facts) + f"。 [{index}]", index

    if (org_preferred and tech_preferred
            and re.search(r"公司架构|组织架构", question)
            and re.search(r"控制架构|控制系统|技术架构|系统架构", question)
            and re.search(r"分别|各|都", question)):
        left = describe(org_preferred[0])
        right = describe(tech_preferred[0])
        if left and right:
            return (
                f"组织架构：{left[0]} 技术架构：{right[0]}",
                [left[1], right[1]],
                1.0,
            )

    if re.search(r"公司架构|组织架构", question) and not org_preferred:
        return None
    if re.search(r"公司架构|组织架构|一级部门|部门|汇报", question):
        preferred = org_preferred
    elif re.search(r"控制架构|控制系统|技术架构|系统架构", question):
        preferred = tech_preferred
    else:
        preferred = []
    target = (preferred or candidates)[0]
    described = describe(target)
    if described is None:
        return None
    return described[0], [described[1]], 1.0


def _comparison_answer(question: str, chunks: list[dict]) -> tuple[str, list[int], float] | None:
    """Build a cited comparison from the deterministic dimension table.

    Slow or truncated model outputs are a real failure mode; exact old/new
    values already exist in the aligned table, so use them directly.
    """
    # ponytail:shortcut 只覆盖 DIMENSIONS 维度表，超出维度仍交回模型。
    # 具体差值题需要模型计算“增加了多少”；组织架构走通用旧/新对照。
    if re.search(r"多少|增加|减少|缩短|提高|降低|提升|下降", question):
        return None
    if re.search(r"公司架构|组织架构|一级部门|部门|汇报", question):
        return None
    table = build_comparison_table(question, chunks)
    if not table:
        return None
    pair = version_pair(chunks, question)
    if not pair:
        return None
    docs = {doc_id: first for doc_id, first in {
        chunk.get("doc_id"): chunk for chunk in pair
    }.items()}
    ordered = sorted(docs.values(),
                     key=lambda chunk: (str(chunk.get("effective_date") or "9999"),
                                        str(chunk.get("version") or "")))
    older, newer = ordered[0], ordered[-1]
    indices = []
    for doc in (older, newer):
        index = next((position for position, chunk in enumerate(chunks, 1)
                      if chunk.get("doc_id") == doc.get("doc_id")), None)
        if index is not None and index not in indices:
            indices.append(index)
    if len(indices) < 2:
        return None
    answer = (
        f"根据旧版 {older.get('version', '')} 与新版 {newer.get('version', '')} 的资料，"
        f"主要差异如下：\n{table}\n上述差异均来自两版原文。 [{indices[0]}][{indices[1]}]"
    )
    return answer, indices, 1.0


def _family_contrast_answer(question: str, chunks: list[dict]) -> tuple[str, list[int], float] | None:
    """Cited old/new contrast for families without architecture dimensions.

    Used for organization charts and policy versions: pick the best chunk per
    version and quote its key sentences instead of asking a slow model to
    reconstruct both versions from scratch.
    """
    # ponytail:shortcut 无维度表的文档族只做两版关键句对照，不逐条语义核验。
    pair = version_pair(chunks, question)
    if not pair:
        return None
    by_doc: dict[str, list[dict]] = {}
    for chunk in pair:
        by_doc.setdefault(chunk.get("doc_id"), []).append(chunk)
    ordered = sorted(by_doc.values(),
                     key=lambda items: (str(items[0].get("effective_date") or "9999"),
                                        str(items[0].get("version") or "")))
    if len(ordered) < 2:
        return None
    older, newer = ordered[0], ordered[-1]
    org_family = any(item.get("doc_family") == "org_chart" for item in pair)
    older_text = " ".join(chunk_body(item) for item in older)
    newer_text = " ".join(chunk_body(item) for item in newer)
    older_only = set(UNIT_RE.findall(older_text)) - set(UNIT_RE.findall(newer_text))
    newer_only = set(UNIT_RE.findall(newer_text)) - set(UNIT_RE.findall(older_text))

    def choose(items: list[dict], other: list[dict], distinctive: set[str]) -> tuple[dict, list[str]]:
        ranked = sorted(items, key=lambda item: item.get("score", 0.0), reverse=True)
        if org_family:
            for chunk in ranked:
                sentences = [sentence.strip() for sentence in
                             re.split(r"(?<=[。！？；\n])", chunk_body(chunk)) if sentence.strip()]
                department_list = next((sentence for sentence in sentences
                                        if "个一级部门" in sentence), "")
                if department_list:
                    others = [sentence for sentence in sentences if sentence != department_list][:2]
                    return chunk, [department_list] + others
        hits: list[str] = []
        first_chunk = None
        for chunk in ranked:
            sentences = [sentence.strip() for sentence in
                         re.split(r"(?<=[。！？；\n])", chunk_body(chunk)) if sentence.strip()]
            for sentence in sentences:
                if sentence in hits:
                    continue
                if any(token in sentence for token in distinctive):
                    hits.append(sentence)
                    first_chunk = first_chunk or chunk
            if len(hits) >= 4:
                break
        if hits and first_chunk is not None:
            return first_chunk, hits[:4]
        best = ranked[0]
        other_text = normalize(" ".join(chunk_body(item) for item in other))
        sentences = [sentence.strip() for sentence in
                     re.split(r"(?<=[。！？；\n])", chunk_body(best)) if sentence.strip()]
        differing = [sentence for sentence in sentences
                     if normalize(sentence) not in other_text]
        return best, (differing or sentences)[:3]

    old_chunk, old_sentences = choose(older, newer, older_only)
    new_chunk, new_sentences = choose(newer, older, newer_only)
    index = {chunk.get("chunk_id"): position for position, chunk in enumerate(chunks, 1)}
    old_index = index.get(old_chunk.get("chunk_id"))
    new_index = index.get(new_chunk.get("chunk_id"))
    if not old_index or not new_index:
        return None
    answer = (
        f"旧版 {old_chunk.get('version', '')} [{old_index}]：" + " ".join(old_sentences) + "\n"
        f"新版 {new_chunk.get('version', '')} [{new_index}]：" + " ".join(new_sentences)
    )
    return answer, [old_index, new_index], 1.0


def _model_validation_error(question: str, answer: str, chunks: list[dict]) -> str | None:
    if unsupported_numeric_tokens(answer, chunks, question):
        return "unsupported_quantity"
    if unsupported_quantitative_trends(answer, chunks):
        return "unsupported_quantitative_trend"
    facts = verify_answer_facts(question, answer, chunks)
    if any(facts.get(key) for key in ("numeric_missing", "period_missing", "conflicts", "unbound_facts")):
        return "fact_period_subject_or_modality_mismatch"
    return None


def _salvage_truncated_claims(raw_answer: str) -> list[dict]:
    """当模型在 max_tokens 处截断时，保留已经完整输出的 claim 对象。

    只接受显式 ``refused:false`` 的 JSON 前缀；最后一个未闭合的 claim 丢弃。
    调用方仍会走逐 claim 的引用、事实与时期核验，不能借截断绕过证据约束。
    """
    if not re.search(r'"refused"\s*:\s*false', raw_answer, re.I):
        return []
    marker = raw_answer.find('"claims"')
    if marker < 0:
        return []
    bracket = raw_answer.find("[", marker)
    if bracket < 0:
        return []
    decoder = json.JSONDecoder()
    index = bracket + 1
    claims = []
    while index < len(raw_answer):
        while index < len(raw_answer) and raw_answer[index] in " \r\n\t,":
            index += 1
        if index >= len(raw_answer) or raw_answer[index] != "{":
            break
        try:
            value, end = decoder.raw_decode(raw_answer, index)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            claims.append(value)
        index = end
    return claims


def _llm_answer(question: str, chunks: list[dict], graph_paths: list[str] | None = None,
                strict: bool = False, budget: RequestBudget | None = None,
                plan: dict | None = None) -> ModelAnswer:
    from app.core.config import get_settings, thinking_extra_body

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY")
    if budget is not None and not budget.can_call_model():
        budget.stop("model_budget_exhausted")
        raise RuntimeError("请求预算不允许继续调用模型")
    from openai import OpenAI

    route = route_for(plan)
    model = route["model"]
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
        timeout=min(config.LLM_TIMEOUT, max(0.001, budget.remaining_seconds())) if budget else config.LLM_TIMEOUT,
        max_retries=0,
    )
    comparison = reasoning_plan(question)["comparison"]
    comparison_block = ""
    if comparison:
        table = build_comparison_table(question, chunks)
        if table:
            comparison_block = table
    enumeration_block = enumeration_hint(question, chunks)
    skills = match_skills(question)
    skill_block = render_skill_block(skills)
    context = build_context(chunks)
    messages = answer_messages(
        question,
        context,
        comparison=comparison,
        strict=strict,
        enumeration=bool(enumeration_block),
        graph_paths=graph_paths,
        comparison_table=comparison_block,
        enumeration_block=enumeration_block,
        skill_block=skill_block,
    )
    if budget is not None:
        budget.charge_model(stage="answer", model=model, detail="attempt")
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=messages,
            max_tokens=route["max_tokens"],
            response_format={"type": "json_object"},
            extra_body=thinking_extra_body(model),
        )
    except Exception:
        if budget is not None:
            budget.note_call_result(ok=False, detail="answer_failed")
        raise
    if budget is not None:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
        budget.note_model_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                total_tokens=total_tokens, available=model_usage_available(usage))
        try:
            from app.core.metrics import record_usage

            record_usage("llm", model, prompt_tokens, completion_tokens, total_tokens)
        except Exception:
            pass
    raw_answer = (response.choices[0].message.content or "").strip()
    rejected = []
    try:
        try:
            payload = json.loads(raw_answer)
        except json.JSONDecodeError:
            # 严格模式允许保留截断前的完整 claim；非严格模式仍按原合同拒绝。
            salvaged = _salvage_truncated_claims(raw_answer) if strict else []
            if not salvaged:
                raise
            payload = {"refused": False, "claims": salvaged}
            rejected.append({"claim_index": 0, "reason": "output_truncated"})
        if not isinstance(payload.get("refused"), bool) or not isinstance(payload.get("claims"), list):
            raise ValueError("invalid schema")
        if payload["refused"]:
            return ModelAnswer(REFUSAL_TEXT, [], 0.0, model, [skill["id"] for skill in skills])
        parts = []
        claim_parts = []
        if not payload["claims"]:
            raise ValueError("empty claims")
        for position, claim in enumerate(payload["claims"], 1):
            text, sources = claim["text"], claim["sources"]
            if not isinstance(text, str) or not text.strip() or not isinstance(sources, list):
                raise ValueError("invalid claim")
            if any(type(i) is not int or not 1 <= i <= len(chunks) for i in sources):
                raise ValueError("invalid source")
            if not sources:
                if not strict:
                    raise ModelAnswerValidationError("claim_missing_sources", raw_answer)
                rejected.append({"claim_index": position, "reason": "missing_citations"})
                continue
            # Structured sources are the single citation representation.
            text = re.sub(r"\[\d+\]", "", text)
            refs = "".join(f"[{i}]" for i in dict.fromkeys(sources))
            # Attach the declared support to every clause/sentence of the claim.
            # verify_answer splits on Chinese semicolons as well, so a citation
            # appended only to the final clause would fail the per-sentence check.
            current = []
            for sentence in re.split(r"(?<=[。！？；;\n])", text):
                if sentence.strip():
                    current.append(sentence.strip().rstrip("。！？；;") + " " + refs + "。")
            parts.extend(current)
            claim_parts.append((position, "\n".join(current)))
        if not parts:
            raise ValueError("no supported claims")
        answer = "\n".join(parts)
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ModelAnswerValidationError("模型返回的事实与引用结构无效", raw_answer)
    indices = [int(marker) for marker in re.findall(r"\[(\d+)\]", answer)]
    if any(not 1 <= index <= len(chunks) for index in indices):
        raise ModelAnswerValidationError("模型返回了上下文外的引用编号", answer)
    from app.capabilities.retrieval.answer import verify_answer
    verified, _, structurally_grounded = verify_answer(answer, chunks)
    if not structurally_grounded:
        raise ModelAnswerValidationError("模型回答未通过逐句引用检查", answer)
    answer = verified
    if model_scope_mismatch(question, chunks):
        raise ModelAnswerValidationError("引用了同系列其他型号的资料", answer)
    error = _model_validation_error(question, answer, chunks)
    if strict and (error or rejected):
        # Only a well-formed second model response is eligible for a partial
        # answer. Do not turn invalid references/schema or failed services into
        # fabricated support. Keep complete claims, never cut a claim's premise.
        accepted = []
        for position, claim in claim_parts:
            reason = _model_validation_error(question, claim, chunks)
            if not reason and not verify_answer_facts(question, claim, chunks)["verified"]:
                reason = "claim_support_not_verified"
            if reason:
                rejected.append({"claim_index": position, "reason": reason})
            else:
                accepted.append(claim)
        if accepted:
            answer = "\n".join(accepted)
            # Recheck the combined retained text: inherited years or a later
            # trend must not become valid merely by being checked in isolation.
            error = _model_validation_error(question, answer, chunks)
            if not verify_answer_facts(question, answer, chunks)["verified"]:
                error = error or "combined_claim_support_not_verified"
            indices = [int(marker) for marker in re.findall(r"\[(\d+)\]", answer)]
        else:
            error = "no_supported_claims"
    if error:
        raise ModelAnswerValidationError(error, answer)
    # 生成分支的可信度由“结构校验 + 引用有效”保证；检索分只作为最低证据门槛，
    # 不能再用 0.28 的句级抽取阈值把语义正确但检索分偏低的答案误判为无依据。
    coverage = 1.0
    return ModelAnswer(answer, indices, coverage, model, [skill["id"] for skill in skills], rejected)



def _renumber(answer: str, old_indices: list[int]) -> tuple[str, list[int]]:
    ordered = []
    for index in old_indices:
        if index not in ordered:
            ordered.append(index)
    mapping = {old: new for new, old in enumerate(ordered, 1)}
    answer = re.sub(
        r"\[(\d+)\]",
        lambda match: f"[{mapping[int(match.group(1))]}]"
        if int(match.group(1)) in mapping
        else match.group(0),
        answer,
    )
    return answer, ordered


def _focus_supported(question: str, chunks: list[dict], model_answer: bool = False) -> bool:
    """Check explicit entities; generic task nouns are not an LLM rejection gate."""
    if not chunks:
        return False
    if model_answer:
        # These are the actual cited chunks, already bounded by visible context.
        # “类型/应用/联系/推断” need not be literal words in an answerable source.
        targets = named_subjects(question)
        visible_names = set().union(*(named_subjects(chunk_body(c) + " " + str(c.get("title", ""))) for c in chunks))
        return (targets <= visible_names and all(
            any(token_supported(model, chunk) for chunk in chunks) for model in MODEL_RE.findall(question)
        ))
    import jieba.posseg as pseg

    # Generation must use the immutable visible context passed by retrieval;
    # reloading a global index here can change namespace/ACL/version mid-answer.
    corpus_text = "\n".join(chunk_body(chunk) + " " + str(chunk.get("title", "")) for chunk in chunks).lower()
    ignore = {
        "澜图", "智能", "公司", "企业", "项目", "产品", "方案", "服务", "中国",
        "多长", "多久", "多少", "几天", "几个", "哪些", "哪里", "什么", "怎样", "如何", "是否",
        "学时", "周期", "去年", "前年", "明年", "上年", "今年",
        "名字", "名称", "全称", "简称", "叫什么", "哪家", "公司名",
        "用途", "类型", "应用", "场景", "联系", "推断", "事实", "依据",
        # Query field names, applicability roles and document words are not
        # unknown business entities. Their values still require cited evidence.
        "数量", "数目", "总数", "比例", "比重", "金额", "时长", "时限", "时效",
        "上限", "下限", "底线", "目标", "标杆", "现场", "员工", "制度", "规定",
    }
    unknown = []
    for word, flag in pseg.cut(question):
        word = word.strip()
        if len(word) < 2 or word in ignore or re.fullmatch(r"20\d{2}", word):
            continue
        if word[0] in "和与的" or word[-1] in "和与的是":
            continue
        if not (flag.startswith("n") or flag in {"eng", "nz", "vn", "t"}):
            continue
        if word.lower() in corpus_text:
            continue
        # 复合词只要有一个二字子词在语料中出现，视为可解释，不判为未登录实体。
        if len(word) > 2 and any(word[i : i + 2].lower() in corpus_text for i in range(len(word) - 1)):
            continue
        unknown.append(word)
    return not unknown


def generate_answer(question: str, chunks: list[dict], use_llm: bool | None = None,
                    evidence_complete: bool = True, graph_paths: list[str] | None = None,
                    budget: RequestBudget | None = None) -> dict:
    question = resolve_relative_years(question)
    use_llm = config.USE_LLM if use_llm is None else use_llm
    budget = budget or new_budget(use_llm=use_llm)
    method = "extractive"
    chunks, context_truncated = select_context(
        chunks, compress=config.CONTEXT_COMPRESSION, question=question
    )
    best_score = chunks[0].get("score", 0.0) if chunks else 0.0
    plan = reasoning_plan(question)
    comparison = plan["comparison"]
    version_comparison = (
        bool(re.search(r"新旧|相比|对比|比较|变化|差异|区别|演进|升级|变更", question))
        and bool(re.search(r"架构|版本|版|系统|方案|设计", question))
    ) or (
        bool(re.search(r"旧版|旧架构", question)) and bool(re.search(r"新版|新架构", question))
    )
    if version_comparison and chunks:
        if version_pair(chunks, question) is None:
            missing = "旧版或新版"
            return {
                "answer": f"当前检索到的资料只包含该主题的一个版本，缺少{missing}资料，无法进行版本变化对比。",
                "citations": [], "grounded": False, "refused": True,
                "method": "evidence_check", "confidence": 0.0,
                "reasoning": {"operation": "version_pair_check", "refused": True},
                "fact_check": {"verified": False, "reason": "version_pair_missing"},
                "refusal_reason": "version_pair_missing",
                "prompt_version": PROMPT_VERSIONS["answer"],
                "model_used": "",
                "context_truncated": context_truncated,
                "context_chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
                "graph_paths": graph_paths or [],
            }
    answer = ""
    indices: list[int] = []
    confidence = 0.0
    model_error = None
    model_used = ""
    skill_packs: list[str] = []
    rejected_claims: list[dict] = []
    if chunks:
        org_result = answer_org_question(question, chunks)
        if org_result is not None:
            answer, indices, confidence = org_result
            method = "org_facts"
            use_llm = False
    if method != "org_facts" and version_comparison and chunks:
        comparison_result = _comparison_answer(question, chunks)
        if comparison_result is not None:
            answer, indices, confidence = comparison_result
            method = "deterministic_comparison"
            use_llm = False
        else:
            family_result = _family_contrast_answer(question, chunks)
            if family_result is not None:
                answer, indices, confidence = family_result
                method = "family_contrast"
                use_llm = False
    if method not in {"deterministic_comparison", "family_contrast", "org_facts"} and chunks:
        metadata_result = _metadata_answer(question, chunks)
        if metadata_result is not None:
            answer, indices, confidence = metadata_result
            method = "document_metadata"
            use_llm = False
    derived = None if method in {"document_metadata", "org_facts"} else infer(
        question, chunks, complete=evidence_complete and not context_truncated
    )
    if derived is not None:
        answer, indices, confidence = derived["answer"], derived["indices"], derived["confidence"]
        if derived["refused"]:
            return {"answer": answer, "citations": [], "grounded": False, "refused": True,
                    "method": "evidence_check", "confidence": 0.0,
                    "reasoning": derived, "fact_check": {"verified": False, "reason": "evidence_gap"},
                    "refusal_reason": "evidence_gap",
                    "prompt_version": PROMPT_VERSIONS["answer"], "model_used": "",
                    "context_truncated": context_truncated}
        method = "deterministic_reasoning"
        use_llm = False

    if derived is None and use_llm and chunks and budget.can_call_model():
        last_error = None
        # 综合题首轮即启用严格证据模式：无来源的因果推断应被剔除并保留有据事实，
        # 而不是先整题失败、再用第二次长调用补救。
        first_strict = bool(plan.get("synthesis"))
        for attempt in range(2):
            if not budget.can_call_model():
                break
            attempt_started = time.monotonic()
            try:
                with budget.stage("answer_generation"):
                    candidate = _llm_answer(
                        question, chunks, graph_paths,
                        strict=bool(attempt) or first_strict,
                        budget=budget, plan=plan,
                    )
                    answer, indices, confidence = candidate.answer, candidate.indices, candidate.confidence
                    model_used, skill_packs = candidate.model, candidate.skill_packs
                    rejected_claims = candidate.rejected_claims
                method = "llm"
                model_error = None
                break
            except Exception as exc:
                last_error = exc
                model_error = type(exc).__name__
                reason = str(exc.args[0])[:160] if getattr(exc, "args", None) else ""
                logger.warning("LLM 回答失败（第 %s 次）：%s%s",
                               attempt + 1, model_error, f" [{reason}]" if reason else "")
                # 超时或单次已耗时过久时，重试只会把总延迟翻倍。
                if model_error in {"APITimeoutError", "APIConnectionError"}:
                    break
                if time.monotonic() - attempt_started > 60:
                    break
        if method != "llm":
            logger.warning("LLM 连续失败，回退抽取式：%s", type(last_error).__name__)
            use_llm = False

    # 模型额度/时限不足不是“无证据”：已有合法块时仍走抽取式回答。
    if derived is None and chunks and not budget.can_call_model():
        use_llm = False

    if derived is None and not use_llm and method not in {
        "deterministic_comparison", "family_contrast", "document_metadata", "org_facts"
    }:
        with budget.stage("extractive_answer"):
            answer, indices, confidence = _extractive_answer(question, chunks)
        if model_error:
            method = "extractive_fallback"
            if reasoning_plan(question)["complex"]:
                answer, indices, confidence = "现有企业资料不足以完成该推断，生成服务暂时不可用，请稍后重试。", [], 0.0

    refused = False
    refusal_reason = None
    min_score = config.MIN_SCORE_COMPARISON if comparison else config.MIN_SCORE
    self_contained = derived is not None and derived.get("operation") == "year_arithmetic"
    if not answer or (not indices and not self_contained) or (derived is None and best_score < min_score):
        refusal_reason = ("empty_answer" if not answer else "model_refusal" if method == "llm" and not indices
                          else "missing_citations" if not indices else "low_retrieval_score")
        if model_error and method == "extractive_fallback" and not indices and chunks:
            refusal_reason = "model_output_invalid" if model_error == "ModelAnswerValidationError" else "generation_unavailable"
            answer = ("已找到相关资料，但生成内容未通过证据核验，暂时无法提供可靠结论。" if refusal_reason == "model_output_invalid"
                      else "已找到相关资料，生成服务暂时不可用，请稍后重试。")
        else:
            answer = REFUSAL_TEXT
        indices = []
        refused = True
    elif method != "llm" and confidence < 0.28:
        refusal_reason = "low_extractive_coverage"
        answer = REFUSAL_TEXT
        indices = []
        refused = True
    elif (derived is None and method not in {"document_metadata", "family_contrast", "org_facts"}
          and not version_comparison and not _focus_supported(
              question, [chunks[i-1] for i in sorted(set(indices))] if method == "llm" else chunks,
              model_answer=method == "llm")):
        refusal_reason = "question_entity_mismatch" if method == "llm" else "unsupported_query_focus"
        answer = REFUSAL_TEXT
        indices = []
        refused = True
    elif answer.startswith(REFUSAL_TEXT) or (
        len(answer) <= 120 and any(marker in answer for marker in REFUSAL_MARKERS)
    ):
        refused = True
        indices = []
        refusal_reason = "model_refusal" if method == "llm" else "evidence_not_covered"

    if not refused:
        answer, indices = _renumber(answer, indices)
        if derived is not None:
            mapping = {old: new for new, old in enumerate(indices, 1)}
            derived["answer"] = answer
            derived["indices"] = list(mapping.values())
            derived["source_chunk_ids"] = [chunks[i-1].get("chunk_id") for i in indices]
            for fact in derived.get("facts", []):
                if isinstance(fact, dict) and "citation" in fact:
                    fact["citation"] = mapping.get(fact["citation"], fact["citation"])

    citations = []
    for new_index, old_index in enumerate(indices, 1):
        chunk = dict(chunks[old_index - 1])
        chunk["citation_index"] = new_index
        citations.append(chunk)

    fact_check = {"verified": False, "reason": "refused"} if refused else verify_answer_facts(
        question, answer, citations
    )
    if not refused and (fact_check.get("trend_missing") or
                        (config.FACT_CHECK_REFUSE_ON_CONFLICT and fact_check.get("conflicts"))):
        answer = REFUSAL_TEXT
        indices = []
        citations = []
        refused = True
        fact_check["refused_for_conflict"] = True
        refusal_reason = "unsupported_quantitative_trend" if fact_check.get("trend_missing") else "fact_conflict"

    if refused:
        confidence = 0.0
    partial = bool(rejected_claims) and not refused
    if partial:
        answer = "部分回答（其余内容缺少可核验证据）：\n" + answer
        if plan.get("synthesis"):
            answer += (
                "\n说明：以上仅列出可核对事实；资料未给出足以证明因果关系的直接证据，"
                "相关联系仍属待验证推断。"
            )

    return {
        "answer": answer,
        "citations": citations,
        "grounded": bool(indices) and not refused,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "partial": partial,
        "rejected_claims": rejected_claims if partial else [],
        "method": method,
        "confidence": round(confidence, 4),
        "max_chunk_score": round(best_score, 4),
        "model_error": model_error,
        "model_used": model_used,
        "skill_packs": skill_packs,
        "reasoning": derived,
        "fact_check": fact_check,
        "prompt_version": PROMPT_VERSIONS["answer"],
        "context_truncated": context_truncated,
        "context_chunk_ids": [c.get("chunk_id") for c in chunks],
        "graph_paths": graph_paths or [],
    }

