"""查询理解：同义词/缩写扩展、语义检索改写、时间范围抽取、检索词清洗。"""
import json
import logging
import re
from datetime import date, timedelta
from functools import lru_cache

from app.capabilities.retrieval.tokenizer import tokenize
from enterprise_rag import config
from enterprise_rag.budget import model_usage_available

logger = logging.getLogger(__name__)

SYNONYMS = {
    "出差": ["差旅", "住宿标准", "出差补贴", "餐补"],
    "差旅": ["出差", "住宿标准", "交通标准"],
    "报销": ["费用报销", "发票", "审批流程"],
    "年假": ["带薪年假", "工龄", "假期"],
    "考勤": ["打卡", "弹性工作", "远程办公", "迟到"],
    "售后": ["售后服务", "客户服务", "维保", "服务网点", "服务等级协议", "SLA"],
    "SLA": ["服务等级协议", "响应时效", "到场时间", "故障分级"],
    "P1": ["服务等级协议", "SLA", "故障分级", "响应时效", "到场时间"],
    "响应时间": ["响应时效", "到场时间", "故障分级"],
    "质保": ["保修", "质保期", "延保"],
    "客诉": ["客户投诉", "投诉处理", "8D"],
    "产能": ["生产能力", "年产能", "扩产"],
    "研发": ["研发投入", "研发费用", "研发人员", "研究所"],
    "折扣": ["价格折让", "折扣审批", "毛利率"],
    "打几折": ["折扣", "折扣区间", "折扣审批"],
    "几折": ["折扣", "折扣区间", "折扣审批"],
    "打折": ["折扣", "折扣审批"],
    "赚": ["净利润", "归母净利润", "利润", "营业收入"],
    "公司叫什么": ["公司名称", "企业名称", "全称", "澜图智能装备股份有限公司"],
    "公司名字": ["公司名称", "企业名称", "全称", "澜图智能装备股份有限公司"],
    "公司名": ["公司名称", "企业名称", "全称"],
    "叫什么": ["公司名称", "企业名称", "全称"],
    "全称": ["公司名称", "企业名称", "澜图智能装备股份有限公司"],
    "哪家公司": ["公司名称", "企业名称", "澜图智能装备股份有限公司"],
    "什么公司": ["公司简介", "主营业务", "工业机器人", "智能制造"],
    "账期": ["付款账期", "月结", "回款"],
    "机器人": ["工业机器人", "六轴", "SCARA", "本体"],
    "移动机器人": ["AMR", "自主移动机器人", "调度"],
    "视觉": ["机器视觉", "3D视觉", "LT-V3D", "缺陷检测"],
    "数字孪生": ["LantuTwin", "仿真", "虚拟调试"],
    "半导体": ["晶圆", "洁净", "Class 10"],
    "锂电": ["动力电池", "叠片", "电芯"],
    "光伏": ["电池片", "串焊"],
    "召回": ["召回通知", "线束", "整改"],
    "专利": ["知识产权", "发明专利", "软件著作权"],
    "员工人数": ["员工", "编制", "人数", "分布"],
    "营收": ["营业收入", "收入", "业绩"],
    "营业收入": ["营收", "收入", "营业额"],
    "归母净利润": ["归属于母公司股东的净利润", "净利润", "利润"],
    "净利润": ["归母净利润", "利润"],
    "毛利率": ["毛利", "盈利能力", "成本"],
    "组织架构": ["一级部门", "汇报关系", "部门职责"],
    "公司架构": ["组织架构", "一级部门", "汇报关系", "部门职责"],
    "采购": ["供应商", "招标", "比价", "付款"],
    "培训": ["课程", "讲师", "认证", "学时"],
    "离职": ["交接", "竞业限制", "离职面谈"],
    "数据": ["数据分级", "信息安全", "外发"],
    "合规": ["反商业贿赂", "出口管制", "举报"],
    "巡检": ["预防性维护", "保养", "点检"],
}

STOPWORDS = {
    "什么", "多少", "怎么", "如何", "哪些", "哪个", "是否", "请问", "一下", "的是",
    "公司", "我们", "目前", "现在", "相关", "具体", "情况", "进行", "以及", "还有",
    "可以", "需要", "应该", "这个", "那个", "吗", "呢", "的", "了", "和", "与", "及",
}

_TABLE_HINTS = ("价格", "参数", "清单", "多少", "金额", "收入", "人数", "台数", "占比", "库存", "表格")


def _end_of_month(year: int, month: int) -> str:
    if not 1 <= month <= 12:
        raise ValueError("月份必须在 1 到 12 之间")
    if month == 12:
        return f"{year}-12-31"
    return (date(year, month + 1, 1) - timedelta(days=1)).isoformat()


def extract_time(question: str) -> tuple[str | None, str]:
    """返回 (as_of, 原文时间表达)。as_of 为包含该时点的最后一天。"""
    patterns = [
        (r"(\d{4})\s*年\s*(\d{1,2})\s*月", "month"),
        (r"(\d{4})\s*年\s*(上|下)半年", "half"),
        (r"(\d{4})\s*年\s*(?:第)?\s*([一二三四1-4])\s*季度", "quarter"),
        (r"(\d{4})\s*[年-]\s*Q\s*([1-4])", "quarter"),
        (r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*版", "month"),
        (r"(\d{4})\s*年版", "year"),
        (r"(\d{4})\s*版", "year"),
        (r"(\d{4})\s*年", "year"),
    ]
    for pattern, kind in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        year = int(match.group(1))
        if kind == "month":
            return _end_of_month(year, int(match.group(2))), match.group(0)
        if kind == "half":
            month = 6 if match.group(2) == "上" else 12
            return _end_of_month(year, month), match.group(0)
        if kind == "quarter":
            raw = match.group(2)
            quarter = {"一": 1, "二": 2, "三": 3, "四": 4}.get(raw, int(raw) if raw.isdigit() else 1)
            return _end_of_month(year, quarter * 3), match.group(0)
        if kind == "year":
            return f"{year}-12-31", match.group(0)
    return None, ""


def resolve_relative_years(question: str, reference_year: int | None = None) -> str:
    """Resolve user-authored relative years; never rewrite documentary clocks."""
    year = date.today().year if reference_year is None else reference_year
    offsets = {"今年":0, "去年":-1, "前年":-2, "明年":1}
    return re.sub(r"今年|去年|前年|明年", lambda m: f"{year+offsets[m.group(0)]}年", question or "")


def rewrite_query(question: str) -> dict:
    question = (question or "").strip()
    lowered = question.lower()
    expansions: list[str] = []
    matched: list[str] = []
    for key, values in SYNONYMS.items():
        if key.lower() in lowered:
            matched.append(key)
            expansions.extend(value for value in values if value not in expansions)
    # Relative query dates use a supplied/operating clock; documentary '年前'
    # inference is separate and never invents a missing documentary anchor.
    today = date.today()
    for label, delta in (("今年", 0), ("去年", -1), ("前年", -2), ("明年", 1)):
        if label in question:
            expansions.append(f"{today.year + delta}年")
    rewritten = question
    if expansions:
        rewritten = f"{question} {' '.join(expansions)}"
    as_of, time_label = extract_time(question)
    return {
        "original": question,
        "rewritten": rewritten,
        "matched_synonyms": matched,
        "expansions": expansions,
        "as_of": as_of,
        "time_label": time_label,
    }


_TOPIC_RULES = (
    ("business", "经营", r"经营|业绩|增长|营收|营业收入|净利润|订单|销售额|经营现金流",
     "经营业绩 营业收入 净利润 新签订单 同比增长",
     r"(?:营业收入|营收|净利润|归母净利润|新签订单|订单金额|销售额|经营现金流)\s*(?:为|达|达到|实现|约|共|总计|合计|[:：])?\s*[-−]?\d[\d,.]*\s*(?:亿元|万元|元)"
     r"|(?:营业收入|营收|净利润|新签订单)\s*(?:同比|较上年|较去年).{0,12}\d[\d.]*\s*%"),
    ("research", "研发", r"研发|技术投入|技术进展|创新投入", "研发投入 研发费用 研发人员 技术进展",
     r"研发(?:投入|费用|费用率|人员|团队)|技术(?:投入|进展|成果)|授权专利"),
    ("production", "生产", r"产能|生产效率|交付周期|制造效率|良率", "产能 生产效率 交付周期 良率",
     r"产能|生产效率|交付周期|制造效率|良率"),
    ("workforce", "人员", r"员工人数|员工总数|人员结构|员工结构|编制", "员工人数 员工总数 人员结构 编制",
     r"员工(?:人数|总数|结构)|人员结构|编制"),
)


def topic_matches(topic_id: str, text: str) -> bool:
    """A topic hit is substantive evidence, not a mention in another metric.

    In particular, “研发投入占营业收入13.8%” supplies research evidence but
    no realized operating revenue. This is coverage, not a claim of causality.
    """
    return any(key == topic_id and re.search(pattern, text)
               for key, _, _, _, pattern in _TOPIC_RULES)


def _bounded_topics(question: str) -> tuple[list[dict], list[str]]:
    from enterprise_rag.facts import named_subjects

    if not re.search(r"结合|联系|关系|与|和|及|分别|同时|对比|比较|占", question):
        return [], []
    matches = []
    for key, label, trigger, terms, _ in _TOPIC_RULES:
        hits = [m for m in re.finditer(trigger, question)
                if not re.search(r"(?:不要讨论|不谈|不涉及|无需讨论|排除)[^，。；]*$", question[:m.start()])]
        if hits:
            matches.append((hits[0].start(), key, label, terms))
    if len(matches) < 2:
        return [], []
    subjects = sorted({re.sub(r"^(?:结合|关于|请问|比较|对比|解释|说明|分析|介绍)+", "", item)
                       for item in named_subjects(question)})
    years = list(dict.fromkeys(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question)))
    anchor = " ".join(subjects + [year + "年" for year in years])
    topics = [{"id":key, "label":label, "query":f"{anchor} {terms}".strip()}
              for _, key, label, terms in sorted(matches)[:3]]
    return topics, subjects


def reasoning_plan(question: str) -> dict:
    """Bounded query decomposition. Fact periods are not policy effective dates."""
    years = list(dict.fromkeys(re.findall(r"(?:19|20)\d{2}", question)))
    count = "部门" in question and bool(re.search(r"多少|几个|总数|共有|统计", question))
    temporal = bool(re.search(r"成立|创立|创办|年前|当前年份", question))
    numeric = bool(re.search(r"合计|相差|差额|多多少|少多少|相隔|自然日", question))
    bridge = "负责人" in question and "部门" in question
    comparison = bool(re.search(
        r"相比|对比|比较|区别|差异|变化|演进|变更|升级|新旧|旧版|新版|旧架构|新架构|上一版|历史版本"
        r"|缩短|增加|减少|提升|降低|扩大|缩小|拉长",
        question,
    ))
    reporting = bool(
        re.search(r"的上级|的直属上级|的下属|直接下属|下属有(?:哪些|谁)|下属是谁|向谁汇报|汇报给谁|汇报关系|什么关系|关系是什么", question)
    )
    # 开放式综合题需要跨主题归纳，但不能无限扩张；单独标记后使用更短、
    # 更严格的首轮生成，避免长 JSON 先撞上客户端超时。
    synthesis = bool(re.search(
        r"结合|有依据的联系|哪些.{0,8}联系|联系.{0,8}(?:增长|投入|经营|技术)"
        r"|区分.{0,8}事实.{0,8}推断|增长.{0,8}技术投入|技术投入.{0,8}增长",
        question,
    )) and not comparison
    topics, topic_subjects = _bounded_topics(question)
    existing_complex = count or temporal or numeric or bridge or comparison or reporting or len(years) > 1
    complex_query = existing_complex or bool(topics)
    # 复杂度决定有限跳数，而不是让模型自由循环。
    relation_cues = len(re.findall(r"负责|属于|上级|下属|汇报|配套|客户|项目|版本|架构", question))
    multi_hop = bool(re.search(
        r"的上级的|的上司的|的下属的|客户的.{0,8}负责|项目的.{0,8}负责|配套.{0,8}客户|"
        r"汇报链|间接|跨部门|通过.{0,6}找到|谁管理.{0,6}的",
        question,
    ))
    max_hops = 2 if (multi_hop or reporting or comparison) else (1 if relation_cues else 0)
    max_hops = min(max_hops, config.MAX_HOPS)
    queries = [question] + [topic["query"] for topic in topics]
    if count:
        queries += [question + " 组织清单 部门名称", question + " 部门职责 组织架构",
                    "公司组织架构 一级部门 部门清单 汇报关系"]
    if temporal:
        queries += [re.sub(r"(?:19|20)\d{2}\s*年", "", question) + " 成立 创立 公司沿革",
                    "当前年份记录 今年 本年"]
    if bridge:
        queries.append(question + " 负责人 员工 部门归属")
    if comparison:
        queries += [
            question + " 旧版 新版 版本差异 变更说明",
            "旧版 历史版本 原架构 变更内容",
            "新版 当前版本 升级内容 新增能力",
        ]
        # “公司架构”指组织架构；只有控制/技术/系统架构才补控制系统查询。
        org_arch = bool(re.search(r"公司架构|组织架构|一级部门|汇报关系|部门职责", question))
        tech_arch = bool(re.search(r"控制架构|控制系统|技术架构|系统架构|机器人架构", question))
        if "架构" in question and not org_arch:
            queries += [
                "控制系统架构 旧版 v2.4 CANopen LT-C600 伺服",
                "控制系统架构 新版 v3.3 EtherCAT LT-C800 版本演进",
            ]
        if org_arch and not tech_arch:
            queries += ["公司组织架构 一级部门 汇报关系 部门职责"]
    if reporting:
        queries += [
            question + " 员工汇报关系 直接上级 下属",
            "汇报链 上级 下属 组织架构",
        ]
    if len(years) > 1:
        queries += [re.sub(r"(?:19|20)\d{2}\s*年", "", question) + f" {year}年" for year in years]
    if re.search(r"公司架构|组织架构|一级部门", question):
        queries.append("公司组织架构 一级部门 版本 更新时间 生效日期")
    # A period in a financial comparison must not discard older facts.
    # A report's 2025 figures can be published in 2026. Only policy/effectivity
    # questions implicitly apply an effective-date snapshot; explicit as_of wins.
    policy_time = bool(re.search(r"制度|标准|政策|办法|规章|差旅|考勤|折扣|服务等级|SLA|有效|生效", question, re.I))
    policy_as_of = not (comparison or len(years) > 1 or temporal or numeric or topics)
    return {"complex": complex_query, "enumeration": count, "temporal": temporal,
            "bridge": bridge, "comparison": comparison, "reporting": reporting,
            "synthesis": synthesis,
            "queries": list(dict.fromkeys(queries))[:10], "policy_as_of": policy_as_of,
            "strict_effective_date": policy_time, "max_hops": max_hops,
            "topics": topics, "topic_subjects": topic_subjects,
            "bounded_topic_only": bool(topics) and not existing_complex,
            "complexity": ("high" if synthesis or multi_hop or reporting or comparison
                           or len(years) > 1 or topics
                           else "medium" if complex_query else "low")}


def _canonical_company_queries(question: str) -> list[str]:
    """把“公司是什么/叫什么/做什么”等意图映射为文档主题查询，不生成答案。"""
    text = question or ""
    identity = bool(re.search(r"(公司|企业|贵司|我们).{0,8}(名称|叫什么|名字|全称|简称|简介|介绍|概况|是哪家|什么公司)", text))
    business = bool(
        re.search(r"(主营业务|经营范围|业务范围|什么业务|哪些业务|做什么|从事什么|靠什么|处理哪些|提供哪些|产品与服务)", text)
        or ("公司" in text and re.search(r"(主要|主营|经营|从事|做|处理).{0,6}(业务|什么|哪些)", text))
    )
    location = bool(re.search(r"(总部|地址|在哪里|located)", text)) and any(w in text for w in ("公司", "企业", "澜图"))
    queries: list[str] = []
    if identity:
        queries += [
            "澜图智能装备股份有限公司 公司简介 公司概况",
            "公司全称 总部 成立时间 主营业务",
        ]
    if business:
        queries += [
            "公司主营业务 经营范围 工业机器人 智能产线 工业软件",
            "公司简介 业务版图 产品与解决方案 智能制造整体解决方案",
        ]
    if location:
        queries += ["公司总部 地址 苏州工业园区 星澜路"]
    return queries


class QueryPlanError(RuntimeError):
    """查询规划调用失败；失败结果不进入缓存。"""


@lru_cache(maxsize=256)
def _llm_query_plan_cached(question: str, model: str) -> tuple[str, ...]:
    """让生成模型只做检索式改写，不回答事实；失败抛出，避免缓存空结果。"""
    from app.core.config import get_settings, thinking_extra_body
    from enterprise_rag import config
    from enterprise_rag.budget import current_budget

    settings = get_settings()
    if not settings.openai_api_key:
        raise QueryPlanError("未配置 OPENAI_API_KEY")
    budget = current_budget()
    timeout = 12.0
    if budget is not None:
        timeout = min(timeout, max(0.001, budget.remaining_seconds()))
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            timeout=timeout,
            max_retries=0,
        )
        prompt = (
            "你是企业知识库的检索查询改写器。只输出 JSON，不要回答问题，不要编造公司事实。\n"
            "把用户问题改写成 2-3 个适合企业文档检索的中文查询，补充同义表达、领域术语和可能的文档主题；"
            "保留原问题的时间、主体、限定条件。\n"
            '输出格式：{"queries":["..."]}。\n'
            f"用户问题：{question}"
        )
        if budget is not None:
            budget.charge_model(stage="query_plan", model=model, detail="attempt")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=220,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                extra_body=thinking_extra_body(model),
            )
        except Exception:
            if budget is not None:
                budget.note_call_result(ok=False, detail="query_plan_failed")
            raise
        if budget is not None:
            usage = getattr(response, "usage", None)
            budget.note_model_usage(
                available=model_usage_available(usage),
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
            )
        payload = json.loads(response.choices[0].message.content or "{}")
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list):
            raise QueryPlanError("invalid_query_plan")
        queries = [item.strip()[:300] for item in raw_queries if isinstance(item, str) and item.strip()]
        if not queries:
            raise QueryPlanError("empty_query_plan")
        return tuple(queries[:3])
    except QueryPlanError:
        raise
    except Exception as exc:
        raise QueryPlanError(type(exc).__name__) from exc


def _llm_query_plan(question: str) -> tuple[str, ...]:
    from enterprise_rag import config
    from enterprise_rag.budget import current_budget
    from enterprise_rag.model_router import route_for

    if not config.USE_LLM:
        return ()
    budget = current_budget()
    if budget is not None and not budget.can_call_model():
        return ()
    route = route_for({"complexity": "low"}, task="query_plan")
    try:
        return _llm_query_plan_cached(question, route["model"])
    except QueryPlanError as exc:
        logger.warning("语义查询规划失败，使用规则扩展: %s", exc)
        return ()


def semantic_query_variants(question: str, allow_llm: bool = True) -> list[str]:
    """在低置信度/泛化问法上补充语义检索式；变式只用于召回，不作为答案。"""
    # ponytail:shortcut 模型查询规划失败时降级为规则变式，保证仍可检索。
    text = question or ""
    variants = _canonical_company_queries(text)
    # 短问句、泛化问法且没有明确数字时，交给生成模型做同义改写。
    generic = (
        len(text) <= 50
        and not re.search(r"\d", text)
        and re.search(r"(是什么|哪些|什么|做什么|介绍|主要|如何|怎么|区别|包括|属于|意思)", text)
    )
    if allow_llm and generic and len(variants) < 3:
        variants.extend(_llm_query_plan(text))
    return list(dict.fromkeys(item for item in variants if item and item != text))[:4]


@lru_cache(maxsize=128)
def _hyde_hypothesis_cached(question: str, model: str) -> tuple[str, ...]:
    """Cache successful hypotheses only; exceptions are never lru-cached."""
    from app.core.config import get_settings, thinking_extra_body

    settings = get_settings()
    if not settings.openai_api_key:
        raise QueryPlanError("未配置 OPENAI_API_KEY")
    from enterprise_rag.budget import current_budget

    budget = current_budget()
    timeout = min(10.0, max(0.001, budget.remaining_seconds())) if budget else 10.0
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            timeout=timeout,
            max_retries=0,
        )
        prompt = (
            "你是企业文档检索引擎的假设片段生成器。不要回答用户，也不要声称事实为真。"
            "请写一段 80-120 字的、可能出现在公司内部文档中的中性描述，"
            "包含用户问题中的主体、术语和可能的章节词。只输出片段文本。\n"
            f"用户问题：{question}"
        )
        if budget is not None:
            budget.charge_model(stage="hyde", model=model, detail="attempt")
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=180,
                messages=[{"role": "user", "content": prompt}],
                extra_body=thinking_extra_body(model),
            )
        except Exception:
            if budget is not None:
                budget.note_call_result(ok=False, detail="hyde_failed")
            raise
        if budget is not None:
            usage = getattr(response, "usage", None)
            budget.note_model_usage(
                available=model_usage_available(usage),
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
            )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise QueryPlanError("empty_hypothesis")
        return (text[:300],)
    except Exception as exc:
        raise QueryPlanError(type(exc).__name__) from exc


def _hyde_hypothesis(question: str) -> tuple[str, ...]:
    """HyDE is an optional retrieval hint, never evidence for answering."""
    from enterprise_rag.budget import current_budget

    if not config.HYDE_ENABLED:
        return ()
    budget = current_budget()
    if budget is not None and not budget.can_call_model():
        return ()
    try:
        return _hyde_hypothesis_cached(question, config.LLM_MODEL)
    except QueryPlanError as exc:
        logger.warning("HyDE 假设片段生成失败: %s", exc)
        return ()


# Existing diagnostics clear the private hypothesis cache through this entrypoint.
_hyde_hypothesis.cache_clear = _hyde_hypothesis_cached.cache_clear


def hyde_variants(question: str, allow_llm: bool = True) -> list[str]:
    """可回退的召回实验：关闭时零调用，失败时返回空。"""
    if not config.HYDE_ENABLED or not allow_llm:
        return []
    return list(_hyde_hypothesis(question or ""))


def contextualize(question: str, messages: list[dict]) -> str:
    """Carry only user-authored topic text, never treat a previous model answer as evidence."""
    if not re.search(r"它|他们|该项目|那(?:个|么)?|再加上|上述|前者|后者|这(?:个|家公司)", question):
        return question
    history = [str(m.get("content", "")) for m in messages
               if m.get("role") == "user" and str(m.get("content", "")) != question]
    if not history:
        return question
    return f"{history[-1][:180]}；追问：{question}"


def query_terms(rewritten: str, original: str) -> list[str]:
    terms = []
    for token in tokenize(rewritten):
        if len(token) < 2 and not token.isascii():
            continue
        if token in STOPWORDS or token in {"的", "了"}:
            continue
        if len(token) == 2 and any(char in token for char in "的了是和与在吗呢"):
            continue
        if token not in terms:
            terms.append(token)
    # 原始问题中的词权重更高，放在前面。
    original_terms = [
        token
        for token in tokenize(original)
        if len(token) >= 2 and token not in STOPWORDS and not (len(token) == 2 and any(char in token for char in "的了是和与在吗呢"))
    ]
    ordered = []
    for token in original_terms + terms:
        if token not in ordered:
            ordered.append(token)
    return ordered


def wants_table(question: str) -> bool:
    return any(hint in question for hint in _TABLE_HINTS)
