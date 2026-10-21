"""查询理解：同义词/缩写扩展、语义检索改写、时间范围抽取、检索词清洗。"""
import json
import logging
import re
from datetime import date, timedelta
from functools import lru_cache

from app.capabilities.retrieval.tokenizer import tokenize

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


def reasoning_plan(question: str) -> dict:
    """Bounded query decomposition. Fact periods are not policy effective dates."""
    years = list(dict.fromkeys(re.findall(r"(?:19|20)\d{2}", question)))
    count = "部门" in question and bool(re.search(r"多少|几个|总数|共有|统计", question))
    temporal = bool(re.search(r"成立|创立|创办|年前|当前年份", question))
    numeric = bool(re.search(r"合计|相差|差额|多多少|少多少|相隔|自然日", question))
    bridge = "负责人" in question and "部门" in question
    complex_query = count or temporal or numeric or bridge or len(years) > 1
    queries = [question]
    if count:
        queries += [question + " 组织清单 部门名称", question + " 部门职责 组织架构"]
    if temporal:
        queries += [re.sub(r"(?:19|20)\d{2}\s*年", "", question) + " 成立 创立 公司沿革",
                    "当前年份记录 今年 本年"]
    if bridge:
        queries.append(question + " 负责人 员工 部门归属")
    if len(years) > 1:
        queries += [re.sub(r"(?:19|20)\d{2}\s*年", "", question) + f" {year}年" for year in years]
    # A period in a financial comparison must not discard older facts.
    # A report's 2025 figures can be published in 2026. Only policy/effectivity
    # questions implicitly apply an effective-date snapshot; explicit as_of wins.
    policy_time = bool(re.search(r"制度|标准|政策|办法|规章|差旅|考勤|折扣|服务等级|SLA|有效|生效", question, re.I))
    policy_as_of = not (len(years) > 1 or temporal or numeric)
    return {"complex": complex_query, "enumeration": count, "temporal": temporal,
            "bridge": bridge, "queries": list(dict.fromkeys(queries))[:5], "policy_as_of": policy_as_of,
            "strict_effective_date": policy_time}


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


@lru_cache(maxsize=256)
def _llm_query_plan(question: str) -> tuple[str, ...]:
    """让生成模型只做检索式改写，不回答事实；失败时返回空。"""
    from app.core.config import get_settings
    from enterprise_rag import config

    settings = get_settings()
    if not settings.openai_api_key or not config.USE_LLM:
        return ()
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            timeout=12,
            max_retries=0,
        )
        prompt = (
            "你是企业知识库的检索查询改写器。只输出 JSON，不要回答问题，不要编造公司事实。\n"
            "把用户问题改写成 2-3 个适合企业文档检索的中文查询，补充同义表达、领域术语和可能的文档主题；"
            "保留原问题的时间、主体、限定条件。\n"
            '输出格式：{"queries":["..."]}。\n'
            f"用户问题：{question}"
        )
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0,
            max_tokens=220,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            extra_body={"enable_thinking": False}
            if settings.openai_enable_thinking.strip().lower() == "false"
            else None,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        queries = [str(item).strip() for item in payload.get("queries", []) if str(item).strip()]
        return tuple(queries[:3])
    except Exception as exc:
        logger.warning("语义查询规划失败，使用规则扩展: %s", type(exc).__name__)
        return ()


def semantic_query_variants(question: str) -> list[str]:
    """在低置信度/泛化问法上补充语义检索式；变式只用于召回，不作为答案。"""
    text = question or ""
    variants = _canonical_company_queries(text)
    # 短问句、泛化问法且没有明确数字时，交给生成模型做同义改写。
    generic = (
        len(text) <= 50
        and not re.search(r"\d", text)
        and re.search(r"(是什么|哪些|什么|做什么|介绍|主要|如何|怎么|区别|包括|属于|意思)", text)
    )
    if generic and len(variants) < 3:
        variants.extend(_llm_query_plan(text))
    return list(dict.fromkeys(item for item in variants if item and item != text))[:4]


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
