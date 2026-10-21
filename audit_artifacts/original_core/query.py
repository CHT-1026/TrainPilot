"""查询理解：同义词/缩写扩展、时间范围抽取、检索词清洗。"""
import re
from datetime import date, timedelta

from app.capabilities.retrieval.tokenizer import tokenize

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
    "去年": ["2025年", "净利润", "营业收入"],
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
