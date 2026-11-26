"""信息来源路由（S11）：文档、当前业务记录、公开信息三类，不互相冒充。

- 当前审批/请假状态只能走服务端业务查询，不能用旧制度文档或摘要代替；
- 公开信息只在任务确需且允许时使用，本项目默认不提供外网兜底；
- 空检索不是外传问题或调用公开搜索的许可。
"""
from __future__ import annotations

import re

SOURCE_DOCUMENT = "document"
SOURCE_BUSINESS = "business_record"
SOURCE_PUBLIC = "public_info"
SOURCE_UNKNOWN = "unknown"

_BUSINESS = (
    "我的申请", "我的请假", "我的审批", "申请进度", "申请状态", "待审批", "待办",
    "审批列表", "已提交", "还能休", "剩余年假", "考勤记录", "我的状态", "撤销申请",
    "取消申请", "谁审批", "审批到哪", "通过了吗", "批了吗",
)
_BUSINESS_RECORD = re.compile(
    r"我的.{0,5}(?:申请|请假|审批|待办|进度|状态|余额|记录)"
    r"|我的.{0,4}(?:年假|考勤|调休|休假).{0,6}(?:还剩|剩余|还有|余额|记录|几天|已休|已用)"
    r"|(?:剩余|还有|剩下).{0,3}(?:年假|调休|休假)"
    r"|(?:年假|调休|休假).{0,4}(?:余额|还剩|剩余)"
    r"|考勤记录"
)
_RECORD_SERVICE = re.compile(r"年假|考勤|调休|休假|余额")
_TEXT_TASK = re.compile(r"翻译|改写|润色|校对|解释(?:这句|这段|以下)|总结(?:这句|这段|以下)")
_POLICY_WORDS = ("制度", "规定", "标准", "办法", "政策", "流程")
_PUBLIC = (
    "新闻", "天气", "股价", "汇率", "网上", "百度", "谷歌", "搜索一下", "外部资料",
    "最新政策", "行业新闻", "公开信息", "互联网",
)
_DOCUMENT = (
    "制度", "流程", "标准", "规定", "办法", "细则", "通知", "政策", "福利", "报销",
    "差旅", "考勤", "年假", "病假", "事假", "绩效", "晋升", "采购", "供应商", "合同",
    "安全", "保密", "数据", "产品", "型号", "参数", "项目", "客户", "部门", "汇报",
    "组织", "架构", "版本", "培训", "招聘", "离职", "IT", "运维", "客服", "质量",
)


def classify_source(question: str) -> dict:
    text = (question or "").strip()
    business = [word for word in _BUSINESS if word in text]
    public = [word for word in _PUBLIC if word in text]
    document = [word for word in _DOCUMENT if word in text]
    # 用户在翻译/改写自己给出的文本时，外部主题词不构成“需要联网”。
    if _TEXT_TASK.search(text):
        return {"kind": SOURCE_DOCUMENT, "matches": [], "allowed": True,
                "reason": "处理用户已提供的文本，不需要外网检索"}
    record = bool(_BUSINESS_RECORD.search(text)) and not any(
        word in text for word in _POLICY_WORDS
    )
    if record:
        return {"kind": SOURCE_BUSINESS, "matches": business or ["record"], "allowed": True,
                "reason": "问题涉及当前用户的业务记录，应走服务端业务查询"}
    if business and not document:
        return {"kind": SOURCE_BUSINESS, "matches": business, "allowed": True,
                "reason": "问题涉及当前用户的业务记录，应走服务端业务查询"}
    if public and not document and not business:
        return {"kind": SOURCE_PUBLIC, "matches": public, "allowed": False,
                "reason": "公开信息检索默认关闭，不能用外网代替企业知识库"}
    if document or business:
        return {"kind": SOURCE_DOCUMENT, "matches": document or business, "allowed": True,
                "reason": "内部制度/业务文档问题"}
    if not text:
        return {"kind": SOURCE_UNKNOWN, "matches": [], "allowed": False, "reason": "空问题"}
    return {"kind": SOURCE_DOCUMENT, "matches": [], "allowed": True,
            "reason": "默认企业知识库，空检索时澄清或拒答，不转外网"}


def looks_text_task(text: str) -> bool:
    """用户要求翻译/改写自己给出的文本，不应被企业知识兜底改道。"""
    return bool(_TEXT_TASK.search(text or ""))


def suggested_task(question: str) -> str:
    source = classify_source(question)
    if source["kind"] == SOURCE_BUSINESS:
        progress = re.search(
            r"(?:申请|审批|请假|报销).{0,4}(?:进度|状态|到哪)|(?:进度|状态).{0,4}(?:到哪|怎么样|如何)",
            question,
        )
        if _RECORD_SERVICE.search(question) and not progress and not re.search(r"待审批|待办|审批列表", question):
            return "record_unsupported"
        if any(word in question for word in ("待审批", "待办", "审批列表")):
            return "query_pending"
        return "query"
    if source["kind"] == SOURCE_PUBLIC:
        return "clarify"
    return "policy_qa"
