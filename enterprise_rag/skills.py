"""版本化、按需加载的业务能力包（S32）。

能力包提供步骤与检查点，不是事实来源；事实仍必须来自检索证据。
只有问题命中触发词才加载，避免把重复 SOP 自动积累成系统指令。
"""
from __future__ import annotations

SKILLS = {
    "leave_submit": {
        "version": "1.0",
        "title": "请假申请办理步骤",
        "triggers": ("请假", "年假", "病假", "事假", "婚假", "产假", "丧假"),
        "steps": (
            "确认请假类型、开始日期、结束日期和原因",
            "系统按工作日计算天数并检查重叠申请",
            "展示草稿，等待用户明确确认后才提交",
            "提交后返回申请编号和当前审批环节",
        ),
        "source_docs": ("employee_handbook", "leave_policy"),
    },
    "approval": {
        "version": "1.0",
        "title": "审批处理检查点",
        "triggers": ("审批", "通过", "驳回", "待审批", "审批意见"),
        "steps": (
            "先查询申请当前状态与审批环节",
            "核对审批人是否为直属主管或 HR",
            "展示将执行的决策与意见，等待确认",
            "写入后返回新的状态与审计记录",
        ),
        "source_docs": ("approval_policy",),
    },
    "expense": {
        "version": "1.0",
        "title": "费用报销核对步骤",
        "triggers": ("报销", "发票", "差旅", "住宿", "餐补"),
        "steps": (
            "确认费用类别、金额、发生日期与票据",
            "按制度核对标准与审批权限",
            "超标准部分单独说明，不直接承诺可报销",
        ),
        "source_docs": ("travel_policy", "expense_policy"),
    },
    "procurement": {
        "version": "1.0",
        "title": "采购与供应商核对步骤",
        "triggers": ("采购", "供应商", "招标", "比价", "合同"),
        "steps": (
            "确认采购类别、金额区间和需求部门",
            "核对供应商准入、比价与合同审批要求",
            "涉及付款账期时引用合同/财务制度原文",
        ),
        "source_docs": ("procurement_policy",),
    },
}


def match_skills(question: str, limit: int = 2) -> list[dict]:
    text = question or ""
    matched = []
    for skill_id, pack in SKILLS.items():
        hits = [word for word in pack["triggers"] if word in text]
        if hits:
            matched.append({"id": skill_id, **pack, "matched_triggers": hits})
    matched.sort(key=lambda item: len(item["matched_triggers"]), reverse=True)
    return matched[:limit]


def render_skill_block(skills: list[dict]) -> str:
    if not skills:
        return ""
    lines = []
    for skill in skills:
        lines.append(f"[{skill['id']} v{skill['version']}] {skill['title']}")
        lines.extend(f"- {step}" for step in skill["steps"])
    return "\n".join(lines)


def describe_skills() -> list[dict]:
    return [
        {"id": skill_id, "version": pack["version"], "title": pack["title"],
         "triggers": list(pack["triggers"]), "source_docs": list(pack["source_docs"])}
        for skill_id, pack in SKILLS.items()
    ]
