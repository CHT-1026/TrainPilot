"""请假结果的用户文案：结构化数据与展示分离，避免让模型改写事实。"""
from datetime import date, datetime
from typing import Any

from app.capabilities.policy_engine import PolicyCheck

STATUS_LABELS = {
    "pending": "待审批",
    "approved": "已批准",
    "rejected": "已驳回",
    "cancelled": "已取消",
    "finished": "已结束",
}

STEP_LABELS = {
    "manager": "待主管审批",
    "hr": "待 HR 审批",
    "done": "审批完成",
}


def _date_text(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value or "")


def serialize_application(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "app_id": row.get("id"),
        "employee_id": row.get("employee_id"),
        "employee_name": row.get("real_name") or row.get("employee_name"),
        "manager_id": row.get("manager_id"),
        "department": row.get("department"),
        "leave_type": row.get("leave_type"),
        "leave_days": row.get("leave_days"),
        "start_date": _date_text(row.get("start_date")),
        "end_date": _date_text(row.get("end_date")),
        "reason": row.get("reason"),
        "status": row.get("status"),
        "status_label": STATUS_LABELS.get(row.get("status"), row.get("status")),
        "approval_step": row.get("approval_step"),
        "step_label": STEP_LABELS.get(row.get("approval_step"), row.get("approval_step")),
        "manager_approval": row.get("manager_approval"),
        "hr_approval": row.get("hr_approval"),
        "comments": row.get("comments"),
        "version": row.get("version", 0),
        "created_at": _date_text(row.get("created_at")),
    }


def format_my_applications(data: dict[str, Any]) -> str:
    items = data.get("items") or []
    total = data.get("total", len(items))
    if not items:
        return "您目前没有进行中的请假申请。"
    lines = [f"【我的请假申请】共 {total} 条"]
    for index, item in enumerate(items, 1):
        step = item.get("step_label") or item.get("status_label")
        lines.append(
            f"{index}. 编号: {item['app_id']} | {item['leave_type']} {item['leave_days']} 天 "
            f"({item['start_date']} 至 {item['end_date']}) | {step}"
        )
        if item.get("comments"):
            lines.append(f"   审批意见: {item['comments']}")
    if total > len(items):
        lines.append(f"仅显示前 {len(items)} 条，共 {total} 条。")
    return "\n".join(lines)


def format_pending(data: dict[str, Any]) -> str:
    items = data.get("items") or []
    total = data.get("total", len(items))
    if not items:
        return "目前没有待审批的申请。"
    lines = [f"【待审批申请】共 {total} 条"]
    for index, item in enumerate(items, 1):
        lines.append(f"{index}. 编号: {item['app_id']}")
        lines.append(
            f"   申请人: {item['employee_name']} | {item['leave_type']} {item['leave_days']} 天"
        )
        lines.append(f"   时间: {item['start_date']} 至 {item['end_date']}")
        lines.append(f"   理由: {item['reason']}")
        lines.append(f"   环节: {item['step_label']}")
    lines.append("\n可直接回复：通过 <申请编号> / 驳回 <申请编号>，系统会先给出确认草稿。")
    if total > len(items):
        lines.append(f"仅显示前 {len(items)} 条，共 {total} 条。")
    return "\n".join(lines)


def format_draft(draft: dict[str, Any], checks: list[PolicyCheck]) -> str:
    lines = ["请确认以下请假申请："]
    lines.append(f"- 类型：{draft.get('leave_type')}")
    lines.append(f"- 时间：{draft.get('start_date')} 至 {draft.get('end_date')}")
    lines.append(f"- 工作日：{draft.get('working_days')} 天")
    lines.append(f"- 原因：{draft.get('reason')}")
    if draft.get("requires_hr"):
        lines.append("- 审批路径：主管审批 → HR 审批")
    else:
        lines.append("- 审批路径：主管审批")
    notices = [check.message for check in checks if not check.blocking and check.ok and check.rule in {"hr_step", "sick_certificate"}]
    if notices:
        lines.append("- 提示：" + "；".join(notices))
    lines.append("\n回复「确认」提交，回复「取消」放弃。")
    return "\n".join(lines)


def format_approval_draft(application: dict[str, Any], decision_label: str, comments: str) -> str:
    lines = ["请确认审批操作："]
    lines.append(f"- 申请编号：{application['app_id']}")
    lines.append(f"- 申请人：{application['employee_name']}（{application.get('department') or '未填部门'}）")
    lines.append(f"- 请假：{application['leave_type']} {application['leave_days']} 天")
    lines.append(f"- 时间：{application['start_date']} 至 {application['end_date']}")
    lines.append(f"- 当前环节：{application['step_label']}")
    lines.append(f"- 审批决定：{decision_label}")
    if comments:
        lines.append(f"- 审批意见：{comments}")
    lines.append("\n回复「确认」执行，回复「取消」放弃。")
    return "\n".join(lines)


def format_cancel_draft(application: dict[str, Any]) -> str:
    return (
        "请确认取消以下申请：\n"
        f"- 申请编号：{application['app_id']}\n"
        f"- 类型：{application['leave_type']} {application['leave_days']} 天\n"
        f"- 时间：{application['start_date']} 至 {application['end_date']}\n"
        "\n回复「确认」取消，回复「取消」放弃。"
    )


def format_receipt(application: dict[str, Any]) -> str:
    if application["status"] == "cancelled":
        return f"申请 {application['app_id']} 已取消。"
    lines = [
        "请假申请已提交成功。",
        "",
        "申请详情：",
        f"- 请假类型：{application['leave_type']}",
        f"- 请假时间：{application['start_date']} 至 {application['end_date']}",
        f"- 工作日：{application['leave_days']} 天",
        f"- 请假原因：{application['reason']}",
        f"- 申请编号：{application['app_id']}",
        f"- 当前状态：{application['step_label'] or application['status_label']}",
    ]
    return "\n".join(lines)


def format_approval_receipt(application: dict[str, Any], decision_label: str) -> str:
    return (
        f"{decision_label}成功。\n"
        f"- 申请编号：{application['app_id']}\n"
        f"- 申请人：{application['employee_name']}\n"
        f"- 当前状态：{application['status_label']}\n"
        f"- 当前环节：{application['step_label']}"
    )


def format_suggestion(data: dict[str, Any]) -> str:
    application = data.get("application") or {}
    lines = [f"【审批建议】申请 {application.get('app_id')}"]
    lines.append(f"结论：{data.get('recommendation')}")
    reasons = data.get("reasons") or []
    if reasons:
        lines.append("依据：")
        for reason in reasons:
            lines.append(f"- {reason}")
    risks = data.get("risks") or []
    if risks:
        lines.append("需核实：")
        for risk in risks:
            lines.append(f"- {risk}")
    citations = data.get("citations") or []
    if citations:
        lines.append("制度依据：")
        for citation in citations:
            lines.append(
                f"- [{citation.get('index')}] {citation.get('title', '')} · "
                f"{citation.get('section', '')}（{citation.get('version') or '未标注版本'}）"
            )
    lines.append("\n以上为规则校验结果，不替代主管判断。")
    return "\n".join(lines)
