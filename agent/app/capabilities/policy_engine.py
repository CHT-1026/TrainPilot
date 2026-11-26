"""确定性请假政策：日期、工作日、上限、审批路径。"""
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel

ALLOWED_LEAVE_TYPES = ("年假", "病假", "事假", "婚假", "产假", "丧假")

MAX_DAYS: dict[str, int] = {
    "年假": 15,
    "病假": 30,
    "事假": 10,
    "婚假": 10,
    "产假": 158,
    "丧假": 3,
}

ANNUAL_HR_THRESHOLD = 3


class PolicyCheck(BaseModel):
    rule: str
    ok: bool
    message: str
    blocking: bool = False


def parse_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def count_working_days(start: date, end: date, holidays: set[date] | None = None) -> int:
    """统计起止日期内的工作日，排除周末与节假日。"""
    if end < start:
        return 0
    holidays = holidays or set()
    days = 0
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5 and cursor not in holidays:
            days += 1
        cursor += timedelta(days=1)
    return days


def requires_hr(leave_type: str, working_days: int) -> bool:
    return leave_type == "年假" and working_days > ANNUAL_HR_THRESHOLD


def normalize_draft(raw: dict[str, Any]) -> dict[str, Any]:
    """把中英文键统一为内部字段，并保留原始值用于确认展示。"""
    leave_type = raw.get("leave_type") or raw.get("请假类型") or ""
    start = parse_date(raw.get("start_date") or raw.get("开始日期"))
    end = parse_date(raw.get("end_date") or raw.get("结束日期"))
    reason = (raw.get("reason") or raw.get("leave_reason") or raw.get("请假原因") or "").strip()
    explicit_days = raw.get("leave_days") or raw.get("请假天数")
    return {
        "leave_type": str(leave_type).strip(),
        "start_date": start,
        "end_date": end,
        "reason": reason,
        "explicit_days": explicit_days,
    }


def validate_leave_draft(
    raw: dict[str, Any],
    *,
    today: date | None = None,
    holidays: set[date] | None = None,
) -> tuple[list[PolicyCheck], dict[str, Any]]:
    """校验请假草稿，返回检查项与规范化后的草稿。"""
    today = today or date.today()
    normalized = normalize_draft(raw)
    checks: list[PolicyCheck] = []

    leave_type = normalized["leave_type"]
    start: date | None = normalized["start_date"]
    end: date | None = normalized["end_date"]
    reason: str = normalized["reason"]

    if not leave_type:
        checks.append(PolicyCheck(rule="leave_type", ok=False, message="缺少请假类型", blocking=True))
    elif leave_type not in ALLOWED_LEAVE_TYPES:
        checks.append(
            PolicyCheck(
                rule="leave_type",
                ok=False,
                message=f"请假类型必须是：{'/'.join(ALLOWED_LEAVE_TYPES)}",
                blocking=True,
            )
        )
    else:
        checks.append(PolicyCheck(rule="leave_type", ok=True, message=leave_type))

    if not start or not end:
        checks.append(PolicyCheck(rule="dates", ok=False, message="缺少开始或结束日期", blocking=True))
        working_days = 0
    elif end < start:
        checks.append(PolicyCheck(rule="dates", ok=False, message="结束日期不能早于开始日期", blocking=True))
        working_days = 0
    elif start < today:
        checks.append(PolicyCheck(rule="dates", ok=False, message="开始日期不能早于今天", blocking=True))
        working_days = count_working_days(start, end, holidays)
    else:
        working_days = count_working_days(start, end, holidays)
        checks.append(
            PolicyCheck(
                rule="dates",
                ok=True,
                message=f"{start.isoformat()} 至 {end.isoformat()}，共 {working_days} 个工作日",
            )
        )

    if start and end and start <= end and working_days == 0:
        checks.append(PolicyCheck(rule="working_days", ok=False, message="所选日期中没有工作日", blocking=True))

    max_days = MAX_DAYS.get(leave_type)
    if max_days and working_days > max_days:
        checks.append(
            PolicyCheck(
                rule="max_days",
                ok=False,
                message=f"{leave_type}单次最多 {max_days} 天",
                blocking=True,
            )
        )
    elif max_days and working_days > 0:
        checks.append(PolicyCheck(rule="max_days", ok=True, message=f"未超过 {leave_type} {max_days} 天上限"))

    if not reason:
        checks.append(PolicyCheck(rule="reason", ok=False, message="缺少请假原因", blocking=True))
    elif len(reason) < 2:
        checks.append(PolicyCheck(rule="reason", ok=False, message="请假原因过短", blocking=True))
    else:
        checks.append(PolicyCheck(rule="reason", ok=True, message="已填写请假原因"))

    if leave_type == "年假" and working_days > ANNUAL_HR_THRESHOLD:
        checks.append(
            PolicyCheck(
                rule="hr_step",
                ok=True,
                message=f"年假超过 {ANNUAL_HR_THRESHOLD} 天，主管批准后需 HR 审批",
                blocking=False,
            )
        )
    if leave_type == "病假":
        checks.append(
            PolicyCheck(
                rule="sick_certificate",
                ok=True,
                message="病假需按制度提供医院证明",
                blocking=False,
            )
        )

    normalized.update(
        {
            "working_days": working_days,
            "requires_hr": requires_hr(leave_type, working_days),
        }
    )
    return checks, normalized


def blocking_failures(checks: list[PolicyCheck]) -> list[PolicyCheck]:
    return [check for check in checks if check.blocking and not check.ok]
