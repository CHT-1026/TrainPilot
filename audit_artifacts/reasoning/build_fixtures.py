"""Rebuild the reviewed fixed synthetic fixture JSONL; does not modify product data."""
import json
from pathlib import Path

cases = []


def doc(i, text, **kwargs):
    return {"id": i, "text": text, **kwargs}


def case(i, category, question, expected, docs, patterns=None, **kwargs):
    cases.append({"id": i, "category": category, "question": question, "expected": expected,
                  "documents": docs, "expected_docs": [d["id"] for d in docs],
                  "answer_patterns": patterns or [], **kwargs})


departments = ["星河研究部", "赤峰制造部", "远帆销售部", "青禾财务部"]
department_docs = [doc(f"dept_{i}", f"星槐公司组织清单第{i}项。星槐公司设有{d}，该部门参与联合运营。", title=f"星槐组织记录{i}") for i, d in enumerate(departments, 1)]
case("R01_four_departments", "set_count", "星槐公司的组织清单中共有多少个不同部门？请列出部门并给出总数。", "四个不同部门，列全四个部门，引用四份记录", department_docs, [r"(?:4|四)\s*个", *departments])
case("R02_duplicate_department", "dedup_count", "星槐公司的组织清单中共有多少个不同部门？请去重后列出并给出总数。", "重复记录不重复计数，四个不同部门", department_docs + [doc("dept_repeat", "星槐公司组织清单复核：星河研究部仍参与联合运营。", title="星槐组织复核记录")], [r"(?:4|四)\s*个", *departments], expected_docs=["dept_1", "dept_2", "dept_3", "dept_4"])
case("R03_foundation_cn", "year_inference", "槐序公司的成立年份是哪一年？请根据当前年份记录和公司沿革记录计算。", "2026 - 70 = 1956 年；两份依据均需引用", [doc("clock", "槐序公司的当前年份记录：今年是2026年。", title="槐序年份记录"), doc("history", "槐序公司沿革记录：本公司于七十年前创立。", title="槐序沿革记录")], [r"1956\s*年"])
case("R04_foundation_arabic", "year_inference", "松原公司的成立年份是哪一年？请根据当前年份记录和公司沿革记录计算。", "2026 - 70 = 1956 年", [doc("clock", "松原公司的当前年份记录：今年是2026年。", title="松原年份记录"), doc("history", "松原公司沿革记录：本公司于70年前创立。", title="松原沿革记录")], [r"1956\s*年"])
case("R05_missing_year_anchor", "uncertainty", "榆光公司的成立年份是哪一年？", "缺少七十年前所相对的年份锚点，拒绝给出确定年份", [doc("history", "榆光公司沿革记录：本公司于七十年前创立。", title="榆光沿革记录", updated="", effective="")], expect_refusal=True, forbidden_answer_patterns=[r"(?:19|20)\d{2}\s*年"])
case("R06_conflicting_year_anchor", "uncertainty", "竹汀公司的成立年份是哪一年？请核对所有当前年份记录。", "相同权威的年份记录2026与2025冲突，拒绝给出唯一成立年", [doc("clock_a", "竹汀公司当前年份记录甲：今年是2026年。", title="竹汀年份甲"), doc("clock_b", "竹汀公司当前年份记录乙：今年是2025年。", title="竹汀年份乙"), doc("history", "竹汀公司沿革记录：本公司于七十年前创立。", title="竹汀沿革记录")], expect_refusal=True, forbidden_answer_patterns=[r"(?:成立于|成立年份.{0,3})(?:1955|1956)"])
case("R07_person_department", "relational_join", "云澈项目负责人的所属部门是什么？请连接项目负责人记录与员工部门记录。", "负责人顾清岚 -> 顾清岚属于霜叶研究部，因此为霜叶研究部；须显式形成项目负责人所属部门结论，仅并列两句原文不通过", [doc("project", "云澈项目负责人记录：云澈项目的负责人是顾清岚。", title="云澈项目负责人记录"), doc("staff", "员工部门记录：顾清岚所属部门为霜叶研究部。", title="员工部门归属表")], [r"霜叶研究部", r"(?:因此[^。]{0,70}霜叶研究部|云澈项目(?:的)?负责人[^。]{0,30}(?:所属部门|属于|隶属)[^。]{0,20}霜叶研究部)"])
case("R08_sum_money_units", "numeric_sum", "苔原项目甲合同与乙合同的金额合计多少元？请统一金额单位。", "1.2万元+8000元=20000元", [doc("contract_a", "苔原项目甲合同金额：1.2万元。", title="苔原甲合同"), doc("contract_b", "苔原项目乙合同金额：8000元。", title="苔原乙合同")], [r"(?:20[ ,，]?000\s*元|2(?:\.0)?\s*万元|二万\s*元)"])
case("R09_difference_money", "numeric_difference", "苔原项目甲合同金额比乙合同金额多多少元？请统一金额单位后计算差额。", "12000-8000=4000元", [doc("contract_a", "苔原项目甲合同金额：1.2万元。", title="苔原甲合同"), doc("contract_b", "苔原项目乙合同金额：8000元。", title="苔原乙合同")], [r"(?:4[ ,，]?000\s*元|0\.4\s*万元|四千\s*元)"])
case("R10_sum_cn_numbers", "numeric_sum", "银柳项目甲合同与乙合同的金额合计多少元？请统一金额单位。", "一万二千元+八千元=20000元", [doc("contract_a", "银柳项目甲合同金额：一万二千元。", title="银柳甲合同"), doc("contract_b", "银柳项目乙合同金额：八千元。", title="银柳乙合同")], [r"(?:20[ ,，]?000\s*元|2(?:\.0)?\s*万元|二万\s*元)"])
case("R11_date_exclusive", "date_difference", "白桦项目从开始日期到结束日期相隔多少天？按结束日减开始日计算，不包含开始日。", "2024-02-28 至2024-03-01，闰年相隔2天", [doc("start", "白桦项目开始日期：2024年2月28日。", title="白桦项目启动记录"), doc("end", "白桦项目结束日期：2024年3月1日。", title="白桦项目结项记录")], [r"(?:2|两|二)\s*天"])
case("R12_date_inclusive", "date_difference", "白桦项目从开始日期到结束日期共涉及多少个自然日？开始日和结束日都计入。", "2024-02-28至2024-03-01，首尾均计入为3个自然日", [doc("start", "白桦项目开始日期：2024年2月28日。", title="白桦项目启动记录"), doc("end", "白桦项目结束日期：2024年3月1日。", title="白桦项目结项记录")], [r"(?:3|三)\s*(?:个)?(?:自然日|天)"])
many_names = ["松石研究部", "银杉制造部", "长风销售部", "青芒财务部", "白露人力部", "秋海法务部", "碧泉采购部", "紫藤运维部", "流萤质量部", "天青市场部", "寒星战略部", "明川交付部"]
case("R13_twelve_evidence", "large_set_count", "栖岚公司的组织清单中共有多少个不同部门？请列出全部部门并统计总数。", "12个不同部门，必须召回并引用全部12份证据", [doc(f"many_{i}", f"栖岚公司组织清单第{i}项。栖岚公司设有{n}，该部门参与联合运营。", title=f"栖岚组织记录{i}") for i, n in enumerate(many_names, 1)], [r"(?:12|十二)\s*个", *many_names])
case("R14_employee_denied", "access_control", "苍梧项目的保密预算金额是多少？", "普通员工看不到finance保密文档，应拒答且检索结果不泄露", [doc("secret", "苍梧项目保密预算金额为987654元。", title="苍梧保密预算", visibility="finance")], expect_refusal=True, expected_docs=[], forbidden_docs=["secret"], forbidden_answer_patterns=[r"987654"])
case("R15_finance_allowed", "access_control", "苍梧项目的保密预算金额是多少？", "财务角色可读取987654元并给出引用", [doc("secret", "苍梧项目保密预算金额为987654元。", title="苍梧保密预算", visibility="finance")], [r"987[ ,，]?654\s*元"], role="finance", user_id="audit-finance")
future_docs = [doc("future_v1", "榛林公司差旅住宿标准：700元/晚。", title="榛林差旅标准v1", family="future_policy", version="v1", effective="2030-01-01", updated="2029-12-01"), doc("future_v2", "榛林公司差旅住宿标准：900元/晚。", title="榛林差旅标准v2", family="future_policy", version="v2", effective="2031-01-01", updated="2030-12-01")]
case("R16_before_first_effective", "version_safety", "截至2026年，榛林公司有效的差旅住宿标准是多少？", "两个版本都尚未生效，应拒答，不能回退到未来版本", future_docs, expect_refusal=True, expected_docs=[], forbidden_docs=["future_v1", "future_v2"], forbidden_answer_patterns=[r"(?:700|900)\s*元"], as_of="2026-12-31")
case("R17_missing_join_target", "uncertainty", "雪杉项目负责人的所属部门是什么？", "有负责人但没有员工部门映射，应拒答，不能答非所问", [doc("project", "雪杉项目负责人记录：雪杉项目负责人是梁知远。", title="雪杉项目负责人")], expect_refusal=True)
case("R18_no_scope_total", "bounded_count", "这份摘录能否确定霁川公司所有部门的总数？", "资料明确仅摘录局部，无法断言公司部门总数，应拒答", [doc("partial", "霁川公司组织资料节选：仅摘录营销部和研发部，完整部门清单未提供。", title="霁川组织资料节选")], expect_refusal=True)
case("R19_employee_after_finance", "access_control", "苍梧项目的保密预算金额是多少？", "财务用户访问后，切换普通员工仍须拒答；无权限结果不可继承", [doc("secret", "苍梧项目保密预算金额为987654元。", title="苍梧保密预算", visibility="finance")], expect_refusal=True, expected_docs=[], forbidden_docs=["secret"], forbidden_answer_patterns=[r"987654"], user_id="audit-second-employee")
case("R20_unrelated_year_anchor", "entity_alignment", "榆枫公司的成立年份是哪一年？请根据榆枫沿革计算。", "2026是另一家公司松舟的锚点，榆枫的七十年前无锚点，必须拒答", [doc("other_clock", "松舟公司的当前年份记录：今年是2026年。", title="松舟年份记录"), doc("target_history", "榆枫公司沿革记录：本公司于七十年前创立。", title="榆枫沿革记录", updated="", effective="")], expect_refusal=True, expected_docs=["target_history"], forbidden_answer_patterns=[r"1956\s*年"])

path = Path(__file__).with_name("cases.jsonl")
path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in cases), encoding="utf-8")
print(path, len(cases))
