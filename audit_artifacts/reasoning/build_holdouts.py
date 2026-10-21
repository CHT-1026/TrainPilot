"""Independent holdouts, frozen before running the repaired code."""
import json
from pathlib import Path

cases = []


def d(i, text, **kwargs):
    return {"id": i, "text": text, **kwargs}


def add(i, category, q, expected, docs, patterns=None, **kwargs):
    cases.append({"id": i, "category": category, "question": q, "expected": expected,
                  "documents": docs, "expected_docs": [x["id"] for x in docs],
                  "answer_patterns": patterns or [], **kwargs})


add("H01_three_departments", "holdout_set", "望舒公司的组织清单共有多少个不同部门？请列出并计数。", "去重后3个部门，不能把元数据里的审计部当正文部门", [d("hr", "望舒公司组织清单：设有绯云人力部。", title="望舒组织清单人力"), d("ops", "望舒公司组织清单：设有山岚运维部。", title="望舒组织清单运维"), d("rd", "望舒公司组织清单：设有沧澜研发部。", title="望舒组织清单研发"), d("dup", "望舒公司组织清单复核：绯云人力部正常运作。", title="望舒组织复核")], [r"(?:3|三)\s*个", "绯云人力部", "山岚运维部", "沧澜研发部"], expected_docs=["hr", "ops", "rd"])
add("H02_changed_years", "holdout_year", "沐川公司的成立年份是哪一年？", "2027-88=1939年，避免固定1956", [d("history", "沐川公司沿革：本公司于八十八年前创立。", title="沐川沿革"), d("clock", "沐川公司当前年份记录：今年是2027年。", title="沐川年份")], [r"1939\s*年"])
add("H03_equal_contract_amounts", "holdout_money", "荒原项目甲合同与乙合同的金额合计多少元？", "两个不同合同恰好同额，3500+3500=7000，金额不能作为去重键", [d("contract_b", "荒原项目乙合同金额：3500元。", title="荒原乙合同"), d("contract_a", "荒原项目甲合同金额：3500元。", title="荒原甲合同")], [r"(?:7[ ,，]?000\s*元|0\.7\s*万元|七千\s*元)"])
add("H04_large_unit_sum", "holdout_money", "西岭项目甲合同与乙合同的金额合计多少元？", "1.25万元+0.0075亿元=762500元", [d("contract_a", "西岭项目甲合同金额：1.25万元。", title="西岭甲合同"), d("contract_b", "西岭项目乙合同金额：0.0075亿元。", title="西岭乙合同")], [r"(?:762[ ,，]?500\s*元|76\.25\s*万元)"])
add("H05_reversed_difference", "holdout_difference", "鹿鸣项目甲合同金额比乙合同金额多多少元？", "按问题主宾顺序计算甲17500-乙9500=8000，不能按文档顺序相减", [d("contract_b", "鹿鸣项目乙合同金额：9500元。", title="鹿鸣乙合同"), d("contract_a", "鹿鸣项目甲合同金额：1.75万元。", title="鹿鸣甲合同")], [r"(?:8[ ,，]?000\s*元|0\.8\s*万元|八千\s*元)"])
add("H06_cross_year_date", "holdout_date", "霜桥项目从开始日期到结束日期相隔多少天？按结束日减开始日。", "2025-12-31到2026-01-01相隔1天", [d("end", "霜桥项目结束日期：2026年1月1日。", title="霜桥结项记录"), d("start", "霜桥项目开始日期：2025年12月31日。", title="霜桥启动记录")], [r"(?:1|一)\s*天"])
add("H07_leap_day_inclusive", "holdout_date", "涧雨项目从开始日期到结束日期共涉及多少个自然日？开始日和结束日都计入。", "2024闰日2月29日到3月1日首尾计入共2个自然日", [d("start", "涧雨项目开始日期：2024年2月29日。", title="涧雨启动记录"), d("end", "涧雨项目结束日期：2024年3月1日。", title="涧雨结项记录")], [r"(?:2|二|两)\s*(?:个)?(?:自然日|天)"])
add("H08_invalid_date_refusal", "holdout_invalid_date", "临溪项目从开始日期到结束日期相隔多少天？", "2025年没有2月29日，必须拒绝确定计算且不抛未处理异常", [d("start", "临溪项目开始日期：2025年2月29日。", title="临溪启动记录"), d("end", "临溪项目结束日期：2025年3月1日。", title="临溪结项记录")], expect_refusal=True)
add("H09_missing_operand", "holdout_missing_operand", "青谷项目甲合同与乙合同的金额合计多少元？", "乙金额未提供，不能把甲金额当合计", [d("contract_a", "青谷项目甲合同金额：6300元。", title="青谷甲合同"), d("contract_b", "青谷项目乙合同仍在磋商中，合同金额尚未确定。", title="青谷乙合同")], expect_refusal=True)
add("H10_unrelated_anchor", "holdout_entity", "岚沂公司的成立年份是哪一年？", "岚沂的八十年前不能与青池的今年2028相减", [d("history", "岚沂公司沿革：本公司于八十年前创立。", title="岚沂沿革", updated="", effective=""), d("clock", "青池公司当前年份记录：今年是2028年。", title="青池年份")], expect_refusal=True, expected_docs=["history"], forbidden_answer_patterns=[r"1948\s*年"])
add("H11_partial_registry", "holdout_incomplete_set", "能否根据这些资料确定锦穗公司的所有部门总数？", "局部摘要不支持公司总数，必须拒答或明确无法确定", [d("partial", "锦穗公司部分部门摘要：列举清漪产品部、墨羽运营部。此记录仅为部分样例，不是完整清单。", title="锦穗部分摘要")], expect_refusal=True)
add("H12_three_hop_join", "holdout_three_hop", "云垂项目负责人的所属部门主管是谁？请连接项目、员工和部门记录。", "云垂负责人夏闻溪 -> 山霁研究部 -> 主管时若尘，三份证据均需引用", [d("project", "云垂项目负责人是夏闻溪。", title="云垂负责人"), d("staff", "夏闻溪所属部门为山霁研究部。", title="员工部门归属表"), d("department", "山霁研究部的部门主管是时若尘。", title="部门主管名录")], [r"时若尘"])
add("H13_unrelated_amount_distractor", "holdout_entity_money", "璃河项目甲合同与乙合同的金额合计多少元？", "璃河甲4000+乙6000=10000；不能加入无关青川合同9000", [d("a", "璃河项目甲合同金额：4000元。", title="璃河甲合同"), d("b", "璃河项目乙合同金额：6000元。", title="璃河乙合同"), d("other", "青川项目甲合同金额：9000元。", title="青川甲合同")], [r"(?:10[ ,，]?000\s*元|1(?:\.0)?\s*万元|一万\s*元)"], expected_docs=["a", "b"], forbidden_answer_patterns=[r"19[ ,，]?000\s*元"])

path = Path(__file__).with_name("holdout_cases.jsonl")
path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in cases), encoding="utf-8")
print(path, len(cases))
