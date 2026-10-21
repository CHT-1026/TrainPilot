"""Evidence-operator regressions: independent values, distractors and boundaries."""
import unittest
from decimal import Decimal

from enterprise_rag.reasoning import infer, number


def c(text, title=""):
    return {"text": text, "title": title}


class NumberTest(unittest.TestCase):
    def test_units_and_mixed_cn(self):
        for raw, expected in [("二零二七", 2027), ("十二万三千零五", 123005), ("一万亿", 10**12), ("一亿零三万", 100030000), ("二点七五", Decimal("2.75"))]:
            with self.subTest(raw=raw):
                self.assertEqual(number(raw), expected)


class ScopeAndMoneyTest(unittest.TestCase):
    def test_subject_distractor_is_excluded(self):
        result = infer("霁月项目甲合同与乙合同金额合计多少元？", [c("霁月项目甲合同金额：1300元。"), c("霁月项目乙合同金额：2700元。"), c("远海项目甲合同金额：9999元。")])
        self.assertIn("= 4000 元", result["answer"])
        self.assertEqual(result["indices"], [1, 2])

    def test_equal_amounts_are_distinct_operands(self):
        result = infer("竹安项目甲合同与乙合同金额合计多少元？", [c("竹安项目乙合同金额：4100元。"), c("竹安项目甲合同金额：4100元。")])
        self.assertIn("= 8200 元", result["answer"])

    def test_missing_requested_contract_cannot_use_other_contract(self):
        result = infer("棠夏项目甲合同与乙合同金额合计多少元？", [c("棠夏项目甲合同金额：1700元。"), c("棠夏项目丙合同金额：2300元。")])
        self.assertTrue(result["refused"])

    def test_duplicate_identical_fact_is_not_added_twice(self):
        result = infer("长亭项目甲合同与乙合同金额合计多少元？", [c("长亭项目甲合同金额：1110元。"), c("长亭项目甲合同金额：1110元。"), c("长亭项目乙合同金额：1890元。")])
        self.assertIn("= 3000 元", result["answer"])

    def test_conflicting_same_contract_refuses(self):
        result = infer("清江项目甲合同与乙合同金额合计多少元？", [c("清江项目甲合同金额：1000元。"), c("清江项目甲合同金额：2000元。"), c("清江项目乙合同金额：3000元。")])
        self.assertTrue(result["refused"])

    def test_direction_keeps_question_order_and_sign(self):
        result = infer("棠溪项目甲合同金额比乙合同金额多多少元？", [c("棠溪项目乙合同金额：7500元。"), c("棠溪项目甲合同金额：2250元。")])
        self.assertIn("2250 − 7500 = -5250 元", result["answer"])
        result = infer("棠溪项目甲合同金额比乙合同金额少多少元？", [c("棠溪项目乙合同金额：7500元。"), c("棠溪项目甲合同金额：2250元。")])
        self.assertIn("7500 − 2250 = 5250 元", result["answer"])

    def test_unit_normalization_keeps_decimal_exactness(self):
        result = infer("西津项目甲合同与乙合同金额合计多少元？", [c("西津项目甲合同金额：0.123万元。"), c("西津项目乙合同金额：二千七百七十元。")])
        self.assertIn("= 4000 元", result["answer"])

    def test_explicit_pair_difference_not_blocked_by_corpus_budget(self):
        result = infer("初阳项目甲合同金额比乙合同金额多多少元？", [c("初阳项目甲合同金额：7100元。"), c("初阳项目乙合同金额：4300元。")], complete=False)
        self.assertIn("= 2800 元", result["answer"])


class TimeAndJoinTest(unittest.TestCase):
    def test_general_qa_is_not_intercepted_by_subject_guard(self):
        chunks = [c("澜图智能装备股份有限公司提供工业机器人。恒星项目合同金额：5000元。"), c("海外项目需要遵守出口管制，项目毛利率底线为25%。")]
        questions = ["澜图智能是一家什么公司？", "2025年公司研发投入和研发人员数量分别是多少？", "海外项目需要遵守哪些出口管制和数据合规要求？", "2025版销售折扣制度规定的项目毛利率底线是多少？", "研发部门的员工数量是多少？", "研发部门有多少名员工？"]
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNone(infer(question, chunks))

    def test_unrelated_clock_refuses(self):
        result = infer("柳溪公司的成立年份是哪一年？", [c("柳溪公司于六十三年前创立。"), c("桃潭公司今年是2029年。")])
        self.assertTrue(result["refused"])

    def test_entirely_different_company_never_infers_requested_subject(self):
        result = infer("乙公司的成立年份是哪一年？", [c("甲公司于三十年前创立。"), c("甲公司今年是2028年。")])
        self.assertTrue(result["refused"])
        self.assertNotIn("1998", result["answer"])

    def test_title_aligns_generic_relative_fact(self):
        result = infer("云泽公司的成立年份是哪一年？", [c("本公司于二十五年前创立。", "云泽公司沿革"), c("今年是2028年。", "云泽公司年份记录")])
        self.assertIn("2003 年", result["answer"])

    def test_ambiguous_generic_history_cannot_borrow_named_clock(self):
        result = infer("竹林公司的成立年份是哪一年？", [c("本公司于四十年前创立。"), c("竹林公司今年是2029年。")])
        self.assertTrue(result["refused"])

    def test_company_age_ignores_later_institute_foundation(self):
        result = infer("到2028年，公司的成立经过多少年？", [c("公司成立于1998年4月3日。2006年研究所成立。2012年分公司成立。")])
        self.assertIn("2028 − 1998 = 30 年", result["answer"])

    def test_foundation_calendar_with_location_ignores_institute(self):
        result = infer("到2029年，公司成立经过多少年？", [c("2009 年 5 月 16 日，公司在常州工业园区成立，创始团队来自高校。2017年公司研究所成立。")])
        self.assertIn("2029 − 2009 = 20 年", result["answer"])

    def test_three_hop_and_unrelated_owner(self):
        result = infer("清音项目负责人的所属部门主管是谁？", [c("清音项目负责人是陆知微。"), c("陆知微所属部门为观澜研究部。"), c("观澜研究部的部门主管是江沅。"), c("遥岑项目负责人是闻舒。"), c("闻舒所属部门为夜航市场部。")])
        self.assertIn("主管是江沅", result["answer"])
        self.assertEqual(result["indices"], [1, 2, 3])

    def test_invalid_calendar_date_refuses_without_crash(self):
        result = infer("清枫项目从开始日期到结束日期相隔多少天？", [c("清枫项目开始日期：2026年2月29日。"), c("清枫项目结束日期：2026年3月1日。")])
        self.assertTrue(result["refused"])

    def test_leap_day_and_inclusive_boundary(self):
        result = infer("澄溪项目共涉及多少个自然日？开始日和结束日都计入。", [c("澄溪项目开始日期：2028年2月28日。"), c("澄溪项目结束日期：2028年3月1日。")])
        self.assertIn("= 3 天", result["answer"])

    def test_compound_department_name_not_split_at_conjunction(self):
        result = infer("北溪公司的组织清单共有多少个部门？", [c("北溪公司设有研发与创新部。"), c("北溪公司设有工程部。"), c("南川公司设有采购部。")])
        self.assertIn("共 2 个部门", result["answer"])
        self.assertIn("研发与创新部", result["answer"])

    def test_same_chunk_single_letter_companies_are_isolated(self):
        result = infer("甲公司的组织清单共有多少个部门？", [c("甲公司设有研发部，乙公司设有采购部。甲公司设有财务部。")])
        self.assertIn("共 2 个部门", result["answer"])
        self.assertNotIn("采购部", result["answer"])


if __name__ == "__main__":
    unittest.main()
