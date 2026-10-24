"""Requested metrics, table headers and statistical periods share one contract."""
import unittest

from enterprise_rag.answer import _extractive_answer, generate_answer
from enterprise_rag.budget import RequestBudget


def row(text, title="2023年企业资料", section="正文", kind="text", score=.8, doc="evidence"):
    return {"text": text, "title": title, "section": section, "chunk_type": kind,
            "doc_id":doc, "chunk_id":doc+"#0", "score":score}


class FinancialAnswerSlotsTest(unittest.TestCase):
    def answer(self, question, rows):
        return _extractive_answer(question, rows)[0]

    def test_revenue_does_not_filter_an_independent_location(self):
        rows = [row("2023年海外营业收入2.7亿元；2022年营业收入18亿元。", score=.95),
                row("公司在芬兰赫尔辛基设立北欧技术中心。", doc="location")]
        text=self.answer("2023年海外营业收入是多少，北欧技术中心设在哪里？", rows)
        self.assertIn("2.7亿元",text)
        self.assertIn("赫尔辛基",text)
        self.assertNotIn("2022",text)
        self.assertNotIn("18亿元",text)

    def test_all_three_requested_budget_slots_and_modality_survive(self):
        rows = [row("2028年营业收入目标21亿元。", "2028年度预算", "收入目标"),
                row("研发投入3.1亿元，资本支出1.7亿元。", "2028年度预算", "费用预算", doc="cost")]
        text=self.answer("2028年营业收入目标、研发投入预算和资本支出分别是多少？",rows)
        for value in ("21亿元","3.1亿元","1.7亿元","预算"):
            self.assertIn(value,text)
        self.assertNotIn("实际",text)

    def test_planned_amount_with_no_body_marker_keeps_source_heading(self):
        text=self.answer("2028年营业收入目标是多少？",[row("2028年营业收入21亿元。","2028年度经营计划","年度目标")])
        self.assertIn("目标",text)
        self.assertIn("经营计划",text)

    def test_monthly_row_inherits_each_column_metric_and_unit(self):
        rows=[row("截至2023年4月30日，公司2023年营业收入99亿元。",score=.99),
              row("| 月份 | 营收（万元） | 订单（万元） |\n| --- | --- | --- |\n| 2023年3月 | 120 | 150 |\n| 2023年4月 | 230 | 260 |",kind="table",doc="monthly")]
        text=self.answer("2023年4月营业收入是多少？",rows)
        self.assertIn("营收230万元",text)
        self.assertNotIn("99亿元",text)
        self.assertNotIn("120万元",text)

    def test_monthly_question_cannot_accept_an_annual_snapshot(self):
        rows=[row("截至2023年4月30日，2023年营业收入99亿元。")]
        result=generate_answer("2023年4月营业收入是多少？",rows,use_llm=False,budget=RequestBudget(offline=True))
        self.assertTrue(result["refused"])

    def test_annual_metric_after_snapshot_date_still_answers_annual_question(self):
        rows=[row("截至2023年12月31日，公司2023年营业收入19亿元。")]
        text=self.answer("2023年营业收入是多少？",rows)
        self.assertIn("19亿元",text)

    def test_single_quarter_excludes_cumulative_even_when_later_clause_omits_it(self):
        rows=[row("前三季度累计营业收入27亿元；新签订单39亿元。","2023年第三季度数据","累计业绩",score=.99),
              row("第三季度新签订单12亿元。","2023年第三季度数据","订单结构",doc="quarter")]
        text=self.answer("2023年第三季度订单金额是多少？",rows)
        self.assertIn("12亿元",text)
        self.assertNotIn("39亿元",text)
        self.assertNotIn("27亿元",text)

    def test_quarter_numeric_and_q_notations_work_without_specific_year_or_value(self):
        rows=[row("第一季度订单2亿元。","2022经营数据"),
              row("第二季度订单5亿元。","2022经营数据",doc="quarter2")]
        for question in ("2022年Q2订单金额是多少？","2022年第2季度订单金额是多少？"):
            text=self.answer(question,rows)
            self.assertIn("5亿元",text)
            self.assertNotIn("2亿元",text)

    def test_annual_question_excludes_quarter_and_previous_year(self):
        rows=[row("2023年营业收入18亿元。"),row("2022年营业收入12亿元。",doc="previous"),
              row("2023年第三季度营业收入4亿元。",doc="quarter")]
        text=self.answer("2023年营业收入是多少？",rows)
        self.assertIn("18亿元",text)
        self.assertNotIn("12亿元",text)
        self.assertNotIn("4亿元",text)

    def test_research_amount_does_not_filter_research_headcount(self):
        rows=[row("2023年研发投入2.9亿元。"),row("2023年研发人员240人。",doc="headcount")]
        text=self.answer("2023年研发投入和研发人员数量分别是多少？",rows)
        self.assertIn("2.9亿元",text)
        self.assertIn("240人",text)

    def test_halfyear_and_cumulative_are_distinct_from_single_periods(self):
        rows=[row("2023年上半年新签订单9亿元。"),row("2023年新签订单23亿元。",doc="annual"),
              row("前三季度累计新签订单17亿元。","2023年第三季度报告","累计数据",doc="cumulative"),
              row("第三季度新签订单8亿元。","2023年第三季度报告",doc="single")]
        half=self.answer("2023年上半年新签订单金额是多少？",rows)
        cumulative=self.answer("2023年前三季度累计订单金额是多少？",rows)
        self.assertIn("9亿元",half)
        self.assertNotIn("23亿元",half)
        self.assertIn("17亿元",cumulative)
        self.assertNotIn("8亿元",cumulative)

    def test_capital_spending_alone_keeps_year_and_budget_modality(self):
        rows=[row("资本开支2.4亿元。","2027年度预算"),
              row("资本开支3.6亿元。","2028年度预算",doc="future")]
        text=self.answer("2027年资本开支预算是多少？",rows)
        self.assertIn("2.4亿元",text)
        self.assertIn("预算",text)
        self.assertNotIn("3.6亿元",text)

    def test_nonfinancial_semicolon_lists_are_not_cut_to_four_atomic_items(self):
        text=self.answer("支持等级如何划分？",[row("支持等级：A级现场支持；B级远程支持；C级电话支持；D级邮件支持；E级社区支持。")])
        for value in ("A级","B级","C级","D级","E级"):
            self.assertIn(value,text)


if __name__=="__main__":
    unittest.main()
