"""Finite metric/direction checking, independent of fixed documents or values."""
import unittest

from enterprise_rag.verification import unsupported_quantitative_trends, verify_answer_facts


def source(text, title="2025年度经营数据"):
    return {"chunk_id":"test#0", "doc_id":"test", "title":title, "text":text, "section":"经营"}


class QuantitativeTrendTest(unittest.TestCase):
    def test_wrapped_amount_and_full_profit_label_keep_their_trend(self):
        chunks=[source('2025年归属于母公司股东的净利润4.12\n亿元，同比增长26.4%。')]
        self.assertFalse(unsupported_quantitative_trends('2025年归母净利润增长 [1]。',chunks))
    def test_current_value_cannot_prove_growth_or_decline(self):
        chunks=[source("2025年新签订单48.90亿元；机器人出货21,400台。")]
        for claim in ("2025年新签订单增长，机器人出货量提升 [1]。",
                      "2025年新签订单下降 [1]。"):
            with self.subTest(claim=claim):
                self.assertTrue(unsupported_quantitative_trends(claim,chunks))

    def test_correct_citation_cannot_borrow_revenue_trend_for_orders(self):
        chunks=[source("2025年营业收入42.63亿元，同比增长19.1%；新签订单48.90亿元；机器人出货21,400台。")]
        self.assertTrue(unsupported_quantitative_trends("2025年新签订单增长 [1]。",chunks))
        self.assertFalse(unsupported_quantitative_trends("2025年营业收入增长 [1]。",chunks))

    def test_direct_same_metric_and_period_supports_direction(self):
        chunks=[source("2025年新签订单48.90亿元，同比增长18.5%；机器人出货量下降。")]
        self.assertFalse(unsupported_quantitative_trends("2025年新签订单增长，机器人出货量下降 [1]。",chunks))
        self.assertTrue(unsupported_quantitative_trends("2025年机器人出货量提升 [1]。",chunks))

    def test_half_year_cannot_support_annual_trend(self):
        chunks=[source("上半年新签订单22.7亿元，同比增长18.5%。","2025年半年度报告")]
        self.assertFalse(unsupported_quantitative_trends("2025年上半年新签订单增长 [1]。",chunks))
        self.assertTrue(unsupported_quantitative_trends("2025年全年新签订单增长 [1]。",chunks))
        self.assertTrue(unsupported_quantitative_trends("2025年新签订单增长 [1]。",chunks))

    def test_two_comparable_years_prove_direction_and_money_units(self):
        chunks=[source("2024年营业收入80,000万元。","2024年度经营数据"),
                source("2025年营业收入10亿元。")]
        self.assertFalse(unsupported_quantitative_trends("2025年营业收入同比增长 [1][2]。",chunks))
        self.assertTrue(unsupported_quantitative_trends("2025年营业收入同比下降 [1][2]。",chunks))

    def test_two_nonmatching_periods_or_subjects_cannot_prove_direction(self):
        chunks=[source("2024年上半年营业收入8亿元。","2024年半年度报告"),
                source("2025年全年营业收入10亿元。")]
        self.assertTrue(unsupported_quantitative_trends("2025年营业收入增长 [1][2]。",chunks))
        chunks=[source("2024年甲公司营业收入8亿元。","2024经营数据"),
                source("2025年乙公司营业收入10亿元。")]
        self.assertTrue(unsupported_quantitative_trends("2025年甲公司营业收入增长 [1][2]。",chunks))

    def test_equal_values_do_not_prove_increase(self):
        chunks=[source("2024年营业收入8亿元。","2024年度报告"),source("2025年营业收入8亿元。")]
        self.assertTrue(unsupported_quantitative_trends("2025年营业收入增长 [1][2]。",chunks))

    def test_only_actual_sentence_citations_count(self):
        chunks=[source("2025年新签订单48.90亿元。"),source("2025年新签订单同比增长18.5%。")]
        self.assertTrue(unsupported_quantitative_trends("2025年新签订单增长 [1]。",chunks))
        self.assertFalse(unsupported_quantitative_trends("2025年新签订单增长。 [2]",chunks))

    def test_qualitative_improvement_and_unknown_or_negated_claims_are_outside_guard(self):
        for claim in ("研发投入提升技术能力 [1]。", "研发提升能力 [1]。",
                      "无法判断2025年新签订单是否增长 [1]。", "资料未提供新签订单增长的证据 [1]。",
                      "2025年新签订单未增长 [1]。", "没有证据表明机器人出货量提升 [1]。"):
            with self.subTest(claim=claim):
                self.assertFalse(unsupported_quantitative_trends(claim,[source("现有数据不足。")]))

    def test_question_or_unconfirmed_trend_is_not_an_asserted_comparison(self):
        chunks=[source("现有数据不足。")]
        self.assertFalse(unsupported_quantitative_trends("新签订单是否增长尚不明确 [1]。",chunks))
        self.assertFalse(unsupported_quantitative_trends("新签订单增长尚未证实 [1]。",chunks))

    def test_yoy_and_qoq_do_not_substitute_for_each_other(self):
        chunks=[source("2025年营业收入环比增长。")]
        self.assertTrue(unsupported_quantitative_trends("2025年营业收入同比增长 [1]。",chunks))
        self.assertFalse(unsupported_quantitative_trends("2025年营业收入环比增长 [1]。",chunks))
        chunks=[source("2024年营业收入8亿元。","2024年度报告"),source("2025年营业收入10亿元。")]
        self.assertTrue(unsupported_quantitative_trends("2025年营业收入环比增长 [1][2]。",chunks))

    def test_inference_label_does_not_excuse_an_unsupported_premise(self):
        self.assertTrue(unsupported_quantitative_trends(
            "合理推断研发投入与新签订单增长同步 [1]。",[source("2025年新签订单48.90亿元。")]))

    def test_forecast_is_not_actual_direction(self):
        chunks=[source("预计2025年新签订单增长。")]
        self.assertTrue(unsupported_quantitative_trends("2025年新签订单增长 [1]。",chunks))
        self.assertFalse(unsupported_quantitative_trends("预计2025年新签订单增长 [1]。",chunks))

    def test_direction_inherits_year_from_previous_claim(self):
        claim="2025年营业收入42.63亿元 [1]。同年新签订单增长 [2]。"
        chunks=[source("2025年营业收入42.63亿元。"),source("2024年新签订单同比增长。","2024年度报告")]
        self.assertTrue(unsupported_quantitative_trends(claim,chunks))

    def test_same_year_cannot_support_fake_yoy_calculation(self):
        chunks=[source("2025年营业收入8亿元。"),source("2025年营业收入10亿元。")]
        self.assertTrue(unsupported_quantitative_trends("2025年营业收入同比增长 [1][2]。",chunks))

    def test_fact_report_preserves_keys_and_exposes_trend_missing(self):
        report=verify_answer_facts("2025年经营变化","2025年新签订单增长 [1]。",
                                   [source("2025年新签订单48.90亿元。")])
        self.assertFalse(report["verified"])
        self.assertEqual(report["trend_missing"][0]["metric"],"订单")
        self.assertIn("numeric_missing",report)
        self.assertIn("conflicts",report)

    def test_growth_rate_is_not_misread_as_the_metric_level(self):
        self.assertFalse(unsupported_quantitative_trends("新签订单增长率下降 [1]。",
                                                       [source("新签订单增速放缓。")]))


if __name__=="__main__":
    unittest.main()
