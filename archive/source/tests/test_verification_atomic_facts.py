"""Positive formatting fixes must keep period, subject, sign, unit and identity guards."""
import unittest

from enterprise_rag.verification import verify_answer_facts


def chunk(text, title="公司资料"):
    return {"chunk_id":"d#0", "doc_id":"d", "title":title, "text":text, "section":"说明"}


class AtomicVerificationTest(unittest.TestCase):
    def verify(self, answer, text, title="公司资料"):
        return verify_answer_facts("说明资料内容",answer,[chunk(text,title)])

    def test_percentage_ranges_are_not_negative_upper_endpoints(self):
        for text in ("质保金5%-10%。", "标准折扣8%-15%，特殊折扣15%-25%。"):
            with self.subTest(text=text):
                self.assertTrue(self.verify(text+" [1]",text)["verified"])

    def test_shared_unit_range_checks_both_endpoints(self):
        self.assertTrue(self.verify("单线配置15-16台 [1]。","单线配置15-16台。")["verified"])
        for claim in ("单线配置14-16台 [1]。", "单线配置15-17台 [1]。"):
            self.assertFalse(self.verify(claim,"单线配置15-16台。")["verified"])

    def test_standalone_negative_and_real_negative_range_are_preserved(self):
        self.assertTrue(self.verify("温度-10℃ [1]。","温度-10℃。")["verified"])
        self.assertFalse(self.verify("毛利率-10% [1]。","毛利率10%。")["verified"])
        self.assertFalse(self.verify("毛利率10% [1]。","毛利率-10%。")["verified"])
        self.assertTrue(self.verify("温度-20℃--10℃ [1]。","温度-20℃--10℃。")["verified"])

    def test_multiple_event_years_do_not_apply_to_every_value(self):
        text="甲公司成立于2011-06-18，注册资本6.8亿元，2023年上市。"
        self.assertTrue(self.verify(text+" [1]",text)["verified"])
        self.assertFalse(self.verify(text.replace("2023年","2024年")+" [1]",text)["verified"])
        self.assertFalse(self.verify(text.replace("6.8亿元","8.6亿元")+" [1]",text)["verified"])

    def test_undated_values_before_dated_fact_are_not_relabelled(self):
        text="公司研发人员1,286\n人，占员工总数33.3%，2025年研发投入5.87亿元，占营业收入13.8%。"
        answer="公司研发人员1,286人，占员工总数33.3%，2025年研发投入5.87亿元，占营业收入13.8% [1]。"
        self.assertTrue(self.verify(answer,text)["verified"])
        for year in (2025,2026):
            claim=f"{year}年研发人员占比33.3% [1]。"
            self.assertFalse(self.verify(claim,text)["verified"])

    def test_enum_separator_keeps_model_and_following_quantity_boundaries(self):
        text="单条叠片线配置15-16台LT-R20、2台AMR，良率99.5%。"
        self.assertTrue(self.verify(text+" [1]",text)["verified"])
        self.assertFalse(self.verify(text.replace("2台AMR","3台AMR")+" [1]",text)["verified"])

    def test_units_and_adjacent_quantity_fragments_cannot_be_borrowed(self):
        self.assertFalse(self.verify("配置3台 [1]。","配置3人、13台设备。")["verified"])
        self.assertFalse(self.verify("配置2人 [1]。","配置2台设备。")["verified"])
        self.assertFalse(self.verify("研发投入5.87万元 [1]。","研发投入5.87亿元。")["verified"])

    def test_company_continuation_does_not_cross_an_explicit_new_company(self):
        source="甲公司成立于2011年，注册资本6.8亿元，2023年上市。乙公司2024年注册资本8.6亿元。"
        self.assertTrue(self.verify("甲公司成立于2011年，2023年上市 [1]。",source)["verified"])
        self.assertFalse(self.verify("甲公司2024年注册资本8.6亿元 [1]。",source)["verified"])
        self.assertFalse(self.verify("丙公司成立于2011年 [1]。",source)["verified"])

    def test_product_base_identity_cannot_match_pro_suffix(self):
        self.assertFalse(self.verify("LT-R20适用于搬运 [1]。","LT-R20 Pro适用于搬运。")["verified"])

    def test_trend_guard_remains_strict(self):
        checked=self.verify("2025年新签订单增长 [1]。","2025年新签订单48.90亿元。")
        self.assertFalse(checked["verified"])
        self.assertTrue(checked["trend_missing"])


if __name__=="__main__":
    unittest.main()
