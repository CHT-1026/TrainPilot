"""Independent boundary regressions; all provider calls are fake and no index is written."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from enterprise_rag import answer as answer_module, config, query
from enterprise_rag.budget import RequestBudget, bind_budget
from enterprise_rag.evidence import MODEL_RE, enumeration_hint, token_supported, unsupported_numeric_tokens
from enterprise_rag.prompts import answer_messages
from enterprise_rag.query import resolve_relative_years
from enterprise_rag.reasoning import infer
from enterprise_rag.verification import verify_answer_facts
from scripts.eval_assertions import evaluate_case


def chunk(text, title="", doc="d1"):
    return {"text": text, "title": title, "doc_id": doc, "chunk_id": doc + "#1", "score": .8}


class SynthesisPlanTest(unittest.TestCase):
    def test_open_ended_synthesis_is_high_complexity_and_marked(self):
        plan = query.reasoning_plan(
            "结合2025年经营情况和研发投入，说明公司增长与技术投入之间有哪些有依据的联系，区分事实与推断。"
        )
        self.assertTrue(plan["synthesis"])
        self.assertEqual(plan["complexity"], "high")
        self.assertTrue(plan["bounded_topic_only"])


class BoundFactsTest(unittest.TestCase):
    def test_title_period_and_mixed_currency_units(self):
        rows = [chunk("营业收入为96000万元。", "2022年度报告"),
                chunk("营业收入为6.4亿元。", "2021年度报告", "d2")]
        result = infer("2022年营业收入比2021年增加多少亿元？", rows)
        self.assertFalse(result["refused"])
        self.assertIn("3.2 亿元", result["answer"])
        self.assertTrue(verify_answer_facts("2022年营业收入比2021年增加多少亿元？", result["answer"], rows)["verified"])

    def test_signed_growth_and_decline_and_zero_base(self):
        rows = [chunk("2023年营业收入为9亿元。"), chunk("2022年营业收入为12亿元。", doc="d2")]
        growth = infer("2023年营业收入比2022年增长率是多少？", rows)
        decline = infer("2023年营业收入比2022年下降率是多少？", rows)
        self.assertIn("-25%", growth["answer"])
        self.assertIn("下降率为25%", decline["answer"])
        zero = infer("2023年营业收入比2022年增长率是多少？", [rows[0], chunk("2022年营业收入为0元。", doc="d2")])
        self.assertTrue(zero["refused"])

    def test_budget_is_not_realized_revenue_but_explicit_target_is_allowed(self):
        rows = [chunk("2028年营业收入目标为9亿元。")]
        self.assertTrue(infer("2028年实际营业收入是多少？", rows)["refused"])
        self.assertIsNone(infer("2028年营业收入目标是多少？", rows))

    def test_complete_chinese_offset_does_not_truncate_hundreds(self):
        answer = infer("2025年的两百零三年前是哪一年？", [])
        self.assertIn("1822", answer["answer"])
        self.assertEqual(answer["indices"], [])

    def test_unit_and_number_must_be_adjacent(self):
        rows = [chunk("11人操作111台设备。")]
        self.assertTrue(unsupported_numeric_tokens("设备数量为11台 [1]。", rows))
        self.assertFalse(unsupported_numeric_tokens("设备数量为111台 [1]。", rows))

    def test_signed_value_and_decimal_format_are_preserved(self):
        self.assertTrue(unsupported_numeric_tokens("温度为3℃ [1]。", [chunk("当前温度为-3℃。")]))
        self.assertFalse(unsupported_numeric_tokens("温度为-3℃ [1]。", [chunk("当前温度为-3.0℃。")]))

    def test_subject_and_project_must_both_match(self):
        rows = [chunk("北辰公司蓝海项目有7台设备。南星公司蓝海项目有9台设备。")]
        self.assertFalse(verify_answer_facts("北辰公司蓝海项目设备数？", "北辰公司蓝海项目有9台设备 [1]。", rows)["verified"])
        self.assertTrue(verify_answer_facts("北辰公司蓝海项目设备数？", "北辰公司蓝海项目有7台设备 [1]。", rows)["verified"])

    def test_both_polarities_and_unrelated_action(self):
        for source, claim in [("设备不支持远程启动。", "设备支持远程启动"),
                              ("设备支持远程启动。", "设备不支持远程启动")]:
            self.assertFalse(verify_answer_facts("设备支持远程启动吗？", claim + " [1]。", [chunk(source)])["verified"])
        result = verify_answer_facts("设备支持远程启动吗？", "设备支持远程启动 [1]。", [chunk("设备支持远程启动，但不支持远程停止。")])
        self.assertTrue(result["verified"])

    def test_proved_sum_restatement_preserves_unit_and_sources(self):
        rows = [chunk("北区有1,200人；南区有300人。"), chunk("另一个仓库有1500台设备。", doc="d2")]
        self.assertFalse(unsupported_numeric_tokens("1200+300=1500人 [1]。两区共1500人 [1]。", rows))
        self.assertTrue(unsupported_numeric_tokens("1200+300=1500人 [1]。两区共1500台 [1]。", rows))
        self.assertTrue(unsupported_numeric_tokens("1200+300=1500人 [1]。两区共1500人 [2]。", rows))
        self.assertTrue(unsupported_numeric_tokens("1200+300=1600人 [1]。两区共1600人 [1]。", rows))


class EvaluationContractTest(unittest.TestCase):
    def payload(self, **kw):
        result = {"answer": "设备有7台 [1]。", "refused": False,
                  "fact_check":{"verified":True},
                  "citations": [{"index": 1, "doc_id": "d1", "visibility":"all"}], "retrieval": {"chunks": [{"doc_id": "d1"}]}}
        result.update(kw)
        return result

    def test_missing_gold_cannot_vacuously_pass(self):
        result = evaluate_case({"question": "设备多少？"}, self.payload(), 1)
        self.assertFalse(result["passed"])
        self.assertIn("missing_answer_gold", result["assessment_issues"])

    def test_correct_refusal_remains_na_but_cited_refusal_needs_gold(self):
        case = {"question": "未知？", "expect_refusal": True}
        self.assertTrue(evaluate_case(case, self.payload(refused=True, answer="现有资料不足。", citations=[]), 1)["passed"])
        self.assertFalse(evaluate_case(case, self.payload(refused=True), 1)["passed"])

    def test_excluded_sources_and_invalid_indices_are_failures(self):
        case = {"question": "设备多少？", "keywords": ["7台"], "expected_doc_ids": ["d1"], "excluded_doc_ids": ["d1"]}
        self.assertFalse(evaluate_case(case, self.payload(), 1)["dimensions"]["source"])
        result = evaluate_case({"question": "设备多少？", "keywords": ["7台"]}, self.payload(citations=[{"index": "bad", "doc_id": "d1"}]), 1)
        self.assertFalse(result["dimensions"]["citation"])

    def test_source_renumbering_matches_returned_citations(self):
        result = self.payload(retrieval={"chunks": [{"doc_id": "irrelevant"}, {"doc_id": "d1"}]})
        self.assertTrue(evaluate_case({"question": "设备多少？", "keywords": ["7台"], "expected_doc_ids": ["d1"]}, result, 1)["passed"])


class GeneratedAnswerBoundaryTest(unittest.TestCase):
    def test_range_endpoint_is_not_negative_but_standalone_sign_remains(self):
        self.assertTrue(token_supported("17%", chunk("费用率区间为12%-17%。")))
        self.assertFalse(token_supported("17%", chunk("增长率为-17%。")))
        self.assertTrue(token_supported("-17%", chunk("增长率为-17%。")))

    def test_complete_model_version_and_pro_suffix(self):
        self.assertEqual(MODEL_RE.findall("LantuTwin 4.2发布，LT-C900控制器量产"), ["LantuTwin 4.2", "LT-C900"])
        self.assertTrue(token_supported("LantuTwin4.2", chunk("LantuTwin 4.2已发布。")))
        self.assertFalse(token_supported("LantuTwin4.1", chunk("LantuTwin 4.2已发布。")))
        self.assertFalse(token_supported("LT-R30", chunk("LT-R30 Pro的额定负载为30kg。")))
        self.assertTrue(token_supported("LT-R30 Pro", chunk("LT-R30 Pro的额定负载为30kg。")))

    def test_identity_can_inherit_title_but_parameters_cannot(self):
        source = chunk("适用于仓储物料配送。", "LT-AMR 产品手册")
        self.assertTrue(token_supported("LT-AMR", source))
        self.assertFalse(token_supported("800kg", source))

    def test_non_enumeration_analysis_does_not_receive_unrelated_list(self):
        source = chunk("研发中心下设3个研究所：视觉研究所、控制器研究所、机器人研究所。")
        self.assertEqual(enumeration_hint("结合研发投入解释增长有哪些有依据的联系。", [source]), "")
        self.assertTrue(enumeration_hint("研发中心有哪些研究所？", [source]))

    def test_generic_question_nouns_do_not_trigger_llm_focus_rejection(self):
        source = chunk("工业机器人覆盖装配、焊接场景。")
        with patch("enterprise_rag.store.load_index", side_effect=AssertionError("LLM focus must use its cited context")):
            self.assertTrue(answer_module._focus_supported("工业机器人有哪些类型，各适合什么应用？", [source], model_answer=True))
        self.assertFalse(answer_module._focus_supported("乙公司有哪些产品？", [chunk("甲公司提供工业机器人。")], model_answer=True))
        self.assertFalse(answer_module._focus_supported("LT-R30有哪些应用？", [chunk("LT-R60用于装配。")], model_answer=True))

    def test_extractive_focus_uses_only_visible_context_and_distinguishes_unknown_topic(self):
        source = chunk("工业机器人用于焊接和搬运。")
        with patch("enterprise_rag.store.load_index", side_effect=AssertionError("Generation must not reload index")):
            good = answer_module.generate_answer("工业机器人有哪些用途？", [source], use_llm=False)
            bad = answer_module.generate_answer("工业机器人的食堂菜单是什么？", [source], use_llm=False)
        self.assertFalse(good["refused"])
        self.assertTrue(bad["refused"])
        self.assertEqual(bad["confidence"], 0)

    def test_question_field_words_are_not_unknown_business_entities(self):
        pairs = [
            ("研发人员数量是多少？", "研发人员有120人。"),
            ("住宿上限和报销时限是多少？", "住宿不超过500元；报销须在20天内提交。"),
            ("毛利率底线是多少？", "项目毛利率不得低于28%。"),
            ("响应时效是多少？", "故障响应时间为2小时。"),
            ("标杆客户是谁？", "典型客户为北辰制造。"),
        ]
        for question, body in pairs:
            with self.subTest(question=question):
                self.assertTrue(answer_module._focus_supported(question, [chunk(body)]))
        self.assertFalse(answer_module._focus_supported("食堂菜单是什么？", [chunk("项目毛利率不得低于28%。")]))

    def test_each_extracted_semicolon_clause_has_its_source(self):
        rows = [chunk("控制器支持9轴联动；总线周期为2ms；额定负载为8kg。")]
        result = answer_module.generate_answer("控制器支持几轴联动，总线周期是多少？", rows, use_llm=False)
        self.assertFalse(result["refused"])
        self.assertEqual(result["fact_check"]["unbound_sentences"], 0)
        self.assertTrue(result["fact_check"]["verified"])

    def test_relative_year_filters_other_years_goals_and_interim_reports(self):
        rows = [chunk("2029年营业收入为31亿元；归母净利润为4亿元。", "2029年度报告"),
                chunk("2028年营业收入为24亿元。", "2028年度报告", "d2"),
                chunk("2030年营业收入目标为40亿元。", "2030经营预算", "d3"),
                chunk("2029年上半年营业收入为12亿元。", "2029半年度报告", "d4")]
        original = resolve_relative_years
        with patch.object(answer_module, "resolve_relative_years", side_effect=lambda q: original(q, reference_year=2030)):
            result = answer_module.generate_answer("公司去年赚了多少钱？", rows, use_llm=False)
        self.assertFalse(result["refused"])
        self.assertIn("31", result["answer"])
        self.assertNotIn("2028", result["answer"])
        self.assertNotIn("2030", result["answer"])
        self.assertNotIn("上半年", result["answer"])
        self.assertTrue(result["fact_check"]["verified"])

    def test_soft_metric_quantity_wrap_keeps_year_but_not_next_year(self):
        source = chunk("2027年公司营业收入为9亿元，归母净利润\n2亿元。\n2026年公司营业收入为8亿元。")
        good = verify_answer_facts("2027年利润？", "2027年公司归母净利润为2亿元 [1]。", [source])
        bad = verify_answer_facts("2027年收入？", "2027年公司营业收入为8亿元 [1]。", [source])
        self.assertTrue(good["verified"])
        self.assertFalse(bad["verified"])

    def test_late_rejection_has_zero_confidence_and_controlled_reason(self):
        with patch.object(answer_module, "_llm_answer", return_value=answer_module.ModelAnswer("设备支持装配 [1]。", [1], 1.0, "fake", [])):
            low = answer_module.generate_answer("设备用于什么？", [dict(chunk("设备支持装配。"), score=0)], use_llm=True)
        self.assertTrue(low["refused"])
        self.assertEqual(low["confidence"], 0)
        self.assertEqual(low["refusal_reason"], "low_retrieval_score")
        with patch.object(answer_module, "_llm_answer", return_value=answer_module.ModelAnswer(answer_module.REFUSAL_TEXT, [], 1.0, "fake", [])):
            refused = answer_module.generate_answer("设备用于什么？", [chunk("设备支持装配。")], use_llm=True)
        self.assertEqual(refused["refusal_reason"], "model_refusal")
        self.assertEqual(refused["confidence"], 0)

    def test_partial_evidence_prompt_preserves_missing_data_and_causal_boundary(self):
        messages = answer_messages("解释增长与研发联系，区分事实与推断。", "仅有研发投入。", strict=True)
        system, user = messages[0]["content"], messages[1]["content"]
        self.assertIn("部分覆盖", system)
        self.assertIn("不能证明的因果", system)
        self.assertNotIn("不得用计算结果替代", user)


class OptionalModelHintTest(unittest.TestCase):
    def setUp(self):
        query._hyde_hypothesis.cache_clear()
        query._llm_query_plan_cached.cache_clear()
        self.calls = []
        self.responses = []
        owner = self
        class FakeClient:
            def __init__(self, **kw):
                owner.timeout = kw["timeout"]
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            def create(self, **kw):
                owner.calls.append(kw)
                value = owner.responses.pop(0)
                if isinstance(value, Exception):
                    raise value
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))],
                                       usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10))
        self.patches = [patch("openai.OpenAI", FakeClient),
                        patch("app.core.config.get_settings", return_value=SimpleNamespace(openai_api_key="fake", openai_api_base="https://example.invalid")),
                        patch("app.core.config.thinking_extra_body", return_value=None),
                        patch.object(config, "HYDE_ENABLED", True), patch.object(config, "USE_LLM", True)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_budget_denial_is_not_cached(self):
        self.responses = ["可检索的章节片段"]
        with bind_budget(RequestBudget(allow_model=False)):
            self.assertEqual(query.hyde_variants("设备维护章节"), [])
        with bind_budget(RequestBudget()):
            self.assertTrue(query.hyde_variants("设备维护章节"))
        self.assertEqual(len(self.calls), 1)

    def test_failed_attempt_counted_and_recovery_not_cached(self):
        self.responses = [RuntimeError("mock"), "可检索的章节片段"]
        first = RequestBudget()
        with bind_budget(first):
            self.assertEqual(query.hyde_variants("传感器维护"), [])
        self.assertEqual(first.model_calls, 1)
        self.assertFalse(first.snapshot()["calls"][0]["ok"])
        second = RequestBudget()
        with bind_budget(second):
            self.assertTrue(query.hyde_variants("传感器维护"))
        self.assertEqual(second.snapshot()["usage"]["total_tokens"], 10)

    def test_empty_hypothesis_is_not_cached_and_timeout_respects_deadline(self):
        self.responses = ["", "可检索的章节片段"]
        with bind_budget(RequestBudget(deadline_seconds=.25)):
            self.assertEqual(query.hyde_variants("控制器维护"), [])
            self.assertLessEqual(self.timeout, .25)
        with bind_budget(RequestBudget()):
            self.assertTrue(query.hyde_variants("控制器维护"))
        self.assertEqual(len(self.calls), 2)

    def test_empty_query_plan_does_not_poison_cache(self):
        self.responses = ['{"queries":[]}', '{"queries":["维护手册"]}']
        with bind_budget(RequestBudget()):
            self.assertEqual(query._llm_query_plan("控制器故障如何处理"), ())
        with bind_budget(RequestBudget()):
            self.assertEqual(query._llm_query_plan("控制器故障如何处理"), ("维护手册",))


if __name__ == "__main__":
    unittest.main()
