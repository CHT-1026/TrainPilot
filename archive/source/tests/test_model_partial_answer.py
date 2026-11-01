"""Partial evidence must not disappear with one unrelated invalid model claim."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from enterprise_rag.answer import _llm_answer, generate_answer, ModelAnswerValidationError
from enterprise_rag.budget import RequestBudget
from enterprise_rag.evidence import supported_derived_quantities, quantity_key
from enterprise_rag.verification import verify_answer_facts, unsupported_quantitative_trends


CHUNKS=[{'doc_id':'annual','chunk_id':'annual#1','title':'2025年度经营数据','text':'2025年营业收入12亿元，同比增长20%。','score':.9},
        {'doc_id':'people','chunk_id':'people#1','title':'人员概况','text':'研发人员8人，占员工总数20%。','score':.8}]


def provider(payload):
    client=MagicMock()
    client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(payload,ensure_ascii=False)))],usage=None)
    return patch('openai.OpenAI',return_value=client)


def raw_provider(content):
    client=MagicMock()
    client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content))],usage=None)
    return patch('openai.OpenAI',return_value=client)


class PartialModelAnswerTest(unittest.TestCase):
    def setUp(self):
        self.settings=patch('app.core.config.get_settings',return_value=SimpleNamespace(
            openai_api_key='fake',openai_api_base='https://example.invalid'))
        self.thinking=patch('app.core.config.thinking_extra_body',return_value=None)
        self.settings.start();self.thinking.start()
        self.addCleanup(self.settings.stop);self.addCleanup(self.thinking.stop)

    def test_first_attempt_rejects_but_second_preserves_valid_claim(self):
        payload={'refused':False,'claims':[
            {'text':'2025年研发人员8人，占员工总数20%。','sources':[2]},
            {'text':'2025年营业收入12亿元，同比增长20%。','sources':[1]}]}
        with provider(payload):
            with self.assertRaises(ModelAnswerValidationError):_llm_answer('2025年经营与研发情况',CHUNKS)
            result=_llm_answer('2025年经营与研发情况',CHUNKS,strict=True)
        self.assertNotIn('8人',result.answer)
        self.assertIn('12亿元',result.answer)
        self.assertEqual(result.rejected_claims,[{'claim_index':1,'reason':'fact_period_subject_or_modality_mismatch'}])
        self.assertTrue(verify_answer_facts('',result.answer,CHUNKS)['verified'])

    def test_truncated_json_keeps_only_complete_claims_in_strict_mode(self):
        raw=('{"refused":false,"claims":['
             '{"text":"2025年营业收入12亿元，同比增长20%。","sources":[1]},'
             '{"text":"2025年研发人员8')
        with raw_provider(raw):
            result=_llm_answer('2025年经营与研发情况',CHUNKS,strict=True)
        self.assertIn('12亿元',result.answer)
        self.assertNotIn('研发人员',result.answer)
        self.assertIn({'claim_index':0,'reason':'output_truncated'},result.rejected_claims)

    def test_schema_and_invalid_sources_never_become_partial_answers(self):
        for payload in ({'refused':False,'claims':[{'text':'2025年营业收入12亿元','sources':[1]},
                                                 {'text':'坏引用','sources':[3]}]},
                        {'refused':False,'claims':[]}):
            with self.subTest(payload=payload),provider(payload),self.assertRaises(ModelAnswerValidationError):
                _llm_answer('2025年经营与研发情况',CHUNKS,strict=True)

    def test_no_supported_claim_still_fails(self):
        payload={'refused':False,'claims':[{'text':'2025年研发人员8人，占员工总数20%。','sources':[2]}]}
        with provider(payload),self.assertRaises(ModelAnswerValidationError):
            _llm_answer('2025年研发情况',CHUNKS,strict=True)

    def test_full_generation_marks_partial_and_renumbers_retained_sources(self):
        payload={'refused':False,'claims':[{'text':'2025年研发人员8人，占员工总数20%。','sources':[2]},
                                         {'text':'2025年营业收入12亿元，同比增长20%。','sources':[1]}]}
        with provider(payload):
            result=generate_answer('2025年经营情况',CHUNKS,use_llm=True,budget=RequestBudget())
        self.assertFalse(result['refused']);self.assertTrue(result['partial'])
        self.assertTrue(result['answer'].startswith('部分回答'))
        self.assertTrue(result['fact_check']['verified'])
        self.assertEqual(len(result['citations']),1)
        self.assertEqual(result['citations'][0]['doc_id'],'annual')
        self.assertNotIn('[2]',result['answer'])

    def test_unicode_minus_keeps_correct_equation_but_wrong_result_fails(self):
        chunks=[{'text':'全年新签订单48.90亿元，前三季度新签订单35.8亿元。'}]
        self.assertIn(quantity_key('13.10亿元'),supported_derived_quantities('48.90−35.8=13.10亿元',chunks))
        self.assertFalse(supported_derived_quantities('48.90−35.8=14.10亿元',chunks))

    def test_production_offline_is_not_shipment_count(self):
        chunks=[{'text':'公司第100,000台机器人正式下线。','title':'生产里程碑'}]
        wrong=verify_answer_facts('累计出货多少','累计出货10万台 [1]。',chunks)
        self.assertFalse(wrong['verified']);self.assertTrue(wrong['numeric_missing'])
        self.assertTrue(verify_answer_facts('下线多少','下线10万台 [1]。',chunks)['verified'])
        self.assertFalse(verify_answer_facts('下线多少','下线11万台 [1]。',chunks)['verified'])

    def test_bare_orders_alias_still_requires_comparison_evidence(self):
        chunks=[{'text':'2025年新签订单18亿元。','title':'2025年度经营数据'}]
        self.assertTrue(unsupported_quantitative_trends('2025年订单增长 [1]。',chunks))
        self.assertFalse(unsupported_quantitative_trends('2025年订单18亿元 [1]。',chunks))

    def test_empty_source_claim_can_be_omitted_without_accepting_invalid_indices(self):
        payload={'refused':False,'claims':[{'text':'2025年营业收入12亿元。','sources':[1]},
                                         {'text':'没有证据的额外推断','sources':[]}]}
        with provider(payload):
            result=_llm_answer('2025年经营情况',CHUNKS,strict=True)
        self.assertIn('12亿元',result.answer)
        self.assertNotIn('额外推断',result.answer)
        self.assertEqual(result.rejected_claims,[{'claim_index':2,'reason':'missing_citations'}])

    def test_two_production_metrics_keep_their_own_quantities(self):
        chunks=[{'text':'2025年机器人出货21,400台，第100,000台机器人于9月12日下线。','title':'2025里程碑'}]
        self.assertTrue(verify_answer_facts('',
            '2025年机器人出货21,400台，第10万台机器人于9月12日下线 [1]。',chunks)['verified'])
        self.assertFalse(verify_answer_facts('', '累计出货10万台 [1]。',chunks)['verified'])

    def test_count_metric_spans_survive_connectors_and_reversed_word_order(self):
        for connector in ('，', '、', '及', '以及', '和', '与'):
            for source in (
                f'2025年机器人出货21,400台{connector}第100,000台机器人于9月12日下线。',
                f'2025年21,400台机器人出货{connector}第100,000台机器人于9月12日下线。',
            ):
                with self.subTest(source=source):
                    chunks=[{'text':source,'title':'2025年里程碑'}]
                    self.assertTrue(verify_answer_facts('',source+' [1]',chunks)['verified'])
                    self.assertFalse(verify_answer_facts('','累计出货10万台 [1]。',chunks)['verified'])
                    self.assertFalse(verify_answer_facts('','下线21,400台 [1]。',chunks)['verified'])

    def test_production_delivery_shipment_and_offline_counts_are_distinct(self):
        source='2025年生产10台，交付3台，下线9台，出货2台。'
        chunks=[{'text':source,'title':'2025年经营资料'}]
        for metric,value in [('生产',10),('交付',3),('下线',9),('出货',2)]:
            with self.subTest(metric=metric):
                self.assertTrue(verify_answer_facts('',f'2025年{metric}{value}台 [1]。',chunks)['verified'])
                for wrong in {10,3,9,2}-{value}:
                    self.assertFalse(verify_answer_facts('',f'2025年{metric}{wrong}台 [1]。',chunks)['verified'])
        offline=[{'text':'公司第100,000台机器人正式下线。','title':'生产里程碑'}]
        self.assertFalse(verify_answer_facts('','累计交付10万台 [1]。',offline)['verified'])

    def test_count_table_uses_own_metric_column(self):
        chunks=[{'text':'| 月份 | 出货（台） | 交付（台） | 下线（台） |\n| 2025年12月 | 2 | 3 | 9 |',
                 'title':'2025年经营资料','chunk_type':'table'}]
        for metric,value in [('出货',2),('交付',3),('下线',9)]:
            with self.subTest(metric=metric):
                self.assertTrue(verify_answer_facts('',f'2025年{metric}{value}台 [1]。',chunks)['verified'])
                self.assertFalse(verify_answer_facts('',f'2025年{metric}{9 if value!=9 else 2}台 [1]。',chunks)['verified'])

    def test_count_metric_range_keeps_both_endpoints(self):
        chunks=[{'text':'本期交付15-16台。','title':'经营资料'}]
        self.assertTrue(verify_answer_facts('','本期交付15-16台 [1]。',chunks)['verified'])
        self.assertFalse(verify_answer_facts('','本期交付15-17台 [1]。',chunks)['verified'])

    def test_failed_generation_does_not_claim_the_retrieval_was_empty(self):
        with patch('enterprise_rag.answer._llm_answer',side_effect=ModelAnswerValidationError('bad','UNTRUSTED_OUTPUT')):
            result=generate_answer('结合2025年经营与研发投入分析联系',CHUNKS,use_llm=True,budget=RequestBudget())
        self.assertTrue(result['refused'])
        self.assertEqual(result['refusal_reason'],'model_output_invalid')
        self.assertIn('已找到相关资料',result['answer'])
        self.assertNotIn('未覆盖',result['answer'])
        self.assertNotIn('UNTRUSTED_OUTPUT',result['answer'])

    def test_empty_sources_do_not_hide_an_unverified_retained_subject(self):
        payload={'refused':False,'claims':[{'text':'月球基地已启用。','sources':[1]},
                                         {'text':'没有来源的推断','sources':[]}]}
        with provider(payload),self.assertRaises(ModelAnswerValidationError):
            _llm_answer('基地情况',[{'text':'总部位于苏州。','title':'总部'}],strict=True)

    def test_conjunctions_do_not_let_production_count_become_shipments(self):
        for separator in ('、','及','以及'):
            with self.subTest(separator=separator):
                body='2025年机器人出货21,400台'+separator+'第100,000台机器人于9月12日下线。'
                chunks=[{'text':body,'title':'2025里程碑'}]
                self.assertFalse(verify_answer_facts('', '累计出货10万台 [1]。',chunks)['verified'])
                self.assertTrue(verify_answer_facts('', '2025年机器人出货21,400台'+separator+'第10万台机器人于9月12日下线 [1]。',chunks)['verified'])
