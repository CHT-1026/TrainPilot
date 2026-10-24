"""Render structured citation sources once and keep source bounds enforced."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from enterprise_rag.answer import _llm_answer, ModelAnswerValidationError


def render_claim(text, sources, source_text='工业机器人用于搬运。'):
    client=MagicMock()
    client.chat.completions.create.return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"refused":False,"claims":[{"text":text,"sources":sources}]})))],usage=None)
    chunk={"chunk_id":"d#0","doc_id":"d","text":source_text,"title":"工业机器人","section":"用途","score":.9}
    with patch('openai.OpenAI',return_value=client), \
         patch('app.core.config.get_settings',return_value=SimpleNamespace(openai_api_key='fake',openai_api_base='https://example.invalid')), \
         patch('app.core.config.thinking_extra_body',return_value=None):
        return _llm_answer('工业机器人有什么用途？',[chunk]).answer


class ClaimRenderingTest(unittest.TestCase):
    def test_undated_headcount_cannot_borrow_later_research_year(self):
        source='研发人员8人，占员工总数20%，2025年研发投入2亿元。'
        with self.assertRaises(ModelAnswerValidationError):
            render_claim('2025年研发人员8人，占员工总数20%。',[1],source)
        self.assertIn('20%',render_claim('研发人员8人，占员工总数20%（统计期未注明）。',[1],source))
    def test_model_answer_cannot_turn_single_order_value_into_growth(self):
        with self.assertRaises(ModelAnswerValidationError) as raised:
            render_claim('2025年新签订单增长。',[1],'2025年新签订单18亿元。')
        self.assertEqual('unsupported_quantitative_trend',str(raised.exception))
    def test_inline_markers_do_not_duplicate_structured_sources(self):
        result=render_claim('工业机器人用于搬运[1]。',[1])
        self.assertEqual(result.count('[1]'),1)
        self.assertIn('工业机器人用于搬运',result)

    def test_invalid_structured_source_is_still_rejected(self):
        with self.assertRaises(ModelAnswerValidationError):
            render_claim('工业机器人用于搬运[1]。',[2])
