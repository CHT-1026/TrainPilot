import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.models.exceptions import IntentOutputError
from app.workflows.chat_nodes.recognition import _parse_payload, create_recognition_nodes


@pytest.mark.parametrize('frame', ['{}', '```json\n{}\n```'])
def test_valid_intent_in_declared_frame(frame):
    text = json.dumps({'intent': 'policy_qa', 'extracted_info': {}}, ensure_ascii=False)
    assert _parse_payload(frame.format(text))['intent'] == 'policy_qa'


@pytest.mark.parametrize('payload', [
    '{"intent":"made_up"}', '{"intent":"policy_qa","extracted_info":{"page":"2"}}',
    '{"intent":"policy_qa","role":"admin"}', 'explanation {"intent":"policy_qa"}',
    '{"intent":"policy_qa"}}',
])
def test_bad_output_is_not_coerced_or_guessed(payload):
    with pytest.raises(ValidationError):
        _parse_payload(payload)


def test_model_format_failure_is_not_user_clarification():
    model = SimpleNamespace(invoke=lambda _: SimpleNamespace(content='not JSON'))
    node = create_recognition_nodes(model)['intent_recognition']
    with pytest.raises(IntentOutputError) as exc:
        node({'user_message':'2024年上海出差住宿上限多少元？'})
    assert exc.value.status_code == 502


def test_legitimate_user_clarification_is_preserved():
    model = SimpleNamespace(invoke=lambda _: SimpleNamespace(content='{"intent":"clarify","extracted_info":{}}'))
    result = create_recognition_nodes(model)['intent_recognition']({'user_message':'那个呢？'})
    assert result['intent'] == 'clarify'


@pytest.mark.parametrize('text,expected', [
    ('2024年5月1日住宿标准', '2024-05-01'), ('2024-05-01住宿标准', '2024-05-01'),
    ('2024年2月29日', '2024-02-29'), ('不是2025年，查2024年5月1日', '2024-05-01'),
    ('不是2025年，查2024年的住宿标准', '2024-12-31'),
])
def test_exact_and_negated_time(text, expected):
    from enterprise_rag.query import extract_time
    assert extract_time(text)[0] == expected


def test_date_correction_retains_travel_topic_and_overrides_old_period():
    from app.workflows.dialog_state import apply_context, update_dialog_state
    from enterprise_rag.query import reasoning_plan
    first = '根据澜图智能差旅费用报销标准，2025年8月1日去上海，一晚住宿上限多少元？'
    previous = update_dialog_state(None, first)
    follow = '改一下，我要问的是2024年5月1日去上海的标准，不是2025年的。一晚多少元？'
    state = update_dialog_state(previous, follow)
    question, _ = apply_context(follow, [], state)
    assert '差旅' in question and '住宿' in question
    assert state['as_of'] == '2024-05-01'
    assert reasoning_plan(question)['policy_as_of'] is True


def test_real_period_comparison_not_removed():
    from enterprise_rag.query import reasoning_plan
    plan = reasoning_plan('对比2024年与2025年的住宿标准，而不是2023年')
    assert plan['comparison']
    assert plan['policy_as_of'] is False


def test_iso_date_is_not_cleared_or_merged_with_old_date():
    from app.workflows.dialog_state import apply_context, update_dialog_state
    from enterprise_rag.query import extract_time
    first=update_dialog_state(None,'澜图智能差旅住宿标准按2025-08-01查')
    assert first['as_of']=='2025-08-01'
    follow='改一下，按2024-05-01的标准呢？'
    state=update_dialog_state(first,follow)
    question,_=apply_context(follow,[],state)
    assert state['as_of']=='2024-05-01'
    assert '2025-08-01' not in question
    assert extract_time(question)[0]=='2024-05-01'
