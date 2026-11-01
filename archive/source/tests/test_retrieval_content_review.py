import json

import pytest

from scripts.review_retrieval_content import inspect_content


def sample(text, *, turn=0):
    return {'qid': 'x', 'status': 'invalid_final_format', 'answer': None,
        'rounds': [{'turn': turn, 'response': {'choices': [{'message': {'content': text}}]}}],
        'tool_calls_detail': [{'turn': turn, 'result': {'segments': [
            {'doc_id': 'a', 'locator': 'line:1@0', 'text': '共四个部门'}]}}]}


GOLD = {'qid': 'x', 'retrieval_rubric': {'expected': {'count': 4},
    'accepted_statuses': ['answerable'], 'evidence_groups': [[{'doc_id': 'a', 'quote': '共四个部门'}]]}}
ANSWER = {'status': 'answerable', 'answer': {'count': 4},
    'citations': [{'doc_id': 'a', 'locator': 'line:1@0', 'quote': '共四个部门'}]}
CASE = {'qid': 'x', 'question': '几个部门？'}


def test_surplus_brace_content_does_not_change_protocol_failure():
    trace = sample(json.dumps(ANSWER, ensure_ascii=False) + '}')
    review = inspect_content(trace, GOLD, CASE)
    assert review['source_supported_content'] is True
    assert review['strict_task_success'] is False
    assert review['residual'] == '}'
    assert trace['answer'] is None and trace['status'] == 'invalid_final_format'


@pytest.mark.parametrize('suffix', ['，但其实是五个', ' trailing prose', '}}'])
def test_other_suffixes_remain_unreviewed(suffix):
    review = inspect_content(sample(json.dumps(ANSWER) + suffix), GOLD, CASE)
    assert review['source_supported_content'] is None


def test_duplicate_json_keys_remain_unreviewed():
    text = json.dumps(ANSWER).replace('"count": 4', '"count": 5, "count": 4')
    assert inspect_content(sample(text), GOLD, CASE)['source_supported_content'] is None


def test_correct_literal_without_seen_source_does_not_pass():
    trace = sample(json.dumps(ANSWER) + '}')
    trace['tool_calls_detail'] = []
    assert inspect_content(trace, GOLD, CASE)['source_supported_content'] is False


def test_previous_turn_answer_is_not_graded_against_unsent_final_question():
    # Even an accidentally matching answer must not become a successful final-turn result.
    case = {'qid': 'x', 'turns': ['查旧部门', '改为新部门']}
    review = inspect_content(sample(json.dumps(ANSWER)), GOLD, case)
    assert review['source_supported_content'] is None
    assert not review['target_response_received']
    assert review['unreviewed_reason'] == 'target_turn_not_answered'
