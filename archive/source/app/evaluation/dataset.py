"""Freeze questions separately from private grading material."""
from datetime import datetime, timezone
import random
from uuid import uuid4

from app.evaluation.config import EvaluationConfig
from app.evaluation.contracts import BlindQuestion, GradingRule
from app.evaluation.files import digest, index_rows, load_jsonl, write_json, write_jsonl
from app.evaluation.prompts import PROMPT_VERSION, SYSTEM_PROMPT


def prepare_run(config: EvaluationConfig) -> dict:
    source_questions = load_jsonl(config.path(config.question_bank))
    gold = load_jsonl(config.path(config.answer_bank))
    rules = [GradingRule.model_validate(row).model_dump() for row in load_jsonl(config.path(config.grading_rules))]
    questions_by_id = index_rows(source_questions, 'qid')
    answers_by_id = index_rows(gold, 'qid')
    rule_ids = index_rows(rules, 'qid')
    if not questions_by_id or questions_by_id.keys() != answers_by_id.keys():
        raise ValueError('题目与答案必须非空且qid一一对应')
    if not rule_ids.keys() <= questions_by_id.keys():
        raise ValueError('评分规则含题库之外的qid')
    for qid, question in questions_by_id.items():
        if question['question'] != answers_by_id[qid]['question']:
            raise ValueError(f'{qid}的题目与标准答案文本不对应')
    questions = [BlindQuestion(qid=row['qid'], question=row['question']).model_dump() for row in source_questions]
    random.Random(config.seed).shuffle(questions)
    type_order = list(dict.fromkeys(row['question_type'] for row in source_questions))
    type_by_id = {row['qid']: row['question_type'] for row in source_questions}
    if config.question_order == 'type_then_seeded_shuffle':
        questions.sort(key=lambda row: type_order.index(type_by_id[row['qid']]))
    run_id = datetime.now(timezone.utc).strftime('blind_%Y%m%dT%H%M%SZ_') + uuid4().hex[:8]
    directory = config.path(config.runs_dir) / run_id
    directory.mkdir(parents=True)
    for name in ('blind', 'private', 'predictions', 'scores'):
        (directory / name).mkdir()
    write_jsonl(directory/'blind/questions.jsonl', questions)
    write_json(directory/'blind/prompt.json', {'version': PROMPT_VERSION, 'system': SYSTEM_PROMPT})
    write_jsonl(directory/'private/answers.jsonl', gold)
    write_jsonl(directory/'private/grading_rules.jsonl', rules)
    write_jsonl(directory/'private/metadata.jsonl', [
        {**{key: row[key] for key in ('qid', 'question_type', 'difficulty', 'answer_type')},
         'evaluation_track':row.get('evaluation_track','unspecified')} for row in source_questions])
    frozen = ('blind/questions.jsonl', 'blind/prompt.json', 'private/answers.jsonl',
              'private/grading_rules.jsonl', 'private/metadata.jsonl')
    manifest = {'run_id': run_id, 'status': 'prepared_not_run', 'mode': config.mode,
                'reference_errata': None,
                'total': len(questions), 'seed': config.seed,
                'question_order': config.question_order, 'question_type_order': type_order,
                'question_type_by_id': type_by_id,
                'inference_parameters': config.inference_parameters(),
                'labels': {label: slot.slot for label, slot in config.models.items()},
                'sha256': {name: digest(directory/name) for name in frozen},
                'training_overlap_status': 'not_audited',
                'accuracy': {label: None for label in config.models}}
    write_json(directory/'manifest.json', manifest)
    return manifest
