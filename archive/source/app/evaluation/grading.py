"""Offline answer comparison. Never imports or calls a model client."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import unicodedata
from itertools import combinations

from app.evaluation.config import EvaluationConfig, ModelLabel
from app.evaluation.contracts import GradingRule, Prediction, ReferenceErratum, ReviewDecision
from app.evaluation.files import digest, index_rows, load_jsonl, write_json, write_jsonl, run_directory
from app.evaluation.sampling import selected_question_ids


def normalized(text: str) -> str:
    return unicodedata.normalize('NFC', text).strip()


def reference_version(question: str, gold: dict, accepted_answers: list[str], issue: dict | None) -> str:
    basis={'question':question,'reference':gold['answer'],'rubric':gold.get('rubric'),
           'accepted_answers':accepted_answers,'erratum':issue}
    return hashlib.sha256(json.dumps(basis,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()


def review_item_id(run: str, label: str, prediction: dict, version: str) -> str:
    return hashlib.sha256(('\0'.join((run,label,prediction['qid'],prediction['text'],version))).encode('utf-8')).hexdigest()


def equal_json(actual, expected) -> bool:
    if expected is None or isinstance(expected,bool):
        return actual is expected
    if isinstance(expected,(int,float)):
        return type(actual) in (int,float) and actual==expected
    if isinstance(expected,str):
        return type(actual) is str and actual==expected
    if isinstance(expected,list):
        return (type(actual) is list and len(actual)==len(expected)
                and all(equal_json(a,e) for a,e in zip(actual,expected)))
    return (type(actual) is dict and actual.keys()==expected.keys()
            and all(equal_json(actual[key],value) for key,value in expected.items()))


def grade_json(text: str, expected: dict) -> tuple[bool,str]:
    try:
        actual=json.loads(text)
    except ValueError:
        return False,'未按题目要求输出单个有效JSON对象'
    passed=equal_json(actual,expected)
    return passed,'JSON字段、类型和内容符合要求' if passed else 'JSON字段、类型或内容与题设不符'


def summarize(rows: list[dict], total: int) -> dict:
    counts = Counter(row['verdict'] for row in rows)
    correct, wrong, pending, excluded = counts['correct'], counts['incorrect'], counts['pending_review'], counts['excluded']
    complete = len(rows) == total and pending == 0
    api_errors = sum(row['prediction_status'] == 'api_error' for row in rows)
    incomplete = sum(row['prediction_status'] == 'incomplete' for row in rows)
    valid_answers = sum(row['prediction_status'] == 'completed' and row['verdict'] != 'excluded' for row in rows)
    eligible_total = total - excluded
    return {'total': total, 'recorded': len(rows), 'decided': correct+wrong, 'correct': correct, 'incorrect': wrong,
            'eligible_total': eligible_total, 'excluded': excluded,
            'pending_review': pending, 'status': 'final' if complete else 'pending_review',
            'accuracy': correct/eligible_total if complete and eligible_total else None,
            'valid_answers': valid_answers,
            'incorrect_answers': sum(row['verdict'] == 'incorrect' and row['prediction_status'] == 'completed' for row in rows),
            'answer_accuracy': correct/valid_answers if complete and valid_answers else None,
            'api_errors': api_errors, 'incomplete_outputs': incomplete}


def score(config: EvaluationConfig, run_id: str, labels: list[ModelLabel],
          review_file: Path | None = None) -> dict:
    directory = run_directory(config, run_id)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] == 'infrastructure_invalid':
        raise ValueError('该轮已确认基础设施故障，保留原始失败记录，但不能发布为模型能力成绩')
    for name, expected in manifest['sha256'].items():
        if digest(directory/name) != expected:
            raise ValueError('冻结输入已变化，不能沿用该运行评分')
    questions = index_rows(load_jsonl(directory/'blind/questions.jsonl'), 'qid')
    errata_file = directory/manifest['reference_errata'] if manifest['reference_errata'] else None
    errata = index_rows([ReferenceErratum.model_validate(row).model_dump() for row in load_jsonl(errata_file)], 'qid') if errata_file else {}
    if not errata.keys() <= questions.keys():
        raise ValueError('参考勘误包含本批题库之外的qid')
    gold = index_rows(load_jsonl(directory/'private/answers.jsonl'), 'qid')
    metadata = index_rows(load_jsonl(directory/'private/metadata.jsonl'), 'qid')
    rules = {row['qid']: GradingRule.model_validate(row) for row in load_jsonl(directory/'private/grading_rules.jsonl')}
    decisions = index_rows([ReviewDecision.model_validate(row).model_dump() for row in load_jsonl(review_file)], 'item_id') if review_file else {}
    report, review_queue, review_index, expected_review_ids, rows_by_label = {}, [], [], set(), {}
    verdicts_by_model = {}
    for label in labels:
        path = directory/f'predictions/{label}.jsonl'
        seal = json.loads((directory/f'predictions/{label}.seal.json').read_text(encoding='utf-8'))
        if digest(path) != seal['sha256']:
            raise ValueError('作答文件与已封存结果不一致')
        predictions = index_rows([Prediction.model_validate(row).model_dump() for row in load_jsonl(path)], 'qid')
        selected = set(selected_question_ids(directory, manifest, label))
        if label in manifest.get('sampling', {}) and seal.get('sample_sha256') != manifest['sampling'][label]['sha256']:
            raise ValueError('作答封存与抽样清单不一致')
        if predictions.keys() != selected:
            raise ValueError('必须完成该模型固定题单的作答记录，失败题也不能删除')
        rows = []
        for qid, prediction in predictions.items():
            aliases=rules[qid].accepted_answers if qid in rules else []
            version=reference_version(questions[qid]['question'],gold[qid],aliases,errata.get(qid))
            item_id = review_item_id(run_id,label,prediction,version)
            row = {'qid': qid, 'item_id': item_id, 'prediction': prediction['text'],
                   'reference_version':version,
                   'prediction_status': prediction['status'],
                   'reference': gold[qid]['answer'], 'verdict': 'pending_review', 'method': 'unresolved'}
            if qid in errata:
                issue = errata[qid]
                row.update(original_reference=gold[qid]['answer'], reference_issue=issue)
                if issue['kind'] == 'reference_correction':
                    row['reference'] = issue['corrected_answer']
            if qid in errata and errata[qid]['kind'] == 'invalid_question':
                row.update(verdict='excluded', method='invalid_question', reason=errata[qid]['reason'])
            elif prediction['status'] != 'completed':
                row.update(verdict='incorrect', method=prediction['status'])
            elif gold[qid].get('rubric',{}).get('method')=='json_object_equal' and qid not in errata:
                passed,reason=grade_json(prediction['text'],gold[qid]['rubric']['expected'])
                row.update(verdict='correct' if passed else 'incorrect',method='json_object_equal',reason=reason)
            else:
                accepted = [row['reference']]
                if qid in rules and qid not in errata:
                    accepted += rules[qid].accepted_answers
                if normalized(prediction['text']) in {normalized(answer) for answer in accepted}:
                    row.update(verdict='correct', method='exact_or_reviewed_alias')
                else:
                    expected_review_ids.add(item_id)
                    review_item = {'item_id': item_id, 'question': questions[qid]['question'],
                        'prediction': prediction['text'], 'reference': row['reference'],
                        'reference_version':version,'rubric':gold[qid].get('rubric'),
                        'correct': None, 'reason': '', 'reviewer': ''}
                    review_index.append(review_item)
                    if item_id in decisions:
                        decision = decisions[item_id]
                        row.update(verdict='correct' if decision['correct'] else 'incorrect',
                                   method='recorded_review', reason=decision['reason'], reviewer=decision['reviewer'])
                    else:
                        review_queue.append(review_item)
            rows.append(row)
        rows_by_label[label] = rows
        summary = summarize(rows, len(selected))
        grouped = {}
        for dimension in ('question_type', 'difficulty', 'evaluation_track'):
            groups = defaultdict(list)
            for row in rows:
                groups[metadata[row['qid']].get(dimension,'unspecified')].append(row)
            grouped[dimension] = {name: summarize(items, len(items)) for name, items in groups.items()}
        report[label] = {**summary, 'bank_total': manifest['total'], 'breakdown': grouped}
        verdicts_by_model[label] = {row['qid']: row['verdict'] for row in rows}
    if not decisions.keys() <= expected_review_ids:
        raise ValueError('复核文件包含不属于本批作答的item_id')
    for label, rows in rows_by_label.items():
        write_jsonl(directory/f'scores/{label}.jsonl', rows)
    random.Random(manifest['seed']).shuffle(review_queue)
    write_jsonl(directory/'private/review_queue.jsonl', review_queue)
    write_jsonl(directory/'private/review_index.jsonl', review_index)
    comparisons = []
    for left, right in combinations(labels, 2):
        if report[left]['status'] == report[right]['status'] == 'final':
            common_ids = verdicts_by_model[left].keys() & verdicts_by_model[right].keys()
            eligible_ids = {qid for qid in common_ids if verdicts_by_model[left][qid] != 'excluded'
                            and verdicts_by_model[right][qid] != 'excluded'}
            paired = Counter((verdicts_by_model[left][qid], verdicts_by_model[right][qid]) for qid in eligible_ids)
            left_correct = sum(verdicts_by_model[left][qid] == 'correct' for qid in eligible_ids)
            right_correct = sum(verdicts_by_model[right][qid] == 'correct' for qid in eligible_ids)
            valid_ids = {row['qid'] for row in rows_by_label[left] if row['prediction_status'] == 'completed' and row['verdict'] != 'excluded'}
            valid_ids &= {row['qid'] for row in rows_by_label[right] if row['prediction_status'] == 'completed' and row['verdict'] != 'excluded'}
            valid_paired = Counter((verdicts_by_model[left][qid], verdicts_by_model[right][qid]) for qid in valid_ids)
            wins, losses = valid_paired['incorrect', 'correct'], valid_paired['correct', 'incorrect']
            comparisons.append({'left_model': left, 'right_model': right,
                'compared_questions': len(eligible_ids),
                'comparison_scope': 'common_selected_questions',
                'left_correct': left_correct, 'right_correct': right_correct,
                'accuracy_delta': (right_correct-left_correct)/len(eligible_ids) if eligible_ids else None,
                'right_wins': paired['incorrect', 'correct'], 'right_losses': paired['correct', 'incorrect'],
                'common_valid_answers': {'total': len(valid_ids), 'right_wins': wins, 'right_losses': losses,
                                         'accuracy_delta': (wins-losses)/len(valid_ids) if valid_ids else None}})
    result = {'run_id': run_id, 'mode': manifest['mode'], 'models': report,
              'comparisons': comparisons, 'training_overlap_status': manifest['training_overlap_status'],
              'grading': {'review_file_sha256': digest(review_file) if review_file else None,
                          'errata_file_sha256': digest(errata_file) if errata_file else None,
                          'reviewers': dict(Counter(item['reviewer'] for item in decisions.values())),
                          'accuracy_denominator': 'scorable questions after disclosed benchmark exclusions; includes interface failures',
                          'answer_accuracy_denominator': 'completed answers on scorable questions only; no API failures or truncated outputs'}}
    write_json(directory/'summary.json', result)
    snapshot=hashlib.sha256(json.dumps({'summary':result,'rows':rows_by_label},ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
    history=directory/'scores/history'/snapshot
    history.mkdir(parents=True,exist_ok=True)
    for label,rows in rows_by_label.items():
        write_jsonl(history/f'{label}.jsonl',rows)
    write_json(history/'summary.json',result)
    return result
