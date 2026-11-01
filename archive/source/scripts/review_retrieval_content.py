"""Offline content review candidates; never repairs runtime answers or changes protocol failures.

This is a post-hoc diagnostic, separate from the strict task metric. A complete literal
JSON value followed only by one surplus brace can be examined without guessing facts.
Other malformed/prose responses remain unreviewed. All residual text is retained.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.retrieval_corpus import read_jsonl, write_json
from app.evaluation.retrieval_grading import grade
from app.evaluation.retrieval_runtime import ANSWER_SCHEMA


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field; semantic review required')
        result[key] = value
    return result


def inspect_content(trace, gold, case):
    final_turn = len(case['turns']) - 1 if 'turns' in case else 0
    final_response_received = any(r['turn'] == final_turn and 'response' in r for r in trace['rounds'])
    candidate = trace.get('answer')
    residual = ''
    original = None
    if not final_response_received:
        candidate = None
    elif candidate is None and trace['status'] == 'invalid_final_format':
        original = trace['rounds'][-1]['response']['choices'][0]['message'].get('content') or ''
        text = original.strip()
        if text.startswith('```json\n') and text.endswith('\n```'):
            text = text[8:-4]
        try:
            candidate, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(text)
        except ValueError:
            candidate = None
        else:
            residual = text[end:]
            if residual.strip() not in {'', '}'}:
                candidate = None
    verdict = None
    if candidate is not None:
        schema = copy.deepcopy(ANSWER_SCHEMA)
        schema['properties']['status'] = {'type': 'string'}
        try:
            jsonschema.validate(candidate, schema)
        except jsonschema.ValidationError:
            candidate = None
        else:
            observed = {**trace, 'answer': candidate}
            verdict = grade(observed, gold)
    return {'qid': trace['qid'], 'protocol_status': trace['status'], 'actual_answer': trace.get('answer'),
        'target_turn': final_turn, 'target_response_received': final_response_received,
        'unreviewed_reason': 'target_turn_not_answered' if not final_response_received else None,
        'literal_content_candidate': candidate, 'original_nonconforming_content': original,
        'residual': residual, 'source_supported_content': verdict['correct_answer'] and verdict['evidence_supported'] if verdict else None,
        'content_checks': verdict, 'review': 'auxiliary_literal_and_source_check_not_independent_semantic_review',
        'strict_task_success': grade(trace, gold)['success']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT/'research/retrieval_repair_20261101')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    golds = {g['qid']:g for g in read_jsonl(args.root/'private/gold.jsonl')}
    out = {}
    for name in args.runs:
        run = args.root/'runs'/name
        cases = {c['qid']:c for c in json.loads((run/'manifest.json').read_text(encoding='utf-8'))['contract']['cases']}
        out[name] = [inspect_content(json.loads(f.read_text(encoding='utf-8')), golds[f.stem], cases[f.stem])
                     for f in sorted((run/'traces').glob('*.json'))]
    write_json(args.output, {'status': 'posthoc_auxiliary_review_not_official_model_accuracy', 'runs': out})
    print(json.dumps({n:{'cases':len(rs),'content_supported':sum(r['source_supported_content'] is True for r in rs),
                        'unreviewed':sum(r['source_supported_content'] is None for r in rs)} for n,rs in out.items()}))


if __name__ == '__main__':
    main()
