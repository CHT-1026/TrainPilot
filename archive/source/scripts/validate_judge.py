"""Validate typed assertions and evidence coverage; synthetic checks are not model accuracy."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.retrieval_corpus import read_jsonl, write_json
from app.evaluation.retrieval_grading import grade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    checks = []
    for gold in read_jsonl(args.gold):
        rubric = gold['retrieval_rubric']
        citations = [group[0] for group in rubric['evidence_groups']]
        trace = {'status': 'answered', 'answer': {'status': rubric['status'], 'answer': rubric['expected'], 'citations': citations},
            'tool_calls_detail': [{'turn': rubric.get('evidence_after_turn', 0), 'result': {
                'segments': [{**r, 'text': r['quote']} for r in citations]}}]}
        checks.append({'qid': gold['qid'], 'case': 'exact_typed_with_all_evidence', 'passed': grade(trace, gold)['success']})
        for key, value in rubric['expected'].items():
            changed = copy.deepcopy(trace)
            replacement = not value if type(value) is bool else value + 0.001 if type(value) in (int,float) else 'WRONG'
            changed['answer']['answer'][key] = replacement
            checks.append({'qid': gold['qid'], 'case': 'mutate_' + key, 'passed': not grade(changed, gold)['success']})
        trace['tool_calls_detail'] = []
        checks.append({'qid': gold['qid'], 'case': 'unseen_quotes', 'passed': not grade(trace, gold)['success']})
    result = {'purpose': 'synthetic_judge_regression_not_model_results', 'checks': checks,
              'passed': sum(c['passed'] for c in checks), 'total': len(checks)}
    write_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}))
    return 0 if result['passed'] == result['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
