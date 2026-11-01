"""Audit saved requests and outputs offline; never opens a provider connection."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory: Path, labels: list[str]):
    manifest = read_json(directory/'manifest.json')
    questions = read_rows(directory/'blind/questions.jsonl')
    prompt = read_json(directory/'blind/prompt.json')
    indexed = {row['qid']: row for row in questions}
    result = {'run_id': manifest['run_id'], 'calls_to_models': 0,
              'question_count': len(questions), 'mode': manifest['mode'],
              'training_overlap_status': manifest['training_overlap_status'],
              'frozen_files_match': {name: sha256(directory/name) == expected
                                     for name, expected in manifest['sha256'].items()},
              'blind_fields_only': all(set(row) == {'qid', 'question'} for row in questions),
              'question_types': dict(Counter(manifest['question_type_by_id'].values())), 'models': {}}
    for label in labels:
        identity = read_json(directory/f'private/model_{label}.json')
        rows = read_rows(directory/f'predictions/{label}.jsonl')
        hashes_match = True
        for row in rows:
            payload = {'model': identity['model'], 'temperature': identity['temperature'],
                       identity['token_limit_parameter']: identity['max_tokens'], 'stream': False,
                       'messages': [{'role': 'system', 'content': prompt['system']},
                                    {'role': 'user', 'content': indexed[row['qid']]['question']}],
                       **identity['request_options']}
            actual = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode('utf-8')).hexdigest()
            hashes_match &= actual == row['request_sha256']
        seal_path = directory/f'predictions/{label}.seal.json'
        seal_matches = sha256(directory/f'predictions/{label}.jsonl') == read_json(seal_path)['sha256'] if seal_path.exists() else None
        completed = [row for row in rows if row['status'] == 'completed']
        latency = sorted(row['latency_seconds'] for row in completed)
        usage = Counter()
        missing_usage = 0
        for row in completed:
            if row['usage'] is None:
                missing_usage += 1
                continue
            for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if name in row['usage']:
                    usage[name] += row['usage'][name]
            details = row['usage'].get('completion_tokens_details') or {}
            if 'reasoning_tokens' in details:
                usage['reported_reasoning_tokens'] += details['reasoning_tokens']
        qids = [row['qid'] for row in rows]
        reused=[row['qid'] for row in rows if (row.get('provenance') or {}).get('kind')=='reused_exact_request']
        reused_set=set(reused)
        expected_order=([q['qid'] for q in questions if q['qid'] in reused_set]+
                        [q['qid'] for q in questions if q['qid'] not in reused_set])
        result['models'][label] = {
            'identity': identity, 'recorded': len(rows), 'unique_qids': len(set(qids)),
            'same_frozen_question_order': qids == [row['qid'] for row in questions[:len(qids)]],
            'order_matches_reuse_then_unanswered':qids==expected_order[:len(qids)],
            'reused_completed_answers':len(reused),'new_requests_recorded':len(rows)-len(reused),
            'request_hashes_match_question_only_payloads': hashes_match,
            'seal_matches': seal_matches,
            'statuses': dict(Counter(row['status'] for row in rows)),
            'response_models': dict(Counter(row['response_model'] for row in completed)),
            'provider_reported_usage': dict(usage), 'completed_responses_without_usage': missing_usage,
            'completed_latency_seconds': {
                'mean': round(statistics.mean(latency), 6),
                'median': round(statistics.median(latency), 6),
                'p95_nearest_rank': latency[math.ceil(len(latency)*.95)-1],
            } if latency else None,
            'sum_request_seconds': round(sum(row['latency_seconds'] for row in rows), 6),
            'sum_request_seconds_note': 'Sum of request durations; not wall time, GPU time, or token price',
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--models', nargs='+', choices=('A', 'B', 'C', 'D'), required=True)
    args = parser.parse_args()
    directory = (ROOT/'eval_bench/runs'/args.run).resolve()
    if directory.parent != (ROOT/'eval_bench/runs').resolve():
        parser.error('--run must identify one run directory')
    result = audit(directory, args.models)
    (directory/'request_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
