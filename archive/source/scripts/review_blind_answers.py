"""Offline prefix snapshots and recorded assisted reviews; no model client or network."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.grading import reference_version, review_item_id
REFERENCE = ROOT/'research/llm_blind_eval_20261029_execution/current_run.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def read_source(sources, name):
    receipt = read(sources/f'{name}.json')
    content = (sources/f'{name}.jsonl').read_bytes()
    if hashlib.sha256(content).hexdigest() != receipt['sha256']:
        raise ValueError('回答快照与封存哈希不符，不能据此记录判分')
    return receipt, [json.loads(line) for line in content.decode('utf-8').splitlines() if line.strip()]


def merge_reviews(review_files, review_index, output):
    expected = {row['item_id'] for row in rows(review_index)}
    decisions, skipped = {}, []
    for path in review_files:
        for row in rows(path):
            item_id = row['item_id']
            if item_id not in expected:
                skipped.append({'file': str(path), 'item_id': item_id})
                continue
            if item_id in decisions:
                raise ValueError('同一当前答案有重复判分，请先明确采用哪份复核记录')
            decisions[item_id] = row
    with output.open('x', encoding='utf-8') as handle:
        for item_id in sorted(decisions):
            handle.write(json.dumps(decisions[item_id], ensure_ascii=False)+'\n')
    receipt = {'review_index_sha256': hashlib.sha256(review_index.read_bytes()).hexdigest(),
               'output': str(output), 'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'included': len(decisions), 'remaining': len(expected-decisions.keys()),
               'not_in_current_index': skipped,
               'review_files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in review_files},
               'recorded_at': datetime.now(timezone.utc).isoformat(), 'model_calls': 0}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    snap = commands.add_parser('snapshot')
    snap.add_argument('--model', choices=('A', 'B', 'C', 'D'), required=True)
    snap.add_argument('--count', type=int, required=True)
    preview = commands.add_parser('preview')
    preview.add_argument('--source', required=True)
    preview.add_argument('--limit', type=int, default=20)
    record = commands.add_parser('record')
    record.add_argument('--source', required=True)
    record.add_argument('--decisions', type=Path, required=True)
    merge = commands.add_parser('merge', help='按当前参考版本索引合并复核，旧结论保留但不沿用')
    merge.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = read(REFERENCE)['run_id']
    directory = ROOT/'eval_bench/runs'/run
    sources = directory/'private/review_sources'
    reviews = directory/'private/reviews'
    sources.mkdir(exist_ok=True)
    reviews.mkdir(exist_ok=True)
    if args.operation == 'merge':
        receipt = merge_reviews(sorted(reviews.glob('*.jsonl')), directory/'private/review_index.jsonl', args.output)
        print(json.dumps(receipt, ensure_ascii=False))
        return
    if args.operation == 'snapshot':
        prefix = b''.join((directory/f'predictions/{args.model}.jsonl').read_bytes().splitlines(keepends=True)[:args.count])
        count = len(prefix.splitlines())
        if count != args.count:
            raise ValueError('请求的前缀尚未完成，未创建快照')
        name = f'{args.model}_{count:04d}'
        with (sources/f'{name}.jsonl').open('xb') as output:
            output.write(prefix)
        receipt = {'run_id': run, 'label': args.model, 'records': count,
                   'sha256': hashlib.sha256(prefix).hexdigest(),
                   'statuses': dict(Counter(json.loads(line)['status'] for line in prefix.splitlines())),
                   'recorded_at': datetime.now(timezone.utc).isoformat(),
                   'accuracy': None, 'scope': 'Immutable saved response prefix; original statuses retained, whole-run score pending'}
        (sources/f'{name}.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(receipt, ensure_ascii=False))
        return
    source, predictions = read_source(sources, args.source)
    gold = {row['qid']: row for row in rows(directory/'private/answers.jsonl')}
    manifest=read(directory/'manifest.json')
    errata = {row['qid']: row for row in rows(directory/manifest['reference_errata'])} if manifest['reference_errata'] else {}
    rules={row['qid']:row['accepted_answers'] for row in rows(directory/'private/grading_rules.jsonl')}
    versions={qid:reference_version(row['question'],row,rules.get(qid,[]),errata.get(qid)) for qid,row in gold.items()}
    if args.operation == 'preview':
        decided = {row['item_id'] for path in reviews.glob('*.jsonl') for row in rows(path)}
        pending = []
        for row in predictions:
            qid = row['qid']
            if qid in errata and errata[qid]['kind'] == 'invalid_question':
                continue
            if gold[qid].get('rubric',{}).get('method') == 'json_object_equal' and qid not in errata:
                continue
            reference = errata[qid]['corrected_answer'] if qid in errata else gold[qid]['answer']
            normalize = lambda value: unicodedata.normalize('NFC', value).strip()
            if row['status'] != 'completed' or normalize(row['text']) == normalize(reference):
                continue
            if review_item_id(run,source['label'],row,versions[qid]) not in decided:
                pending.append({'qid': qid, 'question': gold[qid]['question'],
                                'reference': reference, 'reference_version':versions[qid], 'prediction': row['text']})
        print(json.dumps({'source': args.source, 'remaining_in_snapshot': len(pending),
                          'items': pending[:args.limit]}, ensure_ascii=False, indent=2))
        return
    by_id = {row['qid']: row for row in predictions}
    decisions = read(args.decisions)
    target = reviews/(args.decisions.stem+'.jsonl')
    with target.open('x', encoding='utf-8') as output:
        for qid, decision in decisions.items():
            output.write(json.dumps({'item_id': review_item_id(run,source['label'],by_id[qid],versions[qid]),
                'correct': decision['correct'], 'reason': decision['reason'],
                'reviewer': 'Codex-assisted-offline-v2-reference-bound'}, ensure_ascii=False)+'\n')
    print(json.dumps({'file': str(target), 'records': len(decisions)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
