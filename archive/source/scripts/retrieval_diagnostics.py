"""Compare raw source availability, answer correctness and citation use without new model calls."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.retrieval_corpus import CorpusTools, read_jsonl, write_json
from app.evaluation.retrieval_grading import compact, grade, visible_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT/'research/retrieval_repair_20261101')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    golds = {g['qid']:g for g in read_jsonl(args.root/'private/gold.jsonl')}
    tools = CorpusTools(args.root/'public')
    r0 = []
    for qid, gold in golds.items():
        hits = tools.run('search', {'query': gold['question'], 'top_k': 10})['hits']
        groups = gold['retrieval_rubric']['evidence_groups']
        docs = [any(h['doc_id']==r['doc_id'] for h in hits for r in g) for g in groups]
        quotes = [any(h['doc_id']==r['doc_id'] and compact(r['quote']) in compact(h['text']) for h in hits for r in g) for g in groups]
        r0.append({'qid':qid,'source_groups':docs,'quote_groups':quotes,'hits':hits})
    runs = {}
    for name in args.runs:
        items = []
        for file in sorted((args.root/'runs'/name/'traces').glob('*.json')):
            trace = json.loads(file.read_text(encoding='utf-8'))
            gold = golds[trace['qid']]
            seen = visible_segments(trace)
            available = [any(s['doc_id']==r['doc_id'] and compact(r['quote']) in compact(s['text']) for s in seen for r in g)
                         for g in gold['retrieval_rubric']['evidence_groups']]
            verdict = grade(trace,gold)
            if trace['status'] != 'answered':
                cause = trace['status']
            elif not all(available):
                cause = 'required_evidence_not_obtained'
            elif not verdict['correct_answer']:
                cause = 'facts_or_computation_mismatch_after_evidence'
            elif not verdict['evidence_supported']:
                cause = 'citation_quote_or_locator_failure'
            elif not verdict['correct_status']:
                cause = 'status_mismatch'
            else:
                cause = 'success'
            items.append({'qid':trace['qid'],'actual_answer':trace['answer'],'status':trace['status'],
                'raw_final_text':trace['rounds'][-1]['response']['choices'][0]['message'].get('content') if trace.get('rounds') and 'response' in trace['rounds'][-1] else None,
                'available_evidence_groups':available,'grade':verdict,'diagnosis':cause,
                'expected':gold['retrieval_rubric']['expected'],'evidence_reference':gold['retrieval_rubric']['evidence_groups']})
        runs[name]=items
    write_json(args.output, {'purpose':'R0 fixed full-question BM25 query is diagnostic only; R1 classifications are observational, not causal proof',
        'retriever_only':r0, 'model_runs':runs})
    print(json.dumps({'r0_complete_source_groups':sum(all(r['source_groups']) for r in r0),
        'r0_complete_quote_groups':sum(all(r['quote_groups']) for r in r0),'r0_tasks':len(r0),
        'runs':{n:{'cases':len(rs),'success':sum(r['grade']['success'] for r in rs)} for n,rs in runs.items()}}))


if __name__=='__main__':
    main()
