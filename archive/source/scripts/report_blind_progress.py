"""Summarize saved evidence only; no provider request and no answer judging."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.sampling import selected_question_ids

OUT = ROOT/'research/llm_blind_eval_20261029_execution'


def main():
    reference = json.loads((OUT/'current_run.json').read_text(encoding='utf-8'))
    generation_state = reference['generation_state']
    directory = ROOT/'eval_bench/runs'/reference['run_id']
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    errata = ([json.loads(line) for line in (directory/manifest['reference_errata']).read_text(encoding='utf-8').splitlines()]
              if manifest['reference_errata'] else [])
    excluded = [row['qid'] for row in errata if row['kind'] == 'invalid_question']
    selections = {label: selected_question_ids(directory, manifest, label) for label in reference['labels']}
    eligible = {label: len(set(qids)-set(excluded)) for label, qids in selections.items()}
    grading = {'eligible_planned_by_model': eligible, 'excluded_question_ids': excluded,
               'reference_corrections': sum(row['kind'] == 'reference_correction' for row in errata)}
    results = {}
    for label, name in reference['labels'].items():
        path = directory/f'predictions/{label}.jsonl'
        raw = path.read_bytes() if path.exists() else b''
        rows = [json.loads(line) for line in raw[:raw.rfind(b'\n')+1].splitlines()]
        states = Counter(row['status'] for row in rows)
        types = Counter(manifest['question_type_by_id'][row['qid']] for row in rows)
        reused=sum((row.get('provenance') or {}).get('kind')=='reused_exact_request' for row in rows)
        results[label] = {'model': name, 'recorded': len(rows), 'planned': len(selections[label]),
                          'reused_completed':reused,'new_requests_recorded':len(rows)-reused,
                          'valid_responses': states['completed'], 'api_errors': states['api_error'],
                          'incomplete_outputs': states['incomplete'], 'type_counts': dict(types),
                          'sealed': (directory/f'predictions/{label}.seal.json').exists(), 'accuracy': None}
    receipt = {'run_id': reference['run_id'], 'recorded_at': datetime.now(timezone.utc).isoformat(),
               'scope': 'generation progress, not a correctness score', 'generation_state': generation_state,
               'grading': grading, 'models': results}
    (OUT/'progress.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 本轮模型盲测进度（非最终成绩）', '',
             f"运行：{reference['run_id']}；更新：{receipt['recorded_at']}", '',
             '| 对象 | 已记录/计划 | 复用回答 | 新请求已落盘 | 完整回答 | 接口失败 | 输出未完成 | 正确率 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in results.values():
        lines.append(f"| {row['model']} | {row['recorded']}/{row['planned']} | {row['reused_completed']} | {row['new_requests_recorded']} | {row['valid_responses']} | {row['api_errors']} | {row['incomplete_outputs']} | 本页不统计 |")
    if 'C' in reference['labels']:
        schedule = '本地Q6/Q4按各自冻结抽样清单顺序加载与作答；计划题数以逐模型清单为准。云端原回答已封存，评分在本地完成。'
    elif generation_state == 'held_for_full_reference_audit':
        schedule = '当前已按用户要求停止模型请求，先全面审核题库与标准答案；本地Q6/Q4自动等待任务已停止。'
    else:
        schedule = '本轮云端MiMo Flash与Qwen Flash；本地是否执行以当前运行范围回执为准。'
    lines += ['', schedule,
              f"原文勘误：{grading['reference_corrections']}条参考更正；{len(excluded)}条基准缺陷单列（{'、'.join(excluded)}）。逐模型可评分题计划："+'、'.join(f'{label}={count}' for label,count in eligible.items())+'。',
              '第一轮基础设施故障记录完整保留，见infrastructure_failure_1.json；不把DNS失败当模型能力成绩。',
              '千问最终指定为qwen3.6-flash；Plus只有更正前的连通探测，没有进入正式题库。',
              '原始作答在项目eval_bench/runs/<run_id>/predictions/；此报告不调用模型，也不生成正确性结论。', '']
    (OUT/'进度.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
