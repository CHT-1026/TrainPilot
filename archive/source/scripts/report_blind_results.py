"""Export completed assisted scores and common-question comparisons; no model calls."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.grading import summarize


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def percentage(value):
    return '—' if value is None else f'{100*value:.2f}%'


def grouped(rows, metadata, dimension):
    groups = defaultdict(list)
    for row in rows:
        groups[metadata[row['qid']][dimension]].append(row)
    return {name: summarize(items, len(items)) for name, items in sorted(groups.items())}


def export_results(directory, labels, output):
    source = read_json(directory/'summary.json')
    manifest = read_json(directory/'manifest.json')
    errata = directory/manifest['reference_errata'] if manifest['reference_errata'] else None
    errata_sha = hashlib.sha256(errata.read_bytes()).hexdigest() if errata else None
    if errata_sha != source['grading']['errata_file_sha256']:
        raise ValueError('当前参考勘误尚未重新判分，请先执行score再导出成绩')
    for label in labels:
        if label not in source['models'] or source['models'][label]['status'] != 'final':
            raise ValueError(f'{label}尚未完成全部语义判分，不能生成完成报告')
    scores = {label: read_rows(directory/f'scores/{label}.jsonl') for label in labels}
    metadata = {row['qid']: row for row in read_rows(directory/'private/metadata.jsonl')}
    identities = {label: read_json(directory/f'private/model_{label}.json') for label in labels}
    by_id = {label: {row['qid']: row for row in rows} for label, rows in scores.items()}
    common = set.intersection(*(set(rows) for rows in by_id.values()))
    common = {qid for qid in common if all(by_id[label][qid]['verdict'] != 'excluded' for label in labels)}
    common_summary = {}
    for label in labels:
        rows = [by_id[label][qid] for qid in sorted(common)]
        common_summary[label] = {**summarize(rows, len(rows)),
                                 'by_track': grouped(rows, metadata, 'evaluation_track'),
                                 'by_type': grouped(rows, metadata, 'question_type')}
    pairs = [item for item in source['comparisons']
             if item['left_model'] in labels and item['right_model'] in labels]
    result = {'run_id': source['run_id'], 'generated_at': datetime.now(timezone.utc).isoformat(),
              'labels': labels, 'model_calls': 0, 'model_identities': identities,
              'models': {label: source['models'][label] for label in labels},
              'common_questions': {'total': len(common), 'models': common_summary}, 'pairs': pairs,
              'grading': source['grading'], 'training_overlap_status': source['training_overlap_status'],
              'scope': 'Completed assisted scores for the requested labels; not independent double-blind certification or training-effect proof.'}
    lines = ['# 模型测评离线汇总', '', f"运行：{source['run_id']}；生成：{result['generated_at']}。本报告仅覆盖候选{', '.join(labels)}。", '',
             '|候选|模型/槽位|正确/可评分|整体准确率|完整回答准确率|接口失败|输出未完成|',
             '|---|---|---:|---:|---:|---:|---:|']
    for label in labels:
        s, identity = source['models'][label], identities[label]
        lines.append(f"|{label}|{identity['model']} / {identity['slot']}|{s['correct']}/{s['eligible_total']}|{percentage(s['accuracy'])}|{percentage(s['answer_accuracy'])}|{s['api_errors']}|{s['incomplete_outputs']}|")
    lines += ['', '整体分母包含接口失败和截断；完整回答准确率另列。各自题单不同，不能直接用整体比例相减。', '',
              f"## 所列候选共同{len(common)}题", '', '|候选|正确/共同题数|准确率|', '|---|---:|---:|']
    for label in labels:
        s = common_summary[label]
        lines.append(f"|{label}|{s['correct']}/{s['eligible_total']}|{percentage(s['accuracy'])}|")
    for title, key in [('共同题的轨道', 'by_track'), ('共同题的题型', 'by_type')]:
        names = sorted({name for item in common_summary.values() for name in item[key]})
        lines += ['', f'## {title}', '', '|分组|共同题数|'+'|'.join(f'{label}正确' for label in labels)+'|',
                  '|---|---:|'+'|'.join('---:' for label in labels)+'|']
        for name in names:
            counts = [common_summary[label][key][name] for label in labels]
            lines.append(f"|{name}|{counts[0]['total']}|"+'|'.join(str(item['correct']) for item in counts)+'|')
    lines += ['', '## 各自题单的分组结果', '', '|候选|轨道/题型|题数|正确|准确率|', '|---|---|---:|---:|---:|']
    for label in labels:
        for dimension in ['evaluation_track', 'question_type']:
            for name, item in sorted(source['models'][label]['breakdown'][dimension].items()):
                lines.append(f"|{label}|{name}|{item['eligible_total']}|{item['correct']}|{percentage(item['accuracy'])}|")
    lines += ['', '## 两两共同题比较', '', '|左/右|同题数|左正确|右正确|右减左（百分点）|右胜/右负（含故障）|双方完整题数|完整题右胜/右负|', '|---|---:|---:|---:|---:|---|---:|---|']
    for item in pairs:
        valid = item['common_valid_answers']
        delta = '—' if item['accuracy_delta'] is None else f"{100*item['accuracy_delta']:+.2f}"
        lines.append(f"|{item['left_model']}/{item['right_model']}|{item['compared_questions']}|{item['left_correct']}|{item['right_correct']}|{delta}|{item['right_wins']}/{item['right_losses']}|{valid['total']}|{valid['right_wins']}/{valid['right_losses']}|")
    relative = lambda path: Path(os.path.relpath(path, output)).as_posix()
    lines += ['', '## 原始记录', '']
    for label in labels:
        lines.append(f"- {label}：[完整最终回答](<{relative(directory/f'predictions/{label}.jsonl')}>)；[逐题参考与判分理由](<{relative(directory/f'scores/{label}.jsonl')}>)；[回答封存回执](<{relative(directory/f'predictions/{label}.seal.json')}>)。")
    lines += ['', '项目README明确语料面向虚构企业。闭卷领域题主要检验对项目企业设定的掌握，通用模型未见这些资料时，低命中不能直接解释为推理能力差；还应检查题干及共同提示词是否明确默认企业主体。带材料题是受控文本夹具，工具/MCP/多轮等不等于真实系统集成验收。', '',
              '本轮本地题单采用保留原前缀的条件覆盖抽样，不能外推全库。不同原厂推理预算和量化配置的比较是本次配置下的描述性结果，不证明统计显著差异、训练因果或完整业务能力。', '',
              '语义标签为辅助离线复核；核心通过不认证答案所有扩展为公司事实或现行法律。独立复查尚未执行。原始最终回答、参考、理由和封存回执保留在运行目录；未来改参考可离线重判，改输入须另建运行。', '']
    output.mkdir()
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output/'common_question_results.jsonl').open('x', encoding='utf-8') as handle:
        for qid in sorted(common):
            handle.write(json.dumps({'qid': qid, 'metadata': metadata[qid],
                'models': {label: {key: by_id[label][qid][key] for key in ['verdict', 'prediction_status', 'item_id']} for label in labels}}, ensure_ascii=False)+'\n')
    files = [directory/'summary.json', directory/'manifest.json', directory/'blind/prompt.json', directory/'private/metadata.jsonl']
    if errata:
        files.append(errata)
    for label in labels:
        files += [directory/f'scores/{label}.jsonl', directory/f'private/model_{label}.json',
                  directory/f'predictions/{label}.jsonl', directory/f'predictions/{label}.seal.json']
    hashes = lambda paths: {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    receipt = {'at': result['generated_at'], 'run_id': source['run_id'], 'labels': labels, 'model_calls': 0,
               'inputs_sha256': hashes(files),
               'outputs_sha256': hashes([output/'report.md', output/'report.json', output/'common_question_results.jsonl'])}
    (output/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--models', nargs='+', choices=['A', 'B', 'C', 'D'], default=['A', 'B', 'C', 'D'])
    parser.add_argument('--output', required=True, type=Path, help='New output directory; earlier reports remain unchanged')
    args = parser.parse_args()
    result = export_results(ROOT/'eval_bench/runs'/args.run, args.models, args.output)
    print(json.dumps({'output': str(args.output), 'models': args.models,
                      'common_questions': result['common_questions']['total'], 'model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
