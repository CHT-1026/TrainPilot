"""Export reproducible results and original final text after the selected runs finish."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from app.evaluation.retrieval_corpus import read_jsonl, write_json
from app.evaluation.retrieval_grading import grade
from app.evaluation.retrieval_reporting import report_runs, run_rows
from review_retrieval_content import inspect_content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    base = ROOT/'research/retrieval_repair_20261101'
    runs = [base/'runs'/name for name in args.runs]
    for run in runs:
        expected, rows = run_rows(run)
        if set(expected) != set(rows):
            raise ValueError(f'Run not finished: {run.name}')
    summary = report_runs(runs, base/'private/gold.jsonl', args.output)
    golds = {g['qid']:g for g in read_jsonl(base/'private/gold.jsonl')}
    records, by_run = [], {}
    lines = ['# 逐题原始回答与复核依据', '',
        '严格接口结果与内容辅助复核分开。多余括号不从原始回答中删除；下列原文供独立复查。', '']
    for run in runs:
        _, traces = run_rows(run)
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))['contract']
        cases = {c['qid']:c for c in manifest['cases']}
        current = []
        for qid, trace in sorted(traces.items()):
            review = inspect_content(trace, golds[qid], cases[qid])
            raw = trace['rounds'][-1].get('response',{}).get('choices',[{'message':{}}])[0]['message'].get('content') if trace.get('rounds') else None
            row = {'run':run.name, 'model':manifest['model'], 'qid':qid,'question':golds[qid]['question'],
                'raw_final_text':raw, 'trace_file':str((run/'traces'/f'{qid}.json').resolve()),
                'raw_text_is_target_turn':review['target_response_received'],
                'planned_turns':cases[qid].get('turns', [cases[qid].get('question')]),
                'grade':grade(trace,golds[qid]), 'content_review':review, 'gold':golds[qid]}
            records.append(row);current.append(row)
            lines += [f"## {run.name} / {qid}", '', golds[qid]['question'], '',
                f"执行状态：{trace['status']}；严格任务通过：{row['grade']['success']}；内容辅助核验：{review['source_supported_content']}。", '',
                f"目标用户轮次：{review['target_turn']+1}；已取得该轮响应：{review['target_response_received']}。未到目标轮的先前回答不与最终题目比对。", '',
                '实际最后一次模型正文（可能在目标用户轮次前终止）：', '', '````text', raw if raw is not None else '（未取得正文；详见原始轨迹）', '````', '',
                '参考字段：`'+json.dumps(golds[qid]['retrieval_rubric']['expected'],ensure_ascii=False)+'`', '',
                '证据组：`'+json.dumps(golds[qid]['retrieval_rubric']['evidence_groups'],ensure_ascii=False)+'`', '',
                '判分：`'+json.dumps(row['grade'],ensure_ascii=False)+'`', '']
        by_run[run.name] = {'model':manifest['model'],'tasks':len(current),
            'strict_pass':sum(r['grade']['success'] for r in current),
            'content_supported_auxiliary':sum(r['content_review']['source_supported_content'] is True for r in current),
            'content_unreviewed':sum(r['content_review']['source_supported_content'] is None for r in current)}
    for name, rows in [("answers_with_review.jsonl",records)]:
        (args.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (args.output/'逐题原始回答.md').write_text('\n'.join(lines),encoding='utf-8')
    common = set.intersection(*[{r['qid'] for r in records if r['run']==name} for name in by_run])
    paired = {name:{'tasks':len(common), 'strict_pass':sum(r['grade']['success'] for r in records if r['run']==name and r['qid'] in common),
                   'content_supported_auxiliary':sum(r['content_review']['source_supported_content'] is True for r in records if r['run']==name and r['qid'] in common)} for name in by_run}
    # Seal earlier artifacts without trusting previous completion claims.
    old = json.loads((base/'old_artifacts_seal.json').read_text(encoding='utf-8'))
    mismatches = [r['path'] for r in old if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']]
    result = {'scope':'targeted_regression_not_formal_400_800','models':by_run,'paired_common_tasks':paired,
        'common_qids':sorted(common), 'old_artifacts_checked':len(old),'old_artifacts_mismatched':mismatches,
        'formal_400_800_completed':False, 'auxiliary_content_review_is_posthoc_not_independent':True,
        'comparison_limit':'native_v2 A/B/D and native_v3 C have different executor revisions; descriptive comparison only, not a controlled quantization or training effect',
        'public_case_sha256':hashlib.sha256((base/'cases.jsonl').read_bytes()).hexdigest(),
        'current_gold_sha256':hashlib.sha256((base/'private/gold.jsonl').read_bytes()).hexdigest()}
    write_json(args.output/'final_results.json',result)
    rows = ['# 本轮修复与复测结果', '', '定向回归；不是完整正式测评。所有原始回答和失败终态均保留。', '',
        '| 模型 | 任务 | 严格接口及证据通过 | 内容可核验通过（事后辅助） | 内容未判定 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for value in by_run.values():
        rows.append(f"| {value['model']} | {value['tasks']} | {value['strict_pass']} | {value['content_supported_auxiliary']} | {value['content_unreviewed']} |")
    rows += ['', '严格结果包含JSON/状态等接口约束。内容列只做字面字段与实际原文证据核验，不改变协议失败，不是独立双盲评审或通用准确率。', '',
        f"四者共同题 {len(common)} 项；描述性结果如下。题量不同的总体比例不可直接比较。", '',
        '| 运行 | 共同任务 | 严格通过 | 内容辅助可核验 |',
        '| --- | ---: | ---: | ---: |']
    for name,value in paired.items():
        rows.append(f"| {name} | {value['tasks']} | {value['strict_pass']} | {value['content_supported_auxiliary']} |")
    rows += ['', 'A/B/D主回归与C使用不同执行器修订，详见修复说明；本表不作严格模型排名或训练效果结论。', '',
        f"旧封存文件核验：{len(old)}项，SHA不一致{len(mismatches)}项。", '',
        '正式测评仍需逐题事实rubric审阅、补Q4追加192道合格任务、补长文/连续跨文档线索/其他来源能力，之后重新冻结400/800题单。', '',
        '自动辅助检查未覆盖的8条格式异常和B的一条引用失败另经本助手逐字复核，业务值及实际证据获取均正确；见../../author_review_format_exceptions.json。该补充复核不混入上表，不是独立评审。', '',
        '详见[修复说明与后续正式测评条件](../../修复说明与后续正式测评条件.md)、[逐题原始回答](逐题原始回答.md)、answers_with_review.jsonl和各运行traces目录。']
    (args.output/'结果总览.md').write_text('\n'.join(rows),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
