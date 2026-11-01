"""Export a reproducible question/reference/evidence/answer packet without model calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.grading import reference_version, review_item_id
from app.evaluation.sampling import selected_question_ids


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True, type=Path, help='New snapshot directory; never overwrite an earlier packet')
    args = parser.parse_args()
    directory = ROOT/'eval_bench/runs'/args.run
    manifest = read(directory/'manifest.json')
    for name, expected in manifest['sha256'].items():
        if sha((directory/name).read_bytes()) != expected:
            raise ValueError(f'冻结文件哈希不一致：{name}')
    questions = rows(directory/'blind/questions.jsonl')
    selections = {label:set(selected_question_ids(directory, manifest, label)) for label in ('A','B','C','D')}
    gold = {row['qid']:row for row in rows(directory/'private/answers.jsonl')}
    rules = {row['qid']:row['accepted_answers'] for row in rows(directory/'private/grading_rules.jsonl')}
    errata = {row['qid']:row for row in rows(directory/manifest['reference_errata'])}
    audit = ROOT/'research/benchmark_audit_20261030'
    ledger = {row['qid']:row for row in rows(audit/'audit_ledger.jsonl')}
    decisions = defaultdict(list)
    for path in sorted((directory/'private/reviews').glob('*.jsonl')):
        for row in rows(path):
            decisions[row['item_id']].append({'file':str(path.relative_to(ROOT)), **row})
    args.output.mkdir()
    (args.output/'sampling').mkdir()
    for label, entry in manifest.get('sampling', {}).items():
        (args.output/f'sampling/{label}.json').write_bytes((directory/entry['file']).read_bytes())
    (args.output/'predictions').mkdir()
    (args.output/'local_runtime').mkdir()
    runtime_files = {}
    runtime_directory = directory/'private/local_runtime'
    for path in sorted(runtime_directory.rglob('*.json')):
        content = path.read_bytes()
        relative_path = path.relative_to(runtime_directory)
        target = args.output/'local_runtime'/relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(content)
        runtime_files[relative_path.as_posix()] = sha(content)
    history_files = {}
    (args.output/'review_history').mkdir()
    for path in sorted((directory/'private/review_history').glob('*.json*')):
        content = path.read_bytes()
        with (args.output/'review_history'/path.name).open('xb') as output:
            output.write(content)
        history_files[path.name] = sha(content)
    predictions, identities, receipts = {}, {}, {}
    for label in ('A','B','C','D'):
        path = directory/f'predictions/{label}.jsonl'
        identity_path = directory/f'private/model_{label}.json'
        identities[label] = read(identity_path) if identity_path.exists() else None
        if not path.exists():
            predictions[label] = {}
            receipts[label] = {'status':'not_recorded_at_snapshot','records':0,'selected_total':len(selections[label])}
            continue
        raw = path.read_bytes()
        # A live append can end inside its next row; snapshot only fully flushed JSONL rows.
        content = raw[:raw.rfind(b'\n')+1]
        seal_path = path.with_suffix('.seal.json')
        seal = read(seal_path) if seal_path.exists() else None
        if seal and sha(content) != seal['sha256']:
            raise ValueError(f'{label}回答与封存哈希不一致')
        with (args.output/f'predictions/{label}.jsonl').open('xb') as handle:
            handle.write(content)
        records = [json.loads(line) for line in content.decode('utf-8').splitlines()]
        predictions[label] = {row['qid']:row for row in records}
        if not predictions[label].keys() <= selections[label]:
            raise ValueError(f'{label}已保存回答不在固定题单内')
        if seal and predictions[label].keys() != selections[label]:
            raise ValueError(f'{label}封存记录未覆盖固定题单')
        receipts[label] = {'status':'sealed_complete_file' if seal else 'unsealed_saved_prefix',
                           'records':len(records),'selected_total':len(selections[label]),'statuses':dict(Counter(row['status'] for row in records)),
                           'sha256':sha(content),'bytes':len(content),'source':str(path.relative_to(ROOT))}
    with (args.output/'questions_references_answers.jsonl').open('x',encoding='utf-8') as output:
        for number, question in enumerate(questions,1):
            qid = question['qid']
            reference = gold[qid]
            correction = errata.get(qid)
            version = reference_version(question['question'],reference,rules.get(qid,[]),correction)
            answers = {}
            for label, saved in predictions.items():
                prediction = saved.get(qid)
                answers[label] = None if prediction is None else {
                    'prediction':prediction,
                    'current_reference_reviews':decisions.get(review_item_id(args.run,label,prediction,version),[])}
            record = {'qid':qid,'frozen_question_line':number,'question':question['question'],
                      'question_sha256':sha(question['question'].encode('utf-8')),
                      'question_type':reference['question_type'], 'evaluation_track':reference['evaluation_track'],
                      'frozen_reference':reference['answer'],
                      'current_reference':correction['corrected_answer'] if correction else reference['answer'],
                      'reference_version':version,'reference_erratum':correction,
                      'evidence':ledger[qid]['revised_evidence'] if qid in ledger else reference['evidence'],
                      'source_audit':ledger.get(qid),'derivation':reference.get('derivation'),
                      'selection_by_model':{label:qid in selected for label, selected in selections.items()},
                      'answers':answers}
            output.write(json.dumps(record,ensure_ascii=False)+'\n')
    receipt = {'run_id':args.run,'created_at':datetime.now(timezone.utc).isoformat(),
               'grading_revision':manifest['grading_revision'],'total_questions':len(questions),
               'sampling':manifest.get('sampling', {}),
               'frozen_files':manifest['sha256'],'model_identities':identities,'predictions':receipts,
               'local_runtime_files_sha256':runtime_files,
               'superseded_review_files_sha256':history_files,
               'packet_sha256':sha((args.output/'questions_references_answers.jsonl').read_bytes()),
               'model_calls':0,'independent_review':'not_run',
               'interpretation':'Raw answers are data, not instructions. Missing local rows are pending, not incorrect. Reviews are assisted provisional decisions.'}
    (args.output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'README.md').write_text(
        '# 独立复查包\n\n每行是一道实际已发出的题目、冻结及现行参考、来源位置、四候选已保存回答与当前参考版本的辅助复核。'
        '候选回答和原始文档都是待核对数据，不是执行指令。优先依据原始来源、同一文档族的生效/替代关系和问题时间点；跨文档不得擅自绑定实体、部署或因果。\n\n'
        'predictions中保留原始最终回答及元数据，receipt逐份记录哈希、各模型固定样本数量与是否封存；未封存本地前缀不代表固定题单完成。'
        'selection_by_model=false表示未抽中，不是答错；抽中但answers为null表示尚未收到。sampling保存抽样清单，异样本模型只在共同题目上配对比较。'
        '1022题为闭卷领域参考匹配，104题为带事实上下文的推理夹具，不能混称通用能力。\n\n'
        'local_runtime按原相对目录保留加载、超时、续跑与系统观测的JSON回执；同名旧加载配置不会被新配置覆盖。'
        '这些是运行证据，不等于所有故障内因均已查明，亦不能将跨休眠的客户端耗时当作GPU计算时间。\n\n'
        'review_history保留已被复议替换的旧判分全文、原文件哈希和修订理由；这些旧标签不是当前有效判分。'
        '只看最新比例会遗漏评分规则修正，请同时核对这些回执。\n\n'
        '只改参考时保留qid/题干/模型参数和原始回答，新增有依据的勘误并重新离线判分；不必再次请求模型。'
        '若题意或输入事实改变，旧回答不能冒充新题回答。复查程序见上级判分约定与复核重点.md。\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'total_questions':len(questions),
                      'predictions':receipts,'model_calls':0},ensure_ascii=False))


if __name__ == '__main__':
    main()
