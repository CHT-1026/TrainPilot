"""Reuse completed, byte-identical requests. Never reads gold or calls a provider."""
from collections import Counter
import hashlib
import json

from app.evaluation.config import EvaluationConfig, ModelLabel, resolve_model
from app.evaluation.contracts import Prediction
from app.evaluation.files import digest, index_rows, load_jsonl, run_directory, write_json


def reuse_completed(config: EvaluationConfig, target_run: str, source_run: str, label: ModelLabel) -> dict:
    source,target=run_directory(config,source_run),run_directory(config,target_run)
    old=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    new=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
    if old['status']=='infrastructure_invalid':
        raise ValueError('不能从已确认为基础设施无效的运行导入正式成绩')
    for directory,manifest in ((source,old),(target,new)):
        for name in ('blind/questions.jsonl','blind/prompt.json'):
            if digest(directory/name)!=manifest['sha256'][name]:
                raise ValueError('复用需要未变更的冻结题目和提示词')
    identity,_=resolve_model(config,label)
    previous_identity=json.loads((source/f'private/model_{label}.json').read_text(encoding='utf-8'))
    prompt=json.loads((target/'blind/prompt.json').read_text(encoding='utf-8'))
    previous_prompt=json.loads((source/'blind/prompt.json').read_text(encoding='utf-8'))
    if identity!=previous_identity or prompt!=previous_prompt:
        raise ValueError('模型、推理参数或共同提示词不同，不能复用旧回答')
    predictions_path=source/f'predictions/{label}.jsonl'
    source_hash=digest(predictions_path)
    old_questions=index_rows(load_jsonl(source/'blind/questions.jsonl'),'qid')
    old_rows=load_jsonl(predictions_path)
    previous=index_rows(old_rows,'qid')
    lines={r['qid']:i for i,r in enumerate(old_rows,1)}
    imported=[]
    skipped=Counter()
    for question in load_jsonl(target/'blind/questions.jsonl'):
        qid=question['qid']
        if qid not in previous:
            skipped['not_previously_answered']+=1
            continue
        prediction=Prediction.model_validate(previous[qid]).model_dump()
        if prediction['status']!='completed':
            skipped['previous_interface_or_incomplete']+=1
            continue
        if question['question']!=old_questions[qid]['question']:
            skipped['question_text_changed']+=1
            continue
        payload={'model':identity['model'],'temperature':identity['temperature'],
                 identity['token_limit_parameter']:identity['max_tokens'],'stream':False,
                 'messages':[{'role':'system','content':prompt['system']},
                             {'role':'user','content':question['question']}],**identity['request_options']}
        expected=hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode('utf-8')).hexdigest()
        if expected!=prediction['request_sha256']:
            raise ValueError(f'{qid}旧请求摘要不对应当前输入，不能推断复用资格')
        prediction['provenance']={'kind':'reused_exact_request','source_run':source_run,'source_label':label,
            'source_file_sha256':source_hash,'source_line':lines[qid],
            'question_sha256':hashlib.sha256(question['question'].encode('utf-8')).hexdigest()}
        imported.append(prediction)
    if digest(predictions_path)!=source_hash:
        raise ValueError('导入期间旧回答文件发生变化，请先停止写入该源文件')
    destination=target/f'predictions/{label}.jsonl'
    with destination.open('x',encoding='utf-8') as output:
        for prediction in imported:
            output.write(json.dumps(prediction,ensure_ascii=False)+'\n')
    write_json(target/f'private/model_{label}.json',identity)
    receipt={'source_run':source_run,'target_run':target_run,'label':label,'reused_completed':len(imported),
             'remaining':new['total']-len(imported),'not_reused':dict(skipped),
             'source_prediction_sha256':source_hash,'imported_prediction_sha256':digest(destination),
             'generation_calls_during_import':0,'answer_correctness_used_for_selection':False}
    write_json(target/f'private/reuse_{label}.json',receipt)
    return receipt
