"""Sequential generation and resume. This module never opens the private answer bank."""
import json
import os
from collections import Counter
from datetime import datetime, timezone

from app.evaluation.client import ChatCompletionsClient
from app.evaluation.config import EvaluationConfig, ModelLabel, resolve_model
from app.evaluation.contracts import BlindQuestion, Prediction
from app.evaluation.files import digest, index_rows, load_jsonl, write_json, run_directory
from app.evaluation.sampling import selected_question_ids


def generate(config: EvaluationConfig, run_id: str, label: ModelLabel) -> dict:
    if not config.execution_enabled:
        raise ValueError('模型请求未启用；execution_enabled=false，未调用模型')
    identity, api_key = resolve_model(config, label)
    directory = run_directory(config, run_id)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if config.inference_parameters() != manifest['inference_parameters']:
        raise ValueError('统一推理参数已变化，请新建运行，不混合不同条件的成绩')
    # Only the two model-facing files are read and verified here.
    for name in ('blind/questions.jsonl', 'blind/prompt.json'):
        if digest(directory/name) != manifest['sha256'][name]:
            raise ValueError('冻结题目或提示词已变化，请新建运行')
    questions = [BlindQuestion.model_validate(row) for row in load_jsonl(directory/'blind/questions.jsonl')]
    selected = set(selected_question_ids(directory, manifest, label))
    questions = [question for question in questions if question.qid in selected]
    prompt = json.loads((directory/'blind/prompt.json').read_text(encoding='utf-8'))
    identity_path = directory/f'private/model_{label}.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('模型或推理参数已变化，不允许混入同一份作答记录')
    else:
        with identity_path.open('x', encoding='utf-8') as handle:
            json.dump(identity, handle, ensure_ascii=False, indent=2)
    prediction_path = directory/f'predictions/{label}.jsonl'
    previous = ([Prediction.model_validate(row).model_dump() for row in load_jsonl(prediction_path)]
                if prediction_path.exists() else [])
    completed = index_rows(previous, 'qid')
    if not completed.keys() <= {question.qid for question in questions}:
        raise ValueError('已有作答包含冻结题库之外的qid')
    client = ChatCompletionsClient(identity, api_key, prompt['system'])
    statuses = Counter(row['status'] for row in previous)
    with prediction_path.open('a', encoding='utf-8') as output:
        for question in questions:
            if question.qid in completed:
                continue
            result = client.answer(question)
            output.write(result.model_dump_json() + '\n')
            output.flush()
            completed[question.qid] = result.model_dump()
            statuses[result.status] += 1
            progress = {'label': label, 'pid': os.getpid(), 'qid': question.qid,
                        'question_type': manifest['question_type_by_id'][question.qid],
                        'recorded': len(completed), 'total': len(questions), 'statuses': dict(statuses),
                        'updated_at': datetime.now(timezone.utc).isoformat(),
                        'last_latency_seconds': result.latency_seconds, 'accuracy': None}
            write_json(directory/f'predictions/{label}.progress.json', progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            # A question exhausting its time budget is an observed result, not a retry.
            if result.status == 'api_error' and result.error != 'TimeoutError':
                raise RuntimeError(f'{label}返回{result.error}，已记录实际失败；先处理接口故障，再继续剩余题目')
    receipt = {'label': label, 'total': len(questions), 'sha256': digest(prediction_path),
               'status': 'generation_finished', 'accuracy': None}
    if label in manifest.get('sampling', {}):
        receipt['sample_sha256'] = manifest['sampling'][label]['sha256']
    write_json(directory/f'predictions/{label}.seal.json', receipt)
    return receipt
