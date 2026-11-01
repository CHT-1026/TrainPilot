"""Explicit connectivity probes for the two authorized remote targets only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local', override=False)
    from app.evaluation.config import load_config, resolve_model
    from app.evaluation.client import ChatCompletionsClient
    from app.evaluation.contracts import BlindQuestion
    from app.evaluation.files import write_json
    from app.evaluation.prompts import SYSTEM_PROMPT
    config = load_config()
    expected = {'A': 'mimo-v2.6-flash', 'B': 'qwen3.6-flash'}
    targets = {label: resolve_model(config, label) for label in expected}
    if {label: identity['model'] for label, (identity, _) in targets.items()} != expected:
        raise ValueError('仅允许用户指定的两个云端模型')
    receipt = {'purpose': 'connectivity_only_not_scored', 'probes': []}
    for label, (identity, key) in targets.items():
        result = ChatCompletionsClient(identity, key, SYSTEM_PROMPT).answer(
            BlindQuestion(qid='connectivity_probe', question='请仅回复OK。'))
        receipt['probes'].append({'label': label, 'identity': identity, 'result': result.model_dump()})
        write_json(ROOT/'research/llm_blind_eval_20261029_execution/remote_probe_flash.json', receipt)
        print(json.dumps({'label': label, 'requested': identity['model'], **result.model_dump()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
