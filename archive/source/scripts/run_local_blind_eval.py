"""One finite local evaluation phase after its cloud dependencies have sealed."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/llm_blind_eval_20261029_execution'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--label', required=True, choices=('C', 'D'))
    args = parser.parse_args()
    directory = (ROOT/'eval_bench/runs'/args.run).resolve()
    if directory.parent != (ROOT/'eval_bench/runs').resolve():
        parser.error('--run must identify one run directory')
    receipt_dir = directory/'private/local_runtime'
    receipt_dir.mkdir(exist_ok=True)
    phase_path = receipt_dir/f'{args.label}_phase.json'

    def stage(name):
        receipt = {'run_id': args.run, 'label': args.label, 'stage': name,
                   'updated_at': datetime.now(timezone.utc).isoformat()}
        phase_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(receipt, ensure_ascii=False), flush=True)

    def command(script, *arguments):
        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts'/script), *arguments],
                       cwd=ROOT, check=True)

    dependencies = ['A', 'B'] if args.label == 'C' else ['A', 'B', 'C']
    stage('waiting_for_'+'_'.join(dependencies))
    while not all((directory/f'predictions/{label}.seal.json').exists() for label in dependencies):
        time.sleep(30)
    stage('verifying_selected_file')
    source = next(row for row in json.loads((OUT/'local_file_checksums.json').read_text(encoding='utf-8'))
                  if row['label'] == args.label)
    weights = Path(source['path'])
    with weights.open('rb') as handle:
        source['sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    source.update(bytes=weights.stat().st_size, mtime_ns=weights.stat().st_mtime_ns)
    stage('loading_selected_variant')
    command('manage_eval_local.py', 'load', '--label', args.label, '--run', args.run)
    loaded = json.loads((receipt_dir/f'{args.label}_load.json').read_text(encoding='utf-8'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:1235/api/v1/models', timeout=15) as response:
        inventory = json.load(response)
    target = next(row for row in inventory['models'] if row['key'] == 'qwen/qwen3.8-27b')
    source.update(run_id=args.run, recorded_at=datetime.now(timezone.utc).isoformat(),
                  local_api_model=target, effective_load_config=loaded['response']['load_config'],
                  instance_id=loaded['response']['instance_id'])
    (receipt_dir/f'{args.label}_source.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    stage('generating')
    try:
        command('llm_blind_eval.py', 'generate', '--run', args.run, '--model', args.label)
    except subprocess.CalledProcessError:
        stage('generation_failed')
        raise
    stage('unloading_own_instance')
    command('manage_eval_local.py', 'unload', '--label', args.label, '--instance', source['instance_id'], '--run', args.run)
    stage('generation_finished_and_unloaded')


if __name__ == '__main__':
    main()
