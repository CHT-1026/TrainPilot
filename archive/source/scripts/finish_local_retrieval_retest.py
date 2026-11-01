"""Finish frozen Q4 tasks, then load only Q6 for its predeclared diagnostic subset."""
import asyncio
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.retrieval_corpus import read_jsonl, write_json
from app.evaluation.retrieval_runtime import DEFAULT_BUDGET, execute, model_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--q6-only',action='store_true',help='Continue after Q4 was already completed; do not repeat paid Q4 tasks')
    args=parser.parse_args()
    out = ROOT/'research/retrieval_repair_20261101'
    py = ROOT/'.venv312/Scripts/python.exe'
    server = ROOT/'scripts/local_model_server.py'
    expected = {c['qid'] for c in read_jsonl(out/'cases.jsonl')}
    deadline = time.monotonic() + 2400
    while True:
        committed = {p.stem for p in (out/'runs/native_v2_D/traces').glob('*.json')}
        if committed == expected:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Q4 frozen run is incomplete; Q6 was not started')
        time.sleep(10)
    if not args.q6_only:
        asyncio.run(execute(read_jsonl(out/'cases_local_smoke2.jsonl'), model_config('D'), out/'public',
            out/'runs/native_v3_smoke_D', DEFAULT_BUDGET))
    subprocess.run([str(py), str(server), 'stop'], check=True)
    log = (out/'runtime/q6_owner.log').open('w',encoding='utf-8')
    loader = subprocess.Popen([str(py), str(server), 'start', '--quant', 'Q6_K'],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    write_json(out/'runtime/q6_owner.json', {'pid':loader.pid, 'purpose':'authorized_Q6_only'})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while True:
            if loader.poll() is not None:
                raise RuntimeError('Q6 loader exited before readiness; see q6_owner.log')
            try:
                with opener.open('http://127.0.0.1:1235/v1/models',timeout=3) as response:
                    inventory = json.load(response)
                if 'qwen3.8-27b-Q6_K' not in [m['id'] for m in inventory['data']]:
                    raise RuntimeError('Unexpected local model; no Q6 calls made')
                write_json(out/'runtime/q6_inventory.json',inventory)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if time.monotonic()>=deadline:
                    raise RuntimeError('Q6 readiness timed out')
                time.sleep(5)
        conf = model_config('C')
        asyncio.run(execute(read_jsonl(out/'cases_q6_diagnostic16.jsonl'), conf, out/'public',
            out/'runs/native_v3_C16', DEFAULT_BUDGET))
    finally:
        subprocess.run([str(py), str(server), 'stop'], check=True)
        loader.wait(timeout=15)
        log.close()
    q6_expected={c['qid'] for c in read_jsonl(out/'cases_q6_diagnostic16.jsonl')}
    q6_recorded={p.stem for p in (out/'runs/native_v3_C16/traces').glob('*.json')}
    write_json(out/'local_retest_complete.json', {'Q4_frozen_tasks':32,'Q6_diagnostic_expected':len(q6_expected),
        'Q6_diagnostic_recorded':len(q6_recorded),'all_tasks_recorded':q6_recorded==q6_expected,
        'formal_400_800_completed':False,'temporary_model_servers_stopped':True})
    if q6_recorded!=q6_expected:
        raise RuntimeError('Q6 tasks incomplete; see individual terminal traces, no completion claim')


if __name__=='__main__':
    main()
