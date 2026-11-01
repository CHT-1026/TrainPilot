"""Start the stopped local app against existing services; no migration or ingestion."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/tool_mcp_20261027'


def main():
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8080)) == 0:
            raise RuntimeError('8080 already in use; no process was replaced')
    state = json.loads((ROOT / 'local_runtime/tools-review-processes.json').read_text(encoding='utf-8'))
    log = ROOT / 'local_runtime/logs/backend.log'
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen([
            str(ROOT / '.venv312/Scripts/python.exe'), '-u', '-m', 'uvicorn', 'local_runtime.web_app:app',
            '--env-file', str(ROOT / '.env.local'), '--host', '127.0.0.1', '--port', '8080'],
            cwd=ROOT, env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
    receipt = {'backend_pid': process.pid, 'wsl_pid': state['wsl_pid'],
               'wsl_marker': state['wsl_marker'], 'port': 8080, 'url': 'http://127.0.0.1:8080',
               'scope': 'Existing MySQL/Redis only. No migrations, ingestion or business writes.'}
    (ROOT / 'local_runtime/processes.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 45
    while True:
        try:
            with opener.open(receipt['url'] + '/api/ready', timeout=3) as response:
                receipt['ready'] = json.load(response)
            break
        except (urllib.error.URLError, TimeoutError):
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('App did not become ready; inspect local_runtime/logs/backend.log') from None
            time.sleep(.5)
    (OUT / 'runtime_start.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
