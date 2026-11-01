"""Start the project's existing WSL services; no migrations or corpus writes."""
import hashlib,json,socket,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
marker='oa-local-'+hashlib.sha256(str(ROOT).encode('utf-8')).hexdigest()[:12]
log=ROOT/'local_runtime/logs/tools-wsl.log';log.parent.mkdir(exist_ok=True)
with log.open('ab',buffering=0) as stream:
    process=subprocess.Popen(['wsl.exe','-d','Ubuntu-24.04','-u','root','--exec','bash','-lc',
        'systemctl start mysql redis-server && exec -a '+marker+' sleep infinity'],
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
receipt={'wsl_pid':process.pid,'wsl_marker':marker,'purpose':'Start existing services only; no migrations or ingest.'}
(ROOT/'local_runtime/tools-review-processes.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
deadline=time.monotonic()+45
while True:
    try:
        for port in (3306,6379):
            with socket.create_connection(('127.0.0.1',port),timeout=.5):pass
        break
    except OSError:
        if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Project WSL services did not become ready; inspect tools-wsl.log.')
        time.sleep(.5)
print(json.dumps({'mysql':True,'redis':True,**receipt}))
