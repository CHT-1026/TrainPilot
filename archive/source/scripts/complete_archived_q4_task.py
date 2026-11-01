"""Finish ONLY unexecuted RET032 using the frozen v2 runtime; RET031 stays failed."""
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/retrieval_repair_20261101'
FROZEN=OUT/'source_versions/native_v2'


def main():
    run=OUT/'runs/native_v2_D'
    target=run/'traces/RET032.json'
    if target.exists():
        raise ValueError('RET032 already has a terminal trace; no repeat call')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))['contract']
    for relative, digest in manifest['code_sha256'].items():
        if hashlib.sha256((FROZEN/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Archived v2 source checksum mismatch')
    # Package markers let the archived modules import their own archived corpus/server.
    for relative in ['app/__init__.py','app/evaluation/__init__.py']:
        (FROZEN/relative).touch(exist_ok=True)
    sys.path.insert(0,str(FROZEN))
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env.local',override=False)
    from app.evaluation.retrieval_runtime import MCPTools, model_config, run_case
    case=next(c for c in manifest['cases'] if c['qid']=='RET032')
    async def once():
        async with MCPTools(Path(manifest['workspace'])) as tools:
            return await run_case(case,model_config('D'),Path(manifest['workspace']),manifest['budget'],tools=tools)
    trace=asyncio.run(once())
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(target)
    records=list((run/'traces').glob('*.json'))
    (run/'completion.json').write_text(json.dumps({'expected':32,'recorded':len(records),
        'trace_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(records)}},indent=2),encoding='utf-8')
    (run/'continuation.json').write_text(json.dumps({'reason':'An HTTP400 context overflow halted the old batch. Failed RET031 retained; only never-executed RET032 continued.',
        'archived_source_hashes_verified':True,'repeated_failed_tasks':0,'status':trace['status']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'qid':'RET032','status':trace['status']}),flush=True)


if __name__=='__main__':
    main()
