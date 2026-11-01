"""Verify preserved answers, executed source versions and key redaction without API calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/retrieval_repair_20261101'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads((BASE/'old_artifacts_seal.json').read_text(encoding='utf-8'))
    old_mismatches = [r['path'] for r in old if sha(ROOT/r['path']) != r['sha256']]
    snapshot = json.loads((BASE/'public/manifest.json').read_text(encoding='utf-8'))
    snapshot_mismatches = [p for p,h in snapshot['files'].items() if sha(BASE/'public'/p) != h]
    original_mismatches = [p for p,h in snapshot['source_files'].items()
        if sha(BASE/'public/originals'/p) != h]
    selected = {
        'native_v2_A':BASE/'source_versions/native_v2',
        'native_v2_B':BASE/'source_versions/native_v2',
        'native_v2_D':BASE/'source_versions/native_v2',
        'native_v3_smoke_A':BASE/'source_versions/native_v3_smoke',
        'native_v3_smoke_B':BASE/'source_versions/native_v3_smoke',
        'native_v3_smoke_D':BASE/'runs/native_v3_smoke_D/source',
        'native_v3_C16':BASE/'runs/native_v3_C16/source',
    }
    versions = []
    for name, archive in selected.items():
        contract = json.loads((BASE/'runs'/name/'manifest.json').read_text(encoding='utf-8'))['contract']
        versions.append({'run':name,'source_archive':str(archive.relative_to(ROOT)),
            'files_checked':len(contract['code_sha256']),
            'mismatches':[p for p,h in contract['code_sha256'].items() if sha(archive/p) != h]})
    inventory = []
    for run in sorted((BASE/'runs').iterdir()):
        if not run.is_dir():
            continue
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in (run/'traces').glob('*.json')]
        models = sorted({r['response']['model'] for t in rows for r in t['rounds'] if 'response' in r})
        inventory.append({'run':run.name,'planned_tasks':len(manifest['contract']['cases']),
            'terminal_records':len(rows),'status_counts':dict(Counter(t['status'] for t in rows)),
            'actual_response_models':models,
            'purpose':'early_aborted_development_attempt' if run.name in {'native_A','native_B','native_D'} else 'targeted_regression_or_smoke'})
    secrets = {v for k,v in dotenv_values(ROOT/'.env.local').items()
        if k.endswith('API_KEY') and v and len(v)>=16}
    secret_bytes = [v.encode('utf-8') for v in secrets]
    secret_files = []
    for p in BASE.rglob('*'):
        if p.is_file():
            raw = p.read_bytes()
            if any(v in raw for v in secret_bytes):
                secret_files.append(str(p.relative_to(ROOT)))
    paths = [
        'app/evaluation/retrieval_runtime.py','app/evaluation/retrieval_corpus.py',
        'app/evaluation/retrieval_grading.py','app/evaluation/retrieval_reporting.py',
        'app/workflows/chat_nodes/recognition.py','app/workflows/dialog_state.py',
        'app/workflows/chat_agent.py','app/core/config.py','app/models/exceptions.py',
        'enterprise_rag/query.py','scripts/retrieval_eval.py','scripts/mcp_retrieval_server.py',
        'scripts/local_model_server.py','scripts/review_retrieval_content.py',
        'scripts/export_retrieval_repair.py','config/retrieval_evaluation.json',
    ]
    result = {'old_files_checked':len(old),'old_file_mismatches':old_mismatches,
        'public_derived_files_checked':len(snapshot['files']),'public_derived_mismatches':snapshot_mismatches,
        'public_originals_checked':len(snapshot['source_files']),'public_original_mismatches':original_mismatches,
        'executed_source_versions':versions,'run_inventory':inventory,
        'artifact_files_containing_configured_api_keys':secret_files,
        'current_source_sha256':{p:sha(ROOT/p) for p in paths},
        'public_cases_sha256':sha(BASE/'cases.jsonl'),'gold_v3_sha256':sha(BASE/'private/gold.jsonl'),
        'auxiliary_review_sha256':sha(BASE/'author_review_format_exceptions.json'),
        'all_checks_passed':not old_mismatches and not snapshot_mismatches and not original_mismatches
            and not secret_files and not any(v['mismatches'] for v in versions)}
    (BASE/'final_provenance_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'old_files_checked':len(old),'source_versions_checked':len(versions),
        'secret_file_matches':len(secret_files),'all_checks_passed':result['all_checks_passed']}))
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
