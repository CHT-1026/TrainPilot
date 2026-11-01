"""Run the actual platform with an isolated public tool root and authorized candidate B only."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from app.evaluation.retrieval_corpus import write_json


def main():
    load_dotenv(ROOT / '.env.local', override=False)
    out = ROOT / 'research/retrieval_repair_20261101'
    workspace = out / 'public'
    tools = {'workspace_root': str(workspace / 'originals'), 'write_roots': [], 'local_roles': ['admin'],
        'max_rounds': 8, 'max_calls': 12, 'agent_timeout_seconds': 240,
        'commands': {}, 'mcp_servers': [{'name': 'public_docs', 'transport': 'stdio', 'command': '{python}',
            'args': [str(ROOT / 'scripts/mcp_retrieval_server.py'), '--workspace', str(workspace)],
            'tools': {name: {'read_only': True} for name in ['search', 'read', 'list']}}]}
    write_json(out / 'http_tools.json', tools)
    for suffix, target in [('API_KEY', 'OPENAI_API_KEY'), ('BASE_URL', 'OPENAI_API_BASE'), ('MODEL', 'OPENAI_MODEL')]:
        os.environ[target] = os.environ['BLIND_EVAL_B_' + suffix]
    if os.environ['OPENAI_MODEL'] != 'qwen3.6-flash':
        raise ValueError('Platform retest permits qwen3.6-flash only')
    os.environ.update(OPENAI_ENABLE_THINKING='true', LLM_MAX_RETRIES='0', LLM_TIMEOUT_SECONDS='240',
        TOOLS_CONFIG_PATH=str(out / 'http_tools.json'), RAG_LLM_MODEL='qwen3.6-flash',
        RAG_MODEL_SIMPLE='qwen3.6-flash', RAG_MODEL_COMPLEX='qwen3.6-flash',
        RAG_EMBEDDING_PROVIDER='dashscope', RAG_EMBEDDING_MODEL='text-embedding-v3',
        RAG_EMBEDDING_DIM='1024', RAG_MAX_EMBEDDING_CALLS='0', LANGSMITH_TRACING='false')
    write_json(out / 'http_runtime.json', {'pid': os.getpid(), 'model': os.environ['OPENAI_MODEL'],
        'port': 8088, 'tools_root': str(workspace), 'other_model_calls': False, 'retries': 0})
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8088)


if __name__ == '__main__':
    main()
