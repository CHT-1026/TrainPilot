"""Actual RAG API diagnostic, explicitly offline with zero model/encoder calls."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_retrieval_http import OUT, request


def main():
    _, login = request('/api/auth/login', {'username': 'admin', 'password': 'admin123'})
    token = login['data']['token']
    cases = [('founding', '澜图智能成立于哪一年？', '2011'),
             ('manager', '沈亦舟的直接上级是谁？', '顾承澜'),
             ('historic_travel', '2024年5月1日适用的一线城市住宿标准是多少元每晚？', '450')]
    records = []
    for name, question, expected in cases:
        status, body = request('/api/rag/query', {'question': question, 'use_llm': False, 'offline': True, 'top_k': 8}, token)
        records.append({'id': name, 'http_status': status, 'request': {'question': question, 'use_llm': False, 'offline': True},
            'response': body, 'answer_contains_expected': expected in str(body.get('data', {}).get('answer', ''))})
    (OUT / 'rag_api_diagnostic.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{'id': r['id'], 'http_status': r['http_status'], 'answer_contains_expected': r['answer_contains_expected']} for r in records]))


if __name__ == '__main__':
    main()
