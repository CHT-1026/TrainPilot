"""Real platform HTTP/SSE retest. Scoped candidate B runtime; own sessions only."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/retrieval_repair_20261101'
BASE = 'http://127.0.0.1:8088'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path, payload=None, token=None, method=None, stream=False):
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
        headers={'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + token} if token else {})})
    try:
        with OPENER.open(req, timeout=300) as response:
            return response.status, response.read().decode() if stream else json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def main():
    result = {'model': 'qwen3.6-flash', 'scope': 'real_production_routes_isolated_readonly_tools', 'checks': [], 'answers': []}
    sessions = []

    def check(name, ok, evidence):
        result['checks'].append({'name': name, 'passed': bool(ok), 'evidence': evidence})
        (OUT / 'http_retest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    try:
        code, ready = request('/api/ready')
        check('ready_mysql_redis', code == 200, ready)
        check('authentication_required', request('/api/tools')[0] == 401, {})
        code, admin = request('/api/auth/login', {'username': 'admin', 'password': 'admin123'})
        if code != 200:
            raise RuntimeError(f'Existing test admin unavailable: HTTP {code}')
        token = admin['data']['token']
        code, catalog = request('/api/tools', token=token)
        names = [t['name'] for t in catalog['data']['tools']]
        check('real_mcp_discovery', code == 200 and 'mcp_public_docs_search' in names, names)
        for name, args, expected in [
            ('read_file', {'path': '../../private/gold.jsonl'}, 'PATH_NOT_ALLOWED'),
            ('read_file', {'path': 'org/lantu_org_reporting_lines.md', 'line_count': '2'}, 'INVALID_TOOL_ARGS'),
        ]:
            code, body = request('/api/tools/call', {'name': name, 'arguments': args}, token)
            check('boundary_' + expected, code == 200 and body['data']['code'] == expected, body)
        code, raw = request('/api/tools/call', {'name': 'mcp_public_docs_search', 'arguments': {'query': '住宿标准', 'top_k': 2}}, token)
        check('real_mcp_execution', code == 200 and raw['data']['ok'], raw)
        for streaming in (False, True):
            code, session = request('/api/chat/session', {}, token)
            thread = session['data']['thread_id']
            sessions.append((token, thread))
            prompt = '/tool 使用mcp_public_docs_search检索“2024 住宿标准”，再读取返回的2024版差旅原文。回答该版一线城市住宿每晚多少元，并说明实际工具与原文出处。'
            started = time.monotonic()
            code, body = request('/api/chat/stream' if streaming else '/api/chat', {'thread_id': thread, 'user_message': prompt}, token, stream=streaming)
            if streaming:
                events = []
                for frame in body.replace('\r\n', '\n').split('\n\n'):
                    lines = frame.splitlines()
                    if any(l.startswith('data:') for l in lines):
                        event = next((l[6:].strip() for l in lines if l.startswith('event:')), 'message')
                        text = '\n'.join(l[5:].strip() for l in lines if l.startswith('data:'))
                        events.append({'event': event, 'data': json.loads(text)})
                final_events = [e for e in events if e['event'] == 'complete']
                check('stream_terminal_event', code == 200 and len(final_events) == 1, events)
                result['answers'].append({'streaming': True, 'raw_sse': body, 'elapsed_seconds': time.monotonic() - started})
                final_text = final_events[0]['data']['response'] if final_events else ''
                check('stream_answer_and_source', code == 200 and '450' in final_text and 'lantu_policy_travel_2024' in final_text, {})
            else:
                result['answers'].append({'streaming': False, 'response': body, 'elapsed_seconds': time.monotonic() - started})
                data = body['data']
                check('chat_answer_and_tool_events', code == 200 and '450' in data['response'] and bool(data['tool_events']), body)
    finally:
        result['cleanup'] = [{'thread_id': thread, 'http_status': request('/api/chat/session/' + thread, token=token, method='DELETE')[0]} for token, thread in sessions]
        (OUT / 'http_retest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'checks': len(result['checks']), 'passed': sum(c['passed'] for c in result['checks'])}))


if __name__ == '__main__':
    main()
