"""Live app + configured model verification. Deletes only its own sessions and scratch file."""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env.local', override=False)
from app.database.connection import get_redis_client

OUT = ROOT / 'research/tool_mcp_20261027'
BASE = 'http://127.0.0.1:8080'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(path, payload=None, token=None, method=None, timeout=150):
    request = urllib.request.Request(BASE + path,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(), method=method,
        headers={'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + token} if token else {})})
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def main():
    run = uuid4().hex[:12]
    receipt = {'run': run, 'started_at': datetime.now().astimezone().isoformat(), 'checks': [],
        'scope': 'Real HTTP, configured DeepSeek model, existing Redis/checkpointer; scratch file only. No leave/approval business writes.'}
    sessions = []
    action_keys = []
    relative = f'tool_workspace/tool_http_{run}.md'
    scratch = ROOT / relative
    cancelled = ROOT / f'tool_workspace/tool_cancel_{run}.md'
    content = f'工具真实验证 {run}'
    client = get_redis_client()

    def save():
        (OUT / f'http_{run}.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / 'http_latest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')

    def check(name, passed, **data):
        receipt['checks'].append({'id': name, 'passed': bool(passed), **data})
        save()
        print(f'{name}: {bool(passed)}', flush=True)

    def new_session(token):
        status, response = api('/api/chat/session', {}, token)
        assert status == 200
        thread = response['data']['thread_id']
        sessions.append((token, thread))
        return thread

    def chat(token, thread, question):
        started = time.monotonic()
        status, response = api('/api/chat', {'thread_id': thread, 'user_message': question}, token)
        return status, response, round(time.monotonic() - started, 3)

    try:
        status, ready = api('/api/ready')
        check('runtime_ready', status == 200 and ready['status'] == 'ready')
        receipt['model'] = api('/local/status')[1]['model']
        admin_user = api('/api/auth/login', {'username': 'admin', 'password': 'admin123'})[1]['data']
        employee_user = api('/api/auth/login', {'username': 'employee_zhao', 'password': 'admin123'})[1]['data']
        admin, employee = admin_user['token'], employee_user['token']
        catalog = api('/api/tools', token=admin)[1]['data']
        names = {item['name'] for item in catalog['tools']}
        check('real_catalog_6_local_2_mcp', len(names) == 8 and not catalog['unavailable'], catalog=catalog)
        check('authentication_required', api('/api/tools')[0] == 401)
        check('employee_has_no_privileged_tools', api('/api/tools', token=employee)[1]['data']['tools'] == [])
        for name, tool, args, code in (
            ('direct_read', 'read_file', {'path': 'README.md', 'line_count': 2}, 'OK'),
            ('direct_write_no_bypass', 'write_file', {'path': relative, 'content': content}, 'CONFIRMATION_REQUIRED'),
            ('bad_arguments', 'read_file', {'path': 'README.md', 'line_count': '2'}, 'INVALID_TOOL_ARGS'),
            ('credential_path_boundary', 'read_file', {'path': '.env.local'}, 'PATH_NOT_ALLOWED'),
        ):
            status, response = api('/api/tools/call', {'name': tool, 'arguments': args}, admin)
            check(name, status == 200 and response['data']['code'] == code, result=response)

        for name, question, expected in (
            ('model_config', '/tool 调用query_project_config，section=model。根据实际结果回答当前模型名称和thinking设置。', {'query_project_config'}),
            ('model_file_read', '/tool 调用read_file读取README.md，从第1行开始读取4行，说明第一行是什么。', {'read_file'}),
            ('model_list_and_code_search', '/tool 先用list_files列出app/workflows/chat_nodes目录前5项，再用search_code在app/workflows/chat_nodes/tools.py中查找execute_confirmed，limit=5。报告真实行号。', {'list_files', 'search_code'}),
            ('model_mcp_documents', '/tool 必须通过MCP的mcp_project_docs_search_documents查询“斯图加特”，limit=2。根据结果说明这个地点对应什么，并注明来源路径。', {'mcp_project_docs_search_documents'}),
        ):
            thread = new_session(admin)
            status, response, seconds = chat(admin, thread, question)
            events = response.get('data', {}).get('tool_events', [])
            calls = {event['tool'] for event in events if event['result']['ok']}
            check(name, status == 200 and expected <= calls, question=question, seconds=seconds, actual=response)

        thread = new_session(admin)
        started = time.monotonic()
        request = urllib.request.Request(BASE + '/api/chat/stream', data=json.dumps({
            'thread_id': thread, 'user_message': '/tool 调用run_command，command=python_version，只报告真实Python版本。'}, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + admin})
        events, streamed = [], ''
        with opener.open(request, timeout=150) as stream:
            block = []
            for raw in stream:
                line = raw.decode('utf-8').rstrip('\r\n')
                if line:
                    block.append(line)
                    continue
                if not block:
                    continue
                kind = next(row[7:] for row in block if row.startswith('event: '))
                data = json.loads('\n'.join(row[6:] for row in block if row.startswith('data: ')))
                events.append({'event': kind, 'data': data})
                if kind == 'token':
                    streamed += data['token']
                elif kind == 'replace':
                    streamed = data['response']
                block = []
        complete = next((item['data'] for item in events if item['event'] == 'complete'), {})
        calls = complete.get('tool_events', [])
        check('model_command_sse_consistency', bool(complete) and streamed == complete['response'] and
            any(call['tool'] == 'run_command' and call['result']['ok'] and 'Python 3.12' in json.dumps(call['result']) for call in calls) and
            any(item['event'] == 'status' and item['data']['stage'] == 'tools' for item in events) and
            not any(item['event'] == 'error' for item in events), seconds=round(time.monotonic()-started, 3), events=events)

        thread = new_session(admin)
        arguments = {'path': relative, 'content': content}
        question = '/tool 调用write_file，参数JSON对象如下；直接解析字段值，不要对content再次编码或额外添加字符：\n' + json.dumps(arguments, ensure_ascii=False)
        status, proposed, seconds = chat(admin, thread, question)
        pending = proposed.get('data', {}).get('pending_action')
        check('model_write_proposal_no_side_effect', status == 200 and pending and not scratch.exists(), question=question, seconds=seconds, actual=proposed)
        if pending:
            action_keys.append(f"tool_action:{admin_user['username']}:{pending['action_id']}")
            status, confirmed, seconds = chat(admin, thread, '确认')
            check('real_chat_confirm_executes_once', status == 200 and confirmed['data']['last_result']['ok'] and
                confirmed['data']['pending_action'] is None and scratch.read_text(encoding='utf-8') == content,
                seconds=seconds, actual=confirmed)
            before = scratch.stat().st_mtime_ns
            status, repeated, seconds = chat(admin, thread, '确认')
            check('repeat_confirm_no_reexecution', status == 200 and scratch.stat().st_mtime_ns == before and
                not repeated['data']['tool_events'], seconds=seconds, actual=repeated)

        thread = new_session(admin)
        question = f'/tool 调用write_file创建新文件 tool_workspace/tool_cancel_{run}.md，内容为cancel probe。'
        status, proposed, seconds = chat(admin, thread, question)
        pending = proposed.get('data', {}).get('pending_action')
        status, aborted, seconds = chat(admin, thread, '取消')
        check('real_chat_cancel_no_write', pending and status == 200 and aborted['data']['pending_action'] is None and not cancelled.exists(), actual=aborted)

        thread = new_session(employee)
        status, rejected, seconds = chat(employee, thread, '/tool 读取本地README.md')
        check('employee_chat_cannot_trigger_tool', status == 200 and not rejected['data']['tool_events'] and '没有可调用' in rejected['data']['response'], actual=rejected)
        status, rag = api('/api/rag/query', {'question': '公司全称是什么？', 'use_llm': False, 'top_k': 8}, employee, timeout=45)
        check('existing_rag_still_works', status == 200 and '澜图智能装备股份有限公司' in rag['data']['answer'] and bool(rag['data']['citations']), actual=rag)
    except Exception as exc:
        receipt['error_type'] = type(exc).__name__
        save()
        raise
    finally:
        cleanup = []
        for token, thread in sessions:
            try:
                status, result = api('/api/chat/session/' + thread, token=token, method='DELETE', timeout=15)
                cleanup.append({'thread': thread, 'deleted': status == 200})
            except (urllib.error.URLError, TimeoutError):
                cleanup.append({'thread': thread, 'deleted': False})
        receipt['session_cleanup'] = cleanup
        for path in (scratch, cancelled):
            # Exact run-owned names only; no recursive workspace deletion.
            if path.parent == ROOT / 'tool_workspace' and run in path.name:
                path.unlink(missing_ok=True)
        if action_keys:
            client.delete(*action_keys)
        receipt['owned_action_remaining'] = sum(client.exists(key) for key in action_keys)
        client.close()
        receipt['finished_at'] = datetime.now().astimezone().isoformat()
        receipt['summary'] = {'passed': sum(row['passed'] for row in receipt['checks']), 'total': len(receipt['checks'])}
        receipt['passed'] = (not receipt.get('error_type') and all(row['passed'] for row in receipt['checks']) and
            all(row['deleted'] for row in cleanup) and receipt['owned_action_remaining'] == 0)
        save()
        print(json.dumps({'passed': receipt['passed'], **receipt['summary']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
