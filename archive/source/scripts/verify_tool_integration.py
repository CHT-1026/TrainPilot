"""Explicit local/MCP integration test; no model charges or production SQL writes."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from redis.exceptions import RedisError

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env.local',override=False)
from app.capabilities.tools.actions import execute_confirmed
from app.capabilities.tools.config import Command,MCPServerConfig,MCPToolPolicy,ToolsConfig,load_config
from app.capabilities.tools.local import process_env
from app.capabilities.tools.registry import build_registry
from app.database.connection import close_async_redis,get_async_redis_client
from app.domain.user.models import UserContext

OUT=ROOT/'research/tool_mcp_20261027';OUT.mkdir(parents=True,exist_ok=True)
RUN=uuid4().hex[:12]
report={'run':RUN,'started_at':datetime.now().astimezone().isoformat(),'checks':[],
        'scope':'Real temporary filesystem, local CLI, Redis action claim, SDK stdio and authenticated Streamable HTTP. No model calls, no production SQL writes.'}


def check(name,passed,**evidence):
    report['checks'].append({'id':name,'passed':bool(passed),**evidence})
    (OUT/'integration_latest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,bool(passed),flush=True)


def content(result):
    return json.loads(result.data['content'][0]['text'])


async def main():
    os.environ['NO_PROXY']=','.join(filter(None,[os.getenv('NO_PROXY',''),'127.0.0.1','localhost']))
    redis=get_async_redis_client();http_process=None;owned_action=None
    log=(OUT/f'mcp_http_{RUN}.log').open('w',encoding='utf-8')
    try:
        with tempfile.TemporaryDirectory(prefix='lantu-integration-') as temporary:
            root=Path(temporary);(root/'notes').mkdir();(root/'example.py').write_text('value = 42\n',encoding='utf-8')
            config=ToolsConfig(workspace_root=str(root),write_roots=['notes'],command_timeout_seconds=.8,
                commands={
                    'version':Command(argv=['{python}','--version'],description='version',read_only=True),
                    'env_check':Command(argv=['{python}','-c','import os;print(bool(os.getenv("OPENAI_API_KEY")))'],description='safe test',read_only=True),
                    'sleep':Command(argv=['{python}','-c','import time;time.sleep(10)'],description='timeout test',read_only=True),
                },mcp_servers=load_config().mcp_servers)
            actor=UserContext(user_id='tool_probe_'+RUN,name='工具协议测试',role='admin')
            registry=await build_registry(actor,config)
            read=await registry.invoke('read_file',{'path':'example.py'})
            check('real_file_read',read.ok and read.data['text']=='value = 42',result=read.to_dict())
            search=await registry.invoke('search_code',{'query':'value','path':'example.py'})
            check('real_ripgrep',search.ok and any('value = 42' in value for value in search.data['matches']),result=search.to_dict())
            command=await registry.invoke('run_command',{'command':'version'})
            check('real_cli',command.ok and 'Python 3.12' in command.data['output'],result=command.to_dict())
            clean_env=await registry.invoke('run_command',{'command':'env_check'})
            check('cli_no_ambient_model_key',clean_env.ok and clean_env.data['output'].strip()=='False')
            slow=await registry.invoke('run_command',{'command':'sleep'})
            check('cli_timeout',slow.code=='TOOL_TIMEOUT',result=slow.to_dict())
            pending={'action_id':'integration_'+RUN,'draft_version':1,'expires_at':time.time()+600,
                     'draft':{'name':'write_file','arguments':{'path':'notes/result.md','content':'integration write'}}}
            pending['binding']=registry.action_binding(registry.prepare(**pending['draft']))
            owned_action=f"tool_action:{actor.user_id}:{pending['action_id']}"
            first=await execute_confirmed(registry,pending,1);again=await execute_confirmed(registry,pending,1)
            check('real_redis_write_confirmation_and_replay',first.ok and first==again and (root/'notes/result.md').read_text()=='integration write',result=first.to_dict())
            external=await registry.invoke('mcp_project_docs_search_documents',{'query':'研发','limit':2})
            check('stdio_real_project_documents',external.ok and len(content(external)['hits'])==2,result=external.to_dict())
            employee=await build_registry(UserContext(user_id='employee',name='普通员工',role='employee'),config)
            check('role_scope_before_discovery',not employee.tools and not employee.unavailable)
            fixture=MCPServerConfig(name='fixture',transport='stdio',command='{python}',
                args=['-X','utf8','tests/mcp_fixture_server.py'],env={'MCP_FIXTURE_SECRET':'test-only-value'},
                tools={name:MCPToolPolicy(read_only=True) for name in ('echo','fail','slow')},timeout_seconds=5)
            fixture_config=config.model_copy(update={'mcp_servers':[fixture]})
            remote=await build_registry(actor,fixture_config)
            echoed=await remote.invoke('mcp_fixture_echo',{'text':'原样参数 "MCP" 123'})
            check('stdio_parameter_and_env',echoed.ok and content(echoed)=={'text':'原样参数 "MCP" 123','fixture_env_present':True})
            error=await remote.invoke('mcp_fixture_fail',{})
            check('remote_error_is_not_success_or_secret_echo',error.code=='MCP_TOOL_ERROR' and 'DO_NOT_EXPOSE' not in str(error.to_dict()),result=error.to_dict())
            from app.integrations.mcp_client import call_tool
            started=time.monotonic()
            try:
                await call_tool(fixture.model_copy(update={'timeout_seconds':.3}),'slow',{})
                timed_out=False
            except Exception as exc:
                timed_out=getattr(exc,'code',None)=='MCP_TIMEOUT'
            check('stdio_timeout',timed_out,seconds=round(time.monotonic()-started,3))
            with socket.socket() as port_socket:
                port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1]
            auth='Bearer test-'+RUN;os.environ['MCP_LANTU_TOOL_PROBE_AUTH']=auth
            http_process=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'tests/mcp_fixture_server.py'),'--http','--port',str(port)],
                cwd=ROOT,env={**process_env(),'MCP_FIXTURE_AUTH':auth},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            deadline=time.monotonic()+12
            while True:
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
                except OSError:
                    if http_process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Owned MCP HTTP fixture did not start')
                    await asyncio.sleep(.1)
            http=MCPServerConfig(name='httpfixture',transport='streamable_http',url=f'http://127.0.0.1:{port}/mcp',
                headers={'Authorization':'${MCP_LANTU_TOOL_PROBE_AUTH}'},tools={'echo':MCPToolPolicy(read_only=True)},timeout_seconds=5)
            http_registry=await build_registry(actor,config.model_copy(update={'mcp_servers':[http]}))
            check('http_tool_registration_allowlist',set(http_registry.tools)-set(registry.tools)=={'mcp_httpfixture_echo'},unavailable=http_registry.unavailable)
            echoed=await http_registry.invoke('mcp_httpfixture_echo',{'text':'HTTP原样参数'})
            check('http_auth_and_parameters',echoed.ok and content(echoed)['text']=='HTTP原样参数',result=echoed.to_dict())
            wrong=http.model_copy(update={'headers':{}})
            degraded=await build_registry(actor,config.model_copy(update={'mcp_servers':[wrong]}))
            check('http_auth_failure_keeps_local_tools',len(degraded.tools)==6 and bool(degraded.unavailable),unavailable=degraded.unavailable)
            check('http_failure_local_read_still_works',(await degraded.invoke('read_file',{'path':'example.py'})).ok)
    finally:
        if owned_action:
            try:
                await redis.delete(owned_action)
                report['owned_action_remaining']=await redis.exists(owned_action)
            except RedisError:
                report['cleanup_error']='Redis unavailable: owned action cleanup could not be verified'
        await close_async_redis()
        if http_process and http_process.poll() is None:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(http_process.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            else:http_process.terminate()
            http_process.wait(timeout=10)
        report['http_fixture_stopped']=http_process is None or http_process.poll() is not None
        log.close()
        report['finished_at']=datetime.now().astimezone().isoformat()
        report['passed']=all(row['passed'] for row in report['checks']) and not report.get('cleanup_error') and report.get('owned_action_remaining')==0
        report['summary']={'passed':sum(row['passed'] for row in report['checks']),'total':len(report['checks'])}
        for name in ('integration_latest.json',f'integration_{RUN}.json'):
            (OUT/name).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary']),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(asyncio.run(main()))
