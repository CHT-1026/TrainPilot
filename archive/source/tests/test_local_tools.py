"""Tool boundary tests: real temporary files, fake process/network boundaries."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
from redis.exceptions import ConnectionError as RedisConnectionError

from app.capabilities.tools.actions import execute_confirmed
from app.capabilities.tools.agent import run_agent
from app.capabilities.tools.config import Command, MCPServerConfig, MCPToolPolicy, ToolsConfig
from app.capabilities.tools.local import ToolError
from app.capabilities.tools.registry import ToolRegistry, build_registry
from app.domain.user.models import UserContext
from app.workflows.chat_nodes.common import actor_from_state

ADMIN = UserContext(user_id="tool_test", name="工具测试", role="admin")
EMPLOYEE = UserContext(user_id="employee", name="员工", role="employee")


class FakeRedis:
    def __init__(self): self.data = {}
    async def set(self, key, value, *, nx=False, ex=None):
        if nx and key in self.data:return None
        self.data[key] = value
        return True
    async def get(self, key): return self.data.get(key)


class ToolModel:
    def __init__(self, replies): self.replies = iter(replies); self.received = []; self.schemas = []
    def bind_tools(self, schemas): self.schemas = schemas; return self
    async def ainvoke(self, messages): self.received.append(list(messages)); return next(self.replies)


def tool_reply(name, args, identifier="call1"):
    return AIMessage(content="", tool_calls=[{"name":name,"args":args,"id":identifier}])


class LocalToolsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="lantu-tools-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "notes").mkdir()
        (self.root / "sample.py").write_text("first\nsecond\nthird\n", encoding="utf-8")
        self.config = ToolsConfig(workspace_root=str(self.root), write_roots=["notes"], commands={
            "version":Command(argv=["{python}","--version"], description="version", read_only=True),
            "mutate":Command(argv=["{python}","script.py"], description="mutate"),
        })
        self.registry = ToolRegistry(ADMIN, self.config)

    async def test_registry_registers_six_schemas_and_reads_real_lines(self):
        self.assertEqual(len(self.registry.tools), 6)
        result = await self.registry.invoke("read_file", {"path":"sample.py","start_line":2,"line_count":1})
        self.assertTrue(result.ok);self.assertEqual(result.data["text"],"second")
        self.assertEqual(result.data["total_lines"],3)
        self.assertEqual(len(result.data["sha256"]),64)

    async def test_invalid_types_unknown_params_and_path_escape_are_explicit_errors(self):
        for args in ({"path":"sample.py","line_count":"2"}, {"path":"sample.py","actor":"admin"}):
            result = await self.registry.invoke("read_file",args)
            self.assertEqual(result.code,"INVALID_TOOL_ARGS")
        self.assertEqual((await self.registry.invoke("read_file",{"path":"../outside.py"})).code,"PATH_NOT_ALLOWED")

    async def test_credentials_and_virtual_environment_are_not_read_or_listed(self):
        (self.root / ".env.local").write_text("SECRET=not_for_model",encoding="utf-8")
        self.assertEqual((await self.registry.invoke("read_file",{"path":".env.local"})).code,"PATH_NOT_ALLOWED")
        listing=await self.registry.invoke("list_files",{})
        self.assertNotIn('.env.local',str(listing.data))
        self.assertEqual((await self.registry.invoke("read_file",{"path":".venv312/x.py"})).code,"PATH_NOT_ALLOWED")

    async def test_normal_tokenizer_code_and_text_extensions_are_not_blocked(self):
        (self.root/'tokenizer.rst').write_text('normal code documentation',encoding='utf-8')
        (self.root/'.env.example').write_text('OPENAI_API_KEY=placeholder',encoding='utf-8')
        self.assertTrue((await self.registry.invoke('read_file',{'path':'tokenizer.rst'})).ok)
        self.assertTrue((await self.registry.invoke('read_file',{'path':'.env.example'})).ok)

    async def test_write_requires_confirmation_then_checks_current_hash(self):
        args={"path":"notes/new.md","content":"one"}
        self.assertEqual((await self.registry.invoke("write_file",args)).code,"CONFIRMATION_REQUIRED")
        plan=self.registry.prepare("write_file",args)
        self.assertTrue(plan.needs_confirmation)
        created=await self.registry.execute(plan);self.assertTrue(created.ok)
        wrong=await self.registry.execute(self.registry.prepare("write_file",{**args,"content":"two"}))
        self.assertEqual(wrong.code,"FILE_CHANGED")
        updated=await self.registry.execute(self.registry.prepare("write_file",{**args,"content":"two","expected_sha256":created.data["sha256"]}))
        self.assertTrue(updated.ok);self.assertEqual((self.root/'notes/new.md').read_text(),'two')
        denied=await self.registry.execute(self.registry.prepare("write_file",{"path":"sample.py","content":"x"}))
        self.assertEqual(denied.code,"PATH_NOT_ALLOWED")

    async def test_process_receipt_and_failure_do_not_silently_fallback(self):
        with patch('app.capabilities.tools.local.run_process',new=AsyncMock(return_value={"exit_code":0,"output":"Python 3.12","truncated":False})) as process:
            result=await self.registry.invoke('run_command',{'command':'version'})
        self.assertTrue(result.ok);self.assertTrue(process.await_args.args[0][0].endswith('python.exe'))
        self.assertEqual((await self.registry.invoke('run_command',{'command':'mutate'})).code,'CONFIRMATION_REQUIRED')
        self.assertEqual((await self.registry.invoke('run_command',{'command':'unregistered'})).code,'COMMAND_NOT_REGISTERED')
        with patch('app.capabilities.tools.local.run_process',new=AsyncMock(return_value={"exit_code":2,"output":"failure","truncated":False})):
            self.assertEqual((await self.registry.invoke('run_command',{'command':'version'})).code,'COMMAND_FAILED')

    async def test_nonzero_exit_is_not_created_by_output_truncation(self):
        with patch('app.capabilities.tools.local.run_process',new=AsyncMock(return_value={"exit_code":0,"output":"x"*30000,"truncated":True})):
            result=await self.registry.invoke('run_command',{'command':'version'})
        self.assertTrue(result.ok);self.assertTrue(result.data['truncated']);self.assertEqual(result.data['exit_code'],0)

    async def test_search_passes_query_as_literal_not_options(self):
        with patch('app.capabilities.tools.local.run_process',new=AsyncMock(return_value={"exit_code":0,"output":"sample.py:1:first\n","truncated":False})) as process:
            result=await self.registry.invoke('search_code',{'query':'--files'})
        self.assertTrue(result.ok)
        argv=process.await_args.args[0];self.assertEqual(argv[-3:],['--','--files','.'])

    async def test_role_and_server_identity_cannot_come_from_conversation(self):
        self.assertEqual(ToolRegistry(EMPLOYEE,self.config).tools,{})
        actor=actor_from_state({'user':EMPLOYEE.model_dump(),'conversation_context':{'user_role':'admin'}})
        self.assertEqual(actor.role,'employee')
        with self.assertRaises(KeyError):actor_from_state({'conversation_context':{'user_role':'admin'}})

    async def test_auth_user_mapping_uses_canonical_service_fields_without_role_defaults(self):
        data = {'username': 'employee', 'name': '员工', 'role': 'employee',
                'department': None, 'position': None, 'manager_id': None}
        self.assertEqual(UserContext.from_dict(data).user_id, 'employee')
        del data['role']
        with self.assertRaises(KeyError):
            UserContext.from_dict(data)

    async def test_model_actually_gets_schemas_and_tool_result_on_next_round(self):
        model=ToolModel([tool_reply('read_file',{'path':'sample.py','line_count':1}), AIMessage(content='工具read_file返回first')])
        result=await run_agent(self.registry,model,'读取文件',0)
        self.assertIn('first',result['task_result']);self.assertEqual(len(result['tool_events']),1)
        self.assertEqual(model.received[1][-1].type,'tool')
        self.assertEqual(json.loads(model.received[1][-1].content)['data']['text'],'first')
        self.assertEqual(len(model.schemas),6)

    async def test_model_write_draft_does_not_execute_and_has_bound_arguments(self):
        model=ToolModel([tool_reply('write_file',{'path':'notes/new.md','content':'hello'})])
        result=await run_agent(self.registry,model,'写入文件',2)
        self.assertFalse((self.root/'notes/new.md').exists())
        self.assertEqual(result['draft_version'],3)
        self.assertEqual(result['pending_action']['draft']['name'],'write_file')
        self.assertGreater(result['pending_action']['expires_at'],time.time())

    async def test_confirmation_has_persistent_claim_and_does_not_run_twice(self):
        redis=FakeRedis();pending={'action_id':'test','draft_version':1,'expires_at':time.time()+600,
            'draft':{'name':'write_file','arguments':{'path':'notes/new.md','content':'hello'}}}
        pending['binding']=self.registry.action_binding(self.registry.prepare(**pending['draft']))
        with patch('app.capabilities.tools.actions.get_async_redis_client',return_value=redis):
            first=await execute_confirmed(self.registry,pending,1)
            repeated=await execute_confirmed(self.registry,pending,1)
        self.assertTrue(first.ok);self.assertEqual(first,repeated)
        self.assertEqual((self.root/'notes/new.md').read_text(),'hello')

    async def test_stale_unknown_and_storage_failure_never_replay_write(self):
        pending={'action_id':'test','draft_version':1,'expires_at':time.time()+600,
            'draft':{'name':'write_file','arguments':{'path':'notes/new.md','content':'hello'}}}
        pending['binding']=self.registry.action_binding(self.registry.prepare(**pending['draft']))
        self.assertEqual((await execute_confirmed(self.registry,pending,2)).code,'STALE_TOOL_DRAFT')
        with patch('app.capabilities.tools.actions.get_async_redis_client',side_effect=RedisConnectionError()):
            self.assertEqual((await execute_confirmed(self.registry,pending,1)).code,'ACTION_STORE_UNAVAILABLE')
        redis=FakeRedis();redis.data['tool_action:tool_test:test']=json.dumps({'status':'running'})
        with patch('app.capabilities.tools.actions.get_async_redis_client',return_value=redis):
            self.assertEqual((await execute_confirmed(self.registry,pending,1)).code,'ACTION_OUTCOME_UNKNOWN')
        self.assertFalse((self.root/'notes/new.md').exists())

    async def test_confirmation_cannot_silently_switch_workspace(self):
        draft={'name':'write_file','arguments':{'path':'notes/new.md','content':'hello'}}
        pending={'action_id':'test','draft_version':1,'expires_at':time.time()+600,'draft':draft,
                 'binding':self.registry.action_binding(self.registry.prepare(**draft))}
        changed=ToolRegistry(ADMIN,self.config.model_copy(update={'workspace_root':str(self.root/'other')}))
        self.assertEqual((await execute_confirmed(changed,pending,1)).code,'STALE_TOOL_DRAFT')

    async def test_unavailable_mcp_preserves_local_tools_without_fake_success(self):
        server=MCPServerConfig(name='offline',transport='stdio',command='missing',tools={'search':MCPToolPolicy(read_only=True)})
        config=self.config.model_copy(update={'mcp_servers':[server]})
        with patch('app.integrations.mcp_client.discover',new=AsyncMock(side_effect=ConnectionError('PRIVATE_TOKEN'))):
            registry=await build_registry(ADMIN,config)
        self.assertEqual(len(registry.tools),6);self.assertEqual(registry.unavailable,[{'server':'offline','code':'MCP_UNAVAILABLE'}])
        self.assertNotIn('PRIVATE_TOKEN',str(registry.describe()))

    async def test_model_without_tool_call_cannot_claim_an_operation_completed(self):
        model=ToolModel([AIMessage(content='已经创建了文件')])
        result=await run_agent(self.registry,model,'写入文件',0)
        self.assertTrue(result['task_result'].startswith('本轮未执行任何工具操作。'))
        self.assertEqual(result['tool_events'], [])
        self.assertIsNone(result['pending_action'])

    async def test_missing_information_can_be_clarified_without_a_fake_tool_execution(self):
        model=ToolModel([AIMessage(content='请提供要读取的文件路径。')])
        result=await run_agent(self.registry,model,'读取文件',0)
        self.assertIn('请提供要读取的文件路径', result['task_result'])
        self.assertEqual(result['tool_events'], [])
