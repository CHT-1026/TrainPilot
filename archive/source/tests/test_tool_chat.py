"""Production graph wiring with a fake model/store and real temporary file tools."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.capabilities.tools.config import ToolsConfig
from app.capabilities.tools.registry import ToolRegistry
from app.routes import tools as tools_api
from app.routes.auth import get_user_context
from app.workflows.chat_nodes.tools import is_tool_request
from tests.test_knowledge_flow import build_graph
from tests.test_local_tools import ADMIN, EMPLOYEE, FakeRedis, ToolModel, tool_reply


class ToolGraphTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='tool-graph-');self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);(self.root/'notes').mkdir()
        (self.root/'readme.rst').write_text('real local content',encoding='utf-8')
        self.registry=ToolRegistry(ADMIN,ToolsConfig(workspace_root=str(self.root),write_roots=['notes']))

    async def test_read_request_traverses_the_existing_graph_and_publishes_once(self):
        model=ToolModel([tool_reply('read_file',{'path':'readme.rst'}),AIMessage(content='read_file: real local content')])
        graph=build_graph(model)
        with patch('app.workflows.chat_nodes.tools.build_registry',new=AsyncMock(return_value=self.registry)):
            result=await graph.ainvoke({'user_message':'/tool 读取 readme.rst','user':ADMIN.model_dump(),'messages':[]},
                                      {'configurable':{'thread_id':'read'}})
        self.assertEqual(result['current_task'],'tool_action')
        self.assertEqual(result['response'],'read_file: real local content')
        self.assertEqual(len(result['tool_events']),1)
        self.assertEqual(len(model.received),2)

    async def test_confirmed_write_clears_pending_and_repeated_confirm_does_not_write(self):
        model=ToolModel([tool_reply('write_file',{'path':'notes/result.md','content':'verified write'})])
        graph=build_graph(model);config={'configurable':{'thread_id':'write'}};redis=FakeRedis()
        with patch('app.workflows.chat_nodes.tools.build_registry',new=AsyncMock(return_value=self.registry)), \
             patch('app.capabilities.tools.actions.get_async_redis_client',return_value=redis):
            draft=await graph.ainvoke({'user_message':'/tool 创建 notes/result.md','user':ADMIN.model_dump(),'messages':[]},config)
            self.assertFalse((self.root/'notes/result.md').exists())
            self.assertIsNotNone(draft['pending_action'])
            final=await graph.ainvoke({'user_message':'确认','user':ADMIN.model_dump(),'tool_events':[]},config)
            self.assertEqual(final['current_task'],'confirm_tool');self.assertTrue(final['last_result']['ok'])
            self.assertIsNone(final['pending_action']);self.assertEqual((self.root/'notes/result.md').read_text(),'verified write')
            def recognize_confirm(prompt):
                self.assertIn('当前待确认动作：无',prompt)
                return AIMessage(content=json.dumps({'intent':'confirm_action','extracted_info':{}}))
            model.invoke=recognize_confirm
            repeated=await graph.ainvoke({'user_message':'确认','intent':'confirm_action','user':ADMIN.model_dump()},config)
        self.assertEqual(len(redis.data),1)
        self.assertIsNone(repeated['pending_action'])

    async def test_cancel_discards_only_the_pending_tool_and_does_not_write(self):
        model=ToolModel([tool_reply('write_file',{'path':'notes/cancel.md','content':'never written'})])
        graph=build_graph(model);config={'configurable':{'thread_id':'cancel'}}
        with patch('app.workflows.chat_nodes.tools.build_registry',new=AsyncMock(return_value=self.registry)):
            await graph.ainvoke({'user_message':'/tool 创建 notes/cancel.md','user':ADMIN.model_dump()},config)
            result=await graph.ainvoke({'user_message':'取消','user':ADMIN.model_dump()},config)
        self.assertIsNone(result['pending_action']);self.assertFalse((self.root/'notes/cancel.md').exists())

    async def test_policy_file_questions_are_not_forced_to_project_tools(self):
        for question in ('查看年假制度文件规定','请读取员工年假制度文件并说明天数','公司年假制度文件有哪些？'):
            self.assertFalse(is_tool_request(question))
        self.assertTrue(is_tool_request('读取本地文件 readme.rst'))


class ToolRouteTest(unittest.TestCase):
    def test_authenticated_catalog_role_scope_and_write_no_bypass(self):
        app=FastAPI();app.include_router(tools_api.router)
        admin_registry=ToolRegistry(ADMIN,ToolsConfig())
        app.dependency_overrides[get_user_context]=lambda: ADMIN
        with TestClient(app) as client,patch('app.routes.tools.build_registry',new=AsyncMock(return_value=admin_registry)):
            listed=client.get('/api/tools').json()['data']['tools']
            self.assertEqual(len(listed),6)
            result=client.post('/api/tools/call',json={'name':'write_file','arguments':{'path':'tool_workspace/test.md','content':'x'}})
            self.assertEqual(result.json()['data']['code'],'CONFIRMATION_REQUIRED')
            self.assertEqual(client.post('/api/tools/call',json={'name':'read_file','arguments':{},'actor':'admin'}).status_code,422)
        app.dependency_overrides[get_user_context]=lambda: EMPLOYEE
        with TestClient(app) as client,patch('app.routes.tools.build_registry',new=AsyncMock(return_value=ToolRegistry(EMPLOYEE,ToolsConfig()))):
            self.assertEqual(client.get('/api/tools').json()['data']['tools'],[])
