"""Untrusted server schemas stop at one registry boundary; roles precede connection."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.capabilities.tools.config import MCPServerConfig, MCPToolPolicy, ToolsConfig
from app.capabilities.tools.registry import build_registry
from tests.test_local_tools import ADMIN, EMPLOYEE


class MCPBoundaryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = MCPServerConfig(name='fixture', transport='stdio', command='fixture',
            tools={'echo': MCPToolPolicy(read_only=True)})
        self.config = ToolsConfig(mcp_servers=[self.server])

    async def test_internal_schema_refs_validate_and_forward_arguments_once(self):
        schema = {'type': 'object', 'properties': {'text': {'$ref': '#/$defs/text'}},
                  '$defs': {'text': {'type': 'string'}}, 'required': ['text'], 'additionalProperties': False}
        remote = SimpleNamespace(name='echo', description='echo', input_schema=schema)
        with patch('app.integrations.mcp_client.discover', new=AsyncMock(return_value=[remote])), \
             patch('app.integrations.mcp_client.call_tool', new=AsyncMock(return_value={'echo': '原样'})) as call:
            registry = await build_registry(ADMIN, self.config)
            result = await registry.invoke('mcp_fixture_echo', {'text': '原样'})
            invalid = await registry.invoke('mcp_fixture_echo', {'text': 1})
        self.assertTrue(result.ok)
        self.assertEqual(invalid.code, 'INVALID_TOOL_ARGS')
        call.assert_awaited_once_with(self.server, 'echo', {'text': '原样'})

    async def test_external_schema_ref_is_not_fetched_or_forwarded(self):
        remote = SimpleNamespace(name='echo', description='echo', input_schema={
            'type': 'object', 'properties': {'text': {'$ref': 'https://schema.invalid/private'}}})
        with patch('app.integrations.mcp_client.discover', new=AsyncMock(return_value=[remote])), \
             patch('app.integrations.mcp_client.call_tool', new=AsyncMock()) as call:
            registry = await build_registry(ADMIN, self.config)
            result = await registry.invoke('mcp_fixture_echo', {'text': 'value'})
        self.assertEqual(result.code, 'INVALID_TOOL_ARGS')
        call.assert_not_awaited()

    async def test_bad_remote_schema_keeps_local_tools_and_unauthorized_role_never_connects(self):
        remote = SimpleNamespace(name='echo', description='echo', input_schema={'type': 'not_a_type'})
        with patch('app.integrations.mcp_client.discover', new=AsyncMock(return_value=[remote])) as discover:
            registry = await build_registry(ADMIN, self.config)
            employee = await build_registry(EMPLOYEE, self.config)
        self.assertEqual(len(registry.tools), 6)
        self.assertEqual(registry.unavailable, [{'server': 'fixture', 'code': 'MCP_SCHEMA_INVALID'}])
        self.assertEqual(employee.tools, {})
        discover.assert_awaited_once_with(self.server)
