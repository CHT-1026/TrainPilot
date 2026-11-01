"""Read-only MCP server using the same schemas and boundary validation as local tools."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from app.evaluation.retrieval_corpus import CorpusTools, TOOL_SPECS


async def serve(workspace: Path):
    tools = CorpusTools(workspace)

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(
            name=spec['function']['name'], description=spec['function']['description'],
            input_schema=spec['function']['parameters'],
            annotations=types.ToolAnnotations(read_only_hint=True, destructive_hint=False))
            for spec in TOOL_SPECS])

    async def call_tool(ctx, params):
        # CorpusTools validates the original arguments once; no decorator type coercion.
        result = tools.run(params.name, params.arguments or {})
        return types.CallToolResult(content=[types.TextContent(type='text',
            text=json.dumps(result, ensure_ascii=False))], is_error='error' in result)

    server = Server('blind-public-retrieval', on_list_tools=list_tools, on_call_tool=call_tool)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(serve(args.workspace))


if __name__ == '__main__':
    main()
