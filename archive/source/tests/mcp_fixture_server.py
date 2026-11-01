"""Explicit protocol test fixture, never discovered as a unit test."""
import argparse
import asyncio
import os

from mcp.server import MCPServer
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

mcp=MCPServer('lantu-protocol-fixture',log_level='CRITICAL')


@mcp.tool()
def echo(text: str) -> dict:
    """Echo test input and report whether a configured credential was supplied, never its value."""
    return {'text':text,'fixture_env_present':bool(os.getenv('MCP_FIXTURE_SECRET'))}


@mcp.tool()
def fail() -> str:
    """Return a deliberate tool execution failure."""
    raise ValueError('DO_NOT_EXPOSE_FIXTURE_SECRET')


@mcp.tool()
async def slow() -> str:
    """Exercise timeout cancellation."""
    await asyncio.sleep(5)
    return 'late'


class RequireHeader(BaseHTTPMiddleware):
    async def dispatch(self,request,call_next):
        if request.headers.get('authorization') != os.environ['MCP_FIXTURE_AUTH']:
            return JSONResponse({'error':'unauthorized'},status_code=401)
        return await call_next(request)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--http',action='store_true');parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    if args.http:
        import uvicorn
        application=mcp.streamable_http_app(stateless_http=True,json_response=True)
        application.add_middleware(RequireHeader)
        uvicorn.run(application,host='127.0.0.1',port=args.port,log_level='warning')
    else:
        mcp.run()
