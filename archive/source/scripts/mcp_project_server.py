"""Separate MCP process exposing real read-only project documents and Git metadata.

Run with stdio (default), or --http --port 8765 for Streamable HTTP.
"""
import argparse
import os
import subprocess
from pathlib import Path
from typing import Annotated
from pydantic import Field

from mcp.server import MCPServer

ROOT = Path(__file__).resolve().parents[1]
mcp = MCPServer("lantu-project-docs")


@mcp.tool()
def search_documents(query: Annotated[str, Field(min_length=1, max_length=500, pattern=r"\S")],
                     limit: Annotated[int, Field(ge=1, le=20)] = 5) -> dict:
    """搜索本项目公开企业Markdown资料，返回原文片段与来源路径；非员工私有数据查询。"""
    hits = []
    for path in sorted((ROOT / "enterprise_rag/corpus/docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        position = text.lower().find(query.lower())
        if position >= 0:
            hits.append({"source": path.relative_to(ROOT).as_posix(),
                         "text": text[max(0, position - 100):position + 500]})
            if len(hits) == limit:
                break
    return {"query": query, "hits": hits, "scope": "project public document files"}


@mcp.tool()
def repository_summary() -> dict:
    """读取Git最近提交；不修改仓库，不返回remote URL或凭据。"""
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    result = subprocess.run(["git", "log", "-1", "--format=%h %s"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=5, check=True, **options)
    return {"latest_commit": result.stdout.strip()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.http:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    else:
        mcp.run()
