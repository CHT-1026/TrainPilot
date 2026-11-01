"""Isolated read-only MCP surface over a fixed public corpus snapshot."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.server.mcpserver import MCPServer
from app.evaluation.retrieval_corpus import CorpusTools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    tools = CorpusTools(args.workspace)
    server = MCPServer("blind-public-retrieval")

    @server.tool()
    def search(query: str, top_k: int = 5) -> dict:
        return tools.run("search", {"query": query, "top_k": top_k})

    @server.tool()
    def read(doc_id: str, offset: int = 0, limit: int = 40) -> dict:
        return tools.run("read", {"doc_id": doc_id, "offset": offset, "limit": limit})

    @server.tool(name="list")
    def list_documents(keyword: str = "", offset: int = 0) -> dict:
        return tools.run("list", {"keyword": keyword, "offset": offset})

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
