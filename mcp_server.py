import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ["LOG_DIR"] = str(Path(os.getenv("LOG_DIR", "logs")) / "mcp")

from mcp.server.fastmcp import FastMCP

import mcp_tool
import rag
from logger import logger

mcp = FastMCP("healthcare-kb", host="127.0.0.1", port=8001)

for _fn in (mcp_tool.kb_search, mcp_tool.kb_list_sources, mcp_tool.kb_reindex, mcp_tool.web_search_save):
    mcp.tool()(_fn)
    logger.info("MCP tool registered | %s", _fn.__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()

    mcp.settings.host, mcp.settings.port = a.host, a.port

    logger.info("MCP server start | transport=%s host=%s port=%s", a.transport, a.host, a.port)
    rag.sync_data_dir()
    mcp.run(transport=a.transport)


if __name__ == "__main__":
    main()