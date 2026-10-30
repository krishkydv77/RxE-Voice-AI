import asyncio
import os
from contextlib import AsyncExitStack

from mcp import ClientSession

try:
    from mcp.client.streamable_http import streamablehttp_client as _http_client
except ImportError:
    from mcp.client.streamable_http import streamable_http_client as _http_client

from logger import logger, log_metric, Timer

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")
MCP_CONNECT_TIMEOUT = float(os.getenv("MCP_CONNECT_TIMEOUT", "5"))
MCP_CALL_TIMEOUT = float(os.getenv("MCP_CALL_TIMEOUT", "20"))


class MCPToolClient:
    def __init__(self, url: str = MCP_SERVER_URL) -> None:
        self.url = url
        self.active = False
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        await self.close()
        logger.info("MCP client connect | url=%s", self.url)
        stack = AsyncExitStack()
        try:
            async def _open():
                streams = await stack.enter_async_context(_http_client(self.url))
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                return session

            with Timer() as t:
                self._session = await asyncio.wait_for(_open(), MCP_CONNECT_TIMEOUT)
            self._stack = stack
            tools = [x.name for x in (await self._session.list_tools()).tools]
            self.active = True
            logger.info("MCP client connected | tools=%s took=%.1f ms", tools, t.ms)
            return True
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.error("MCP client connect FAIL | url=%s err=%s: %s", self.url, type(e).__name__, e)
            try:
                await stack.aclose()
            except BaseException:
                pass
            self._session, self._stack, self.active = None, None, False
            return False

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except BaseException as e:
                logger.info("MCP client close: %s", type(e).__name__)
        self._session, self._stack, self.active = None, None, False

    async def call(self, tool: str, args: dict) -> str:
        for attempt in (1, 2):
            if not self._session:
                if not await self.connect():
                    raise RuntimeError("Failed to connect to MCP server")
            logger.info("MCP call start | tool=%s args=%s", tool, {k: (str(v)[:60]) for k, v in args.items()})
            try:
                with Timer() as t:
                    res = await asyncio.wait_for(self._session.call_tool(tool, args), MCP_CALL_TIMEOUT)
                text = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
                if res.isError:
                    raise RuntimeError(f"tool error: {text[:200]}")
                log_metric("mcp_call", tool=tool, latency_ms=t.ms, result_chars=len(text), attempt=attempt)
                logger.info("MCP call done | tool=%s took=%.1f ms result_chars=%d", tool, t.ms, len(text))
                return text
            except (asyncio.TimeoutError, OSError, ConnectionError) as e:
                logger.error("MCP call fail | tool=%s attempt=%d err=%s: %s", tool, attempt, type(e).__name__, e)
                await self.close()
                if attempt == 2:
                    raise
        raise RuntimeError("unreachable")