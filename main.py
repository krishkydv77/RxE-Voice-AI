import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

import agent
import rag
from logger import logger, log_metric, request_id_var, LOG_DIR

BASE_DIR = Path(__file__).resolve().parent
MAX_SESSIONS = 200
MAX_AUDIO_MB = 10

_sessions: "OrderedDict[str, list[dict]]" = OrderedDict()


def get_history(session_id: str) -> list[dict]:
    if session_id in _sessions:
        _sessions.move_to_end(session_id)
    else:
        _sessions[session_id] = []
        logger.info("SESSION new | id=%s active=%d", session_id, len(_sessions))
        while len(_sessions) > MAX_SESSIONS:
            old, _ = _sessions.popitem(last=False)
            logger.info("SESSION evicted | id=%s", old)
    return _sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    request_id_var.set("startup")
    logger.info("=" * 70)
    logger.info("APP start | log_dir=%s data_dir=%s", LOG_DIR.resolve(), rag.DATA_DIR)
    for key in ("GROQ_API_KEY", "SARVAM_API_KEY", "SERPAPI_API_KEY"):
        if not (os.getenv(key) or (key == "SERPAPI_API_KEY" and os.getenv("SERPAPI_KEY"))):
            logger.error("ENV missing | %s is not set in .env", key)
    logger.info("CONFIG | llm=%s stt=%s tts=%s speaker=%s embed=%s chunk=%d/%d top_k=%d core_target=%.0f ms",
                agent.voice.GROQ_MODEL, agent.voice.STT_MODEL, agent.voice.TTS_MODEL, agent.voice.TTS_SPEAKER,
                rag.EMBED_MODEL, rag.CHUNK_SIZE, rag.CHUNK_OVERLAP, rag.TOP_K, agent.CORE_TARGET_MS)

    if agent.USE_MCP and await agent.mcp.connect():
        logger.info("TOOLS mode = MCP server (%s)", agent.mcp._session and agent.mcp.url)
    else:
        if agent.USE_MCP:
            logger.error("MCP server not found -> switching to in-process mode")
        logger.info("TOOLS mode = in-process")
        rag.sync_data_dir()
    await agent.warmup()
    logger.info("APP ready")
    yield
    logger.info("APP shutdown")
    await agent.mcp.close()
    await agent.voice.http_client.aclose()


app = FastAPI(title="Healthcare & Medical E-Commerce Voice Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.middleware("http")
async def request_logger(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    request_id_var.set(rid)
    start = time.perf_counter()
    logger.info("REQUEST start | %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("REQUEST crashed | %s %s", request.method, request.url.path)
        raise
    ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    if request.url.path.startswith("/api"):
        log_metric("request", method=request.method, path=request.url.path, status=response.status_code, total_ms=ms)
    logger.info("REQUEST end | %s %s -> %d in %.1f ms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
async def health():
    if agent.mcp.active:
        return {"status": "ok", "tools": "mcp", "sources": await agent.mcp.call("kb_list_sources", {}),
                "sessions": len(_sessions)}
    s = rag.stats()
    return {"status": "ok", "tools": "in-process", "chunks": s["total_chunks"],
            "files": list(s["files"].keys()), "sessions": len(_sessions)}


@app.post("/api/reindex")
async def reindex(force: bool = False):
    import asyncio
    logger.info("API reindex | force=%s", force)
    if agent.mcp.active:
        return {"result": await agent.mcp.call("kb_reindex", {"force": force})}
    return await asyncio.to_thread(rag.sync_data_dir, force)


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    existed = _sessions.pop(session_id, None) is not None
    logger.info("SESSION cleared | id=%s existed=%s", session_id, existed)
    return {"cleared": existed}


@app.post("/api/voice")
async def voice_endpoint(audio: UploadFile = File(...), session_id: str = Form("default")):
    data = await audio.read()
    logger.info("API voice | session=%s file=%s type=%s size=%.1f KB",
                session_id, audio.filename, audio.content_type, len(data) / 1024)
    if not data:
        raise HTTPException(400, "Empty audio")
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(413, f"Audio file must be smaller than {MAX_AUDIO_MB} MB")

    try:
        result = await agent.run_voice(data, get_history(session_id),
                                       audio.filename or "audio.webm", audio.content_type or "audio/webm")
        
        # (Optional) Response aane ke baad yahan metadata log dekh sakte hain
        meta = result.get("metadata", {})
        logger.info("API voice completed | path=%s total_cost=₹%.4f total_ms=%.1f", 
                    result.get("path"), meta.get("total_cost_inr", 0.0), meta.get("total_ms", 0.0))

    except Exception as e:
        logger.exception("API voice failed")
        return JSONResponse({"error": str(e)}, status_code=502)
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")),
                reload=False, log_level="warning")