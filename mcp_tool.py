import asyncio
import hashlib
import os
import re
from datetime import datetime

import httpx
from dotenv import load_dotenv

import rag
from logger import logger, log_metric, Timer

load_dotenv()

SERPAPI_URL = "https://serpapi.com/search.json"
SERPAPI_TIMEOUT = float(os.getenv("SERPAPI_TIMEOUT", "5"))
SERPAPI_FILE = "serpapi.md"
SERPAPI_HEADER = (
    "# Web Answers (SerpAPI)\n\n"
    "This file is dynamically populated. When user queries are not resolved within local "
    "healthcare and medical documentation, external product/medical information is fetched and saved.\n"
)

NO_WEB_RESULT = "NO_WEB_RESULT"
_http = httpx.AsyncClient(timeout=SERPAPI_TIMEOUT)
_bg: set = set()


def _api_key() -> str:
    return os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY", "")


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", q.lower())).strip()


async def serpapi_search(query: str, lang_code: str = "en-IN") -> dict:
    key = _api_key()
    if not key:
        logger.error("SERPAPI | SERPAPI_API_KEY missing - web search skipped")
        return {"found": False, "answer": "", "evidence": "", "sources": [], "latency_ms": 0.0}

    hl = lang_code.split("-")[0] if lang_code else "en"
    params = {"engine": "google", "q": query, "api_key": key, "num": 5, "hl": hl, "gl": "in"}
    logger.info("SERPAPI start | q=%r hl=%s", query, hl)

    with Timer() as t:
        try:
            r = await _http.get(SERPAPI_URL, params=params)
        except httpx.HTTPError as e:
            logger.error("SERPAPI network error | %s: %s", type(e).__name__, e)
            log_metric("serpapi", status="error", latency_ms=t.elapsed_ms, error=str(e)[:200])
            return {"found": False, "answer": "", "evidence": "", "sources": [], "latency_ms": t.elapsed_ms}

    if r.status_code != 200:
        logger.error("SERPAPI http %s | %s", r.status_code, r.text[:200])
        log_metric("serpapi", status="error", http_status=r.status_code, latency_ms=t.ms)
        return {"found": False, "answer": "", "evidence": "", "sources": [], "latency_ms": t.ms}

    j = r.json()
    answer = ""
    ab = j.get("answer_box") or {}
    for field in ("answer", "result", "snippet"):
        if ab.get(field):
            answer = str(ab[field])
            break
    if not answer:
        answer = (j.get("knowledge_graph") or {}).get("description", "")

    lines, sources = [], []
    for item in (j.get("organic_results") or [])[:4]:
        snip = (item.get("snippet") or "").strip()
        if snip:
            lines.append(f"- {item.get('title', '')}: {snip}")
        if item.get("link"):
            sources.append(item["link"])
    if not answer and lines:
        answer = lines[0][2:]
    evidence = ("; ".join([answer] if answer else []) + "\n" + "\n".join(lines)).strip()[:1500]

    found = bool(evidence)
    meta = {"status": "ok", "found": found, "latency_ms": t.ms, "organic": len(j.get("organic_results") or []),
            "has_answer_box": bool(ab), "evidence_chars": len(evidence)}
    log_metric("serpapi", **meta)
    logger.info("SERPAPI done | found=%s organic=%d answer_box=%s took=%.1f ms | answer=%r",
                found, meta["organic"], meta["has_answer_box"], t.ms, answer[:100])
    return {"found": found, "answer": answer, "evidence": evidence, "sources": sources[:3], "latency_ms": t.ms}


def _save_sync(question: str, search_query: str, answer: str, evidence: str, sources: list[str]) -> int:
    key = hashlib.sha1(_norm(question).encode("utf-8")).hexdigest()[:12]
    entry = (
        f"## Q: {question.strip()}\n\n"
        f"- Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"- Search query: {search_query}\n"
        f"- Source: web (SerpAPI) general medical/product data\n"
        f"- Links: {', '.join(sources) if sources else 'n/a'}\n\n"
        f"Answer: {answer.strip()}\n\n"
        f"Details: {evidence.strip()}\n"
    )
    return rag.append_markdown_entry(SERPAPI_FILE, entry, key, SERPAPI_HEADER)


async def save_web_answer(question: str, search_query: str, answer: str, evidence: str, sources: list[str]) -> int:
    with Timer() as t:
        n = await asyncio.to_thread(_save_sync, question, search_query, answer, evidence, sources)
    logger.info("SERPAPI save | file=%s new_chunks=%d took=%.1f ms", SERPAPI_FILE, n, t.ms)
    log_metric("serpapi_save", new_chunks=n, latency_ms=t.ms)
    return n


# ------------------------------------------------------------------ MCP TOOLS
async def kb_search(query: str, top_k: int = 3) -> str:
    """Healthcare & Medical E-Commerce knowledge base semantic search."""
    logger.info("TOOL kb_search | query=%r top_k=%d", query, top_k)
    hits = await asyncio.to_thread(rag.search, query, max(1, min(top_k, 8)))
    if not hits:
        return "No results found. The medical knowledge base might be empty. Run kb_reindex."
    return "\n\n".join(
        f"[{i}] score={h['score']} | {h['source']} | {h['section']}\n{h['text']}" for i, h in enumerate(hits, 1)
    )


async def kb_list_sources() -> str:
    """List indexed files in vector database."""
    logger.info("TOOL kb_list_sources")
    s = await asyncio.to_thread(rag.stats)
    lines = [f"data_dir={s['data_dir']} | total_chunks={s['total_chunks']} | embed_model={s['embed_model']}"]
    for name, m in s["files"].items():
        lines.append(f"- {name} ({m.get('type')}) chunks={m.get('chunks')} indexed_at={m.get('indexed_at')}")
    return "\n".join(lines)


async def kb_reindex(force: bool = False) -> str:
    """Reindex updated or new Markdown/PDF files from data folder."""
    logger.info("TOOL kb_reindex | force=%s", force)
    s = await asyncio.to_thread(rag.sync_data_dir, force)
    return (f"indexed={s['indexed']} skipped={s['skipped']} removed={s['removed']} "
            f"total_chunks={s['total_chunks']} took_ms={s['sync_ms']}")


async def web_search_save(question: str, search_query: str = "") -> str:
    """Web search fallback for general external medical specifications."""
    q = (search_query or question).strip()
    logger.info("TOOL web_search_save | question=%r search_query=%r", question, q)
    res = await serpapi_search(q)
    if not res["found"]:
        return f"{NO_WEB_RESULT}: No search results obtained."
    task = asyncio.create_task(
        save_web_answer(question, q, res["answer"] or res["evidence"][:300], res["evidence"], res["sources"]))
    _bg.add(task)
    task.add_done_callback(lambda t: (_bg.discard(t), t.exception() and logger.error("BG save fail: %r", t.exception())))
    return (f"Answer: {res['answer']}\n\nDetails:\n{res['evidence']}\n\nSources: {', '.join(res['sources'])}\n"
            f"(saving to {SERPAPI_FILE} in background)")