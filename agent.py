import asyncio
import os
import re

import mcp_tool
import rag
import voice
from mcp_client import MCPToolClient
from logger import logger, log_metric, Timer

WORD_LIMIT = int(os.getenv("REPLY_MAX_WORDS", "15"))
CORE_TARGET_MS = float(os.getenv("CORE_TARGET_MS", "2000"))  # STT + LLM + TTS target
SUPPORT_HELPLINE = os.getenv("SUPPORT_HELPLINE", "1800-123-4567")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "160"))
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")

USE_MCP = os.getenv("USE_MCP", "1") == "1"   # 1 = MCP server mode; 0 = in-process mode
mcp = MCPToolClient()
SCORE_RE = re.compile(r"score=([0-9.]+)")
SRC_RE = re.compile(r"^\[\d+\] score=[0-9.]+ \| (.+?) \|", re.M)

SEARCH_RE = re.compile(r"^\s*SEARCH\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)

FALLBACK = {
    "hi-IN": "क्षमा करें, मुझे इसकी सटीक चिकित्सा जानकारी नहीं मिली। कृपया हमारे हेल्थकेयर सपोर्ट से संपर्क करें।",
    "en-IN": "Sorry, I couldn't find precise medical details regarding that. Please reach out to healthcare support.",
}

_bg_tasks: set[asyncio.Task] = set()


# ------------------------------------------------------------------ prompts
def _base_rules(lang: str) -> str:
    return (
        f"AI Voice Assistant for Healthcare E-Commerce (speech output). Reply in the question's language (Hindi in Devanagari script; others in native script; ignore guessed: {lang}).\n"
        f"- Max {WORD_LIMIT} words, 1-2 short sentences. Plain text only (no markdown, emojis, or symbols).\n"
        "- Do not guess medical conditions or prescriptions without data. Maintain compliance.\n"
        f"- For emergencies, severe symptoms, or critical billing, prompt user to call support ({SUPPORT_HELPLINE}).\n"
        "- Small talk: reply politely and offer help with medicines, lab tests, devices, or orders.\n"
        "- Unrelated queries: state in one short sentence you only assist with healthcare/medical queries."
    )


def kb_system_prompt(lang: str) -> str:
    return _base_rules(lang) + (
        "- Use ONLY the CONTEXT for medical product, prescription, return, delivery, or health order facts.\n"
        "- If CONTEXT does not answer the question, output exactly one line: "
        "SEARCH: <short English web search query> and nothing else (this is the only case where you do not use the user's language)."
    )


def web_system_prompt(lang: str) -> str:
    return _base_rules(lang) + (
        "- Use ONLY the WEB RESULTS for medical facts. Mention briefly that this is general medical information. "
        "Never output SEARCH."
    )


# ------------------------------------------------------------------ helpers
def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_`#>~|]+", " ", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)  # emoji
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_SCRIPTS = [(0x0900, 0x097F, "hi-IN"), (0x0980, 0x09FF, "bn-IN"), (0x0A00, 0x0A7F, "pa-IN"),
            (0x0A80, 0x0AFF, "gu-IN"), (0x0B00, 0x0B7F, "od-IN"), (0x0B80, 0x0BFF, "ta-IN"),
            (0x0C00, 0x0C7F, "te-IN"), (0x0C80, 0x0CFF, "kn-IN"), (0x0D00, 0x0D7F, "ml-IN")]


def detect_reply_lang(text: str, stt_lang: str) -> str:
    latin, counts = 0, {}
    for ch in text:
        if ch.isascii() and ch.isalpha():
            latin += 1
            continue
        o = ord(ch)
        for lo, hi, code in _SCRIPTS:
            if lo <= o <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts and not latin:
        return stt_lang
    code, n = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
    if latin >= n:
        return "en-IN"
    if code == "hi-IN" and stt_lang == "mr-IN":
        return "mr-IN"
    return code


def calculate_groq_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    # Rates per 1M tokens converted to INR (~₹83 per USD)
    if "8b" in model.lower() or "instant" in model.lower():
        input_usd, output_usd = 0.05, 0.08
    else:
        input_usd, output_usd = 0.15, 0.60
    cost_usd = ((prompt_tokens / 1_000_000) * input_usd) + ((completion_tokens / 1_000_000) * output_usd)
    return round(cost_usd * 83.0, 6)


async def _chat(stage: str, messages: list[dict]) -> tuple[str, dict]:
    logger.info("LLM start | stage=%s model=%s msgs=%d", stage, voice.GROQ_MODEL, len(messages))
    kwargs = dict(model=voice.GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=LLM_MAX_TOKENS)
    if "gpt-oss" in voice.GROQ_MODEL.lower():
        kwargs["max_tokens"] = max(LLM_MAX_TOKENS, 500)
        kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}
        logger.info("LLM reasoning model | effort=%s max_tokens=%d", REASONING_EFFORT, kwargs["max_tokens"])
    with Timer() as t:
        resp = await voice.groq_client.chat.completions.create(**kwargs)
    reply = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    meta = {
        "step": stage, "latency_ms": t.ms, "model": voice.GROQ_MODEL,
        "finish_reason": resp.choices[0].finish_reason,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reply_words": len(reply.split()),
    }
    log_metric("llm", **meta)
    logger.info("LLM done | stage=%s took=%.1f ms reply=%r", stage, t.ms, reply[:120])
    voice._warn_if_slow("llm", t.ms, voice.SLOW_LLM_MS)
    if meta["finish_reason"] == "length":
        logger.warning("LLM max_tokens limit reached (finish_reason=length)")
    return reply, meta


def _spawn(coro, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _bg_tasks.add(task)

    def _done(tk: asyncio.Task) -> None:
        _bg_tasks.discard(tk)
        if not tk.cancelled() and tk.exception():
            logger.error("BG task fail | %s: %r", name, tk.exception())

    task.add_done_callback(_done)


# ------------------------------------------------------------------ pipeline
async def run_voice(audio_bytes: bytes, history: list[dict], filename: str, content_type: str) -> dict:
    with Timer() as total:
        # 1) STT
        transcript, lang, stt_meta = await voice.speech_to_text(audio_bytes, filename, content_type)
        logger.info("PIPELINE transcript=%r lang=%s", transcript, lang)
        if not transcript:
            logger.warning("PIPELINE empty transcript - skipping steps")
            return {"transcript": "", "reply": "", "language": lang, "audios": [], "path": "empty",
                    "metadata": {"status": "empty_transcript", "stt": stt_meta, "total_ms": total.elapsed_ms}}

        lang_name = voice.LANGUAGES.get(lang, "Hindi")
        recent = history[-4:]

        # 2) Retrieval: MCP tool kb_search (or in-process fallback)
        via_mcp = mcp.active
        with Timer() as t_ret:
            if via_mcp:
                try:
                    kb_text = await mcp.call("kb_search", {"query": transcript, "top_k": rag.TOP_K})
                except Exception as e:
                    logger.error("PIPELINE MCP kb_search fail: %s: %s", type(e).__name__, e)
                    kb_text = ""
                context = kb_text[:1800]
                scores = SCORE_RE.findall(kb_text)
                top_score = float(scores[0]) if scores else None
                sources = sorted(set(SRC_RE.findall(kb_text)))
            else:
                hits = await asyncio.to_thread(rag.search, transcript, rag.TOP_K)
                context = rag.format_context(hits)
                top_score = hits[0]["score"] if hits else None
                sources = sorted({h["source"] for h in hits})
        logger.info("PIPELINE retrieval done | via=%s top_score=%s took=%.1f ms context_chars=%d",
                    "MCP kb_search" if via_mcp else "in-process", top_score, t_ret.ms, len(context))

        # 3) LLM (with Healthcare KB context)
        msgs = [{"role": "system", "content": kb_system_prompt(lang_name)}, *recent,
                {"role": "user", "content": f"CONTEXT:\n{context or '(empty)'}\n\nQUESTION: {transcript}"}]
        reply, llm1 = await _chat("kb", msgs)
        llm_ms, web_ms, path = llm1["latency_ms"], 0.0, "kb"
        fallback_text = FALLBACK["hi-IN"] if lang == "hi-IN" else FALLBACK["en-IN"]

        llm2 = None

        # 4) Fallback to Web Search if KB has no relevant answer
        m = SEARCH_RE.match(reply)
        if not reply:
            logger.warning("PIPELINE LLM empty reply - fallback text used")
            path, reply = "llm_empty", fallback_text
        elif m:
            search_q = m.group(1).strip().splitlines()[0]
            path = "web"
            logger.info("PIPELINE KB answer missing -> web search | query=%r", search_q)
            if via_mcp:
                with Timer() as tw:
                    try:
                        txt = await mcp.call("web_search_save", {"question": transcript, "search_query": search_q})
                    except Exception as e:
                        logger.error("PIPELINE MCP web_search_save fail: %s: %s", type(e).__name__, e)
                        txt = mcp_tool.NO_WEB_RESULT
                head, _, tail = txt.partition("\n\nSources:")
                web = {"found": not txt.startswith(mcp_tool.NO_WEB_RESULT), "evidence": head,
                       "sources": [x.strip() for x in tail.splitlines()[0].split(",") if x.strip()] if tail else [],
                       "latency_ms": tw.ms}
            else:
                web = await mcp_tool.serpapi_search(search_q, lang)
            web_ms = web["latency_ms"]
            if web["found"]:
                msgs2 = [{"role": "system", "content": web_system_prompt(lang_name)}, *recent,
                         {"role": "user", "content": f"WEB RESULTS:\n{web['evidence']}\n\nQUESTION: {transcript}"}]
                reply, llm2 = await _chat("web", msgs2)
                llm_ms += llm2["latency_ms"]
                sources = web["sources"]
                if not reply or SEARCH_RE.match(reply):
                    logger.warning("PIPELINE web LLM reply invalid - fallback text used")
                    path, reply = "web_llm_failed", fallback_text
                elif not via_mcp:
                    _spawn(mcp_tool.save_web_answer(transcript, search_q, reply, web["evidence"], web["sources"]),
                           "save_serpapi")
            else:
                path, reply, sources = "web_failed", fallback_text, []

        reply = clean_for_speech(reply)
        tts_lang = detect_reply_lang(reply, lang)
        logger.info("PIPELINE reply language | stt=%s tts=%s", lang, tts_lang)

        # 5) TTS
        audios, tts_meta = await voice.text_to_speech(reply, tts_lang)

        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": reply})

    stt_ms, tts_ms = stt_meta["latency_ms"], tts_meta["latency_ms"]
    core_ms = round(stt_ms + llm_ms + tts_ms, 1)

    # Costs and Tokens breakdown calculation
    stt_cost = stt_meta.get("stt_cost_inr", 0.0)
    tts_cost = tts_meta.get("tts_cost_inr", 0.0)

    prompt_tokens = (llm1.get("prompt_tokens") or 0) + (llm2.get("prompt_tokens") if llm2 else 0)
    completion_tokens = (llm1.get("completion_tokens") or 0) + (llm2.get("completion_tokens") if llm2 else 0)
    total_llm_tokens = prompt_tokens + completion_tokens
    llm_cost = calculate_groq_cost(prompt_tokens, completion_tokens, voice.GROQ_MODEL)

    web_cost = 0.40 if path == "web" else 0.0
    vector_db_cost = 0.0
    other_infra_cost = 0.05

    total_cost_inr = round(stt_cost + tts_cost + llm_cost + web_cost + vector_db_cost + other_infra_cost, 6)

    summary = {
        "status": "ok", "path": path, "language": lang, "top_score": top_score, "via_mcp": via_mcp,
        "stt_cost_inr": stt_cost,
        "llm_cost_inr": llm_cost,
        "tts_cost_inr": tts_cost,
        "web_cost_inr": web_cost,
        "vector_db_cost_inr": vector_db_cost,
        "other_infra_cost_inr": other_infra_cost,
        "total_cost_inr": total_cost_inr,
        "stt_ms": stt_ms, "retrieval_ms": t_ret.ms, "llm_ms": round(llm_ms, 1), "web_ms": web_ms,
        "tts_ms": tts_ms, "core_ms": core_ms, "total_ms": total.ms,
        "overhead_ms": round(total.ms - stt_ms - llm_ms - tts_ms - t_ret.ms - web_ms, 1),
    }
    log_metric("pipeline", **summary)
    
    verdict = "OK" if core_ms <= CORE_TARGET_MS else "OVER TARGET"
    log = logger.info if core_ms <= CORE_TARGET_MS else logger.warning
    log("LATENCY SUMMARY | tools=%s path=%s | STT=%.0f + LLM=%.0f + TTS=%.0f = CORE %.0f ms (target %.0f) [%s] | "
        "retrieval=%.0f web=%.0f | TOTAL=%.0f ms | TOTAL COST = ₹%.4f",
        "MCP" if via_mcp else "in-process", path, stt_ms, llm_ms, tts_ms, core_ms, CORE_TARGET_MS, verdict, t_ret.ms, web_ms, total.ms, total_cost_inr)

    # Detailed Token & Cost breakdown log for console
    logger.info("DETAILED METRICS | STT [Cost: ₹%.4f] | LLM [In: %d | Out: %d | Total: %d | Cost: ₹%.4f] | TTS [Cost: ₹%.4f] | Web [Cost: ₹%.4f] | VectorDB [Cost: ₹%.4f] | OTHER [Cost: ₹%.4f]",
                stt_cost, prompt_tokens, completion_tokens, total_llm_tokens, llm_cost, tts_cost, web_cost, vector_db_cost, other_infra_cost)

    return {"transcript": transcript, "reply": reply, "language": lang, "audios": audios,
            "path": path, "sources": sources, "metadata": summary}


async def warmup() -> None:
    logger.info("WARMUP start")
    with Timer() as t:
        if mcp.active:
            try:
                await mcp.call("kb_search", {"query": "medicine order status", "top_k": 1})
            except Exception as e:
                logger.warning("WARMUP mcp fail: %s", e)
        else:
            await asyncio.to_thread(rag.search, "medicine order status", 1)
        try:
            await voice.groq_client.chat.completions.create(
                model=voice.GROQ_MODEL, messages=[{"role": "user", "content": "hello"}], max_tokens=1)
        except Exception as e:
            logger.warning("WARMUP groq fail: %s", e)
        try:
            await voice.http_client.get("https://api.sarvam.ai")
        except Exception as e:
            logger.info("WARMUP sarvam connect: %s", type(e).__name__)
    logger.info("WARMUP done | took=%.1f ms", t.ms)