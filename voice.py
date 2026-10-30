import os
import re
import asyncio
import httpx
from dotenv import load_dotenv
from groq import AsyncGroq

from logger import logger, log_metric, Timer

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

SLOW_STT_MS = float(os.getenv("SLOW_STT_MS", "2000"))
SLOW_LLM_MS = float(os.getenv("SLOW_LLM_MS", "2000"))
SLOW_TTS_MS = float(os.getenv("SLOW_TTS_MS", "3000"))

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

http_client = httpx.AsyncClient(
    timeout=60,
    http2=True,
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
    )
)

LANGUAGES = {
    "hi-IN": "Hindi", "en-IN": "English", "bn-IN": "Bengali",
    "gu-IN": "Gujarati", "kn-IN": "Kannada", "ml-IN": "Malayalam",
    "mr-IN": "Marathi", "od-IN": "Odia", "pa-IN": "Punjabi",
    "ta-IN": "Tamil", "te-IN": "Telugu",
}
DEFAULT_LANG = "hi-IN"


def system_prompt(lang_code: str) -> str:
    lang = LANGUAGES.get(lang_code, "Hindi")
    return f"- ALWAYS reply in {lang}. 2 short sentences. Maximum 15 words.\n"


def _warn_if_slow(stage: str, ms: float, limit: float) -> None:
    if ms > limit:
        logger.warning("%s slow: %.1f ms (limit %.0f ms)", stage.upper(), ms, limit)


# ---------------------------------------------------------------- Cost Helpers
def calculate_stt_cost(audio_bytes: bytes) -> float:
    # Rate: ₹0.49 per minute (~₹0.00816 per second)
    audio_kb = len(audio_bytes) / 1024
    duration_sec = audio_kb / 16.0  # Approx estimation for webm/opus
    rate_per_sec = 0.49 / 60.0
    return round(duration_sec * rate_per_sec, 6)


def calculate_tts_cost(text: str) -> float:
    # Rate: ₹3.00 per 1000 characters for Bulbul V3
    rate_per_char = 3.00 / 1000.0
    return round(len(text) * rate_per_char, 6)


# ---------------------------------------------------------------- STT
async def speech_to_text(audio_bytes: bytes, filename: str = "audio.webm",
                         content_type: str = "audio/webm") -> tuple[str, str, dict]:
    raw_content_type = content_type
    content_type = (content_type or "audio/webm").split(";")[0].strip().lower()
    if content_type in ("", "application/octet-stream"):
        content_type = "audio/webm"

    headers = {"api-subscription-key": SARVAM_API_KEY}
    files = {"file": (filename, audio_bytes, content_type)}
    data = {"model": STT_MODEL, "language_code": "unknown"}

    logger.info("STT start | file=%s type=%s size=%.1f KB", filename, content_type, len(audio_bytes) / 1024)

    with Timer() as t:
        r = await http_client.post(SARVAM_STT_URL, headers=headers, files=files, data=data)

    if r.status_code != 200:
        log_metric("stt", status="error", http_status=r.status_code, latency_ms=t.ms, error=r.text[:300])
        raise RuntimeError(f"Sarvam STT error {r.status_code}: {r.text}")

    j = r.json()
    transcript = (j.get("transcript") or "").strip()
    detected = j.get("language_code") or DEFAULT_LANG
    lang = detected if detected in LANGUAGES else DEFAULT_LANG

    stt_cost = calculate_stt_cost(audio_bytes)

    meta = {
        "status": "ok",
        "latency_ms": t.ms,
        "model": STT_MODEL,
        "provider": "sarvam",
        "http_status": r.status_code,
        "sarvam_request_id": j.get("request_id"),
        "audio_kb": round(len(audio_bytes) / 1024, 1),
        "audio_type": content_type,
        "audio_type_raw": raw_content_type,
        "detected_language": detected,
        "language_used": lang,
        "language_fallback": detected != lang,
        "language_probability": j.get("language_probability"),
        "transcript_chars": len(transcript),
        "transcript_words": len(transcript.split()),
        "stt_cost_inr": stt_cost,
    }
    log_metric("stt", **meta)
    _warn_if_slow("stt", t.ms, SLOW_STT_MS)
    return transcript, lang, meta


# ---------------------------------------------------------------- LLM
async def ask_llm(history: list[dict], user_text: str, lang_code: str) -> tuple[str, dict]:
    messages = [{"role": "system", "content": system_prompt(lang_code)}]
    messages += history[-4:]
    messages.append({"role": "user", "content": user_text})

    logger.info("LLM start | model=%s history_msgs=%d", GROQ_MODEL, len(history[-4:]))

    with Timer() as t:
        resp = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=160,
        )

    reply = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)

    meta = {
        "status": "ok",
        "latency_ms": t.ms,
        "model": GROQ_MODEL,
        "provider": "groq",
        "groq_request_id": getattr(resp, "id", None),
        "finish_reason": resp.choices[0].finish_reason,
        "history_msgs_used": len(history[-4:]),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None),
        "groq_queue_time_s": getattr(usage, "queue_time", None),
        "groq_total_time_s": getattr(usage, "total_time", None),
        "tokens_per_sec": round(completion_tokens / (t.ms / 1000), 1) if completion_tokens and t.ms else None,
        "reply_chars": len(reply),
        "reply_words": len(reply.split()),
    }
    log_metric("llm", **meta)
    _warn_if_slow("llm", t.ms, SLOW_LLM_MS)
    return reply, meta


# ---------------------------------------------------------------- TTS
def split_text(text: str, limit: int = 400) -> list[str]:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= limit:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s[:limit]
    if cur:
        chunks.append(cur)
    return chunks


async def text_to_speech(text: str, lang_code: str) -> tuple[list[str], dict]:
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    chunks = split_text(text)

    logger.info("TTS start | model=%s speaker=%s lang=%s chunks=%d chars=%d",
                TTS_MODEL, TTS_SPEAKER, lang_code, len(chunks), len(text))

    async def fetch_chunk(chunk: str):
        payload = {
            "text": chunk,
            "target_language_code": lang_code,
            "speaker": TTS_SPEAKER,
            "model": TTS_MODEL,
            "pace": 1.1,
            "speech_sample_rate": 16000,
        }
        with Timer() as ct:
            r = await http_client.post(SARVAM_TTS_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Sarvam TTS error {r.status_code}: {r.text}")
        return r.json(), ct.ms

    with Timer() as total:
        tasks = [fetch_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

    audios: list[str] = []
    chunk_latencies: list[float] = []
    sarvam_ids: list[str] = []

    for j, latency in results:
        audios.extend(j.get("audios", []))
        chunk_latencies.append(latency)
        if j.get("request_id"):
            sarvam_ids.append(j["request_id"])

    audio_b64_kb = round(sum(len(a) for a in audios) / 1024, 1)
    tts_cost = calculate_tts_cost(text)

    meta = {
        "status": "ok",
        "latency_ms": total.ms,
        "chunk_latencies_ms": chunk_latencies,
        "model": TTS_MODEL,
        "provider": "sarvam",
        "speaker": TTS_SPEAKER,
        "language": lang_code,
        "sample_rate": 16000,
        "chunks": len(chunks),
        "input_chars": len(text),
        "audio_clips": len(audios),
        "audio_b64_kb": audio_b64_kb,
        "sarvam_request_ids": sarvam_ids,
        "tts_cost_inr": tts_cost,
    }
    log_metric("tts", **meta)
    _warn_if_slow("tts", total.ms, SLOW_TTS_MS)
    return audios, meta


# ---------------------------------------------------------------- Pipeline
async def voice_pipeline(audio_bytes: bytes, history: list[dict],
                         filename: str, content_type: str) -> dict:
    with Timer() as total:
        transcript, lang, stt_meta = await speech_to_text(audio_bytes, filename, content_type)

        if not transcript:
            logger.warning("Empty transcript - skipped LLM/TTS")
            return {
                "transcript": "", "reply": "", "language": lang, "audios": [],
                "metadata": {"status": "empty_transcript", "stt": stt_meta, "total_ms": total.elapsed_ms},
            }

        reply, llm_meta = await ask_llm(history, transcript, lang)
        audios, tts_meta = await text_to_speech(reply, lang)

        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": reply})

    stt_ms, llm_ms, tts_ms = stt_meta["latency_ms"], llm_meta["latency_ms"], tts_meta["latency_ms"]
    summary = {
        "status": "ok",
        "language": lang,
        "stt_ms": stt_ms,
        "llm_ms": llm_ms,
        "tts_ms": tts_ms,
        "total_ms": total.ms,
        "overhead_ms": round(total.ms - stt_ms - llm_ms - tts_ms, 1),
        "history_len": len(history),
    }
    log_metric("pipeline", **summary)
    logger.info("LATENCY SUMMARY | STT=%.0f ms | LLM=%.0f ms | TTS=%.0f ms | TOTAL=%.0f ms",
                stt_ms, llm_ms, tts_ms, total.ms)

    return {
        "transcript": transcript,
        "reply": reply,
        "language": lang,
        "audios": audios,
        "metadata": {**summary, "stt": stt_meta, "llm": llm_meta, "tts": tts_meta},
    }