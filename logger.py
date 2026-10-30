import json
import logging
import os
import time
import contextvars
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def _build_logger(name: str = "healthcare_voicebot") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(LOG_LEVEL)
    log.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(request_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    flt = _RequestIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(flt)

    file_h = RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_h.setFormatter(fmt)
    file_h.addFilter(flt)

    log.addHandler(console)
    log.addHandler(file_h)
    return log


logger = _build_logger()


class Timer:
    def __init__(self) -> None:
        self.ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = round((time.perf_counter() - self._start) * 1000, 1)

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 1)


def log_metric(stage: str, **fields) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "request_id": request_id_var.get(),
        "stage": stage,
        **fields,
    }
    logger.info("[%s] %s", stage.upper(), json.dumps(fields, ensure_ascii=False))
    try:
        with open(LOG_DIR / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("Failed to write metrics.jsonl: %s", e)