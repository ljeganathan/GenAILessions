"""
logging_config.py — Structured logging setup for the RAG application.

Call `setup_logging()` once at application startup.
"""

import logging
import sys
from typing import Optional

from backend.core.config import get_settings


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger with a clean, readable format."""
    settings = get_settings()
    log_level = (level or settings.log_level).upper()

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=fmt,
        datefmt=date_fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialised at level %s", log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (call after setup_logging)."""
    return logging.getLogger(name)
