"""Logging configuration helpers.

This module provides a :func:`setup_logging` function mirroring the common
configuration used across the command line tools.  It supports both text and
JSON output formats and silences noisy third‑party loggers.
"""
from __future__ import annotations

import json
import logging
from typing import Optional


class _JsonFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    fmt: str = "text",
    log_file: Optional[str] = None,
    name: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger.

    Parameters
    ----------
    level:
        Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    fmt:
        ``"text"`` or ``"json"`` output format.
    log_file:
        Optional path to a log file.  If ``None`` only stdout is used.
    name:
        Logger name to return.  If omitted the root logger is returned.
    """

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter: logging.Formatter
    if fmt.lower() == "json":
        formatter = _JsonFormatter()
    else:  # pragma: no cover - formatter is trivial
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s - %(message)s", "%H:%M:%S"
        )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quiet noisy dependencies
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.ERROR)

    return logging.getLogger(name) if name else root


__all__ = ["setup_logging"]
