"""Logging configuration helpers.

This module contains a small helper that mirrors the setup performed in
multiple entrypoints.  Scripts can import :func:`setup_logging` to apply a
consistent configuration.
"""
from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    log_file:
        Optional path to a log file.  If ``None`` only stdout is used.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        filename=log_file,
    )


__all__ = ["setup_logging"]
