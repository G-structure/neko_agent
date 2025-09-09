"""Miscellaneous small utilities shared across scripts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os


def now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def safe_mkdir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return its path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_bool(name: str, default: bool) -> bool:
    """Parse an environment variable as a boolean.

    Values like ``1``, ``true``, ``yes`` and ``on`` are treated as ``True`` while
    ``0``/``false``/``no``/``off`` map to ``False``.  If the variable is not set
    the ``default`` value is returned.
    """

    val = os.getenv(name)
    if val is None:
        return default
    try:
        return bool(int(val))
    except ValueError:
        return val.strip().lower() in {"true", "t", "yes", "y", "on"}


__all__ = ["now_iso", "safe_mkdir", "env_bool"]
