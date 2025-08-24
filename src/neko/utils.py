"""Miscellaneous small utilities shared across scripts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def safe_mkdir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return its path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = ["now_iso", "safe_mkdir"]
