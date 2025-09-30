"""Shared reasoning token utilities for vision agents."""

from __future__ import annotations

import re
from typing import Optional

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def extract_think_segments(text: Optional[str]) -> Optional[str]:
    """Extract concatenated reasoning segments enclosed in <think> tags."""
    if not text:
        return None

    matches = _THINK_PATTERN.findall(text)
    cleaned = [segment.strip() for segment in matches if segment and segment.strip()]
    if not cleaned:
        return None
    return "\n".join(cleaned)


def strip_think_segments(text: Optional[str]) -> str:
    """Remove <think>...</think> blocks from text, returning the remainder."""
    if not text:
        return ""
    return _THINK_PATTERN.sub("", text).strip()
