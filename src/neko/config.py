"""Environment driven configuration helpers.

The goal of this module is to centralize environment variable parsing
for the different entrypoint scripts.  Only a small subset of settings
are represented for now; additional fields can be added as the
monolithic scripts are refactored.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class Settings:
    """Basic runtime settings for the Neko tools."""

    neko_ws: Optional[str] = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Construct settings from environment variables.

        Only a limited subset of the eventual configuration surface is
        implemented so that existing scripts can progressively migrate to
        this module.
        """

        return cls(
            neko_ws=os.getenv("NEKO_WS"),
            log_level=os.getenv("NEKO_LOGLEVEL", "INFO"),
        )


__all__ = ["Settings"]
