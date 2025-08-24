"""Neko core package.

This package hosts reusable components shared across the various
command line tools in ``src``.  The modules currently provide only
lightweight placeholders so that future refactors can import from a
consistent location.
"""

__all__ = [
    "config",
    "logging",
    "websocket",
    "webrtc",
    "utils",
]
