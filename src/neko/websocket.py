"""Asynchronous WebSocket client utilities.

The implementation here is intentionally lightweight.  It exposes a
:class:`NekoWebSocketClient` class that follows the connection pattern used
throughout the existing scripts.  Future work can expand it with
reconnection logic and message routing.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import websockets


@dataclass
class NekoWebSocketClient:
    """Minimal wrapper around :mod:`websockets` for the Neko API."""

    uri: str
    token: Optional[str] = None

    async def connect(self) -> websockets.WebSocketClientProtocol:
        """Open the WebSocket connection."""
        headers = {}
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        self._ws = await websockets.connect(self.uri, extra_headers=headers)
        return self._ws

    async def send(self, message: str) -> None:
        """Send a text message to the server."""
        await self._ws.send(message)

    async def __aiter__(self) -> AsyncIterator[str]:
        """Yield incoming messages from the server."""
        async for msg in self._ws:
            yield msg

    async def close(self) -> None:
        """Close the WebSocket connection."""
        await self._ws.close()


__all__ = ["NekoWebSocketClient"]
