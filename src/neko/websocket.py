"""Shared WebSocket helpers.

This module centralizes the async WebSocket patterns that were previously
repeated across multiple scripts.  It provides a small message broker for
fan-out and a :class:`Signaler` client that manages the connection with
background read/write tasks and exponential backoff reconnects.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
from typing import Any, Dict, Optional, Set

import websockets


class LatestOnly:
    """Container that always returns the most recently set value."""

    def __init__(self) -> None:
        self._val: Any = None
        self._event = asyncio.Event()

    def set(self, value: Any) -> None:
        """Store ``value`` and notify waiters."""
        self._val = value
        self._event.set()

    async def get(self) -> Any:
        """Wait for a value to be set and return it."""
        await self._event.wait()
        self._event.clear()
        return self._val


class Broker:
    """Topic based fan-out for incoming WebSocket messages."""

    def __init__(self) -> None:
        self.queues: Dict[str, asyncio.Queue] = {}
        self.latest: Dict[str, LatestOnly] = {}
        self.waiters: Dict[str, asyncio.Future] = {}

    def topic_queue(self, topic: str, maxsize: int = 512) -> asyncio.Queue:
        if topic not in self.queues:
            self.queues[topic] = asyncio.Queue(maxsize=maxsize)
        return self.queues[topic]

    def topic_latest(self, topic: str) -> LatestOnly:
        if topic not in self.latest:
            self.latest[topic] = LatestOnly()
        return self.latest[topic]

    def publish(self, msg: Dict[str, Any]) -> None:
        ev = msg.get("event", "")
        if (rid := msg.get("reply_to")) and (fut := self.waiters.pop(rid, None)):
            if not fut.done():
                fut.set_result(msg)
            return
        if ev.startswith("signal/"):
            if ev == "signal/video":
                self.topic_latest("video").set(msg)
            elif ev == "signal/candidate":
                self.topic_queue("ice").put_nowait(msg)
            elif ev in {"signal/offer", "signal/provide", "signal/answer", "signal/close"}:
                self.topic_queue("control").put_nowait(msg)
            else:
                self.topic_queue("signal").put_nowait(msg)
        elif ev.startswith(("system/", "control/", "screen/", "keyboard/", "session/", "error/")):
            self.topic_queue("system").put_nowait(msg)
        elif ev.startswith("chat/") or ev.startswith("send/"):
            self.topic_queue("chat").put_nowait(msg)
        else:
            self.topic_queue("misc").put_nowait(msg)


class Signaler:
    """WebSocket client with reconnect and message fan-out."""

    def __init__(self, url: str, **wsopts: Any) -> None:
        self.url = url
        self.wsopts = dict(
            ping_interval=30,
            ping_timeout=60,
            max_queue=1024,
            max_size=10_000_000,
            **wsopts,
        )
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.broker = Broker()
        self._sendq: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._tasks: Set[asyncio.Task] = set()
        self._closed = asyncio.Event()

    async def _cleanup_tasks(self) -> None:
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def connect_with_backoff(self) -> "Signaler":
        backoff = 1
        log = logging.getLogger(__name__)
        while not self._closed.is_set():
            try:
                await self._cleanup_tasks()
                self.ws = await websockets.connect(self.url, **self.wsopts)
                log.info("WebSocket connected to %s", self.url)
                self._closed.clear()
                self._tasks.add(asyncio.create_task(self._read_loop(), name="ws-read"))
                self._tasks.add(asyncio.create_task(self._send_loop(), name="ws-send"))
                return self
            except Exception as e:  # pragma: no cover - network errors
                jitter = random.uniform(0, max(0.25, backoff * 0.25))
                delay = min(backoff + jitter, 30)
                log.error("WS connect error: %s - retrying in %.2fs", e, delay)
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 30)
        return self

    async def close(self) -> None:
        self._closed.set()
        await self._cleanup_tasks()
        if self.ws:
            with contextlib.suppress(Exception):
                await self.ws.close()
                if hasattr(self.ws, "wait_closed"):
                    await self.ws.wait_closed()
            self.ws = None

    async def _read_loop(self) -> None:
        log = logging.getLogger(__name__)
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    log.debug("Received non-JSON: %r", raw)
                    continue
                self.broker.publish(msg)
        except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK) as e:
            log.warning("WS closed: %s", e)
        except asyncio.CancelledError:
            log.info("WS read loop cancelled")
            raise
        finally:
            self._closed.set()

    async def _send_loop(self) -> None:
        log = logging.getLogger(__name__)
        try:
            while not self._closed.is_set():
                msg = await self._sendq.get()
                try:
                    await self.ws.send(json.dumps(msg))
                except websockets.ConnectionClosed:
                    log.warning("Send failed: WS closed")
                    break
        except asyncio.CancelledError:
            log.info("WS send loop cancelled")
            raise
        finally:
            self._closed.set()

    async def send(self, msg: Dict[str, Any]) -> None:
        await self._sendq.put(msg)


__all__ = ["LatestOnly", "Broker", "Signaler"]
