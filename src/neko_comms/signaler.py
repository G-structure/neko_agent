"""WebSocket client for handling signaling and messaging with automatic reconnection.

This module provides the Signaler class that manages a WebSocket connection
with built-in reconnection logic, message routing through a Broker, and
separate read/send loops for handling bidirectional communication.
"""

import asyncio
import json
import logging
import random
import websockets
from websockets.protocol import State
from typing import Set, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from .broker import Broker

logger = logging.getLogger(__name__)


class Signaler:
    """WebSocket client for handling signaling and messaging with automatic reconnection.

    This class manages a WebSocket connection with built-in reconnection logic,
    message routing through a Broker, and separate read/send loops for handling
    bidirectional communication.
    """

    def __init__(self, url: str, **wsopts):
        """Initialize the WebSocket signaler.

        :param url: WebSocket URL to connect to
        :type url: str
        :param wsopts: Additional WebSocket connection options
        :type wsopts: Any
        """
        self.url = url
        self.wsopts = dict(
            ping_interval=30,
            ping_timeout=60,
            max_queue=1024,
            max_size=10_000_000,
            **wsopts
        )
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._tasks: Set[asyncio.Task] = set()
        self._sendq: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.broker = Broker()
        self._closed = asyncio.Event()

    async def connect_with_backoff(self):
        """Establish WebSocket connection with exponential backoff retry.

        This method attempts to connect to the WebSocket URL with increasing
        delays between retry attempts. Once connected, it starts the read and
        send loops as background tasks.

        :return: Self for method chaining.
        :rtype: Signaler
        """
        backoff = 1
        while not self._closed.is_set():
            try:
                self.ws = await websockets.connect(self.url, **self.wsopts)
                try:
                    parsed = urlparse(self.url)
                    qs = parse_qs(parsed.query)
                    if "token" in qs:
                        qs["token"] = ["***"]
                    safe_q = urlencode(qs, doseq=True)
                    safe_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, safe_q, parsed.fragment))
                except Exception:
                    safe_url = self.url
                logger.info("WebSocket connected: %s", safe_url)
                self._tasks.add(asyncio.create_task(self._read_loop(), name="ws-read"))
                self._tasks.add(asyncio.create_task(self._send_loop(), name="ws-send"))
                self._closed.clear()
                return self
            except Exception as e:
                jitter = random.uniform(0, max(0.25, backoff * 0.25))
                delay = min(backoff + jitter, 30)
                logger.error("WS connect error: %s - retrying in %.2fs", e, delay)
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 30)

    async def close(self):
        """Close the WebSocket connection and clean up resources.

        This method cancels all running tasks, closes the WebSocket connection,
        and waits for proper cleanup.

        :return: None
        :rtype: None
        """
        self._closed.set()
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._tasks.clear()
        if self.ws:
            try:
                await self.ws.close()
                if hasattr(self.ws, "wait_closed"):
                    await self.ws.wait_closed()
            except Exception:
                pass
            finally:
                self.ws = None

    async def send(self, msg: dict) -> None:
        """Queue a message for sending over the WebSocket.

        This method places a message in the send queue to be transmitted
        by the background send loop.

        :param msg: Message dictionary to send
        :type msg: dict
        :return: None
        :rtype: None
        """
        try:
            self._sendq.put_nowait(msg)
        except asyncio.QueueFull:
            logger.warning("Send queue full, dropping message: %r", msg)

    async def _read_loop(self):
        """Background task that reads messages from WebSocket and routes them.

        This coroutine continuously reads messages from the WebSocket connection
        and forwards them to the broker for routing to appropriate topic queues.
        It handles connection closure and cancellation gracefully.

        :return: None
        :rtype: None
        """
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    logger.debug("Received non-JSON: %r", raw)
                    continue
                self.broker.publish(msg)
        except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK) as e:
            logger.warning("WS closed: %s", e)
        except asyncio.CancelledError:
            logger.info("WS read loop cancelled")
        finally:
            self._closed.set()

    async def _send_loop(self):
        """Background task that sends queued messages to the WebSocket.

        This coroutine continuously processes messages from the send queue and
        transmits them over the WebSocket connection. It handles connection
        closure and cancellation gracefully.

        :return: None
        :rtype: None
        """
        try:
            while not self._closed.is_set():
                try:
                    msg = await asyncio.wait_for(self._sendq.get(), timeout=1.0)
                    if self.ws and self._ws_is_open():
                        await self.ws.send(json.dumps(msg, ensure_ascii=False))
                    self._sendq.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error("Send error: %s", e)
                    break
        except asyncio.CancelledError:
            logger.info("WS send loop cancelled")
        finally:
            self._closed.set()

    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active.

        :return: True if connected, False otherwise.
        :rtype: bool
        """
        return self.ws is not None and self._ws_is_open() and not self._closed.is_set()

    def _ws_is_open(self) -> bool:
        """Return True when the underlying WebSocket connection is open."""
        if not self.ws:
            return False

        closed = getattr(self.ws, "closed", None)
        if closed is not None:
            return not closed

        state = getattr(self.ws, "state", None)
        if state is not None:
            return state == State.OPEN

        return True

