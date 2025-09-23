"""Message broker for routing WebSocket events to topic-based queues.

This module provides the Broker class for distributing incoming messages
to appropriate topic queues based on event types. It supports both queueing
and latest-only delivery patterns for different types of messages.
"""

import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class LatestOnly:
    """A simple async container that holds only the latest value set.

    This class provides a mechanism to await new values while discarding
    older values. Only the most recent value is kept, making it useful
    for scenarios where only the latest data is relevant.
    """

    def __init__(self):
        """Initialize the LatestOnly container with no value set."""
        self._val = None
        self._event = asyncio.Event()

    def set(self, v):
        """Set a new value and notify any waiting consumers.

        :param v: The new value to store.
        :type v: Any
        :return: None
        :rtype: None
        """
        self._val = v
        self._event.set()

    async def get(self):
        """Wait for and return the latest value, then clear the event.

        :return: The most recently set value.
        :rtype: Any
        """
        await self._event.wait()
        self._event.clear()
        return self._val


class Broker:
    """Message broker for routing WebSocket events to topic-based queues.

    This class manages the distribution of incoming messages to appropriate
    topic queues based on event types. It supports both queueing and latest-only
    delivery patterns for different types of messages.
    """

    def __init__(self):
        """Initialize the broker with empty topic collections."""
        self.queues: Dict[str, asyncio.Queue] = {}
        self.latest: Dict[str, LatestOnly] = {}
        self.waiters: Dict[str, asyncio.Future] = {}

    def topic_queue(self, topic: str, maxsize: int = 512) -> asyncio.Queue:
        """Get or create a queue for the specified topic.

        :param topic: The topic name for the queue.
        :type topic: str
        :param maxsize: Maximum queue size. Defaults to 512.
        :type maxsize: int
        :return: The asyncio Queue for the topic.
        :rtype: asyncio.Queue
        """
        if topic not in self.queues:
            self.queues[topic] = asyncio.Queue(maxsize=maxsize)
        return self.queues[topic]

    def topic_latest(self, topic: str) -> LatestOnly:
        """Get or create a LatestOnly container for the specified topic.

        :param topic: The topic name for the container.
        :type topic: str
        :return: The LatestOnly container for the topic.
        :rtype: LatestOnly
        """
        if topic not in self.latest:
            self.latest[topic] = LatestOnly()
        return self.latest[topic]

    def publish(self, msg: Dict[str, Any]) -> None:
        """Route an incoming message to the appropriate topic queue or container.

        Messages are routed based on their 'event' field prefix. Special handling
        is provided for reply messages, signal events, system events, chat events,
        and send channel events.

        :param msg: The message dictionary to route, expected to have an 'event' key.
        :type msg: Dict[str, Any]
        :return: None
        :rtype: None
        """
        ev = msg.get("event", "")

        # Handle RPC-style replies first
        if (rid := msg.get("reply_to")) and (fut := self.waiters.pop(rid, None)):
            if not fut.done():
                fut.set_result(msg)
            return

        # Route based on event prefix
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
        elif ev.startswith("chat/"):
            self.topic_queue("chat").put_nowait(msg)
        elif ev.startswith("send/"):
            # Treat opaque send channel messages as chat-like for task intake.
            self.topic_queue("chat").put_nowait(msg)
        else:
            self.topic_queue("misc").put_nowait(msg)

    async def wait_for_reply(self, request_id: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Wait for a reply message with the specified request ID.

        This method is used for RPC-style request-response patterns where
        a message is sent with a request ID and a response is expected.

        :param request_id: The unique request identifier to wait for.
        :type request_id: str
        :param timeout: Maximum time to wait for the reply in seconds.
        :type timeout: float
        :return: The reply message dictionary.
        :rtype: Dict[str, Any]
        :raises asyncio.TimeoutError: If no reply is received within the timeout.
        """
        future = asyncio.Future()
        self.waiters[request_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.waiters.pop(request_id, None)
            raise