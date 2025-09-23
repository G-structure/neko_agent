"""Base client interface for Neko communication.

This module defines the abstract base class for all Neko clients,
providing a unified interface for connecting to and communicating
with Neko servers using different protocols.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import asyncio

from .types import Action, Frame, ConnectionState


class NekoClient(ABC):
    """Abstract base class for Neko server communication clients.

    This class defines the common interface that all Neko clients must implement,
    whether they use WebRTC, HTTP, or other communication protocols.
    """

    def __init__(self):
        """Initialize the base client."""
        self._connection_state = ConnectionState.DISCONNECTED
        self._frame_size: Tuple[int, int] = (0, 0)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the Neko server.

        :return: None
        :rtype: None
        :raises ConnectionError: If connection fails after retries.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the Neko server and clean up resources.

        :return: None
        :rtype: None
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the client is currently connected to the server.

        :return: True if connected, False otherwise.
        :rtype: bool
        """
        pass

    @abstractmethod
    async def send_action(self, action: Action) -> None:
        """Send an action to be executed on the remote session.

        :param action: The action to execute (click, type, scroll, etc.)
        :type action: Action
        :return: None
        :rtype: None
        :raises ConnectionError: If not connected to server.
        """
        pass

    @abstractmethod
    async def recv_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """Receive the next video frame from the remote session.

        :param timeout: Maximum time to wait for a frame in seconds.
        :type timeout: float
        :return: The next frame, or None if timeout/error.
        :rtype: Optional[Frame]
        """
        pass

    @abstractmethod
    async def publish_topic(self, topic: str, data: Dict[str, Any]) -> None:
        """Publish a message to a specific topic.

        :param topic: The topic name to publish to.
        :type topic: str
        :param data: The message data to publish.
        :type data: Dict[str, Any]
        :return: None
        :rtype: None
        """
        pass

    @abstractmethod
    async def subscribe_topic(self, topic: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Subscribe to a topic and wait for the next message.

        :param topic: The topic name to subscribe to.
        :type topic: str
        :param timeout: Maximum time to wait for a message in seconds.
        :type timeout: float
        :return: The next message from the topic.
        :rtype: Dict[str, Any]
        :raises asyncio.TimeoutError: If no message received within timeout.
        """
        pass

    # Common properties and utilities
    @property
    def connection_state(self) -> ConnectionState:
        """Get the current connection state.

        :return: The current connection state.
        :rtype: ConnectionState
        """
        return self._connection_state

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get the current frame dimensions.

        :return: Frame dimensions as (width, height).
        :rtype: Tuple[int, int]
        """
        return self._frame_size

    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """Wait for the client to become connected.

        :param timeout: Maximum time to wait in seconds.
        :type timeout: float
        :return: True if connected within timeout, False otherwise.
        :rtype: bool
        """
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self.is_connected():
                return True
            await asyncio.sleep(0.1)
        return False

    def __str__(self) -> str:
        """String representation of the client.

        :return: String describing the client state.
        :rtype: str
        """
        return f"{self.__class__.__name__}({self.connection_state.value})"

    def __repr__(self) -> str:
        """Detailed representation of the client.

        :return: Detailed string representation.
        :rtype: str
        """
        return f"{self.__class__.__name__}(state={self.connection_state.value}, frame_size={self.frame_size})"