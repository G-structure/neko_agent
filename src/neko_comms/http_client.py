"""HTTP/WebSocket client for Neko polling and capture.

This module provides the HTTPNekoClient class for simpler
HTTP-based communication with Neko servers, primarily
used for screenshot polling and training data capture.
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Tuple, Dict, Any, Callable
import requests
import websockets
import json
from PIL import Image
import io

from .base import NekoClient
from .types import Action, Frame, ConnectionState
from .frame_source import FrameSource

logger = logging.getLogger(__name__)


class HTTPNekoClient(NekoClient):
    """HTTP/WebSocket client for Neko screenshot polling and capture.

    This client provides a simpler alternative to WebRTC for scenarios
    where real-time video streaming is not required, such as training
    data capture and periodic screenshot monitoring.
    """

    def __init__(self, base_url: Optional[str] = None, ws_url: Optional[str] = None,
                 username: Optional[str] = None, password: Optional[str] = None):
        """Initialize the HTTP Neko client.

        :param base_url: Base HTTP URL for REST API
        :type base_url: Optional[str]
        :param ws_url: Direct WebSocket URL (alternative to REST)
        :type ws_url: Optional[str]
        :param username: Username for REST authentication
        :type username: Optional[str]
        :param password: Password for REST authentication
        :type password: Optional[str]
        """
        super().__init__()
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.ws_url = ws_url or ""
        self.username = username
        self.password = password

        self.session = requests.Session()
        self.token: Optional[str] = None
        self.session_id: Optional[str] = None

        # WebSocket connection
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._chat_listeners: list[Callable[[str, Dict[str, Any]], None]] = []

        # Frame polling
        self._last_frame: Optional[Frame] = None
        self._frame_event = asyncio.Event()
        self._polling_task: Optional[asyncio.Task] = None

        # Frame source wrapper for compatibility with WebRTCNekoClient
        self.frame_source = self._create_frame_source()

    async def connect(self) -> None:
        """Establish connection to the Neko server via HTTP/WebSocket.

        :return: None
        :rtype: None
        :raises ConnectionError: If connection fails.
        """
        try:
            self._connection_state = ConnectionState.CONNECTING

            if self.base_url and self.username and self.password:
                # Use REST authentication
                await self._authenticate_rest()
            elif self.ws_url:
                # Use direct WebSocket URL
                pass
            else:
                raise ConnectionError("Either (base_url, username, password) or ws_url must be provided")

            # Start WebSocket connection
            await self._connect_websocket()

            # Start screenshot polling task
            self._polling_task = asyncio.create_task(self._poll_screenshots())

            self._connection_state = ConnectionState.CONNECTED
            logger.info("HTTP Neko client connected successfully")

        except Exception as e:
            self._connection_state = ConnectionState.FAILED
            logger.error("Failed to connect HTTP client: %s", e)
            raise ConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Close all connections and clean up resources.

        :return: None
        :rtype: None
        """
        self._connection_state = ConnectionState.DISCONNECTED

        # Stop polling task
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        # Stop WebSocket
        self._stop_event.set()
        if self._ws_thread:
            self._ws_thread.join(timeout=5.0)
            self._ws_thread = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        logger.info("HTTP Neko client disconnected")

    def is_connected(self) -> bool:
        """Check if the client is currently connected.

        :return: True if connected, False otherwise.
        :rtype: bool
        """
        return (self._connection_state == ConnectionState.CONNECTED and
                not self._stop_event.is_set())

    async def send_action(self, action: Action) -> None:
        """Send an action to be executed on the remote session.

        Note: HTTP client has limited action support compared to WebRTC.

        :param action: The action to execute
        :type action: Action
        :return: None
        :rtype: None
        :raises ConnectionError: If not connected to server.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Neko server")

        logger.warning("HTTP client has limited action support: %s", action.action)

    async def recv_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """Receive the next frame via HTTP screenshot polling.

        :param timeout: Maximum time to wait for a frame in seconds.
        :type timeout: float
        :return: The next frame, or None if timeout/error.
        :rtype: Optional[Frame]
        """
        try:
            await asyncio.wait_for(self._frame_event.wait(), timeout=timeout)
            self._frame_event.clear()
            return self._last_frame
        except asyncio.TimeoutError:
            return None

    async def wait_for_track(self, timeout: float = 30.0) -> bool:
        """Wait for video track (compatibility method for HTTP client).

        For HTTP client, we just wait for the first screenshot to be captured.

        :param timeout: Maximum time to wait in seconds.
        :type timeout: float
        :return: Always True for HTTP client (no track needed).
        :rtype: bool
        """
        # Wait for first screenshot to be captured
        try:
            await asyncio.wait_for(self._frame_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_frame(self, timeout: float = 20.0) -> Optional[Image.Image]:
        """Wait for the next frame (compatibility method).

        :param timeout: Maximum time to wait in seconds.
        :type timeout: float
        :return: The next frame as PIL Image, or None if timeout.
        :rtype: Optional[Image.Image]
        """
        frame = await self.recv_frame(timeout=timeout)
        return frame.data if frame else None

    async def publish_topic(self, topic: str, data: Dict[str, Any]) -> None:
        """Publish a message to a specific topic.

        :param topic: The topic name to publish to.
        :type topic: str
        :param data: The message data to publish.
        :type data: Dict[str, Any]
        :return: None
        :rtype: None
        """
        if not self.is_connected() or not self._ws:
            raise ConnectionError("Not connected to Neko server")

        await self._ws.send(json.dumps(data))

    async def subscribe_topic(self, topic: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Subscribe to a topic and wait for the next message.

        Note: HTTP client has simplified message handling.

        :param topic: The topic name to subscribe to.
        :type topic: str
        :param timeout: Maximum time to wait for a message in seconds.
        :type timeout: float
        :return: The next message from the topic.
        :rtype: Dict[str, Any]
        :raises asyncio.TimeoutError: If no message received within timeout.
        """
        # Simplified implementation - would need message queuing for full support
        raise NotImplementedError("Topic subscription not fully implemented in HTTP client")

    def get_screenshot(self, jpeg_quality: int = 85) -> Optional[Image.Image]:
        """Get a screenshot via HTTP request.

        :param jpeg_quality: JPEG quality for compression (0-100)
        :type jpeg_quality: int
        :return: Screenshot as PIL Image, or None if failed
        :rtype: Optional[Image.Image]
        """
        if not self.base_url:
            return None

        try:
            url = f"{self.base_url}/api/shot.jpg"
            params = {"quality": jpeg_quality}
            if self.token:
                params["token"] = self.token

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            img = Image.open(io.BytesIO(response.content))
            self._frame_size = (img.width, img.height)

            # Update last frame
            self._last_frame = Frame(
                data=img,
                timestamp=time.time(),
                width=img.width,
                height=img.height
            )
            self._frame_event.set()

            return img

        except Exception as e:
            logger.error("Failed to get screenshot: %s", e)
            return None

    def poll_screenshot(self, fps: float, jpeg_quality: int, callback: Optional[Callable] = None) -> None:
        """Poll screenshots at the specified frame rate.

        This method is designed to run in a separate thread for continuous
        screenshot capture during training data collection.

        :param fps: Frames per second to capture
        :type fps: float
        :param jpeg_quality: JPEG quality for compression
        :type jpeg_quality: int
        :param callback: Optional callback function for each frame
        :type callback: Optional[Callable]
        :return: None
        :rtype: None
        """
        interval = 1.0 / fps
        while not self._stop_event.is_set():
            start_time = time.time()

            img = self.get_screenshot(jpeg_quality)
            if img and callback:
                callback(img)

            # Sleep for remaining interval time
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        """Stop screenshot polling and other background operations."""
        self._stop_event.set()

    async def _poll_screenshots(self) -> None:
        """Background task to poll screenshots at regular intervals."""
        fps = 2.0  # Poll at 2 FPS
        interval = 1.0 / fps

        while self.is_connected():
            try:
                start_time = asyncio.get_event_loop().time()

                # Get screenshot via HTTP
                img = self.get_screenshot(jpeg_quality=85)
                if img:
                    # Update last frame
                    self._last_frame = Frame(
                        data=img,
                        timestamp=asyncio.get_event_loop().time(),
                        width=img.width,
                        height=img.height
                    )
                    self._frame_size = (img.width, img.height)
                    self._frame_event.set()

                # Sleep for remaining interval time
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error("Error polling screenshot: %s", e)
                await asyncio.sleep(interval)

    def _create_frame_source(self) -> FrameSource:
        """Create a frame source wrapper for compatibility with WebRTCNekoClient."""
        class HTTPFrameSource(FrameSource):
            def __init__(self, client):
                self.client = client

            async def start(self, *args):
                pass  # Polling is started automatically in connect()

            async def stop(self):
                pass  # Polling is stopped automatically in disconnect()

            async def get(self):
                if self.client._last_frame:
                    return self.client._last_frame.data
                return None

            async def wait_for_frame(self, timeout: float = 5.0):
                try:
                    await asyncio.wait_for(self.client._frame_event.wait(), timeout=timeout)
                    self.client._frame_event.clear()
                    if self.client._last_frame:
                        return self.client._last_frame.data
                except asyncio.TimeoutError:
                    pass
                return None

        return HTTPFrameSource(self)

    async def _authenticate_rest(self) -> None:
        """Authenticate via REST API and obtain token."""
        if not self.base_url or not self.username or not self.password:
            raise ConnectionError("REST authentication requires base_url, username, and password")

        try:
            # Login request
            login_url = f"{self.base_url}/api/login"
            login_data = {"username": self.username, "password": self.password}

            response = self.session.post(login_url, json=login_data, timeout=10)
            response.raise_for_status()

            data = response.json()
            self.token = data.get("token")
            if not self.token:
                raise ConnectionError("No token received from login")

            # Build WebSocket URL
            self.ws_url = f"{'wss' if self.base_url.startswith('https') else 'ws'}://{self.base_url.split('://', 1)[1]}/api/ws?token={self.token}"

            logger.info("REST authentication successful")

        except Exception as e:
            raise ConnectionError(f"REST authentication failed: {e}")

    async def _connect_websocket(self) -> None:
        """Connect to WebSocket for real-time events."""
        if not self.ws_url:
            raise ConnectionError("No WebSocket URL available")

        try:
            self._ws = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=60,
                max_size=10_000_000
            )

            # Start background message handler
            self._ws_thread = threading.Thread(
                target=self._websocket_handler,
                daemon=True
            )
            self._ws_thread.start()

            logger.info("WebSocket connected")

        except Exception as e:
            raise ConnectionError(f"WebSocket connection failed: {e}")

    def _websocket_handler(self) -> None:
        """Handle WebSocket messages in background thread."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def message_loop():
                while not self._stop_event.is_set() and self._ws:
                    try:
                        # Simple message receiving - process as needed
                        raw_msg = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
                        try:
                            msg = json.loads(raw_msg)
                            self._process_websocket_message(msg)
                        except json.JSONDecodeError:
                            logger.debug("Received non-JSON message: %r", raw_msg)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break
                    except Exception as e:
                        logger.error("WebSocket receive error: %s", e)
                        break

            loop.run_until_complete(message_loop())

        except Exception as e:
            logger.error("WebSocket handler error: %s", e)
        finally:
            self._stop_event.set()

    def _process_websocket_message(self, msg: Dict[str, Any]) -> None:
        """Process incoming WebSocket messages.

        :param msg: Parsed message dictionary
        :type msg: Dict[str, Any]
        :return: None
        :rtype: None
        """
        event = msg.get("event", "")
        payload = msg.get("payload", {})

        # Basic event handling for capture.py integration
        if event.startswith("chat/"):
            # Handle chat messages for capture triggers
            content = payload.get("content", "")
            self._handle_chat_message(content, msg)
        elif event.startswith("system/"):
            # Handle system events
            self._handle_system_message(msg)

    def _handle_chat_message(self, content: str, msg: Dict[str, Any]) -> None:
        """Handle chat messages - can be overridden for capture logic.

        :param content: Message content
        :type content: str
        :param msg: Full message dictionary
        :type msg: Dict[str, Any]
        :return: None
        :rtype: None
        """
        logger.debug("Chat message: %s", content)
        # This can be used by capture.py for /start and /stop commands
        for listener in list(self._chat_listeners):
            try:
                listener(content, msg)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Chat listener failed: %s", exc, exc_info=True)

    def add_chat_listener(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback invoked for each chat message."""

        self._chat_listeners.append(callback)

    def _handle_system_message(self, msg: Dict[str, Any]) -> None:
        """Handle system messages.

        :param msg: Message dictionary
        :type msg: Dict[str, Any]
        :return: None
        :rtype: None
        """
        event = msg.get("event", "")
        payload = msg.get("payload", {})

        if event == "system/init":
            self.session_id = payload.get("session_id")
            logger.info("HTTP session initialized: %s", self.session_id)

    # Additional methods for capture.py integration
    def get_message_queues(self) -> Dict[str, asyncio.Queue]:
        """Get message queues for different event types.

        This is a simplified version for HTTP client - primarily for chat events.

        :return: Dictionary of event type to queue
        :rtype: Dict[str, asyncio.Queue]
        """
        # Simple implementation - could be enhanced with proper broker
        if not hasattr(self, '_queues'):
            self._queues = {
                'chat': asyncio.Queue(),
                'system': asyncio.Queue(),
            }
        return self._queues

    async def send_websocket_message(self, msg: Dict[str, Any]) -> None:
        """Send a message via WebSocket.

        :param msg: Message to send
        :type msg: Dict[str, Any]
        :return: None
        :rtype: None
        :raises ConnectionError: If not connected
        """
        if not self._ws or self._stop_event.is_set():
            raise ConnectionError("WebSocket not connected")

        await self._ws.send(json.dumps(msg))

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information.

        :return: Session information dictionary
        :rtype: Dict[str, Any]
        """
        return {
            "session_id": self.session_id,
            "base_url": self.base_url,
            "ws_url": self.ws_url,
            "connected": self.is_connected(),
            "screen_size": self.frame_size,
        }