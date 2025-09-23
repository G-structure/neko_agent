"""WebRTC client for full-featured Neko communication.

This module provides the WebRTCNekoClient class that handles
WebRTC peer connections, video frame capture, and real-time
action execution on Neko servers.
"""

import asyncio
import json
import logging
import os
import random
import uuid
from typing import Optional, Tuple, Dict, Any, List, Callable

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    VideoStreamTrack,
    AudioStreamTrack,
    MediaStreamError,
)
from aiortc.sdp import candidate_from_sdp
from PIL import Image
import av

from .base import NekoClient
from .signaler import Signaler
from .actions import ActionExecutor
from .frame_source import FrameSource, WebRTCFrameSource, LiteFrameSource, NoFrameSource
from .types import Action, Frame, ConnectionState, IceCandidate

logger = logging.getLogger(__name__)


class WebRTCNekoClient(NekoClient):
    """Full-featured WebRTC client for Neko server communication.

    This client provides real-time video streaming, action execution,
    and bidirectional messaging via WebRTC and WebSocket connections.
    Supports both inbound video and outbound audio tracks.
    """

    def __init__(self, ws_url: str, ice_servers: Optional[List[str]] = None, **kwargs):
        """Initialize the WebRTC Neko client.

        :param ws_url: WebSocket URL for signaling
        :type ws_url: str
        :param ice_servers: List of ICE server URLs
        :type ice_servers: Optional[List[str]]
        :param kwargs: Additional configuration options
        :type kwargs: Any
        """
        super().__init__()
        self.ws_url = ws_url
        self.signaler: Optional[Signaler] = None
        self.pc: Optional[RTCPeerConnection] = None
        self.action_executor: Optional[ActionExecutor] = None

        # Configuration
        self.ice_servers = ice_servers or []
        self.ice_policy = kwargs.get("ice_policy", "strict")
        self.enable_audio = kwargs.get("enable_audio", True)
        self.rtcp_keepalive = kwargs.get("rtcp_keepalive", False)
        self.request_media = kwargs.get("request_media", True)

        # Session management
        self.session_id: Optional[str] = None
        self.host_id: Optional[str] = None
        self.auto_host = kwargs.get("auto_host", True)

        # Video track and frame handling
        self._video_track: Optional[VideoStreamTrack] = None
        self._audio_track: Optional[AudioStreamTrack] = None
        self._last_frame: Optional[Frame] = None
        self._frame_event = asyncio.Event()
        self._ice_candidates_buffer: List[RTCIceCandidate] = []
        self._remote_description_set = False

        # Frame source management
        self.frame_source: FrameSource = NoFrameSource()
        self.is_lite = kwargs.get("is_lite", False)  # video_lite mode support

        # Background tasks
        self._tasks = set()

        # Custom audio track for outbound audio (yap.py support)
        self._outbound_audio_track: Optional[AudioStreamTrack] = None

    async def connect(self) -> None:
        """Establish WebRTC and WebSocket connections to the Neko server.

        :return: None
        :rtype: None
        :raises ConnectionError: If connection fails after retries.
        """
        try:
            self._connection_state = ConnectionState.CONNECTING

            # Initialize WebSocket signaler
            self.signaler = Signaler(self.ws_url)
            await self.signaler.connect_with_backoff()

            # Initialize action executor
            self.action_executor = ActionExecutor(
                send_func=self.signaler.send,
                chat_func=self._send_chat_safe
            )

            # Set up WebRTC peer connection
            await self._setup_peer_connection()

            # Start background tasks
            self._start_background_tasks()

            # Set up video_lite if needed
            await self.setup_video_lite()

            # Request media if enabled
            if self.request_media:
                await self._request_media()

            self._connection_state = ConnectionState.CONNECTED
            logger.info("WebRTC Neko client connected successfully")

        except Exception as e:
            self._connection_state = ConnectionState.FAILED
            logger.error("Failed to connect WebRTC client: %s", e)
            raise ConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Close all connections and clean up resources.

        :return: None
        :rtype: None
        """
        self._connection_state = ConnectionState.DISCONNECTED

        if self.pc:
            await self.pc.close()
            self.pc = None

        if self.signaler:
            await self.signaler.close()
            self.signaler = None

        self._video_track = None
        self.action_executor = None
        logger.info("WebRTC Neko client disconnected")

    def is_connected(self) -> bool:
        """Check if the client is currently connected.

        :return: True if connected, False otherwise.
        :rtype: bool
        """
        return (self._connection_state == ConnectionState.CONNECTED and
                self.signaler and self.signaler.is_connected() and
                self.pc and self.pc.connectionState == "connected")

    async def send_action(self, action: Action) -> None:
        """Send an action to be executed on the remote session.

        :param action: The action to execute
        :type action: Action
        :return: None
        :rtype: None
        :raises ConnectionError: If not connected to server.
        """
        if not self.is_connected() or not self.action_executor:
            raise ConnectionError("Not connected to Neko server")

        await self.action_executor.execute_action(action, self.frame_size)

    async def recv_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """Receive the next video frame from the WebRTC stream or video_lite.

        :param timeout: Maximum time to wait for a frame in seconds.
        :type timeout: float
        :return: The next frame, or None if timeout/error.
        :rtype: Optional[Frame]
        """
        img = await self.frame_source.wait_for_frame(timeout)
        if img:
            return Frame(
                data=img,
                timestamp=asyncio.get_event_loop().time(),
                width=img.width,
                height=img.height
            )
        return None

    async def publish_topic(self, topic: str, data: Dict[str, Any]) -> None:
        """Publish a message to a specific topic.

        :param topic: The topic name to publish to.
        :type topic: str
        :param data: The message data to publish.
        :type data: Dict[str, Any]
        :return: None
        :rtype: None
        """
        if not self.signaler:
            raise ConnectionError("Not connected to Neko server")

        await self.signaler.send(data)

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
        if not self.signaler:
            raise ConnectionError("Not connected to Neko server")

        queue = self.signaler.broker.topic_queue(topic)
        return await asyncio.wait_for(queue.get(), timeout=timeout)

    async def _setup_peer_connection(self) -> None:
        """Initialize the WebRTC peer connection with ICE servers."""
        ice_servers = [RTCIceServer(urls=url) for url in self.ice_servers]
        config = RTCConfiguration(iceServers=ice_servers)

        if self.ice_policy == "strict":
            config.iceTransportPolicy = "relay"

        self.pc = RTCPeerConnection(config)
        self.pc.on("icecandidate", self._on_ice_candidate)
        self.pc.on("track", self._on_track)

        logger.info("WebRTC peer connection initialized")

    async def _handle_signaling_messages(self) -> None:
        """Background task to handle WebSocket signaling messages."""
        if not self.signaler:
            return

        control_queue = self.signaler.broker.topic_queue("control")
        ice_queue = self.signaler.broker.topic_queue("ice")

        while self.is_connected():
            try:
                # Handle control messages (offers, answers)
                try:
                    msg = await asyncio.wait_for(control_queue.get(), timeout=0.1)
                    await self._handle_control_message(msg)
                except asyncio.TimeoutError:
                    pass

                # Handle ICE candidates
                try:
                    msg = await asyncio.wait_for(ice_queue.get(), timeout=0.1)
                    await self._handle_ice_message(msg)
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error("Error handling signaling message: %s", e)
                break

    async def _handle_control_message(self, msg: Dict[str, Any]) -> None:
        """Handle WebRTC control messages (offers, answers, etc.)."""
        event = msg.get("event", "")
        payload = msg.get("payload", {})

        if event == "signal/offer":
            sdp = payload.get("sdp")
            if sdp and self.pc:
                await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
                self._remote_description_set = True

                # Process any buffered ICE candidates
                for candidate in self._ice_candidates_buffer:
                    await self.pc.addIceCandidate(candidate)
                self._ice_candidates_buffer.clear()

                # Create and send answer
                answer = await self.pc.createAnswer()
                await self.pc.setLocalDescription(answer)

                await self.signaler.send({
                    "event": "signal/answer",
                    "payload": {"sdp": answer.sdp, "type": "answer"}
                })

        elif event == "signal/answer":
            sdp = payload.get("sdp")
            if sdp and self.pc:
                await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))
                self._remote_description_set = True

    async def _handle_ice_message(self, msg: Dict[str, Any]) -> None:
        """Handle ICE candidate messages."""
        payload = msg.get("payload", {})
        candidate_str = payload.get("candidate")

        if candidate_str and self.pc:
            try:
                candidate = candidate_from_sdp(candidate_str)
                candidate.sdpMid = payload.get("sdpMid")
                candidate.sdpMLineIndex = payload.get("sdpMLineIndex")

                if self._remote_description_set:
                    await self.pc.addIceCandidate(candidate)
                else:
                    # Buffer candidates until remote description is set
                    self._ice_candidates_buffer.append(candidate)

            except Exception as e:
                logger.warning("Failed to parse ICE candidate: %s", e)

    async def _on_ice_candidate(self, candidate: Optional[RTCIceCandidate]) -> None:
        """Handle outgoing ICE candidates."""
        if candidate and self.signaler:
            await self.signaler.send({
                "event": "signal/candidate",
                "payload": {
                    "candidate": f"candidate:{candidate.candidate}",
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                }
            })

    def _on_track(self, track) -> None:
        """Handle incoming media tracks."""
        if track.kind == "video":
            self._video_track = track
            # Set up WebRTC frame source
            self.frame_source = WebRTCFrameSource()
            asyncio.create_task(self.frame_source.start(track))
            logger.info("Received video track, started WebRTC frame source")
        elif track.kind == "audio":
            self._audio_track = track
            logger.info("Received audio track")

    async def setup_video_lite(self) -> None:
        """Set up video_lite mode for base64-encoded frames over WebSocket."""
        if self.is_lite:
            self.frame_source = LiteFrameSource()
            await self.frame_source.start()
            # Start video_lite handler
            task = asyncio.create_task(self._handle_video_lite())
            self._tasks.add(task)
            logger.info("Set up video_lite frame source")

    async def _handle_video_lite(self) -> None:
        """Handle video_lite messages with base64-encoded frames."""
        if not self.signaler:
            return

        video_queue = self.signaler.broker.topic_latest("video")

        while self.is_connected():
            try:
                msg = await video_queue.get()
                payload = msg.get("payload", {})

                if "data" in payload:
                    # Extract base64 frame data
                    frame_data = payload["data"]
                    if isinstance(self.frame_source, LiteFrameSource):
                        await self.frame_source.update_frame(frame_data)

                        # Update frame size if provided
                        if "width" in payload and "height" in payload:
                            self._frame_size = (payload["width"], payload["height"])

            except Exception as e:
                logger.error("Error handling video_lite frame: %s", e)
                await asyncio.sleep(0.1)

    async def _send_chat_safe(self, text: str) -> None:
        """Safely send a chat message."""
        try:
            if self.signaler:
                await self.signaler.send({
                    "event": "chat/message",
                    "payload": {"content": text}
                })
        except Exception as e:
            logger.warning("Failed to send chat message: %s", e)

    def _start_background_tasks(self) -> None:
        """Start all background tasks for the WebRTC client."""
        if not self.signaler:
            return

        # Signaling message handlers
        task = asyncio.create_task(self._handle_signaling_messages())
        self._tasks.add(task)

        # System event handler (for session management)
        task = asyncio.create_task(self._handle_system_events())
        self._tasks.add(task)

        # Chat event handler
        task = asyncio.create_task(self._handle_chat_events())
        self._tasks.add(task)

    async def _request_media(self) -> None:
        """Request media (video/audio) from the server."""
        if self.signaler:
            await self.signaler.send({"event": "signal/request"})

    async def _handle_system_events(self) -> None:
        """Handle system events like session updates and host control."""
        if not self.signaler:
            return

        system_queue = self.signaler.broker.topic_queue("system")

        while self.is_connected():
            try:
                msg = await asyncio.wait_for(system_queue.get(), timeout=1.0)
                await self._process_system_event(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error handling system event: %s", e)
                break

    async def _handle_chat_events(self) -> None:
        """Handle chat events for online mode and task management."""
        if not self.signaler:
            return

        chat_queue = self.signaler.broker.topic_queue("chat")

        while self.is_connected():
            try:
                msg = await asyncio.wait_for(chat_queue.get(), timeout=1.0)
                await self._process_chat_event(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error handling chat event: %s", e)
                break

    async def _process_system_event(self, msg: Dict[str, Any]) -> None:
        """Process individual system events."""
        event = msg.get("event", "")
        payload = msg.get("payload", {})

        if event == "system/init":
            # Session initialization
            self.session_id = payload.get("session_id")
            self.host_id = payload.get("host_id")
            logger.info("Session initialized: %s", self.session_id)

        elif event == "control/host":
            # Host control status
            if self.auto_host and not payload.get("has_host"):
                await self.request_host_control()

        elif event == "session/created":
            # New session created
            session_id = payload.get("id")
            if session_id:
                self.session_id = session_id
                logger.info("New session created: %s", session_id)

    async def _process_chat_event(self, msg: Dict[str, Any]) -> None:
        """Process chat events - can be overridden for custom chat handling."""
        event = msg.get("event", "")
        payload = msg.get("payload", {})

        if event == "chat/message":
            content = payload.get("content", "")
            member_id = payload.get("member_id")
            logger.debug("Chat message from %s: %s", member_id, content)

    async def request_host_control(self) -> None:
        """Request host control from the server."""
        if self.signaler:
            await self.signaler.send({"event": "control/request"})
            logger.info("Requested host control")

    async def send_heartbeat(self) -> None:
        """Send heartbeat to maintain connection."""
        if self.signaler:
            await self.signaler.send({"event": "system/heartbeat"})

    def add_outbound_audio_track(self, audio_track: AudioStreamTrack) -> None:
        """Add an outbound audio track (for yap.py TTS support).

        :param audio_track: The audio track to add
        :type audio_track: AudioStreamTrack
        :return: None
        :rtype: None
        """
        if self.pc:
            self._outbound_audio_track = audio_track
            self.pc.addTrack(audio_track)
            logger.info("Added outbound audio track")

    def get_video_track(self) -> Optional[VideoStreamTrack]:
        """Get the current video track.

        :return: The video track if available
        :rtype: Optional[VideoStreamTrack]
        """
        return self._video_track

    def get_audio_track(self) -> Optional[AudioStreamTrack]:
        """Get the current audio track.

        :return: The audio track if available
        :rtype: Optional[AudioStreamTrack]
        """
        return self._audio_track

    async def wait_for_frame(self, timeout: float = 5.0) -> Optional[Frame]:
        """Wait for the next frame with a timeout.

        :param timeout: Maximum time to wait in seconds
        :type timeout: float
        :return: The next frame or None if timeout
        :rtype: Optional[Frame]
        """
        return await self.recv_frame(timeout)

    async def wait_for_track(self, timeout: float = 10.0) -> bool:
        """Wait for a video track to be established.

        :param timeout: Maximum time to wait in seconds
        :type timeout: float
        :return: True if track is available, False if timeout
        :rtype: bool
        """
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self._video_track:
                return True
            await asyncio.sleep(0.1)
        return False

    async def create_offer(self) -> None:
        """Create and send a WebRTC offer."""
        if not self.pc or not self.signaler:
            return

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        await self.signaler.send({
            "event": "signal/offer",
            "payload": {"sdp": offer.sdp, "type": "offer"}
        })
        logger.info("WebRTC offer sent")

    async def handle_keyboard_map(self, keymap: Dict[str, int]) -> None:
        """Handle keyboard mapping updates from server.

        :param keymap: Keyboard mapping dictionary
        :type keymap: Dict[str, int]
        :return: None
        :rtype: None
        """
        from .types import KEYSYM
        KEYSYM.update(keymap)
        logger.info("Updated keyboard mapping with %d keys", len(keymap))

    # Coordinate conversion utilities for manual.py support
    def normalize_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """Convert pixel coordinates to normalized (0-1) coordinates.

        :param x: X pixel coordinate
        :type x: int
        :param y: Y pixel coordinate
        :type y: int
        :return: Normalized coordinates as (x, y)
        :rtype: Tuple[float, float]
        """
        w, h = self.frame_size
        if w > 0 and h > 0:
            return x / w, y / h
        return 0.0, 0.0

    def pixel_coordinates(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates.

        :param norm_x: Normalized X coordinate (0-1)
        :type norm_x: float
        :param norm_y: Normalized Y coordinate (0-1)
        :type norm_y: float
        :return: Pixel coordinates as (x, y)
        :rtype: Tuple[int, int]
        """
        w, h = self.frame_size
        return int(norm_x * w), int(norm_y * h)

    def set_screen_size(self, width: int, height: int) -> None:
        """Set the known screen size (for manual.py support).

        :param width: Screen width in pixels
        :type width: int
        :param height: Screen height in pixels
        :type height: int
        :return: None
        :rtype: None
        """
        self._frame_size = (width, height)
        logger.info("Screen size set to %dx%d", width, height)