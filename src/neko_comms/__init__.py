"""Neko Communication Library

A modular library for communicating with Neko servers via WebRTC and HTTP.
Provides unified interfaces for WebSocket signaling, action execution,
and frame capture.

Main exports:
    WebRTCNekoClient: Full WebRTC client for real-time control
    HTTPNekoClient: HTTP/WebSocket client for polling/capture
    Broker: Message routing for WebSocket events
    Signaler: WebSocket client with auto-reconnect
"""

from .base import NekoClient
from .webrtc_client import WebRTCNekoClient
from .http_client import HTTPNekoClient
from .broker import Broker
from .signaler import Signaler
from .types import Action, Frame, IceCandidate
from .actions import ActionExecutor, safe_parse_action
from .auth import rest_login_and_ws_url
from .audio import PCMQueue, YAPAudioTrack
from .frame_source import FrameSource, WebRTCFrameSource, LiteFrameSource, NoFrameSource

__all__ = [
    "NekoClient",
    "WebRTCNekoClient",
    "HTTPNekoClient",
    "Broker",
    "Signaler",
    "Action",
    "Frame",
    "IceCandidate",
    "ActionExecutor",
    "safe_parse_action",
    "rest_login_and_ws_url",
    "PCMQueue",
    "YAPAudioTrack",
    "FrameSource",
    "WebRTCFrameSource",
    "LiteFrameSource",
    "NoFrameSource",
]