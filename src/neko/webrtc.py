"""Helpers for aiortc based WebRTC connections.

Only a very small surface is provided to avoid copy/paste between
scripts.  The goal is to gradually move the detailed logic out of the
entrypoints and into reusable helpers.
"""
from __future__ import annotations

from typing import Iterable, Optional

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection


def build_configuration(stun_url: str, turn_url: Optional[str] = None,
                        turn_user: Optional[str] = None,
                        turn_pass: Optional[str] = None) -> RTCConfiguration:
    """Create an :class:`RTCConfiguration` with optional TURN support."""
    ice_servers: Iterable[RTCIceServer] = [RTCIceServer(stun_url)]
    if turn_url:
        ice_servers.append(RTCIceServer(turn_url, turn_user, turn_pass))
    return RTCConfiguration(iceServers=list(ice_servers))


def create_peer_connection(config: RTCConfiguration) -> RTCPeerConnection:
    """Return a new :class:`RTCPeerConnection` using ``config``."""
    return RTCPeerConnection(configuration=config)


__all__ = ["build_configuration", "create_peer_connection"]
