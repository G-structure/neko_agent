#!/usr/bin/env python3
"""
neko_manual_cli.py — Production-Ready Manual Control REPL for a Neko v3 Server.

This script provides a command-line interface for manually interacting with a
Neko v3 instance. It correctly implements the WebRTC signaling handshake
(via aiortc) required by many modern Neko servers before control events are
accepted, but it does not process or render the media stream.

Hardenings:
- Silences verbose DEBUG logs from underlying libraries (websockets, aiortc).
- Handles all 12 critical protocol fixes for full Neko v3 compatibility.
- Dynamically learns keyboard layouts from the server via 'keyboard/map'.
- Responds to server heartbeats correctly to prevent disconnection.
- Intelligently re-requests host control if it is lost or given to another user.
- Manages asynchronous tasks safely to prevent resource leaks.
- Real reconnection: if the WebSocket drops, rebuild the connection and tasks.
- Strict ICE mapping (only urls/username/credential are used).
- Sequential double-click to preserve ordering/timing.
- Routes and prints `error/...` events.
- Accepts `keyboard/map` payload in both shapes (flat dict or {"map": {...}}).

Features:
- Robust REST login to automatically acquire a WebSocket token.
- Complete WebRTC signaling (SDP offer/answer + ICE candidate exchange).
- REPL for a wide range of commands: move, click, scroll, text input, key presses.
- Live, color-coded logger for important server events.
- Optional normalized (0..1) or pixel coordinates.
"""

import os
import sys
import json
import shlex
import asyncio
import logging
import argparse
import contextlib
from typing import Dict, Any, Tuple, Optional, Set

import requests
from neko.logging import setup_logging
from neko.websocket import Signaler

# aiortc for SDP/ICE handshake (no media consumption)
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
)
from aiortc.sdp import candidate_from_sdp

# ---
# Configure Logging
# ---
logger = setup_logging(
    os.environ.get("NEKO_LOGLEVEL", "INFO"),
    os.environ.get("NEKO_LOG_FORMAT", "text"),
    os.environ.get("NEKO_LOGFILE"),
    name="neko_manual",
)


# ---
# Utilities
# ---

BUTTON_CODES = {"left": 1, "middle": 2, "right": 3}

KEYSYM = {
    # This default map is augmented by the server's `keyboard/map` event.
    'space': 0x020, 'exclam': 0x021, 'quotedbl': 0x022, 'numbersign': 0x023,
    'dollar': 0x024, 'percent': 0x025, 'ampersand': 0x026, 'apostrophe': 0x027,
    'parenleft': 0x028, 'parenright': 0x029, 'asterisk': 0x02a, 'plus': 0x02b,
    'comma': 0x02c, 'minus': 0x02d, 'period': 0x02e, 'slash': 0x02f,
    '0': 0x030, '1': 0x031, '2': 0x032, '3': 0x033, '4': 0x034, '5': 0x035,
    '6': 0x036, '7': 0x037, '8': 0x038, '9': 0x039, 'colon': 0x03a,
    'semicolon': 0x03b, 'less': 0x03c, 'equal': 0x03d, 'greater': 0x03e,
    'question': 0x03f, 'at': 0x040,
    'A': 0x041, 'B': 0x042, 'C': 0x043, 'D': 0x044, 'E': 0x045, 'F': 0x046,
    'G': 0x047, 'H': 0x048, 'I': 0x049, 'J': 0x04a, 'K': 0x04b, 'L': 0x04c,
    'M': 0x04d, 'N': 0x04e, 'O': 0x04f, 'P': 0x050, 'Q': 0x051, 'R': 0x052,
    'S': 0x053, 'T': 0x054, 'U': 0x055, 'V': 0x056, 'W': 0x057, 'X': 0x058,
    'Y': 0x059, 'Z': 0x05a, 'bracketleft': 0x05b, 'backslash': 0x05c,
    'bracketright': 0x05d, 'asciicircum': 0x05e, 'underscore': 0x05f, 'grave': 0x060,
    'a': 0x061, 'b': 0x062, 'c': 0x063, 'd': 0x064, 'e': 0x065, 'f': 0x066,
    'g': 0x067, 'h': 0x068, 'i': 0x069, 'j': 0x06a, 'k': 0x06b, 'l': 0x06c,
    'm': 0x06d, 'n': 0x06e, 'o': 0x06f, 'p': 0x070, 'q': 0x071, 'r': 0x072,
    's': 0x073, 't': 0x074, 'u': 0x075, 'v': 0x076, 'w': 0x077, 'x': 0x078,
    'y': 0x079, 'z': 0x07a, 'braceleft': 0x07b, 'bar': 0x07c, 'braceright': 0x07d,
    'asciitilde': 0x07e,
    'Backspace': 0xff08, 'Tab': 0xff09, 'Return': 0xff0d, 'Enter': 0xff0d,
    'Escape': 0xff1b, 'Delete': 0xffff, 'Home': 0xff50, 'End': 0xff57,
    'PageUp': 0xff55, 'Prior': 0xff55, 'PageDown': 0xff56, 'Next': 0xff56,
    'Left': 0xff51, 'Up': 0xff52, 'Right': 0xff53, 'Down': 0xff54,
    'Insert': 0xff63,
    'F1': 0xffbe, 'F2': 0xffbf, 'F3': 0xffc0, 'F4': 0xffc1, 'F5': 0xffc2,
    'F6': 0xffc3, 'F7': 0xffc4, 'F8': 0xffc5, 'F9': 0xffc6, 'F10': 0xffc7,
    'F11': 0xffc8, 'F12': 0xffc9,
    'Shift_L': 0xffe1, 'Shift': 0xffe1, 'Shift_R': 0xffe2, 'Control_L': 0xffe3,
    'Control': 0xffe3, 'Ctrl': 0xffe3, 'Control_R': 0xffe4, 'Caps_Lock': 0xffe5,
    'Alt_L': 0xffe9, 'Alt': 0xffe9, 'Alt_R': 0xffea, 'Super_L': 0xffeb,
    'Super': 0xffeb, 'Meta': 0xffeb, 'Super_R': 0xffec,
}

def parse_size(s: str) -> Tuple[int, int]:
    """Parse a WxH size string.

    :param s: Size string like "1280x720".
    :type s: str
    :raises ValueError: If the string is not of the form WxH.
    :return: (width, height)
    :rtype: Tuple[int, int]
    """
    if "x" in s.lower():
        w, h = s.lower().split("x", 1)
        return int(w), int(h)
    raise ValueError("size must look like 1280x720")

def ws_from_rest_login(neko_url: str, username: str, password: str, *, timeout: float = 10.0) -> Tuple[str, str, str]:
    """Perform REST login and derive a WS URL with token.

    :param neko_url: Base https://... for the Neko server.
    :type neko_url: str
    :param username: Account username.
    :type username: str
    :param password: Account password.
    :type password: str
    :param timeout: HTTP timeout in seconds.
    :type timeout: float
    :return: (ws_url, base_url, token) tuple
    :rtype: Tuple[str, str, str]
    """
    base = neko_url.rstrip("/")
    login_url = f"{base}/api/login"
    r = requests.post(login_url, json={"username": username, "password": password}, timeout=timeout)
    r.raise_for_status()
    tok = r.json().get("token")
    if not tok:
        raise RuntimeError("REST login successful, but no token in response")
    host = base.split("://", 1)[-1].rstrip("/")
    scheme = "wss" if base.startswith("https") else "ws"
    ws_url = f"{scheme}://{host}/api/ws?token={tok}"
    return ws_url, base, tok

def name_keysym(name: str) -> int:
    """Map a key name (case-insensitive) or single character to a keysym code.

    :param name: Key name or single character.
    :type name: str
    :return: X11 keysym (0 if unknown).
    :rtype: int
    """
    if len(name) == 1:
        return KEYSYM.get(name.lower(), ord(name))
    return KEYSYM.get(name, KEYSYM.get(name.capitalize(), 0))


class ManualCLI:
    """Interactive CLI that speaks the server's control protocol correctly.

    This object owns the lifetime of the connection(s), REPL, and WebRTC
    signaling. It will reconnect when the WebSocket drops and rebuild all
    background tasks and the RTCPeerConnection as needed.

    :ivar ws_url: The full wss://... URL with token.
    :vartype ws_url: str
    :ivar width: Virtual/known screen width.
    :vartype width: int
    :ivar height: Virtual/known screen height.
    :vartype height: int
    :ivar normalized: If True, input coordinates are 0..1 and scaled.
    :vartype normalized: bool
    :ivar auto_host: If True, re-requests host control when lost.
    :vartype auto_host: bool
    :ivar request_media: If True, performs WebRTC signaling on connect.
    :vartype request_media: bool
    :ivar signaler: Active Signaler for the current connection (or None).
    :vartype signaler: Optional[Signaler]
    :ivar running: Main run flag; set False to stop REPL and connections.
    :vartype running: bool
    :ivar _tasks: Background tasks tied to the current connection.
    :vartype _tasks: Set[asyncio.Task]
    :ivar pc: Current RTCPeerConnection (or None if not negotiated).
    :vartype pc: Optional[RTCPeerConnection]
    :ivar session_id: Our server session identifier, once known.
    :vartype session_id: Optional[str]
    :ivar _curx: Last cursor X used for button events.
    :vartype _curx: int
    :ivar _cury: Last cursor Y used for button events.
    :vartype _cury: int
    """
    def __init__(self, ws: str, width: int, height: int, normalized: bool, auto_host: bool, request_media: bool, base_url: Optional[str] = None, token: Optional[str] = None, audio: bool = True):
        """Initialize a new ManualCLI instance.
        
        :param ws: WebSocket URL with authentication token
        :type ws: str
        :param width: Virtual screen width for coordinate scaling
        :type width: int
        :param height: Virtual screen height for coordinate scaling
        :type height: int
        :param normalized: Whether to treat input coordinates as 0..1 normalized values
        :type normalized: bool
        :param auto_host: Whether to automatically request host control
        :type auto_host: bool
        :param request_media: Whether to perform WebRTC signaling
        :type request_media: bool
        :param base_url: Base URL for REST API calls (optional)
        :type base_url: Optional[str]
        :param token: Authentication token for REST API calls (optional)
        :type token: Optional[str]
        :param audio: Whether to enable audio streaming
        :type audio: bool
        """
        self.ws_url = ws
        self.width = width
        self.height = height
        self.normalized = normalized
        self.auto_host = auto_host
        self.request_media = request_media
        self.audio = audio
        self.base_url = base_url
        self.token = token
        self.signaler: Optional[Signaler] = None
        self.running = True
        self._tasks: Set[asyncio.Task] = set()
        self.pc: Optional[RTCPeerConnection] = None
        self.session_id: Optional[str] = None
        self._curx = 0
        self._cury = 0

    def _xy(self, x: float, y: float) -> Tuple[int, int]:
        """Scale normalized coords and clamp to the configured size.

        :param x: X coordinate (0..1 if normalized else pixels).
        :type x: float
        :param y: Y coordinate (0..1 if normalized else pixels).
        :type y: float
        :return: (px, py) integer pixel coordinates.
        :rtype: Tuple[int, int]
        """
        if self.normalized:
            px = max(0, min(int(x * self.width), self.width - 1))
            py = max(0, min(int(y * self.height), self.height - 1))
        else:
            px = max(0, min(int(x), self.width - 1))
            py = max(0, min(int(y), self.height - 1))
        self._curx, self._cury = px, py
        return px, py

    async def start(self) -> None:
        """Start the REPL and a connection manager that handles reconnection."""
        print("\n\033[92mStarting Neko manual CLI. Type 'help' for commands. Ctrl+D or 'quit' to exit.\033[0m")

        # Connection manager runs in the background and rebuilds state on drops.
        conn_task = asyncio.create_task(self._connection_manager(), name="connection-manager")

        # REPL loop continues even across reconnects.
        while self.running:
            try:
                line = await asyncio.to_thread(input, "\033[96mneko> \033[0m")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            try:
                await self.handle(line.strip())
            except Exception as e:
                logger.error("Command failed: %s", e, exc_info=True)

        # Stop: ensure connection manager and any connection-specific tasks are cleaned up.
        self.running = False
        if not conn_task.done():
            conn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await conn_task
        await self._teardown_connection()

    async def _connection_manager(self) -> None:
        """Maintain a live connection; reconnect on drop with backoff."""
        while self.running:
            # Don't create a new signaler each time - reuse the existing one
            if not self.signaler:
                self.signaler = Signaler(self.ws_url)
            
            await self.signaler.connect_with_backoff()
            
            if not self.signaler.ws or getattr(self.signaler.ws, "closed", True):
                if not self.running:
                    break
                await asyncio.sleep(1.0)
                continue

            # Send initial messages with error handling
            try:
                # Initial requests after (re)connect
                if self.request_media:
                    audio_payload = {} if self.audio else {"disabled": True}
                    await self._safe_send({"event": "signal/request", "payload": {"video": {}, "audio": audio_payload}})
                if self.auto_host:
                    await self._safe_send({"event": "control/request"})
                
                # Only start tasks if connection is still open
                if self.signaler.ws and not getattr(self.signaler.ws, "closed", True):
                    self._tasks.add(asyncio.create_task(self._event_logger(), name="event-logger"))
                    self._tasks.add(asyncio.create_task(self._consume_signaling(), name="signal-consumer"))
                    logger.info("Connected to Neko server")
                else:
                    logger.warning("Connection closed before tasks could start")
                    continue
                    
            except Exception as e:
                logger.error("Failed to send initial messages: %s", e)
                await self._teardown_connection()
                await asyncio.sleep(0.5)
                continue

            # Wait until this connection is closed
            await self.signaler._closed.wait()
            
            logger.warning("Disconnected — attempting to reconnect")
            await self._teardown_connection()
            await asyncio.sleep(0.5)

    async def _teardown_connection(self) -> None:
        """Cancel per-connection tasks and close resources."""
        # Cancel per-connection tasks
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._tasks.clear()
        # Close PC
        if self.pc:
            with contextlib.suppress(Exception):
                await self.pc.close()
        self.pc = None
        # Close signaler (if still open)
        if self.signaler:
            with contextlib.suppress(Exception):
                await self.signaler.close()

    async def _safe_send(self, msg: Dict[str, Any]) -> None:
        """Send a message if connected; otherwise warn."""
        if not self.signaler or self.signaler._closed.is_set():
            logger.warning("Not connected")
            return
        await self.signaler.send(msg)

    async def handle(self, line: str) -> None:
        """Parse and execute a single REPL command line.

        :param line: Raw input line from the user.
        :type line: str
        """
        try:
            args = shlex.split(line)
        except ValueError as e:
            print(f"\033[91mError: Invalid command syntax: {e}\033[0m")
            return

        if not args:
            return
        cmd, *rest = args
        cmd = cmd.lower()

        if cmd in ("quit", "exit"):
            self.running = False
            return
        if cmd in ("help", "?"):
            self._print_help()
            return
        if cmd == "host":
            await self._safe_send({"event": "control/request"})
            return
        if cmd == "unhost":
            await self._safe_send({"event": "control/release"})
            return
        if cmd == "size":
            self._handle_size(rest)
            return
        if cmd == "raw":
            await self._handle_raw(rest)
            return
        if cmd == "move":
            await self._handle_move(rest)
            return
        if cmd in ("click", "rclick", "lclick", "dblclick", "tap", "hover"):
            await self._handle_click(cmd, rest)
            return
        if cmd == "input":
            await self._handle_input(rest)
            return
        if cmd == "text":
            await self._type_text(" ".join(rest))
            return
        if cmd == "enter":
            await self._key_once("Enter")
            return
        if cmd == "key":
            if rest:
                await self._key_once(rest[0])
            else:
                print("usage: key <KeyName>")
            return
        if cmd in ("copy", "cut", "select_all"):
            await self._safe_send({"event": f"control/{cmd}"})
            return
        if cmd == "paste":
            # Paste supports optional text argument
            if rest:
                text = " ".join(rest)
                await self._safe_send({"event": "control/paste", "payload": {"text": text}})
            else:
                await self._safe_send({"event": "control/paste", "payload": {"text": ""}})
            return
        if cmd == "scroll":
            await self._handle_scroll(rest)
            return
        if cmd == "swipe":
            await self._handle_swipe(rest)
            return
        if cmd == "force-take":
            if not self.session_id:
                print("\033[91mError: No session ID available. Connect first.\033[0m")
                return
            await self._safe_send({"event": "admin/control", "payload": {"id": self.session_id}})
            return
        if cmd == "force-release":
            if not self.session_id:
                print("\033[91mError: No session ID available. Connect first.\033[0m")
                return
            await self._safe_send({"event": "admin/release", "payload": {"id": self.session_id}})
            return
        if cmd == "kick":
            await self._handle_kick(rest)
            return
        if cmd == "sessions":
            await self._handle_sessions()
            return
        print(f"\033[91mUnknown command: {cmd}. Try 'help'.\033[0m")

    def _print_help(self) -> None:
        """Print comprehensive help text for all available REPL commands.
        
        Displays color-coded command reference including movement, clicking,
        keyboard input, scrolling, and administrative commands with usage
        examples and parameter descriptions.
        """
        print("""\033[1mCommands:\033[0m
  \033[94mhelp\033[0m                         Show this help message.
  \033[94mquit | exit\033[0m                  Close connection and exit.

  \033[94mhost\033[0m                         Request host (mouse/keyboard) control.
  \033[94munhost\033[0m                       Release host control.
  \033[94msize [WxH]\033[0m                   Show or set virtual screen size for normalized coordinates.
  \033[94mraw '<json>'\033[0m                 Send a raw JSON message to the WebSocket.

  \033[91mforce-take\033[0m                   Force take host control (admin only).
  \033[91mforce-release\033[0m                Force release host control (admin only).
  \033[91mkick <sessionId>\033[0m             Force kick a session by ID (admin only).
  \033[91msessions\033[0m                     List all connected users and their session IDs.

  \033[92mmove X Y\033[0m                     Move pointer to coordinates (pixels, or 0..1 if --norm).
  \033[92mclick [btn]\033[0m                  Click a button (left, right, middle) at current location.
  \033[92mrclick | lclick | dblclick\033[0m   Convenience aliases for right, left, or double click.
  \033[92mhover X Y\033[0m                    Move pointer to X, Y without clicking.
  \033[92mtap X Y [btn]\033[0m                Move to X, Y and then click.

  \033[92minput X Y "text"\033[0m             Click at X, Y and then type the given text.
  \033[92mtext "some text"\033[0m              Type text at the current cursor focus.
  \033[92menter\033[0m                         Press the Enter/Return key.
  \033[92mkey <KeyName>\033[0m                Press a specific key by name (e.g., Escape, F5, Control).
  \033[92mcopy | cut | select_all\033[0m      Send dedicated clipboard/selection commands.
  \033[92mpaste [text]\033[0m                 Paste from clipboard, or paste specific text if provided.

  \033[93mscroll <dir> [N]\033[0m             Scroll in a direction (up, down, left, right) N times.
  \033[93mswipe x1 y1 x2 y2\033[0m            Perform a mouse drag gesture from (x1,y1) to (x2,y2).
""")

    def _handle_size(self, args) -> None:
        """Handle the 'size' command to view or set virtual screen dimensions.
        
        :param args: Command arguments - empty to show current size, or ["WxH"] to set new size
        :type args: list
        """
        if not args:
            print(f"size {self.width}x{self.height}  normalized={self.normalized}")
        else:
            self.width, self.height = parse_size(args[0])
            print(f"ok size -> {self.width}x{self.height}")

    async def _handle_raw(self, args) -> None:
        """Handle the 'raw' command to send arbitrary JSON messages.
        
        :param args: Command arguments containing JSON string to send
        :type args: list
        """
        if not args:
            print('usage: raw \'{"event":"...","payload":{}}\'')
            return
        try:
            payload = json.loads(args[0])
        except json.JSONDecodeError as e:
            print(f"\033[91mInvalid JSON: {e}\033[0m")
            return
        await self._safe_send(payload)

    async def _handle_move(self, args) -> None:
        """Handle the 'move' command to position the mouse cursor.
        
        :param args: Command arguments [x, y] coordinates
        :type args: list
        """
        if len(args) < 2:
            print("usage: move X Y")
            return
        px, py = self._xy(float(args[0]), float(args[1]))
        await self._safe_send({"event": "control/move", "payload": {"x": px, "y": py}})

    async def _handle_click(self, cmd, args) -> None:
        """Handle click-related commands including tap, hover, and button presses.
        
        :param cmd: The click command type (click, rclick, lclick, dblclick, tap, hover)
        :type cmd: str
        :param args: Command arguments which may include coordinates and button type
        :type args: list
        """
        button = "left"
        if cmd == "rclick":
            button = "right"
        if cmd == "lclick":
            button = "left"
        if cmd == "click" and args and args[0] in BUTTON_CODES:
            button = args.pop(0)
        if cmd in ("tap", "hover") and len(args) >= 2:
            await self._handle_move(args)
            if cmd == "hover":
                return
        if cmd == "dblclick":
            # Sequential double click with a short gap for ordering/timing correctness
            await self._button_press(button)
            await asyncio.sleep(0.06)
            await self._button_press(button)
        else:
            await self._button_press(button)

    async def _handle_input(self, args) -> None:
        """Handle the 'input' command to click at coordinates and type text.
        
        :param args: Command arguments [x, y, text] where text may be multiple words
        :type args: list
        """
        if len(args) < 3:
            print('usage: input <x> <y> "text"')
            return
        await self._handle_move(args[:2])
        await self._button_press("left")
        await self._type_text(" ".join(args[2:]))

    async def _handle_scroll(self, args) -> None:
        """Handle the 'scroll' command to perform scrolling gestures.
        
        :param args: Command arguments [direction, amount] where amount defaults to 1
        :type args: list
        """
        if not args:
            print('usage: scroll <direction> [amount=1]')
            return
        direction, amount = args[0].lower(), int(args[1]) if len(args) > 1 else 1
        delta_map = {
            "down": (0, 120 * amount),
            "up": (0, -120 * amount),
            "right": (120 * amount, 0),
            "left": (-120 * amount, 0),
        }
        dx, dy = delta_map.get(direction, (0, 0))
        if dx == 0 and dy == 0:
            print(f"invalid scroll direction: {direction}")
            return
        await self._safe_send({"event": "control/scroll", "payload": {"delta_x": dx, "delta_y": dy}})

    async def _handle_swipe(self, args) -> None:
        """Handle the 'swipe' command to perform drag gestures between two points.
        
        :param args: Command arguments [x1, y1, x2, y2] defining start and end coordinates
        :type args: list
        """
        if len(args) < 4:
            print("usage: swipe x1 y1 x2 y2")
            return
        p1 = self._xy(float(args[0]), float(args[1]))
        p2 = self._xy(float(args[2]), float(args[3]))
        await self._safe_send({"event": "control/move", "payload": {"x": p1[0], "y": p1[1]}})
        await self._button_down("left")
        await asyncio.sleep(0.05)
        await self._safe_send({"event": "control/move", "payload": {"x": p2[0], "y": p2[1]}})
        await self._button_up("left")

    async def _handle_kick(self, args) -> None:
        """Handle the 'kick' admin command to forcibly disconnect a session.
        
        :param args: Command arguments [session_id] to identify target session
        :type args: list
        """
        if not args:
            print("usage: kick <sessionId>")
            return
        session_id = args[0]
        await self._safe_send({"event": "admin/kick", "payload": {"id": session_id}})

    async def _handle_sessions(self) -> None:
        """Handle the 'sessions' command to list all connected users.
        
        Fetches session information from the REST API and displays connected
        users with their session IDs and host status. Requires authentication
        credentials from REST login.
        """
        if not self.base_url or not self.token:
            print("\033[91mSessions command requires REST API access (login with --neko-url credentials)\033[0m")
            return
        
        try:
            sessions_url = f"{self.base_url}/api/sessions"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            # Use asyncio.to_thread to make the blocking requests call non-blocking
            response = await asyncio.to_thread(requests.get, sessions_url, headers=headers, timeout=10.0)
            response.raise_for_status()
            
            sessions_data = response.json()
            
            print("\033[1mConnected Sessions:\033[0m")
            connected_count = 0
            
            for session in sessions_data:
                session_id = session.get("id", "unknown")
                profile = session.get("profile", {})
                state = session.get("state", {})
                username = profile.get("name", "unknown")
                is_connected = state.get("is_connected", False)
                
                if is_connected:
                    connected_count += 1
                    is_host = state.get("is_host", False)
                    host_indicator = " \033[93m[HOST]\033[0m" if is_host else ""
                    print(f"  \033[92m{session_id}\033[0m - {username}{host_indicator}")
            
            if connected_count == 0:
                print("  \033[93mNo connected sessions found\033[0m")
            else:
                print(f"\nTotal connected: {connected_count}")
                
        except requests.RequestException as e:
            print(f"\033[91mFailed to fetch sessions: {e}\033[0m")
        except Exception as e:
            print(f"\033[91mError processing sessions: {e}\033[0m")

    async def _button_press(self, button: str) -> None:
        """Send a complete button press (down + up) event.
        
        :param button: Button name ("left", "right", "middle")
        :type button: str
        """
        await self._safe_send({"event": "control/buttonpress", "payload": {"x": self._curx, "y": self._cury, "code": BUTTON_CODES.get(button, 1)}})

    async def _button_down(self, button: str) -> None:
        """Send a button down event without releasing.
        
        :param button: Button name ("left", "right", "middle")
        :type button: str
        """
        await self._safe_send({"event": "control/buttondown", "payload": {"x": self._curx, "y": self._cury, "code": BUTTON_CODES.get(button, 1)}})

    async def _button_up(self, button: str) -> None:
        """Send a button up (release) event.
        
        :param button: Button name ("left", "right", "middle")
        :type button: str
        """
        await self._safe_send({"event": "control/buttonup", "payload": {"x": self._curx, "y": self._cury, "code": BUTTON_CODES.get(button, 1)}})

    async def _key_once(self, key: str) -> None:
        """Send a complete key press (down + up) event.
        
        :param key: Key name or character to press
        :type key: str
        """
        await self._send_key_event("keypress", key)

    async def _key_down(self, key: str) -> None:
        """Send a key down event without releasing.
        
        :param key: Key name or character to press down
        :type key: str
        """
        await self._send_key_event("keydown", key)

    async def _key_up(self, key: str) -> None:
        """Send a key up (release) event.
        
        :param key: Key name or character to release
        :type key: str
        """
        await self._send_key_event("keyup", key)

    async def _send_key_event(self, event: str, key: str) -> None:
        """Send a keyboard event with the specified key.
        
        :param event: Event type ("keypress", "keydown", "keyup")
        :type event: str
        :param key: Key name or character
        :type key: str
        """
        ks = name_keysym(key)
        if not ks:
            print(f"Unknown key: {key}")
            return
        await self._safe_send({"event": f"control/{event}", "payload": {"keysym": ks}})

    async def _type_text(self, text: str) -> None:
        """Type a string of text by sending individual key press events.
        
        :param text: Text string to type, with optional surrounding quotes that are stripped
        :type text: str
        """
        if len(text) >= 2 and text[0] == text[-1] in ('"', "'"):
            text = text[1:-1]
        for ch in text:
            ks = name_keysym(ch)
            if ks:
                await self._key_once(ch)

    async def _event_logger(self) -> None:
        """Background task that processes server events and maintains connection.

        Consumes messages from the system topic queue, handles heartbeat responses
        to prevent disconnection, logs server events, and dispatches specific
        event types to their appropriate handlers.
        """
        q = self.signaler.broker.topic_queue("system")
        while self.running and self.signaler and not self.signaler._closed.is_set():
            try:
                msg = await q.get()
                ev = msg.get("event", "")
                payload = msg.get("payload", {})

                # Heartbeat reply (kept here so it survives reconnects).
                if ev == "system/heartbeat":
                    # Send heartbeat response to prevent disconnection
                    await self._safe_send({"event": "client/heartbeat"})
                    continue

                # Log server events to file instead of CLI
                logger.info("Server event: %s :: %s", ev, json.dumps(payload, ensure_ascii=False))

                # Event-specific handlers
                if ev == "system/init":
                    self._handle_init_event(payload)
                elif ev == "screen/updated":
                    self._handle_screen_updated(payload)
                elif ev == "control/host":
                    self._handle_host_changed(payload)
                elif ev == "keyboard/map":
                    self._handle_keyboard_map(payload)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in event logger: %s", e, exc_info=True)

    def _handle_init_event(self, payload: dict) -> None:
        """Process system initialization event from server.
        
        Extracts session ID, screen dimensions, and server settings from
        the initial system event. Updates local state and logs warnings
        for any server restrictions.
        
        :param payload: Event payload containing initialization data
        :type payload: dict
        """
        self.session_id = payload.get("session_id")
        if (size := payload.get("screen_size")) and isinstance(size, dict):
            if "width" in size and "height" in size:
                self.width, self.height = int(size["width"]), int(size["height"])
                logger.info("Initial screen size set to %dx%d", self.width, self.height)
        
        # Check server settings for restrictions
        if (settings := payload.get("settings")) and isinstance(settings, dict):
            if settings.get("locked_controls"):
                logger.warning("Server has locked controls - admin privileges may be required")
            if settings.get("private_mode"):
                logger.warning("Server is in private mode")
            if settings.get("control_protection"):
                logger.info("Server has control protection enabled")

    def _handle_screen_updated(self, payload: dict) -> None:
        """Process screen size change events from server.
        
        Updates the local width and height when the server reports
        a screen resolution change.
        
        :param payload: Event payload containing new screen dimensions
        :type payload: dict
        """
        if "width" in payload and "height" in payload:
            self.width, self.height = int(payload["width"]), int(payload["height"])
            logger.info("Screen size changed to %dx%d", self.width, self.height)

    def _handle_host_changed(self, payload: dict) -> None:
        """Process host control change events.
        
        Automatically re-requests host control if it was lost or given to
        another user and auto_host mode is enabled.
        
        :param payload: Event payload containing host status information
        :type payload: dict
        """
        host_id = payload.get("host_id")
        if self.auto_host and (not payload.get("has_host") or host_id != self.session_id):
            logger.info("Host control lost or changed, re-requesting.")
            asyncio.create_task(self._safe_send({"event": "control/request"}))

    def _handle_keyboard_map(self, payload: dict) -> None:
        """Process keyboard mapping updates from server.
        
        Updates the global KEYSYM dictionary with server-provided key mappings.
        Accepts both flat dictionary format and nested {'map': {...}} format.
        
        :param payload: Event payload containing keyboard mappings
        :type payload: dict
        """
        mapping = None
        if isinstance(payload, dict):
            if "map" in payload and isinstance(payload["map"], dict):
                mapping = payload["map"]
            else:
                # Some servers send the map as the payload directly.
                mapping = payload
        if isinstance(mapping, dict):
            KEYSYM.update(mapping)
            logger.info("Updated keyboard map with %d new entries from server.", len(mapping))

    async def _consume_signaling(self) -> None:
        """Handle WebRTC signaling protocol for media stream negotiation.

        Performs the complete WebRTC handshake required by Neko servers:
        waits for SDP offer, configures ICE servers with strict mapping,
        creates and sends SDP answer, and processes incoming ICE candidates.
        Media tracks are logged but not processed since this is a control-only client.
        """
        q = self.signaler.broker.topic_queue("signal")

        # Wait for the first offer/provide for this connection.
        offer_msg = None
        while self.running and self.signaler and not self.signaler._closed.is_set():
            msg = await q.get()
            if msg.get("event") in ("signal/offer", "signal/provide"):
                offer_msg = msg
                break

        if not offer_msg or not self.running or not self.signaler or self.signaler._closed.is_set():
            return

        payload = offer_msg.get("payload", {})

        # Strict ICE server mapping.
        ice_servers = []
        for srv in payload.get("iceservers", []):
            if not isinstance(srv, dict):
                continue
            urls = srv.get("urls") or srv.get("url")
            username = srv.get("username")
            credential = srv.get("credential") or srv.get("password")
            if urls:
                ice_servers.append(RTCIceServer(urls=urls, username=username, credential=credential))

        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))

        @self.pc.on("iceconnectionstatechange")
        def on_ice_change() -> None:
            """Log ICE connection state changes."""
            logger.info("ICE connection state is %s", self.pc.iceConnectionState)

        @self.pc.on("connectionstatechange")
        def on_conn_change() -> None:
            """Log peer connection state changes."""
            logger.info("Peer connection state is %s", self.pc.connectionState)

        @self.pc.on("track")
        def on_track(track) -> None:
            """Log received media tracks without processing them.
            
            :param track: The media track received from the peer connection
            """
            logger.info("Received track: kind=%s id=%s", track.kind, getattr(track, "id", "unknown"))
            # Manual CLI doesn't process media streams, just log the track

        @self.pc.on("icecandidate")
        async def on_cand(cand: Optional[RTCIceCandidate]) -> None:
            """Send local ICE candidates to the remote peer.
            
            :param cand: ICE candidate to send, or None when gathering is complete
            :type cand: Optional[RTCIceCandidate]
            """
            if cand:
                await self._safe_send({
                    "event": "signal/candidate",
                    "payload": {
                        "candidate": cand.candidate,
                        "sdpMid": cand.sdpMid,
                        "sdpMLineIndex": cand.sdpMLineIndex
                    }
                })

        # Apply remote SDP and answer.
        remote_type = payload.get("type", "offer")
        remote_sdp = payload.get("sdp")
        if not remote_sdp:
            logger.error("Missing SDP in offer payload; cannot proceed.")
            return

        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=remote_sdp, type=remote_type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await self._safe_send({
            "event": "signal/answer",
            "payload": {
                "sdp": self.pc.localDescription.sdp,
                "type": self.pc.localDescription.type
            }
        })

        # Handle incoming ICE candidates for this connection until stopped.
        try:
            while self.running and self.signaler and not self.signaler._closed.is_set():
                msg = await q.get()
                if msg.get("event") == "signal/candidate":
                    pay = msg.get("payload", {})
                    cand_str = (pay.get("candidate") or "")
                    # Accept both full-line or "candidate:..." shapes.
                    cand_sdp = cand_str.split(":", 1)[-1] if cand_str.startswith("candidate:") else cand_str
                    if not cand_sdp:
                        continue
                    ice = candidate_from_sdp(cand_sdp)
                    ice.sdpMid = pay.get("sdpMid")
                    sdp_mline = pay.get("sdpMLineIndex")
                    if isinstance(sdp_mline, str) and sdp_mline.isdigit():
                        sdp_mline = int(sdp_mline)
                    ice.sdpMLineIndex = sdp_mline
                    await self.pc.addIceCandidate(ice)
        except asyncio.CancelledError:
            pass


async def main() -> None:
    """Main entry point for the Neko manual control CLI.
    
    Parses command line arguments, performs REST authentication if needed,
    creates a ManualCLI instance with the specified configuration, and starts
    the interactive REPL interface.
    """
    ap = argparse.ArgumentParser(
        "neko-manual",
        description="Manual control REPL for Neko v3.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--ws", default=os.environ.get("NEKO_WS"), help="Full wss://... URL with token.")
    ap.add_argument("--neko-url", default=os.environ.get("NEKO_URL"), help="Base https://... URL for REST login.")
    ap.add_argument("--username", default=os.environ.get("NEKO_USER"), help="Username for REST login.")
    ap.add_argument("--password", default=os.environ.get("NEKO_PASS"), help="Password for REST login.")
    ap.add_argument("--norm", action="store_true", help="Treat coordinates as normalized 0..1 floats.")
    ap.add_argument("--size", default=os.environ.get("NEKO_SIZE", "1280x720"), help="Virtual screen size for scaling, e.g., 1920x1080.")
    ap.add_argument("--no-auto-host", action="store_true", help="Do not automatically request host control.")
    ap.add_argument("--no-media", action="store_true", help="Do not perform WebRTC signaling (may break control).")
    ap.add_argument("--no-audio", action="store_true", help="Disable audio streaming.")
    args = ap.parse_args()

    ws_url = args.ws
    base_url = None
    token = None
    
    if not ws_url:
        if not (args.neko_url and args.username and args.password):
            ap.error("must provide --ws OR all of --neko-url, --username, --password")
        try:
            logger.info("Performing REST login to get WebSocket token...")
            ws_url, base_url, token = ws_from_rest_login(args.neko_url, args.username, args.password)
        except Exception as e:
            logger.critical("REST login failed: %s", e)
            sys.exit(1)
    else:
        # If using direct WS URL, try to extract base URL and token for REST API calls
        if args.neko_url:
            base_url = args.neko_url.rstrip("/")
        # Try to extract token from WS URL
        if "token=" in ws_url:
            token = ws_url.split("token=")[1].split("&")[0]

    w, h = parse_size(args.size)
    cli = ManualCLI(
        ws=ws_url, width=w, height=h, normalized=args.norm,
        auto_host=not args.no_auto_host, request_media=not args.no_media,
        audio=not args.no_audio, base_url=base_url, token=token
    )
    await cli.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
