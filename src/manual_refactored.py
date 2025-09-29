#!/usr/bin/env python3
# src/manual_refactored.py
"""
manual_refactored.py — Manual control REPL using WebRTCNekoClient.

What it does
------------
- Provides interactive command-line interface for controlling Neko servers
- Uses WebRTCNekoClient for all communication (simplified from original)
- Handles mouse movements, clicks, keyboard input, and scrolling
- Manages host control acquisition and release
- Supports both REST login and direct WebSocket authentication

Refactored from original manual.py to use modular neko_comms library.
"""

import asyncio
import json
import logging
import os
import shlex
import sys
from typing import Dict, Any, Tuple, Optional, List

import requests

from neko_comms import WebRTCNekoClient, rest_login_and_ws_url
from neko_comms.types import BUTTON_CODES, name_keysym

# Configure logging
log_file = os.environ.get("NEKO_LOGFILE")
if log_file:
    logging.basicConfig(
        level=os.environ.get("NEKO_LOGLEVEL", "INFO"),
        format='[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file,
        filemode='a'
    )
else:
    logging.basicConfig(
        level=os.environ.get("NEKO_LOGLEVEL", "INFO"),
        format='[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s',
        datefmt='%H:%M:%S'
    )

logger = logging.getLogger("neko_manual")

# Suppress verbose logging from dependencies
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)


# rest_login_and_ws_url is imported from neko_comms.auth


class ManualController:
    """Interactive manual control interface using WebRTCNekoClient."""

    def __init__(self, ws_url: str, width: int = 1920, height: int = 1080,
                 normalized: bool = False, auto_host: bool = True,
                 base_url: Optional[str] = None, token: Optional[str] = None):
        """Initialize the manual controller.

        :param ws_url: WebSocket URL for Neko connection
        :param width: Virtual screen width
        :param height: Virtual screen height
        :param normalized: If True, coordinates are 0-1 and scaled
        :param auto_host: If True, automatically request host control
        :param base_url: REST base URL (when available) for admin commands
        :param token: REST API bearer token for admin commands
        """
        self.ws_url = ws_url
        self.width = width
        self.height = height
        self.normalized = normalized
        self.auto_host = auto_host
        self.base_url = base_url.rstrip("/") if base_url else None
        self.token = token

        # State
        self.running = True
        self._curx = 0
        self._cury = 0

        # Neko client
        self.client = WebRTCNekoClient(
            ws_url=ws_url,
            auto_host=auto_host,
            request_media=True  # Enable WebRTC for full functionality
        )

    async def run(self) -> None:
        """Main run loop with connection management and REPL."""
        logger.info("Starting manual control session")

        while self.running:
            try:
                # Connect to Neko server
                await self.client.connect()
                logger.info("Connected to Neko server")

                # Set screen size
                self.client.set_screen_size(self.width, self.height)

                # Start background tasks
                await asyncio.gather(
                    self._event_logger(),
                    self._repl(),
                    return_exceptions=True
                )

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                self.running = False
            except Exception as e:
                logger.error("Connection error: %s", e)
                await asyncio.sleep(2)  # Brief delay before retry
            finally:
                await self.client.disconnect()

        logger.info("Manual control session ended")

    async def _event_logger(self) -> None:
        """Background task that logs server events."""
        try:
            while self.running and self.client.is_connected():
                try:
                    # Log system events
                    msg = await self.client.subscribe_topic("system", timeout=1.0)
                    event = msg.get("event", "")
                    payload = msg.get("payload", {})

                    if event == "system/heartbeat":
                        continue  # Skip noisy heartbeats

                    logger.info("SYS: %s %s", event, payload)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.debug("Event logger error: %s", e)
                    break

        except Exception as e:
            logger.error("Event logger failed: %s", e)

    async def _repl(self) -> None:
        """Interactive command REPL."""
        print("Neko Manual Control - Type 'help' for commands")

        while self.running and self.client.is_connected():
            try:
                # Get user input (non-blocking)
                cmd = await asyncio.get_event_loop().run_in_executor(
                    None, input, "neko> "
                )

                if not cmd.strip():
                    continue

                await self._handle_command(cmd.strip())

            except EOFError:
                self.running = False
                break
            except Exception as e:
                logger.error("REPL error: %s", e)

    async def _handle_command(self, cmd_line: str) -> None:
        """Handle a single command.

        :param cmd_line: Command line input
        """
        try:
            args = shlex.split(cmd_line)
            if not args:
                return

            cmd = args[0].lower()
            args = args[1:]

            # Command dispatch
            if cmd in ("quit", "exit", "q"):
                self.running = False

            elif cmd == "help":
                self._show_help()

            elif cmd == "move":
                await self._handle_move(args)

            elif cmd in ("click", "rclick", "lclick", "dblclick", "tap", "hover"):
                await self._handle_click(cmd, args)

            elif cmd == "scroll":
                await self._handle_scroll(args)

            elif cmd == "swipe":
                await self._handle_swipe(args)

            elif cmd == "raw":
                await self._handle_raw(args)

            elif cmd == "key":
                await self._handle_key(args)

            elif cmd == "enter":
                await self._handle_enter()

            elif cmd == "text":
                await self._handle_type(args)

            elif cmd == "input":
                await self._handle_input(args)

            elif cmd == "type":
                await self._handle_type(args)

            elif cmd in ("copy", "cut", "select_all"):
                await self._handle_clipboard(cmd)

            elif cmd == "paste":
                await self._handle_paste(args)

            elif cmd == "host":
                await self.client.request_host_control()

            elif cmd == "unhost":
                await self.client.release_host_control()

            elif cmd == "size":
                await self._handle_size(args)

            elif cmd == "force-take":
                await self._handle_force(True)

            elif cmd == "force-release":
                await self._handle_force(False)

            elif cmd == "kick":
                await self._handle_kick(args)

            elif cmd == "sessions":
                await self._handle_sessions()

            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")

        except Exception as e:
            logger.error("Command error: %s", e)
            print(f"Error: {e}")

    async def _safe_send(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Send an event if the client is connected."""
        if not self.client.is_connected():
            print("Not connected to server")
            return

        try:
            await self.client.send_event(event, payload)
        except Exception as exc:
            logger.error("Failed to send event %s: %s", event, exc)
            print(f"Error sending event: {exc}")

    def _show_help(self) -> None:
        """Show help text."""
        help_text = """
Available commands:

Movement & Clicking:
  move X Y                - Move cursor to coordinates
  click [X Y] [button]    - Click at coordinates (or current position)
  rclick | lclick         - Right or left click aliases
  dblclick [X Y]          - Double-click at coordinates
  tap X Y [button]        - Move and click at coordinates
  hover X Y               - Hover cursor at coordinates
  swipe X1 Y1 X2 Y2       - Drag from point 1 to point 2

Keyboard & Text:
  key <keyname>           - Press a key (Escape, F5, etc.)
  enter                   - Press the Enter/Return key
  type <text>             - Type text at current focus
  text <text>             - Alias for type
  input X Y "text"        - Click at X,Y then type text
  copy | cut | select_all - Clipboard shortcuts
  paste [text]            - Paste clipboard or provided text

Raw / Admin:
  raw '{{"event":...}}'    - Send raw JSON event
  host / unhost           - Request or release host control
  force-take              - Force host control (admin)
  force-release           - Force release host control (admin)
  kick <sessionId>        - Kick a session (admin)
  sessions                - List sessions (requires REST login)

Other:
  size [WIDTH HEIGHT]     - Show or set screen dimensions
  scroll dir [amount]     - Scroll direction (up/down/left/right)
  help                    - Show this help
  quit / exit / q         - Exit program

Coordinates:
  - Use pixel coordinates (0,0 = top-left) unless --normalized
  - Current screen size: {width}x{height}
""".format(width=self.width, height=self.height)
        print(help_text)

    def _action_executor(self):
        if not self.client.action_executor:
            raise RuntimeError("Action executor not ready")
        return self.client.action_executor

    def _parse_text_argument(self, args: List[str], usage: str) -> Optional[str]:
        """Join and normalize quoted text arguments."""
        if not args:
            print(usage)
            return None

        text = " ".join(args)
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1]
        return text

    def _xy(self, x: float, y: float) -> Tuple[int, int]:
        """Convert input coordinates to pixel coordinates.

        :param x: X coordinate (normalized if self.normalized is True)
        :param y: Y coordinate (normalized if self.normalized is True)
        :return: Pixel coordinates as (x, y)
        """
        if self.normalized:
            # Convert from 0-1 to pixel coordinates
            px = int(x * self.width)
            py = int(y * self.height)
        else:
            px = int(x)
            py = int(y)

        # Clamp to screen bounds
        px = max(0, min(px, self.width - 1))
        py = max(0, min(py, self.height - 1))

        return px, py

    async def _handle_move(self, args: list) -> None:
        """Handle move command."""
        if len(args) < 2:
            print("Usage: move X Y")
            return

        try:
            x, y = float(args[0]), float(args[1])
            px, py = self._xy(x, y)

            # Update cursor position
            self._curx, self._cury = px, py

            # Send move command via action executor
            await self._action_executor().move(px, py)
            print(f"Moved to ({px}, {py})")

        except ValueError:
            print("Invalid coordinates")

    async def _handle_click(self, cmd: str, args: list) -> None:
        """Handle click-related commands."""
        button = "left"

        # Determine button type
        if cmd == "rclick":
            button = "right"
        elif cmd == "lclick":
            button = "left"
        elif cmd == "click" and args and args[0] in BUTTON_CODES:
            button = args.pop(0)

        # Handle coordinates
        if len(args) >= 2:
            try:
                x, y = float(args[0]), float(args[1])
                px, py = self._xy(x, y)
                self._curx, self._cury = px, py
                await self._action_executor().move(px, py)
            except ValueError:
                print("Invalid coordinates")
                return
        else:
            px, py = self._curx, self._cury

        # Execute click
        if cmd == "hover":
            print(f"Hovering at ({px}, {py})")
        elif cmd == "dblclick":
            await self._action_executor().button_press(px, py, button)
            await asyncio.sleep(0.05)
            await self._action_executor().button_press(px, py, button)
            print(f"Double-clicked {button} at ({px}, {py})")
        else:
            await self._action_executor().button_press(px, py, button)
            print(f"Clicked {button} at ({px}, {py})")

    async def _handle_scroll(self, args: list) -> None:
        """Handle scroll command."""
        if not args:
            print("Usage: scroll up|down|left|right [amount]")
            return

        direction = args[0].lower()
        try:
            amount = int(args[1]) if len(args) > 1 else 1
        except ValueError:
            print("Invalid scroll amount")
            return

        # Map direction to scroll deltas
        delta_map = {
            "down": (0, 120 * amount),
            "up": (0, -120 * amount),
            "right": (120 * amount, 0),
            "left": (-120 * amount, 0),
        }

        if direction not in delta_map:
            print("Invalid scroll direction. Use: up, down, left, right")
            return

        dx, dy = delta_map[direction]
        await self._action_executor().scroll(dx, dy)
        print(f"Scrolled {direction} (amount: {amount})")

    async def _handle_swipe(self, args: list) -> None:
        """Handle swipe command."""
        if len(args) < 4:
            print("Usage: swipe X1 Y1 X2 Y2")
            return

        try:
            x1, y1 = float(args[0]), float(args[1])
            x2, y2 = float(args[2]), float(args[3])

            p1 = self._xy(x1, y1)
            p2 = self._xy(x2, y2)

            await self._action_executor().swipe(p1[0], p1[1], p2[0], p2[1])
            print(f"Swiped from {p1} to {p2}")

        except ValueError:
            print("Invalid coordinates")

    async def _handle_key(self, args: list) -> None:
        """Handle key command."""
        if not args:
            print("Usage: key <keyname>")
            return

        key = " ".join(args)
        ks = name_keysym(key)
        if not ks:
            print(f"Unknown key: {key}")
            return

        await self._action_executor().key_once(key)
        print(f"Pressed key: {key}")

    async def _handle_type(self, args: list) -> None:
        """Handle type command."""
        text = self._parse_text_argument(args, "Usage: type <text>")
        if text is None:
            return

        await self._action_executor().type_text(text)
        print(f"Typed: {text}")

    async def _handle_enter(self) -> None:
        """Handle enter command."""
        await self._action_executor().key_once("Enter")
        print("Pressed Enter")

    async def _handle_input(self, args: List[str]) -> None:
        """Handle input command to click and type text."""
        if len(args) < 3:
            print("Usage: input X Y \"text\"")
            return

        try:
            x, y = float(args[0]), float(args[1])
        except ValueError:
            print("Invalid coordinates")
            return

        text = self._parse_text_argument(args[2:], "Usage: input X Y \"text\"")
        if text is None:
            return

        px, py = self._xy(x, y)
        await self._action_executor().move(px, py)
        await self._action_executor().button_press(px, py, "left")
        await asyncio.sleep(0.05)
        await self._action_executor().type_text(text)
        print(f"Clicked ({px}, {py}) and typed: {text}")

    async def _handle_clipboard(self, cmd: str) -> None:
        """Handle clipboard-related commands."""
        executor = self._action_executor()

        if cmd == "copy":
            await executor.copy()
            print("Copied selection")
            return

        if cmd == "cut":
            await executor.key_down("Control")
            await executor.key_once("x")
            await executor.key_up("Control")
            print("Cut selection")
            return

        if cmd == "select_all":
            await executor.key_down("Control")
            await executor.key_once("a")
            await executor.key_up("Control")
            print("Selected all")
            return

    async def _handle_paste(self, args: List[str]) -> None:
        """Handle paste command."""
        text = None
        if args:
            text = self._parse_text_argument(args, "Usage: paste [text]")
            if text is None:
                return

        payload = {"text": text or ""}
        await self._safe_send("control/paste", payload)
        if text:
            print(f"Pasted text: {text}")
        else:
            print("Requested paste from clipboard")

    async def _handle_raw(self, args: List[str]) -> None:
        """Send raw JSON message."""
        if not args:
            print("Usage: raw '{\"event\":..., \"payload\":{...}}'")
            return

        raw = " ".join(args)
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON: {exc}")
            return

        if not isinstance(message, dict) or "event" not in message:
            print("JSON must include 'event'")
            return

        event = message["event"]
        payload = message.get("payload")
        if payload is not None and not isinstance(payload, dict):
            print("payload must be an object if provided")
            return

        await self._safe_send(event, payload)
        print(f"Sent raw event: {event}")

    async def _handle_force(self, take: bool) -> None:
        """Handle force host commands."""
        session_id = self.client.session_id
        if not session_id:
            print("Session ID unknown; wait for connection to stabilize")
            return

        event = "admin/control" if take else "admin/release"
        await self._safe_send(event, {"id": session_id})
        print(("Force-take" if take else "Force-release") + " sent")

    async def _handle_kick(self, args: List[str]) -> None:
        """Handle kick command."""
        if not args:
            print("Usage: kick <sessionId>")
            return

        target = args[0]
        await self._safe_send("admin/kick", {"id": target})
        print(f"Kick sent for session {target}")

    async def _handle_sessions(self) -> None:
        """List active sessions via REST API."""
        if not self.base_url or not self.token:
            print("Sessions command requires REST login (provide --neko-url credentials)")
            return

        url = f"{self.base_url}/api/sessions"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=10.0)
            response.raise_for_status()
        except Exception as exc:
            logger.error("Failed to fetch sessions: %s", exc)
            print(f"Error fetching sessions: {exc}")
            return

        try:
            data = response.json()
        except ValueError as exc:
            print(f"Invalid JSON from sessions endpoint: {exc}")
            return

        if not data:
            print("No active sessions")
            return

        if isinstance(data, list):
            print("Sessions:")
            for sess in data:
                sid = sess.get("id") or sess.get("session_id")
                user = sess.get("username") or sess.get("user", {}).get("username")
                has_host = sess.get("has_host") or sess.get("host")
                print(f"  - id={sid} user={user} host={has_host}")
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False))

    async def _handle_size(self, args: list) -> None:
        """Handle size command."""
        if not args:
            print(f"Current size: {self.width}x{self.height} normalized={self.normalized}")
            return

        if len(args) < 2:
            print("Usage: size WIDTH HEIGHT")
            return

        try:
            width = int(args[0])
            height = int(args[1])
        except ValueError:
            print("Invalid dimensions")
            return

        self.width = width
        self.height = height
        self.client.set_screen_size(width, height)
        print(f"Screen size set to {width}x{height}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Neko manual control")
    parser.add_argument("--neko-url", help="Neko server base URL")
    parser.add_argument("--username", help="Neko username")
    parser.add_argument("--password", help="Neko password")
    parser.add_argument("--ws", help="Direct WebSocket URL")
    parser.add_argument("--width", type=int, default=1920, help="Screen width")
    parser.add_argument("--height", type=int, default=1080, help="Screen height")
    parser.add_argument("--normalized", action="store_true", help="Use normalized coordinates (0-1)")
    parser.add_argument("--no-auto-host", action="store_true", help="Don't auto-request host control")

    args = parser.parse_args()

    base_url = None
    token = None

    # Determine WebSocket URL
    if args.ws:
        ws_url = args.ws
    elif args.neko_url and args.username and args.password:
        try:
            ws_url, base_url, token = rest_login_and_ws_url(args.neko_url, args.username, args.password)
        except ValueError as e:
            print(f"Login failed: {e}")
            sys.exit(1)
    else:
        # Try environment variables
        neko_url = os.environ.get("NEKO_URL")
        username = os.environ.get("NEKO_USER")
        password = os.environ.get("NEKO_PASS")
        ws_url = os.environ.get("NEKO_WS")

        if ws_url:
            pass  # Use direct WebSocket URL
        elif neko_url and username and password:
            try:
                ws_url, base_url, token = rest_login_and_ws_url(neko_url, username, password)
            except ValueError as e:
                print(f"Login failed: {e}")
                sys.exit(1)
        else:
            print("Error: Must provide either --ws URL or --neko-url + credentials")
            sys.exit(1)

    # Create and run controller
    controller = ManualController(
        ws_url=ws_url,
        width=args.width,
        height=args.height,
        normalized=args.normalized,
        auto_host=not args.no_auto_host,
        base_url=base_url,
        token=token
    )

    await controller.run()


if __name__ == "__main__":
    asyncio.run(main())