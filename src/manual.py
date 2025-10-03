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

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"

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
        self._system_events: asyncio.Queue = asyncio.Queue(maxsize=256)

        # Neko client
        self.client = WebRTCNekoClient(
            ws_url=ws_url,
            auto_host=auto_host,
            request_media=True  # Enable WebRTC for full functionality
        )
        self.client.add_system_listener(self._on_system_event)

    async def run(self) -> None:
        """Main run loop with connection management and REPL."""
        logger.info("Starting manual control session")

        while self.running:
            try:
                # Connect to Neko server
                await self.client.connect()
                logger.info("Connected to Neko server")
                self._drain_system_queue()

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

        self.client.remove_system_listener(self._on_system_event)
        logger.info("Manual control session ended")

    async def _event_logger(self) -> None:
        """Background task that logs server events."""
        try:
            while self.running:
                if not self.client.is_connected():
                    await asyncio.sleep(0.1)
                    continue

                try:
                    msg = await asyncio.wait_for(self._system_events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                event = msg.get("event", "")
                payload = msg.get("payload", {})

                if event == "system/heartbeat":
                    await self._safe_send("client/heartbeat")
                    continue

                logger.info("SYS: %s %s", event, payload)

                if event == "system/init":
                    self._handle_system_init(payload)
                elif event == "screen/updated":
                    self._handle_screen_updated(payload)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Event logger failed: %s", e)

    def _handle_system_init(self, payload: Dict[str, Any]) -> None:
        """Process initial system payload to capture screen size."""
        screen = payload.get("screen_size")
        if isinstance(screen, dict):
            self._update_screen_size(screen.get("width"), screen.get("height"))

    def _handle_screen_updated(self, payload: Dict[str, Any]) -> None:
        """Handle screen size updates from the server."""
        self._update_screen_size(payload.get("width"), payload.get("height"))


    def _update_screen_size(self, width: Optional[Any], height: Optional[Any]) -> None:
        """Update local screen dimensions and inform the client."""
        if width is None or height is None:
            return

        try:
            new_width = int(width)
            new_height = int(height)
        except (TypeError, ValueError):
            logger.debug("Ignoring invalid screen size payload: %s x %s", width, height)
            return

        if new_width <= 0 or new_height <= 0:
            logger.debug("Ignoring non-positive screen size: %s x %s", new_width, new_height)
            return

        changed = (self.width != new_width) or (self.height != new_height)
        self.width = new_width
        self.height = new_height
        self.client.set_screen_size(new_width, new_height)
        if changed:
            print(f"{CYAN}Screen size updated to {new_width}x{new_height}{RESET}")

    def _on_system_event(self, msg: Dict[str, Any]) -> None:
        """Fan-out system events to the local logger queue."""
        try:
            self._system_events.put_nowait(msg)
        except asyncio.QueueFull:
            logger.debug("Dropping system event due to full queue: %s", msg.get("event"))

    def _drain_system_queue(self) -> None:
        """Clear any buffered system events."""
        while not self._system_events.empty():
            try:
                self._system_events.get_nowait()
            except asyncio.QueueEmpty:
                break

    @staticmethod
    def _colorize(color: str, message: str) -> str:
        return f"{color}{message}{RESET}"

    def _info(self, message: str) -> None:
        print(self._colorize(GREEN, message))

    def _warn(self, message: str) -> None:
        print(self._colorize(YELLOW, message))

    def _error(self, message: str) -> None:
        print(self._colorize(RED, message))

    def _note(self, message: str) -> None:
        print(self._colorize(BLUE, message))

    async def _repl(self) -> None:
        """Interactive command REPL."""
        print(f"\n{GREEN}Starting Neko manual CLI. Type 'help' for commands. Ctrl+D or 'quit' to exit.{RESET}")

        while self.running and self.client.is_connected():
            try:
                # Get user input (non-blocking)
                cmd = await asyncio.get_event_loop().run_in_executor(
                    None, input, f"{CYAN}neko> {RESET}"
                )

                if not cmd.strip():
                    continue

                await self._handle_command(cmd.strip())

            except EOFError:
                self.running = False
                break
            except Exception as e:
                logger.error("REPL error: %s", e)
                self._error(f"REPL error: {e}")

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
                self._error(f"Unknown command: {cmd}. Type 'help' for available commands.")

        except Exception as e:
            logger.error("Command error: %s", e)
            self._error(f"Error: {e}")

    async def _safe_send(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Send an event if the client is connected."""
        if not self.client.is_connected():
            self._error("Not connected to server")
            return

        try:
            await self.client.send_event(event, payload)
        except Exception as exc:
            logger.error("Failed to send event %s: %s", event, exc)
            self._error(f"Error sending event: {exc}")

    def _show_help(self) -> None:
        """Show help text."""
        help_text = (
            f"{BOLD}Available commands{RESET}\n\n"
            f"{BOLD}Movement & Clicking:{RESET}\n"
            f"  {GREEN}move X Y{RESET}                Move cursor to coordinates\n"
            f"  {GREEN}click [X Y] [button]{RESET}    Click at coordinates (or current position)\n"
            f"  {GREEN}rclick{RESET} | {GREEN}lclick{RESET}         Right or left click aliases\n"
            f"  {GREEN}dblclick [X Y]{RESET}          Double-click at coordinates\n"
            f"  {GREEN}tap X Y [button]{RESET}        Move and click at coordinates\n"
            f"  {GREEN}hover X Y{RESET}               Hover cursor at coordinates\n"
            f"  {GREEN}swipe X1 Y1 X2 Y2{RESET}       Drag from point 1 to point 2\n\n"
            f"{BOLD}Keyboard & Text:{RESET}\n"
            f"  {GREEN}key <keyname>{RESET}           Press a key (Escape, F5, etc.)\n"
            f"  {GREEN}enter{RESET}                   Press the Enter/Return key\n"
            f"  {GREEN}type <text>{RESET}             Type text at current focus\n"
            f"  {GREEN}text <text>{RESET}             Alias for type\n"
            f"  {GREEN}input X Y \"text\"{RESET}        Click at X,Y then type text\n"
            f"  {GREEN}copy{RESET} | {GREEN}cut{RESET} | {GREEN}select_all{RESET} Clipboard shortcuts\n"
            f"  {GREEN}paste [text]{RESET}            Paste clipboard or provided text\n\n"
            f"{BOLD}Raw / Admin:{RESET}\n"
            f"  {BLUE}raw '{{\"event\":...}}'{RESET}    Send raw JSON event\n"
            f"  {BLUE}host{RESET} / {BLUE}unhost{RESET}           Request or release host control\n"
            f"  {BLUE}force-take{RESET}              Force host control (admin)\n"
            f"  {BLUE}force-release{RESET}           Force release host control (admin)\n"
            f"  {BLUE}kick <sessionId>{RESET}        Kick a session (admin)\n"
            f"  {BLUE}sessions{RESET}                List sessions (requires REST login)\n\n"
            f"{BOLD}Other:{RESET}\n"
            f"  {YELLOW}size [WIDTH HEIGHT]{RESET}     Show or set screen dimensions\n"
            f"  {YELLOW}scroll dir [amount]{RESET}     Scroll direction (up/down/left/right)\n"
            f"  {YELLOW}help{RESET}                    Show this help\n"
            f"  {YELLOW}quit / exit / q{RESET}         Exit program\n\n"
            f"{BOLD}Coordinates:{RESET}\n"
            f"  - Use pixel coordinates (0,0 = top-left) unless --normalized\n"
            f"  - Current screen size: {self.width}x{self.height}\n"
        )
        print(help_text)

    def _action_executor(self):
        if not self.client.action_executor:
            raise RuntimeError("Action executor not ready")
        return self.client.action_executor

    def _parse_text_argument(self, args: List[str], usage: str) -> Optional[str]:
        """Join and normalize quoted text arguments."""
        if not args:
            self._warn(usage)
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
            self._warn("Usage: move X Y")
            return

        try:
            x, y = float(args[0]), float(args[1])
            px, py = self._xy(x, y)

            # Update cursor position
            self._curx, self._cury = px, py

            # Send move command via action executor
            await self._action_executor().move(px, py)
            self._info(f"Moved to ({px}, {py})")

        except ValueError:
            self._error("Invalid coordinates")

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
                self._error("Invalid coordinates")
                return
        else:
            px, py = self._curx, self._cury

        # Execute click
        if cmd == "hover":
            self._note(f"Hovering at ({px}, {py})")
        elif cmd == "dblclick":
            await self._action_executor().button_press(px, py, button)
            await asyncio.sleep(0.05)
            await self._action_executor().button_press(px, py, button)
            self._info(f"Double-clicked {button} at ({px}, {py})")
        else:
            await self._action_executor().button_press(px, py, button)
            self._info(f"Clicked {button} at ({px}, {py})")

    async def _handle_scroll(self, args: list) -> None:
        """Handle scroll command."""
        if not args:
            self._warn("Usage: scroll up|down|left|right [amount]")
            return

        direction = args[0].lower()
        try:
            amount = int(args[1]) if len(args) > 1 else 1
        except ValueError:
            self._error("Invalid scroll amount")
            return

        # Map direction to scroll deltas
        delta_map = {
            "down": (0, 120 * amount),
            "up": (0, -120 * amount),
            "right": (120 * amount, 0),
            "left": (-120 * amount, 0),
        }

        if direction not in delta_map:
            self._error("Invalid scroll direction. Use: up, down, left, right")
            return

        dx, dy = delta_map[direction]
        await self._action_executor().scroll(dx, dy)
        self._info(f"Scrolled {direction} (amount: {amount})")

    async def _handle_swipe(self, args: list) -> None:
        """Handle swipe command."""
        if len(args) < 4:
            self._warn("Usage: swipe X1 Y1 X2 Y2")
            return

        try:
            x1, y1 = float(args[0]), float(args[1])
            x2, y2 = float(args[2]), float(args[3])

            p1 = self._xy(x1, y1)
            p2 = self._xy(x2, y2)

            await self._action_executor().swipe(p1[0], p1[1], p2[0], p2[1])
            self._info(f"Swiped from {p1} to {p2}")

        except ValueError:
            self._error("Invalid coordinates")

    async def _handle_key(self, args: list) -> None:
        """Handle key command."""
        if not args:
            self._warn("Usage: key <keyname>")
            return

        key = " ".join(args)
        ks = name_keysym(key)
        if not ks:
            self._error(f"Unknown key: {key}")
            return

        await self._action_executor().key_once(key)
        self._info(f"Pressed key: {key}")

    async def _handle_type(self, args: list) -> None:
        """Handle type command."""
        text = self._parse_text_argument(args, "Usage: type <text>")
        if text is None:
            return

        await self._action_executor().type_text(text)
        self._info(f"Typed: {text}")

    async def _handle_enter(self) -> None:
        """Handle enter command."""
        await self._action_executor().key_once("Enter")
        self._info("Pressed Enter")

    async def _handle_input(self, args: List[str]) -> None:
        """Handle input command to click and type text."""
        if len(args) < 3:
            self._warn("Usage: input X Y \"text\"")
            return

        try:
            x, y = float(args[0]), float(args[1])
        except ValueError:
            self._error("Invalid coordinates")
            return

        text = self._parse_text_argument(args[2:], "Usage: input X Y \"text\"")
        if text is None:
            return

        px, py = self._xy(x, y)
        await self._action_executor().move(px, py)
        await self._action_executor().button_press(px, py, "left")
        await asyncio.sleep(0.05)
        await self._action_executor().type_text(text)
        self._info(f"Clicked ({px}, {py}) and typed: {text}")

    async def _handle_clipboard(self, cmd: str) -> None:
        """Handle clipboard-related commands."""
        executor = self._action_executor()

        if cmd == "copy":
            await executor.copy()
            self._info("Copied selection")
            return

        if cmd == "cut":
            await executor.key_down("Control")
            await executor.key_once("x")
            await executor.key_up("Control")
            self._info("Cut selection")
            return

        if cmd == "select_all":
            await executor.key_down("Control")
            await executor.key_once("a")
            await executor.key_up("Control")
            self._info("Selected all")
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
            self._info(f"Pasted text: {text}")
        else:
            self._info("Requested paste from clipboard")

    async def _handle_raw(self, args: List[str]) -> None:
        """Send raw JSON message."""
        if not args:
            self._warn("Usage: raw '{\"event\":..., \"payload\":{...}}'")
            return

        raw = " ".join(args)
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._error(f"Invalid JSON: {exc}")
            return

        if not isinstance(message, dict) or "event" not in message:
            self._error("JSON must include 'event'")
            return

        event = message["event"]
        payload = message.get("payload")
        if payload is not None and not isinstance(payload, dict):
            self._error("payload must be an object if provided")
            return

        await self._safe_send(event, payload)
        self._info(f"Sent raw event: {event}")

    async def _handle_force(self, take: bool) -> None:
        """Handle force host commands."""
        session_id = self.client.session_id
        if not session_id:
            self._warn("Session ID unknown; wait for connection to stabilize")
            return

        event = "admin/control" if take else "admin/release"
        await self._safe_send(event, {"id": session_id})
        self._info(("Force-take" if take else "Force-release") + " sent")

    async def _handle_kick(self, args: List[str]) -> None:
        """Handle kick command."""
        if not args:
            self._warn("Usage: kick <sessionId>")
            return

        target = args[0]
        await self._safe_send("admin/kick", {"id": target})
        self._info(f"Kick sent for session {target}")

    async def _handle_sessions(self) -> None:
        """List active sessions via REST API."""
        if not self.base_url or not self.token:
            self._warn("Sessions command requires REST login (provide --neko-url credentials)")
            return

        url = f"{self.base_url}/api/sessions"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=10.0)
            response.raise_for_status()
        except Exception as exc:
            logger.error("Failed to fetch sessions: %s", exc)
            self._error(f"Error fetching sessions: {exc}")
            return

        try:
            data = response.json()
        except ValueError as exc:
            self._error(f"Invalid JSON from sessions endpoint: {exc}")
            return

        if not data:
            self._warn("No active sessions")
            return

        if isinstance(data, list):
            print(f"{BOLD}Sessions:{RESET}")
            for sess in data:
                sid = sess.get("id") or sess.get("session_id")
                user = sess.get("username") or sess.get("user", {}).get("username")
                has_host = sess.get("has_host") or sess.get("host")
                print(f"  {GREEN}{sid}{RESET} - user={user} host={has_host}")
        else:
            self._note(json.dumps(data, indent=2, ensure_ascii=False))

    async def _handle_size(self, args: list) -> None:
        """Handle size command."""
        if not args:
            self._note(f"Current size: {self.width}x{self.height} normalized={self.normalized}")
            return

        if len(args) < 2:
            self._warn("Usage: size WIDTH HEIGHT")
            return

        try:
            width = int(args[0])
            height = int(args[1])
        except ValueError:
            self._error("Invalid dimensions")
            return

        self.width = width
        self.height = height
        self.client.set_screen_size(width, height)
        self._info(f"Screen size set to {width}x{height}")


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
            print(f"{RED}Login failed: {e}{RESET}")
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
                print(f"{RED}Login failed: {e}{RESET}")
                sys.exit(1)
        else:
            print(f"{RED}Error: Must provide either --ws URL or --neko-url + credentials{RESET}")
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