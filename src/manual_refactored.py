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
import logging
import os
import shlex
import sys
from typing import Dict, Any, Tuple, Optional

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
                 normalized: bool = False, auto_host: bool = True):
        """Initialize the manual controller.

        :param ws_url: WebSocket URL for Neko connection
        :param width: Virtual screen width
        :param height: Virtual screen height
        :param normalized: If True, coordinates are 0-1 and scaled
        :param auto_host: If True, automatically request host control
        """
        self.ws_url = ws_url
        self.width = width
        self.height = height
        self.normalized = normalized
        self.auto_host = auto_host

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

            elif cmd == "key":
                await self._handle_key(args)

            elif cmd == "type":
                await self._handle_type(args)

            elif cmd == "host":
                await self.client.request_host_control()

            elif cmd == "size":
                await self._handle_size(args)

            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")

        except Exception as e:
            logger.error("Command error: %s", e)
            print(f"Error: {e}")

    def _show_help(self) -> None:
        """Show help text."""
        help_text = """
Available commands:

Movement & Clicking:
  move X Y                - Move cursor to coordinates
  click [X Y] [button]    - Click at coordinates (or current position)
  rclick [X Y]           - Right-click at coordinates
  lclick [X Y]           - Left-click at coordinates
  dblclick [X Y]         - Double-click at coordinates
  tap X Y                - Touch/tap at coordinates
  hover X Y              - Hover at coordinates

Scrolling & Gestures:
  scroll up|down|left|right [amount]  - Scroll in direction
  swipe X1 Y1 X2 Y2      - Swipe from point 1 to point 2

Keyboard:
  key <keyname>          - Press a key (e.g., "Enter", "Escape", "F1")
  type <text>            - Type text string

Control:
  host                   - Request host control
  size WIDTH HEIGHT      - Set screen dimensions
  help                   - Show this help
  quit / exit / q        - Exit program

Coordinates:
  - Use pixel coordinates (0,0 = top-left)
  - Screen size: {width}x{height}
""".format(width=self.width, height=self.height)
        print(help_text)

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
            await self.client.action_executor.move(px, py)
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
                await self.client.action_executor.move(px, py)
            except ValueError:
                print("Invalid coordinates")
                return
        else:
            px, py = self._curx, self._cury

        # Execute click
        if cmd == "hover":
            print(f"Hovering at ({px}, {py})")
        elif cmd == "dblclick":
            await self.client.action_executor.button_press(px, py, button)
            await asyncio.sleep(0.05)
            await self.client.action_executor.button_press(px, py, button)
            print(f"Double-clicked {button} at ({px}, {py})")
        else:
            await self.client.action_executor.button_press(px, py, button)
            print(f"Clicked {button} at ({px}, {py})")

    async def _handle_scroll(self, args: list) -> None:
        """Handle scroll command."""
        if not args:
            print("Usage: scroll up|down|left|right [amount]")
            return

        direction = args[0].lower()
        amount = int(args[1]) if len(args) > 1 else 1

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
        await self.client.action_executor.scroll(dx, dy)
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

            await self.client.action_executor.swipe(p1[0], p1[1], p2[0], p2[1])
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

        await self.client.action_executor.key_once(key)
        print(f"Pressed key: {key}")

    async def _handle_type(self, args: list) -> None:
        """Handle type command."""
        if not args:
            print("Usage: type <text>")
            return

        text = " ".join(args)

        # Remove surrounding quotes if present
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1]

        await self.client.action_executor.type_text(text)
        print(f"Typed: {text}")

    async def _handle_size(self, args: list) -> None:
        """Handle size command."""
        if len(args) < 2:
            print("Usage: size WIDTH HEIGHT")
            return

        try:
            width = int(args[0])
            height = int(args[1])

            self.width = width
            self.height = height
            self.client.set_screen_size(width, height)

            print(f"Screen size set to {width}x{height}")

        except ValueError:
            print("Invalid dimensions")


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

    # Determine WebSocket URL
    if args.ws:
        ws_url = args.ws
    elif args.neko_url and args.username and args.password:
        try:
            ws_url, _, _ = rest_login_and_ws_url(args.neko_url, args.username, args.password)
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
                ws_url, _, _ = rest_login_and_ws_url(neko_url, username, password)
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
        auto_host=not args.no_auto_host
    )

    await controller.run()


if __name__ == "__main__":
    asyncio.run(main())