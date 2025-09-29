"""Action execution primitives for Neko remote control.

This module provides unified action execution interfaces for
interacting with Neko sessions, including mouse movements,
keyboard input, and high-level action interpretation.
"""

import ast
import asyncio
import json
import logging
from typing import Callable, Tuple, List, Optional, Dict, Any

from .types import Action, BUTTON_CODES, name_keysym, clamp_xy, ACTION_SPACES


COMPLETION_ACTIONS = {"DONE"}

logger = logging.getLogger(__name__)


def safe_parse_action(output_text: str, nav_mode: str = "web", logger: Optional[logging.Logger] = None, parse_errors_counter=None) -> Optional[Dict[str, Any]]:
    """Parse and validate LLM output text into a structured action dictionary.

    :param output_text: The raw output text from the LLM to parse
    :param nav_mode: The navigation mode ('web' or 'phone') which determines allowed actions
    :param logger: Optional logger instance for recording parse errors
    :param parse_errors_counter: Optional metrics counter for parse errors
    :return: A validated action dictionary or None if parsing/validation fails
    """
    try:
        act = json.loads(output_text)
    except json.JSONDecodeError:
        try:
            normalized = output_text.replace("'", '"')
            normalized = (
                normalized
                .replace('None', 'null')
                .replace('True', 'true')
                .replace('False', 'false')
            )
            act = json.loads(normalized)
        except Exception:
            try:
                act = ast.literal_eval(output_text)
            except (ValueError, SyntaxError) as e:
                if logger:
                    logger.error("Parse error: %s | Raw=%r", e, output_text)
                if parse_errors_counter:
                    parse_errors_counter.inc()
                return None

    if isinstance(act, (tuple, list)) and len(act) > 0:
        if logger:
            logger.info("Model returned multiple actions, using first one: %r", act)
        act = act[0]

    try:
        assert isinstance(act, dict)
        typ = act.get("action")
        if typ in COMPLETION_ACTIONS:
            act.setdefault("value", None)
            act.setdefault("position", None)
        elif typ not in ACTION_SPACES.get(nav_mode, []):
            if logger:
                logger.warning("Non-whitelisted action: %r", typ)
            if parse_errors_counter:
                parse_errors_counter.inc()
            return None
        for k in ("action", "value", "position"):
            assert k in act, f"Missing key {k}"
        return act
    except AssertionError as e:
        if logger:
            logger.error("Schema validation error: %s | Parsed=%r", e, act)
        if parse_errors_counter:
            parse_errors_counter.inc()
        return None


class ActionExecutor:
    """Executes high-level actions on a Neko session via WebSocket signaling.

    This class translates Action objects into low-level control commands
    (mouse movements, clicks, keyboard input, etc.) and sends them via
    the provided send function.
    """

    def __init__(self, send_func: Callable[[Dict[str, Any]], None], chat_func: Optional[Callable[[str], None]] = None):
        """Initialize the action executor.

        :param send_func: Function to send WebSocket messages
        :type send_func: Callable[[Dict[str, Any]], None]
        :param chat_func: Optional function to send chat messages for action logging
        :type chat_func: Optional[Callable[[str], None]]
        """
        self.send_func = send_func
        self.chat_func = chat_func

    async def execute_action(self, action: Action, frame_size: Tuple[int, int]) -> None:
        """Execute a high-level action on the remote screen.

        This method translates high-level action objects into low-level
        control commands and sends them via the WebSocket connection.

        :param action: Action to execute (click, type, scroll, etc.)
        :type action: Action
        :param frame_size: Screen dimensions as (width, height)
        :type frame_size: Tuple[int, int]
        :return: None
        :rtype: None
        """
        def to_xy(norm_pt: List[float]) -> Tuple[int, int]:
            """Convert normalized coordinates to pixel coordinates."""
            x = int(float(norm_pt[0]) * frame_size[0])
            y = int(float(norm_pt[1]) * frame_size[1])
            return clamp_xy(x, y, frame_size)

        try:
            # Log action for visibility
            if self.chat_func:
                action_preview = {
                    "action": action.action,
                    "value": action.value if isinstance(action.value, (str, int, float)) else None,
                    "position": action.position,
                }
                await self._safe_chat(f"Action: {json.dumps(action_preview, ensure_ascii=False)}")

            # Execute the action based on type
            if action.action in {"CLICK", "TAP", "SELECT", "HOVER"} and isinstance(action.position, list) and len(action.position) == 2:
                x, y = to_xy(action.position)
                await self.move(x, y)
                if action.action in {"CLICK", "TAP", "SELECT"}:
                    btn = "left"
                    if isinstance(action.value, str) and action.value.lower() in BUTTON_CODES:
                        btn = action.value.lower()
                    await self.button_press(x, y, btn)
                return

            if action.action == "INPUT" and action.value and isinstance(action.position, list) and len(action.position) == 2:
                x, y = to_xy(action.position)
                await self.move(x, y)
                await self.button_press(x, y, "left")
                for ch in str(action.value):
                    await self.key_once("Enter" if ch == "\n" else ch)
                return

            if action.action == "ENTER":
                await self.key_once("Enter")
                return

            if action.action == "SCROLL" and action.value:
                direction = str(action.value).lower()
                amount = action.amount or 1
                try:
                    amount = int(amount)
                except Exception:
                    amount = 1
                delta_map = {
                    "down": (0, 120 * amount),
                    "up": (0, -120 * amount),
                    "right": (120 * amount, 0),
                    "left": (-120 * amount, 0),
                }
                dx, dy = delta_map.get(direction, (0, 0))
                if dx or dy:
                    await self.scroll(dx, dy)
                return

            if action.action == "SWIPE" and isinstance(action.position, list) and len(action.position) == 2 and all(isinstance(p, list) and len(p) == 2 for p in action.position):
                x1, y1 = to_xy(action.position[0])
                x2, y2 = to_xy(action.position[1])
                await self.swipe(x1, y1, x2, y2)
                return

            if action.action == "SELECT_TEXT" and isinstance(action.position, list) and len(action.position) == 2 and all(isinstance(p, list) and len(p) == 2 for p in action.position):
                x1, y1 = to_xy(action.position[0])
                x2, y2 = to_xy(action.position[1])
                await self.select_text(x1, y1, x2, y2)
                return

            if action.action == "COPY":
                await self.copy()
                if self.chat_func:
                    await self._safe_chat("Copied selection to clipboard.")
                return

            if action.action == "ANSWER":
                logger.info("[ANSWER] %r", action.value)
                if self.chat_func:
                    await self._safe_chat(f"Answer: {action.value}")
                return

            logger.warning("Unsupported or malformed action: %r", action)

        except Exception as e:
            logger.error("Action execution failed: %s | action=%r", e, action, exc_info=True)

    async def move(self, x: int, y: int) -> None:
        """Move mouse cursor to coordinates.

        :param x: X coordinate
        :type x: int
        :param y: Y coordinate
        :type y: int
        :return: None
        :rtype: None
        """
        await self.send_func({"event": "control/move", "payload": {"x": x, "y": y}})

    async def button_press(self, x: int, y: int, button: str = "left") -> None:
        """Press and release mouse button at coordinates.

        :param x: X coordinate
        :type x: int
        :param y: Y coordinate
        :type y: int
        :param button: Button name ("left", "right", "middle")
        :type button: str
        :return: None
        :rtype: None
        """
        code = BUTTON_CODES.get(button, 1)
        await self.send_func({"event": "control/buttonpress", "payload": {"x": x, "y": y, "code": code}})

    async def button_down(self, x: int, y: int, button: str = "left") -> None:
        """Press down mouse button at coordinates.

        :param x: X coordinate
        :type x: int
        :param y: Y coordinate
        :type y: int
        :param button: Button name ("left", "right", "middle")
        :type button: str
        :return: None
        :rtype: None
        """
        code = BUTTON_CODES.get(button, 1)
        await self.send_func({"event": "control/buttondown", "payload": {"x": x, "y": y, "code": code}})

    async def button_up(self, x: int, y: int, button: str = "left") -> None:
        """Release mouse button at coordinates.

        :param x: X coordinate
        :type x: int
        :param y: Y coordinate
        :type y: int
        :param button: Button name ("left", "right", "middle")
        :type button: str
        :return: None
        :rtype: None
        """
        code = BUTTON_CODES.get(button, 1)
        await self.send_func({"event": "control/buttonup", "payload": {"x": x, "y": y, "code": code}})

    async def key_once(self, name_or_char: str) -> None:
        """Press and release a key.

        :param name_or_char: Key name or character
        :type name_or_char: str
        :return: None
        :rtype: None
        """
        ks = name_keysym(name_or_char)
        if ks:
            await self.send_func({"event": "control/keypress", "payload": {"keysym": ks}})

    async def key_down(self, name_or_char: str) -> None:
        """Press down a key.

        :param name_or_char: Key name or character
        :type name_or_char: str
        :return: None
        :rtype: None
        """
        ks = name_keysym(name_or_char)
        if ks:
            await self.send_func({"event": "control/keydown", "payload": {"keysym": ks}})

    async def key_up(self, name_or_char: str) -> None:
        """Release a key.

        :param name_or_char: Key name or character
        :type name_or_char: str
        :return: None
        :rtype: None
        """
        ks = name_keysym(name_or_char)
        if ks:
            await self.send_func({"event": "control/keyup", "payload": {"keysym": ks}})

    async def scroll(self, delta_x: int, delta_y: int) -> None:
        """Scroll the screen.

        :param delta_x: Horizontal scroll amount
        :type delta_x: int
        :param delta_y: Vertical scroll amount
        :type delta_y: int
        :return: None
        :rtype: None
        """
        await self.send_func({"event": "control/scroll", "payload": {"delta_x": delta_x, "delta_y": delta_y}})

    async def swipe(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Perform a swipe gesture from (x1, y1) to (x2, y2).

        :param x1: Start X coordinate
        :type x1: int
        :param y1: Start Y coordinate
        :type y1: int
        :param x2: End X coordinate
        :type x2: int
        :param y2: End Y coordinate
        :type y2: int
        :return: None
        :rtype: None
        """
        await self.move(x1, y1)
        await self.button_down(x1, y1, "left")
        await asyncio.sleep(0.05)
        await self.move(x2, y2)
        await self.button_up(x2, y2, "left")

    async def select_text(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Select text from (x1, y1) to (x2, y2).

        :param x1: Start X coordinate
        :type x1: int
        :param y1: Start Y coordinate
        :type y1: int
        :param x2: End X coordinate
        :type x2: int
        :param y2: End Y coordinate
        :type y2: int
        :return: None
        :rtype: None
        """
        await self.move(x1, y1)
        await self.button_down(x1, y1, "left")
        await self.move(x2, y2)
        await self.button_up(x2, y2, "left")

    async def copy(self) -> None:
        """Copy selected text to clipboard.

        :return: None
        :rtype: None
        """
        await self.key_down("Control")
        await self.key_once("c")
        await self.key_up("Control")

    async def type_text(self, text: str) -> None:
        """Type a string of text.

        :param text: Text to type
        :type text: str
        :return: None
        :rtype: None
        """
        for ch in text:
            await self.key_once("Enter" if ch == "\n" else ch)

    async def _safe_chat(self, message: str) -> None:
        """Safely send a chat message if chat function is available.

        :param message: Message to send
        :type message: str
        :return: None
        :rtype: None
        """
        if self.chat_func:
            try:
                await self.chat_func(message)
            except Exception as e:
                logger.warning("Failed to send chat message: %s", e)