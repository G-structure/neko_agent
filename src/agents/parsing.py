"""Shared parsing utilities for vision agents.

This module provides robust JSON/dict parsing for LLM outputs,
supporting multiple formats including:
- ShowUI JSON format (standard JSON with validation)
- Qwen tool calling format (OpenRouter/Anthropic computer use tools)
- Python-style dicts with single quotes
- Python literals (None, True, False)
- ast.literal_eval fallback for edge cases

Moved from neko_comms/actions.py to centralize all parsing logic.
"""

import ast
import json
import logging
from typing import Any, Dict, List, Optional


COMPLETION_ACTIONS = {"DONE", "ANSWER"}


def safe_parse_action(
    output_text: str,
    nav_mode: str = "web",
    logger: Optional[logging.Logger] = None,
    parse_errors_counter=None,
    action_spaces: Optional[Dict[str, List[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Parse and validate LLM output text into a structured action dictionary.

    This is the main parsing function used by ShowUI and other text-based agents.
    Supports robust multi-level fallback parsing with validation against action space.

    :param output_text: The raw output text from the LLM to parse
    :param nav_mode: The navigation mode ('web' or 'phone') which determines allowed actions
    :param logger: Optional logger instance for recording parse errors
    :param parse_errors_counter: Optional metrics counter for parse errors
    :param action_spaces: Optional action space dict (defaults to importing from neko_comms.types)
    :return: A validated action dictionary or None if parsing/validation fails
    """
    # Import ACTION_SPACES if not provided (avoid circular import)
    if action_spaces is None:
        try:
            from neko_comms.types import ACTION_SPACES
            action_spaces = ACTION_SPACES
        except ImportError:
            # Fallback if types not available
            action_spaces = {
                "web": ["CLICK", "INPUT", "ENTER", "SCROLL", "HOVER", "SELECT", "TAP"],
                "phone": ["TAP", "SWIPE", "INPUT", "SCROLL"],
            }

    # Parse the raw text using robust multi-level parsing
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
                    logger.error("Parse error: %s | Raw=%r", e, output_text[:200])
                if parse_errors_counter:
                    parse_errors_counter.inc()
                return None

    # Handle arrays of actions (take first one)
    if isinstance(act, (tuple, list)) and len(act) > 0:
        if logger:
            logger.info("Model returned multiple actions, using first one: %r", act)
        act = act[0]

    # Validate action structure
    try:
        assert isinstance(act, dict)
        typ = act.get("action")
        # Normalize action to uppercase for validation
        if isinstance(typ, str):
            typ = typ.upper()
            act["action"] = typ
        if typ in COMPLETION_ACTIONS:
            act.setdefault("value", None)
            act.setdefault("position", None)
        elif typ not in action_spaces.get(nav_mode, []):
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


def parse_tool_call(
    tool_call: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Parse Qwen/Anthropic tool call response to ShowUI action format.

    Converts Qwen's computer use tool call format to our action string format.
    This is used by remote agents (OpenRouter, Anthropic API, etc).

    :param tool_call: Tool call dictionary from API
    :param logger: Optional logger instance
    :return: Formatted action JSON string or None on error
    """
    try:
        function = tool_call["function"]
        args = json.loads(function["arguments"])

        action_type = args["action"]
        coord = args.get("coordinate", [500, 500])  # Default center
        text = args.get("text")

        # Convert Qwen action types to ShowUI format
        action_map = {
            "left_click": "CLICK",
            "double_click": "CLICK",
            "right_click": "CLICK",
            "type": "INPUT",
            "key": "ENTER",
            "mouse_move": "HOVER",
        }

        mapped_action = action_map.get(action_type, "CLICK")

        # Build action dict in ShowUI format
        action_dict = {
            "action": mapped_action,
            "value": text if text else None,
            "position": _convert_qwen_coordinate(coord),
        }

        # Return as JSON string
        return json.dumps(action_dict)

    except (KeyError, json.JSONDecodeError) as e:
        if logger:
            logger.error("Failed to parse tool call: %s", e)
        return None


def parse_text_response(
    content: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Parse text response as fallback for non-tool-calling models.

    Handles models that don't support tool calling by extracting and parsing
    JSON from text content. Uses robust multi-level parsing.

    :param content: Text content from API
    :param logger: Optional logger instance
    :return: Action JSON string or None
    """
    if logger:
        logger.debug("Raw text response: %r", content[:500])

    # Try to extract JSON from text - look for {...} pattern
    json_str = extract_json_from_text(content)
    if not json_str:
        if logger:
            logger.warning("No JSON object found in text response")
        return None

    if logger:
        logger.debug("Extracted JSON string: %r", json_str[:200])

    # Try standard JSON parsing first
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed)
    except json.JSONDecodeError:
        pass

    # Fallback 1: Normalize Python-style to JSON
    try:
        normalized = json_str.replace("'", '"')
        normalized = (
            normalized
            .replace('None', 'null')
            .replace('True', 'true')
            .replace('False', 'false')
        )
        parsed = json.loads(normalized)
        return json.dumps(parsed)
    except json.JSONDecodeError:
        pass

    # Fallback 2: Use ast.literal_eval for Python-style dicts
    try:
        parsed = ast.literal_eval(json_str)
        if isinstance(parsed, dict):
            return json.dumps(parsed)
        else:
            if logger:
                logger.warning("AST parsing returned non-dict: %r", type(parsed))
            return None
    except (ValueError, SyntaxError) as e:
        if logger:
            logger.error("Failed to parse text response: %s", e)
        return None


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text containing other content.

    Looks for the first {...} pattern in the text.

    :param text: Text potentially containing JSON
    :return: Extracted JSON string or None
    """
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return None


def _convert_qwen_coordinate(coord: List[float]) -> List[float]:
    """Convert Qwen's 1000-based coordinate to 0-1 normalized.

    Qwen returns coordinates in [0, 1000] range.
    We need [0, 1] range for our system.

    :param coord: [x, y] in 0-1000 range
    :return: [x, y] in 0-1 range
    """
    return [coord[0] / 1000.0, coord[1] / 1000.0]