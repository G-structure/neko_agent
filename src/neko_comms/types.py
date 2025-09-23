"""Shared types and constants for Neko communication.

This module defines common data structures, protocol constants,
and type definitions used across the Neko communication library.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum


# ----------------------
# Protocol constants
# ----------------------

BUTTON_CODES = {"left": 1, "middle": 2, "right": 3}

# Base X11-style keysyms; will be augmented dynamically via `keyboard/map`.
KEYSYM: Dict[str, int] = {
    # ASCII and controls
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

ALLOWED_ACTIONS = {
    "CLICK","INPUT","SELECT","HOVER","ANSWER","ENTER","SCROLL","SELECT_TEXT","COPY",
    "SWIPE","TAP"
}

ACTION_SPACES = {
    "web":   ["CLICK","INPUT","SELECT","HOVER","ANSWER","ENTER","SCROLL","SELECT_TEXT","COPY"],
    "phone": ["INPUT","SWIPE","TAP","ANSWER","ENTER"],
}

ACTION_SPACE_DESC = {
    "web": """
1. CLICK: Click an element, value=None, position=[x, y].
2. INPUT: Type a string into an element, value=string, position=[x, y].
3. SELECT: Select a value for an element, value=None, position=[x, y].
4. HOVER: Hover on an element, value=None, position=[x, y].
5. ANSWER: Answer a question, value=string, position=None.
6. ENTER: Enter, value=None, position=None.
7. SCROLL: Scroll the screen, value=direction (e.g. "down"), position=None.
8. SELECT_TEXT: Select text, value=None, position=[[x1, y1], [x2, y2]].
9. COPY: Copy text, value=string, position=None.
""",
    "phone": """
1. INPUT: Type a string into an element, value=string, position=[x, y].
2. SWIPE: Swipe the screen, value=None, position=[[x1, y1], [x2, y2]].
3. TAP: Tap on an element, value=None, position=[x, y].
4. ANSWER: Answer a question, value=string, position=None.
5. ENTER: Enter, value=None, position=None.
"""
}


# ----------------------
# Data types
# ----------------------

@dataclass
class Action:
    """High-level action representation for Neko automation.

    Actions describe user interactions like clicks, text input,
    scrolling, and other UI operations that can be executed
    on a remote Neko session.

    :param action: The action type (e.g., "CLICK", "INPUT", "SCROLL")
    :type action: str
    :param value: Optional action value (e.g., text for INPUT)
    :type value: Optional[Union[str, int, float]]
    :param position: Normalized coordinates for position-based actions
    :type position: Optional[Union[List[float], List[List[float]]]]
    :param amount: Optional amount for scroll actions
    :type amount: Optional[int]
    """
    action: str
    value: Optional[Union[str, int, float]] = None
    position: Optional[Union[List[float], List[List[float]]]] = None
    amount: Optional[int] = None


@dataclass
class Frame:
    """Video frame data from Neko session.

    Represents a captured frame from the WebRTC video stream,
    including raw image data and metadata.

    :param data: Raw frame data (PIL Image or bytes)
    :type data: Any
    :param timestamp: Frame timestamp in seconds
    :type timestamp: float
    :param width: Frame width in pixels
    :type width: int
    :param height: Frame height in pixels
    :type height: int
    """
    data: Any
    timestamp: float
    width: int
    height: int


@dataclass
class IceCandidate:
    """WebRTC ICE candidate information.

    Represents an ICE (Interactive Connectivity Establishment)
    candidate used for WebRTC peer-to-peer connections.

    :param candidate: ICE candidate string
    :type candidate: str
    :param sdp_mid: SDP media ID
    :type sdp_mid: Optional[str]
    :param sdp_mline_index: SDP media line index
    :type sdp_mline_index: Optional[int]
    """
    candidate: str
    sdp_mid: Optional[str] = None
    sdp_mline_index: Optional[int] = None


class EventType(Enum):
    """Neko server event types for message routing."""
    SIGNAL = "signal"
    SYSTEM = "system"
    CHAT = "chat"
    CONTROL = "control"
    ICE = "ice"
    VIDEO = "video"
    MISC = "misc"


class ConnectionState(Enum):
    """WebRTC connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


# ----------------------
# Utility functions
# ----------------------

def name_keysym(name: str) -> int:
    """Convert a key name string to an X11 keysym integer code.

    This function maps key names to their corresponding X11 keysym values. For
    single character inputs, it first checks the KEYSYM dictionary directly,
    then with the lowercase version, and finally falls back to the character's
    ASCII value. For multi-character strings, it searches the KEYSYM dictionary
    with the original name and a capitalized version as fallbacks.

    :param name: The key name or character to convert. Can be a single character
                 (e.g., 'a', 'A', ' ') or a multi-character key name
                 (e.g., 'Return', 'Escape', 'F1').
    :type name: str
    :return: The corresponding X11 keysym integer, or 0 if not found.
    :rtype: int
    """
    if len(name) == 1:
        return KEYSYM.get(name, KEYSYM.get(name.lower(), ord(name)))
    return KEYSYM.get(name, KEYSYM.get(name.capitalize(), 0))


def clamp_xy(x: int, y: int, size: Tuple[int, int]) -> Tuple[int, int]:
    """Clamp x,y coordinates to stay within screen boundaries.

    This function ensures that the provided coordinates remain within valid
    screen bounds. The coordinates are clamped to the range [0, width-1] for
    x and [0, height-1] for y, preventing out-of-bounds mouse/click operations.

    :param x: The x coordinate to clamp.
    :type x: int
    :param y: The y coordinate to clamp.
    :type y: int
    :param size: Screen dimensions as (width, height).
    :type size: Tuple[int, int]
    :return: Clamped coordinates as (x, y).
    :rtype: Tuple[int, int]
    """
    w, h = size
    return max(0, min(x, w-1)), max(0, min(y, h-1))