#!/usr/bin/env python3
"""
agent.py - Production-ready ShowUI-2B Neko v3 WebRTC GUI agent with deterministic shutdown.

12-Factor App compliant refactor:
- Centralized configuration via Settings dataclass
- All side effects moved to main() (no side effects in imports/init)
- JSON logging support via NEKO_LOG_FORMAT environment variable
- $PORT takes priority over NEKO_METRICS_PORT for metrics server
- Frame/click saving uses /tmp/neko-agent/ for relative paths with explicit logging
- Added --healthcheck CLI flag for configuration validation
- Preserved all existing CLI flags and behavior including on-demand host control

Highlights (Python 3.11+ / 3.13-ready):
- TaskGroup lifecycle: all child tasks cancelled/awaited for prompt exit.
- WebSocket client: tidy read/send loops; await close() and wait_closed() when available.
- aiortc teardown: stop receivers/transceivers & tracks before RTCPeerConnection.close().
- Early ICE handling: buffer remote ICE candidates before SRD, then apply after SRD.
- Lite mode (WS-video): nav loop stops promptly when WS disconnects.
- Executor shutdown: wait + cancel_futures to prevent lingering threads.
- Optional RTCP keepalive (NEKO_RTCP_KEEPALIVE=1) to stabilize some NAT/TURN links.
- Prometheus metrics server started once and shut down cleanly.
- Optional hard-exit guard (NEKO_FORCE_EXIT_GUARD_MS>0) to terminate if non-daemon threads linger.

"""

from __future__ import annotations

# --- stdlib
import os
import sys
import io
import ast
import json
import time
import base64
import signal
import random
import uuid
import tempfile
import asyncio
import logging
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from abc import ABC, abstractmethod
from dataclasses import dataclass

# --- third-party
import torch
import requests
import websockets
from PIL import Image, ImageFile, ImageDraw, ImageFont
from prometheus_client import start_http_server, Counter, Histogram
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# aiortc for SDP/ICE handshake and media
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    VideoStreamTrack,
    MediaStreamError,  # clean end-of-stream signal
)
from aiortc.sdp import candidate_from_sdp

# Fail fast on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Safe module-level logger (handlers/level are set in setup_logging())
logger = logging.getLogger("neko_agent")

# ----------------------
# Constants (for backward compatibility with default parameters)
# ----------------------
MAX_STEPS = 8
REFINEMENT_STEPS = 5
DEFAULT_METRIC_PORT = 9000
AUDIO_DEFAULT = True

# ----------------------
# Configuration Settings
# ----------------------
@dataclass
class Settings:
    """Centralized configuration settings loaded from environment variables.

    This class follows 12-Factor App principles by reading all configuration
    from environment variables once at startup rather than throughout the code.
    """
    # Model configuration
    repo_id: str
    size_shortest_edge: int
    size_longest_edge: int

    # Network configuration
    default_ws: str
    neko_ice_policy: str
    neko_stun_url: str
    neko_turn_url: Optional[str]
    neko_turn_user: Optional[str]
    neko_turn_pass: Optional[str]

    # Agent behavior
    max_steps: int
    audio_default: bool
    refinement_steps: int
    neko_rtcp_keepalive: bool
    force_exit_guard_ms: int
    neko_skip_initial_frames: int

    # Logging configuration
    log_level: str
    log_file: Optional[str]
    log_format: str  # 'text' or 'json'

    # Paths and storage
    frame_save_path: Optional[str]
    click_save_path: Optional[str]
    offload_folder: str

    # Metrics configuration
    metrics_port: int  # Uses $PORT with fallback to NEKO_METRICS_PORT

    # Other settings
    run_id: Optional[str]

    @classmethod
    def from_env(cls) -> 'Settings':
        """Load settings from environment variables.

        Uses $PORT with fallback to NEKO_METRICS_PORT for metrics server port.
        Resolves relative paths for frame/click saving to use /tmp/neko-agent/.

        :return: Settings instance populated from environment variables
        :rtype: Settings
        """
        # Handle metrics port: $PORT takes priority over NEKO_METRICS_PORT
        port = os.environ.get("PORT")
        if port is not None:
            try:
                metrics_port = int(port)
            except ValueError:
                metrics_port = int(os.environ.get("NEKO_METRICS_PORT", "9000"))
        else:
            metrics_port = int(os.environ.get("NEKO_METRICS_PORT", "9000"))

        # Handle frame/click save paths - use /tmp/neko-agent/ for relative paths
        frame_save_path = os.environ.get("FRAME_SAVE_PATH")
        click_save_path = os.environ.get("CLICK_SAVE_PATH")

        if frame_save_path and not os.path.isabs(frame_save_path):
            frame_save_path = f"/tmp/neko-agent/{frame_save_path}"

        if click_save_path and not os.path.isabs(click_save_path):
            click_save_path = f"/tmp/neko-agent/{click_save_path}"

        return cls(
            repo_id=os.environ.get("REPO_ID", "showlab/ShowUI-2B"),
            size_shortest_edge=int(os.environ.get("SIZE_SHORTEST_EDGE", "224")),
            size_longest_edge=int(os.environ.get("SIZE_LONGEST_EDGE", "1344")),
            default_ws=os.environ.get("NEKO_WS", "wss://neko.example.com/api/ws"),
            max_steps=int(os.environ.get("NEKO_MAX_STEPS", "8")),
            audio_default=bool(int(os.environ.get("NEKO_AUDIO", "1"))),
            frame_save_path=frame_save_path,
            click_save_path=click_save_path,
            offload_folder=os.environ.get("OFFLOAD_FOLDER", "./offload"),
            refinement_steps=int(os.environ.get("REFINEMENT_STEPS", "5")),
            log_file=os.environ.get("NEKO_LOGFILE"),
            log_level=os.environ.get("NEKO_LOGLEVEL", "INFO"),
            log_format=os.environ.get("NEKO_LOG_FORMAT", "text").lower(),
            neko_ice_policy=os.environ.get("NEKO_ICE_POLICY", "strict"),
            neko_stun_url=os.environ.get("NEKO_STUN_URL", "stun:stun.l.google.com:19302"),
            neko_turn_url=os.environ.get("NEKO_TURN_URL"),
            neko_turn_user=os.environ.get("NEKO_TURN_USER"),
            neko_turn_pass=os.environ.get("NEKO_TURN_PASS"),
            neko_rtcp_keepalive=bool(int(os.environ.get("NEKO_RTCP_KEEPALIVE", "0"))),
            force_exit_guard_ms=int(os.environ.get("NEKO_FORCE_EXIT_GUARD_MS", "0")),
            neko_skip_initial_frames=int(os.environ.get("NEKO_SKIP_INITIAL_FRAMES", "5")),
            metrics_port=metrics_port,
            run_id=os.environ.get("NEKO_RUN_ID"),
        )

    def validate(self) -> List[str]:
        """Validate configuration settings and return list of errors.

        Used by --healthcheck flag to validate configuration.

        :return: List of validation error messages, empty if valid
        :rtype: List[str]
        """
        errors = []

        if self.size_shortest_edge <= 0:
            errors.append("SIZE_SHORTEST_EDGE must be positive")
        if self.size_longest_edge <= 0:
            errors.append("SIZE_LONGEST_EDGE must be positive")
        if self.max_steps <= 0:
            errors.append("NEKO_MAX_STEPS must be positive")
        if self.refinement_steps <= 0:
            errors.append("REFINEMENT_STEPS must be positive")
        if self.metrics_port <= 0 or self.metrics_port > 65535:
            errors.append("Metrics port must be between 1 and 65535")
        if self.neko_ice_policy not in ("strict", "all"):
            errors.append("NEKO_ICE_POLICY must be 'strict' or 'all'")
        if self.log_format not in ("text", "json"):
            errors.append("NEKO_LOG_FORMAT must be 'text' or 'json'")
        if self.log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append("NEKO_LOGLEVEL must be valid logging level")

        return errors

# ----------------------
# Logging Setup Functions
# ----------------------
def setup_logging(settings: Settings) -> logging.Logger:
    """Configure logging based on settings.

    Sets up both console and optional file logging with format determined by
    the log_format setting. Clears any existing handlers and configures
    new ones with appropriate formatters.

    :param settings: Configuration settings containing log format, level, and file path
    :type settings: Settings
    :return: Configured logger instance
    :rtype: logging.Logger
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure format based on settings
    if settings.log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s',
            datefmt='%H:%M:%S'
        )

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(settings.log_level.upper())

    # Setup file handler if specified
    if settings.log_file:
        try:
            file_handler = logging.FileHandler(settings.log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to set up file logging to {settings.log_file}: {e}", file=sys.stderr)

    # Configure third-party logger levels
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.ERROR)

    logger = logging.getLogger("neko_agent")

    # Log frame/click saving configuration if enabled
    if settings.frame_save_path:
        logger.info("Frame saving enabled: %s", settings.frame_save_path)
        # Ensure directory exists
        os.makedirs(os.path.dirname(settings.frame_save_path) or '/tmp/neko-agent', exist_ok=True)

    if settings.click_save_path:
        logger.info("Click action saving enabled: %s", settings.click_save_path)
        # Ensure directory exists
        dir_path = settings.click_save_path if os.path.isdir(settings.click_save_path) else os.path.dirname(settings.click_save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        else:
            os.makedirs('/tmp/neko-agent', exist_ok=True)

    return logger

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        :param record: Log record to format
        :type record: logging.LogRecord
        :return: JSON formatted log string
        :rtype: str
        """
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'message'):
                log_entry[key] = value

        return json.dumps(log_entry)

# ----------------------
# Metrics
# ----------------------
frames_received    = Counter("neko_frames_received_total",    "Total video frames received")
actions_executed   = Counter("neko_actions_executed_total",   "Actions executed by type", ["action_type"])
parse_errors       = Counter("neko_parse_errors_total",       "Action parse errors")
navigation_steps   = Counter("neko_navigation_steps_total",   "Navigation step count")
inference_latency  = Histogram("neko_inference_latency_seconds","Inference latency")
reconnects         = Counter("neko_reconnects_total",         "WS reconnect attempts")
resize_duration    = Histogram("neko_resize_duration_seconds","Resize time")

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
_NAV_SYSTEM = (
    "You are an assistant trained to navigate the {_APP} screen. "
    "Given a task instruction, a screen observation, and an action history sequence, "
    "output the next action and wait for the next observation. "
    "Here is the action space:\n{_ACTION_SPACE}\n"
    "Format the action as a dictionary with the following keys:\n"
    "{{'action': 'ACTION_TYPE', 'value': ..., 'position': ...}}\n"
    "If value or position is not applicable, set as None. "
    "Position might be [[x1,y1],[x2,y2]] for range actions. "
    "Do NOT output extra keys or commentary."
)

# ----------------------
# Small helpers
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
                 (e.g., 'Enter', 'Ctrl', 'F1').
    :type name: str
    :return: The X11 keysym integer code. Returns 0 if no mapping is found
             for multi-character names.
    :rtype: int
    """
    if len(name) == 1:
        return KEYSYM.get(name, KEYSYM.get(name.lower(), ord(name)))
    return KEYSYM.get(name, KEYSYM.get(name.capitalize(), 0))

def clamp_xy(x:int,y:int,size:Tuple[int,int]) -> Tuple[int,int]:
    """Clamp x,y coordinates to stay within screen boundaries.

    This function ensures that the provided coordinates remain within valid
    screen bounds. The coordinates are clamped to the range [0, width-1] for
    x and [0, height-1] for y, preventing out-of-bounds mouse/click operations.

    :param x: The x coordinate to clamp.
    :type x: int
    :param y: The y coordinate to clamp.
    :type y: int
    :param size: A tuple containing (width, height) of the screen bounds.
    :type size: Tuple[int, int]
    :return: A tuple of clamped (x, y) coordinates within the valid bounds.
    :rtype: Tuple[int, int]
    """
    w,h = size
    return max(0,min(x,w-1)), max(0,min(y,h-1))

def resize_and_validate_image(image: Image.Image, settings: Settings, logger: logging.Logger) -> Image.Image:
    """Resize image if it exceeds maximum dimensions while preserving aspect ratio.

    This function checks if the image's longest edge exceeds ``SIZE_LONGEST_EDGE``
    and resizes it proportionally if needed. The resizing uses Lanczos resampling
    for high-quality downscaling. The resize operation is timed and logged to
    Prometheus metrics for monitoring performance.

    :param image: The PIL Image to resize and validate
    :type image: Image.Image
    :param settings: Configuration settings containing size limits
    :type settings: Settings
    :param logger: Logger instance for recording resize operations
    :type logger: logging.Logger
    :return: The original image if within size limits, or a resized copy if the
             longest edge exceeded ``SIZE_LONGEST_EDGE``
    :rtype: Image.Image
    """
    ow,oh = image.size
    me = max(ow,oh)
    if me > settings.size_longest_edge:
        scale = settings.size_longest_edge / me
        nw,nh = int(ow*scale), int(oh*scale)
        t0 = time.monotonic()
        image = image.resize((nw,nh), Image.LANCZOS)
        resize_duration.observe(time.monotonic()-t0)
        logger.info("Resized %dx%d -> %dx%d", ow, oh, nw, nh)
    return image

def save_atomic(img: Image.Image, path: str, logger: logging.Logger) -> None:
    """Atomically save an image to disk using a temporary file and rename.

    This function provides atomic file writes by first writing to a temporary
    file in the same directory as the target, then atomically renaming it to
    the final path. This prevents partial writes from being visible and ensures
    the file is either fully written or not present at all. The operation
    includes proper fsync calls to ensure data is written to disk.

    :param img: The PIL Image to save
    :type img: Image.Image
    :param path: The destination file path where the image should be saved
    :type path: str
    :param logger: Logger instance for recording save operations
    :type logger: logging.Logger
    :raises Exception: If the atomic save operation fails, after cleaning up
                       any temporary files
    :return: None
    :rtype: None
    """
    target_dir = os.path.dirname(path) or os.getcwd()
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".tmp",
        prefix=os.path.basename(path) + ".",
        dir=target_dir,
        delete=False,
    ) as tf:
        tmp_path = tf.name
        img.save(tf, format="PNG")
        tf.flush()
        os.fsync(tf.fileno())
    try:
        os.replace(tmp_path, path)
        dir_fd = os.open(target_dir, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        logger.debug("Saved frame atomically to %s", path)
    except Exception as e:
        logger.error("Failed to save frame atomically to %s: %s", path, e, exc_info=True)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def frame_to_pil_image(frame: Any, settings: Settings, logger: logging.Logger) -> Image.Image:
    """Convert a video frame object into a PIL Image, with robustness.

    This function attempts to convert a given video frame (expected to be an
    ``av.VideoFrame`` or similar object) into a `PIL.Image.Image` instance.
    It prioritizes the ``.to_image()`` method for efficiency. If ``.to_image()``
    fails, which can occur with certain lossy video codecs (e.g., VP8/VP9) or
    corrupted frames, it falls back to converting the frame to an RGB NumPy array
    using ``.to_ndarray('rgb24')`` and then creating a PIL Image from that array.

    The function includes validation steps to ensure the resulting image has valid
    dimensions (width and height greater than 0) at both conversion stages.

    If ``FRAME_SAVE_PATH`` is configured, the resulting image is atomically saved
    to the specified path.

    :param frame: The video frame object to convert. Expected to be an
                  `av.VideoFrame` or an object with compatible `to_image()` or
                  `to_ndarray()` methods
    :type frame: Any
    :param settings: Configuration settings for frame processing
    :type settings: Settings
    :param logger: Logger instance for recording frame conversion operations
    :type logger: logging.Logger
    :raises ValueError: If the converted image (either via ``to_image()`` or
                        ``to_ndarray()``) results in invalid dimensions (<= 0)
    :raises RuntimeError: If both ``.to_image()`` and ``.to_ndarray()`` methods
                          fail to produce a valid image from the frame
    :return: A PIL Image (in RGB format) representation of the video frame
    :rtype: Image.Image
    """
    try:
        img = frame.to_image()  # uses Pillow internally
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")
        rgb = img.convert("RGB").copy()
    except Exception as e1:
        try:
            arr = frame.to_ndarray(format="rgb24")
            if getattr(arr, "size", 0) == 0:
                raise ValueError("Empty ndarray from frame")
            rgb = Image.fromarray(arr, "RGB")
            w, h = rgb.size
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid ndarray image dimensions: {w}x{h}")
            logger.debug("Recovered frame via to_ndarray fallback after to_image error: %r", e1)
        except Exception as e2:
            raise RuntimeError(f"frame->image failed: to_image={e1!r}; to_ndarray={e2!r}")
    if settings.frame_save_path:
        with contextlib.suppress(Exception):
            save_atomic(rgb, settings.frame_save_path, logger)
            logger.debug("Saved frame to %s", settings.frame_save_path)
    return rgb

def draw_action_markers(img: Image.Image, action: Dict[str, Any], step: int) -> Image.Image:
    """Draw visual markers on an image to indicate performed actions.

    This function creates a visual overlay on the provided image showing where
    an action was performed (if applicable) and adds a text label indicating
    the step number, action type, and value. Position-based actions get a red
    crosshair marker drawn at the action coordinates.

    :param img: The source image to annotate with action markers.
    :type img: Image.Image
    :param action: Dictionary containing action details with keys 'action',
                   'position', and 'value'.
    :type action: Dict[str, Any]
    :param step: The step number for labeling the action.
    :type step: int
    :return: A copy of the input image with action markers and labels drawn.
    :rtype: Image.Image
    """
    out = img.copy()
    d = ImageDraw.Draw(out)
    action_type = action.get("action","UNKNOWN")
    pos = action.get("position")
    value = action.get("value","")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except Exception:
        with contextlib.suppress(Exception):
            font = ImageFont.load_default()

    if isinstance(pos, list) and len(pos) == 2 and all(isinstance(v,(int,float)) for v in pos):
        x = max(0, min(int(pos[0]*img.width), img.width-1))
        y = max(0, min(int(pos[1]*img.height), img.height-1))
        r = 15
        d.ellipse([x-r, y-r, x+r, y+r], outline=(255,0,0), width=3)
        d.line([x-r-5, y, x+r+5, y], fill=(255,0,0), width=2)
        d.line([x, y-r-5, x, y+r+5], fill=(255,0,0), width=2)

    label = f"Step {step}: {action_type}" + (f" '{value}'" if value else "")
    if 'font' in locals() and font:
        bbox = d.textbbox((0,0), label, font=font)
        tw,th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    else:
        tw,th = len(label)*8, 16
        font = None
    d.rectangle([10,10, 10+tw+10, 10+th+10], fill=(0,0,0,128))
    d.text((15,15), label, fill=(255,255,255), font=font if font else None)
    return out

# ----------------------
# Parsing LLM actions
# ----------------------
def safe_parse_action(output_text: str, nav_mode: str="web", logger: Optional[logging.Logger] = None) -> Optional[Dict[str,Any]]:
    """Parse and validate LLM output text into a structured action dictionary.

    This function attempts to parse LLM output text as JSON or Python literal,
    validates the structure against the expected schema, and ensures the action
    type is whitelisted for the specified navigation mode. It handles both
    single actions and arrays of actions (using the first one).

    :param output_text: The raw output text from the LLM to parse
    :type output_text: str
    :param nav_mode: The navigation mode ('web' or 'phone') which determines
                     the allowed action types. Defaults to 'web'
    :type nav_mode: str
    :param logger: Optional logger instance for recording parse errors
    :type logger: Optional[logging.Logger]
    :return: A validated action dictionary with required keys 'action', 'value',
             and 'position', or None if parsing/validation fails
    :rtype: Optional[Dict[str, Any]]
    """
    try:
        act = json.loads(output_text)
    except json.JSONDecodeError:
        try:
            act = ast.literal_eval(output_text)
        except (ValueError, SyntaxError) as e:
            if logger:
                logger.error("Parse error: %s | Raw=%r", e, output_text)
            parse_errors.inc()
            return None

    if isinstance(act, (tuple, list)) and len(act) > 0:
        if logger:
            logger.info("Model returned multiple actions, using first one: %r", act)
        act = act[0]

    try:
        assert isinstance(act, dict)
        typ = act.get("action")
        if typ not in ACTION_SPACES.get(nav_mode, []):
            if logger:
                logger.warning("Non-whitelisted action: %r", typ)
            parse_errors.inc()
            return None
        for k in ("action","value","position"):
            assert k in act, f"Missing key {k}"
        return act
    except AssertionError as e:
        if logger:
            logger.error("Schema validation error: %s | Parsed=%r", e, act)
        parse_errors.inc()
        return None

# ----------------------
# Event bus + WS Signaler
# ----------------------
class LatestOnly:
    """A simple async container that holds only the latest value set.

    This class provides a mechanism to await new values while discarding
    older values. Only the most recent value is kept, making it useful
    for scenarios where only the latest data is relevant.
    """

    def __init__(self):
        """Initialize the LatestOnly container with no value set."""
        self._val = None
        self._event = asyncio.Event()

    def set(self, v):
        """Set a new value and notify any waiting consumers.

        :param v: The new value to store.
        :type v: Any
        :return: None
        :rtype: None
        """
        self._val = v
        self._event.set()

    async def get(self):
        """Wait for and return the latest value, then clear the event.

        :return: The most recently set value.
        :rtype: Any
        """
        await self._event.wait()
        self._event.clear()
        return self._val

class Broker:
    """Message broker for routing WebSocket events to topic-based queues.

    This class manages the distribution of incoming messages to appropriate
    topic queues based on event types. It supports both queueing and latest-only
    delivery patterns for different types of messages.
    """

    def __init__(self):
        """Initialize the broker with empty topic collections."""
        self.queues: Dict[str, asyncio.Queue] = {}
        self.latest: Dict[str, LatestOnly] = {}
        self.waiters: Dict[str, asyncio.Future] = {}

    def topic_queue(self, topic: str, maxsize: int = 512) -> asyncio.Queue:
        """Get or create a queue for the specified topic.

        :param topic: The topic name for the queue.
        :type topic: str
        :param maxsize: Maximum queue size. Defaults to 512.
        :type maxsize: int
        :return: The asyncio Queue for the topic.
        :rtype: asyncio.Queue
        """
        if topic not in self.queues:
            self.queues[topic] = asyncio.Queue(maxsize=maxsize)
        return self.queues[topic]

    def topic_latest(self, topic: str) -> LatestOnly:
        """Get or create a LatestOnly container for the specified topic.

        :param topic: The topic name for the container.
        :type topic: str
        :return: The LatestOnly container for the topic.
        :rtype: LatestOnly
        """
        if topic not in self.latest:
            self.latest[topic] = LatestOnly()
        return self.latest[topic]

    def publish(self, msg: Dict[str, Any]) -> None:
        """Route an incoming message to the appropriate topic queue or container.

        Messages are routed based on their 'event' field prefix. Special handling
        is provided for reply messages, signal events, system events, chat events,
        and send channel events.

        :param msg: The message dictionary to route, expected to have an 'event' key.
        :type msg: Dict[str, Any]
        :return: None
        :rtype: None
        """
        ev = msg.get("event","")
        if (rid := msg.get("reply_to")) and (fut := self.waiters.pop(rid, None)):
            if not fut.done():
                fut.set_result(msg)
            return
        if ev.startswith("signal/"):
            if ev == "signal/video":
                self.topic_latest("video").set(msg)
            elif ev == "signal/candidate":
                self.topic_queue("ice").put_nowait(msg)
            elif ev in {"signal/offer","signal/provide","signal/answer","signal/close"}:
                self.topic_queue("control").put_nowait(msg)
            else:
                self.topic_queue("signal").put_nowait(msg)
        elif ev.startswith(("system/","control/","screen/","keyboard/","session/","error/")):
            self.topic_queue("system").put_nowait(msg)
        elif ev.startswith("chat/"):
            self.topic_queue("chat").put_nowait(msg)
        elif ev.startswith("send/"):
            # Treat opaque send channel messages as chat-like for task intake.
            self.topic_queue("chat").put_nowait(msg)
        else:
            self.topic_queue("misc").put_nowait(msg)

class Signaler:
    """WebSocket client for handling signaling and messaging with automatic reconnection.

    This class manages a WebSocket connection with built-in reconnection logic,
    message routing through a Broker, and separate read/send loops for handling
    bidirectional communication.
    """

    def __init__(self, url: str, **wsopts):
        """Initialize the WebSocket signaler.

        :param url: WebSocket URL to connect to
        :type url: str
        :param wsopts: Additional WebSocket connection options
        :type wsopts: Any
        """
        self.url = url
        self.wsopts = dict(
            ping_interval=30,
            ping_timeout=60,
            max_queue=1024,
            max_size=10_000_000,
            **wsopts
        )
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._tasks: Set[asyncio.Task] = set()
        self._sendq: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.broker = Broker()
        self._closed = asyncio.Event()

    async def connect_with_backoff(self):
        """Establish WebSocket connection with exponential backoff retry.

        This method attempts to connect to the WebSocket URL with increasing
        delays between retry attempts. Once connected, it starts the read and
        send loops as background tasks.

        :return: Self for method chaining.
        :rtype: Signaler
        """
        backoff = 1
        while not self._closed.is_set():
            try:
                self.ws = await websockets.connect(self.url, **self.wsopts)
                try:
                    parsed = urlparse(self.url)
                    qs = parse_qs(parsed.query)
                    if "token" in qs:
                        qs["token"] = ["***"]
                    safe_q = urlencode(qs, doseq=True)
                    safe_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, safe_q, parsed.fragment))
                except Exception:
                    safe_url = self.url
                logger.info("WebSocket connected: %s", safe_url)
                self._tasks.add(asyncio.create_task(self._read_loop(), name="ws-read"))
                self._tasks.add(asyncio.create_task(self._send_loop(), name="ws-send"))
                self._closed.clear()
                return self
            except Exception as e:
                jitter = random.uniform(0, max(0.25, backoff * 0.25))
                delay = min(backoff + jitter, 30)
                logger.error("WS connect error: %s - retrying in %.2fs", e, delay)
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 30)

    async def close(self):
        """Close the WebSocket connection and clean up resources.

        This method cancels all running tasks, closes the WebSocket connection,
        and waits for proper cleanup.

        :return: None
        :rtype: None
        """
        self._closed.set()
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._tasks.clear()
        if self.ws:
            try:
                await self.ws.close()
                if hasattr(self.ws, "wait_closed"):
                    await self.ws.wait_closed()
            except Exception:
                pass
            finally:
                self.ws = None

    async def _read_loop(self):
        """Background task that reads messages from WebSocket and routes them.

        This coroutine continuously reads messages from the WebSocket connection
        and forwards them to the broker for routing to appropriate topic queues.
        It handles connection closure and cancellation gracefully.

        :return: None
        :rtype: None
        """
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    logger.debug("Received non-JSON: %r", raw)
                    continue
                self.broker.publish(msg)
        except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK) as e:
            logger.warning("WS closed: %s", e)
        except asyncio.CancelledError:
            logger.info("WS read loop cancelled")
        finally:
            self._closed.set()

    async def _send_loop(self):
        """Background task that sends queued messages to the WebSocket.

        This coroutine continuously processes messages from the send queue and
        transmits them over the WebSocket connection. It handles connection
        closure and cancellation gracefully.

        :return: None
        :rtype: None
        """
        try:
            while not self._closed.is_set():
                msg = await self._sendq.get()
                try:
                    await self.ws.send(json.dumps(msg))
                except websockets.ConnectionClosed:
                    logger.warning("Send failed: WS closed - exiting send loop.")
                    break
        except asyncio.CancelledError:
            logger.info("WS send loop cancelled")
        finally:
            self._closed.set()

    async def send(self, msg: Dict[str, Any]) -> None:
        """Queue a message for sending over the WebSocket.

        :param msg: The message dictionary to send.
        :type msg: Dict[str, Any]
        :return: None
        :rtype: None
        """
        await self._sendq.put(msg)

# ----------------------
# Frame Sources
# ----------------------
class FrameSource(ABC):
    """Abstract base class for video frame sources.

    This class defines the interface that all frame source implementations
    must follow, providing methods to start, stop, and retrieve frames.
    """

    @abstractmethod
    async def start(self, *args: Any) -> None:
        """Start the frame source with the provided arguments.

        :param args: Implementation-specific arguments needed to start.
        :type args: Any
        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the frame source and clean up resources.

        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def get(self) -> Optional[Image.Image]:
        """Retrieve the current frame image.

        :return: The current frame as a PIL Image, or None if unavailable.
        :rtype: Optional[Image.Image]
        """
        ...

class WebRTCFrameSource(FrameSource):
    """Frame source that receives video frames from a WebRTC VideoStreamTrack.

    This implementation reads frames from an aiortc VideoStreamTrack in a
    background task, converting them to PIL Images and making them available
    for retrieval. It includes frame filtering and error handling.
    """

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        """Initialize the WebRTC frame source.

        :param settings: Configuration settings for frame processing
        :type settings: Settings
        :param logger: Logger instance for recording frame operations
        :type logger: logging.Logger
        """
        self.image: Optional[Image.Image] = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.first_frame = asyncio.Event()
        self._running = False
        self.settings = settings
        self.logger = logger

    async def start(self, *args: Any) -> None:
        """Start reading frames from a VideoStreamTrack.

        :param args: Must contain a VideoStreamTrack as the first argument.
        :type args: Any
        :raises ValueError: If no VideoStreamTrack is provided.
        :return: None
        :rtype: None
        """
        if not args:
            raise ValueError("WebRTCFrameSource.start(): need VideoStreamTrack")
        track = args[0]
        await self.stop()
        self._running = True
        self.task = asyncio.create_task(self._reader(track))

    async def stop(self) -> None:
        """Stop the frame reader and clean up resources.

        :return: None
        :rtype: None
        """
        self._running = False
        if self.task and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        self.task = None
        async with self.lock:
            self.image = None
            self.first_frame.clear()

    async def _reader(self, track: VideoStreamTrack) -> None:
        """Background task that reads frames from the VideoStreamTrack.

        This method continuously receives frames from the provided track,
        processes them into PIL Images, and stores the latest frame for
        retrieval. It includes frame filtering and error handling.

        :param track: The WebRTC video track to read from.
        :type track: VideoStreamTrack
        :return: None
        :rtype: None
        """
        frame_count = 0
        skip_initial = self.settings.neko_skip_initial_frames
        last_warn = 0.0  # throttle warning spam
        try:
            while self._running:
                try:
                    frame = await track.recv()
                    frame_count += 1
                    if frame_count <= skip_initial:
                        continue
                    if not hasattr(frame, "to_image") and not hasattr(frame, "to_ndarray"):
                        continue
                    img = frame_to_pil_image(frame, self.settings, self.logger)
                    w,h = img.size
                    if w < 32 or h < 32 or w > 8192 or h > 8192:
                        continue
                    async with self.lock:
                        self.image = img
                        if not self.first_frame.is_set():
                            self.first_frame.set()
                    frames_received.inc()
                except MediaStreamError:
                    # Remote track ended / PC closed: exit cleanly
                    break
                except Exception as e:
                    now = time.monotonic()
                    if now - last_warn > 1.0:
                        logger.warning("Frame process failed (continuing): %r", e)
                        last_warn = now
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Frame reader stopped: %s", e)

    async def get(self) -> Optional[Image.Image]:
        """Get the current frame image.

        :return: A copy of the current frame, or None if no frame available.
        :rtype: Optional[Image.Image]
        """
        async with self.lock:
            return self.image.copy() if self.image else None

class LiteFrameSource(FrameSource):
    """Frame source that receives video frames via WebSocket messages.

    This implementation receives base64-encoded video frames through WebSocket
    messages in 'lite mode', decoding them into PIL Images. This is an
    alternative to WebRTC when real-time media transport is not available.
    """

    def __init__(self, signaler: Signaler) -> None:
        """Initialize the lite frame source.

        :param signaler: The WebSocket signaler to receive frames from.
        :type signaler: Signaler
        """
        self.signaler = signaler
        self.image: Optional[Image.Image] = None
        self.first_frame = asyncio.Event()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, *args: Any) -> None:
        """Start receiving frames from WebSocket messages.

        :param args: Unused for this implementation.
        :type args: Any
        :return: None
        :rtype: None
        """
        if self._task:
            await self.stop()
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop receiving frames and clean up resources.

        :return: None
        :rtype: None
        """
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        async with self._lock:
            self.image = None
            self.first_frame.clear()

    async def _run(self) -> None:
        """Background task that processes incoming video messages.

        This method continuously listens for 'signal/video' messages from the
        broker, decodes the base64-encoded frame data, and stores the resulting
        PIL Images for retrieval.

        :return: None
        :rtype: None
        """
        ch = self.signaler.broker.topic_latest("video")
        try:
            while self._running and not self.signaler._closed.is_set():
                try:
                    msg = await asyncio.wait_for(ch.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if msg.get("event") != "signal/video":
                    continue
                data = msg.get("data")
                if not data:
                    continue
                img = self._decode_frame_base64(data)
                if img is None:
                    continue
                w,h = img.size
                if w <= 0 or h <= 0 or w > 8192 or h > 8192:
                    continue
                async with self._lock:
                    self.image = img
                    if not self.first_frame.is_set():
                        self.first_frame.set()
                frames_received.inc()
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning("LiteFrameSource consumer exiting: %s", e)

    def _decode_frame_base64(self, data: str) -> Optional[Image.Image]:
        """Decode a base64-encoded image string into a PIL Image.

        This method validates the base64 data, checks for common image formats
        (PNG, JPEG), verifies the image integrity, and handles EXIF orientation.

        :param data: Base64-encoded image data string.
        :type data: str
        :return: The decoded PIL Image in RGB format, or None if decoding fails.
        :rtype: Optional[Image.Image]
        """
        try:
            img_bytes = base64.b64decode(data, validate=True)
            if not (img_bytes.startswith(b"\x89PNG") or img_bytes.startswith(b"\xff\xd8\xff")):
                return None
            bio = io.BytesIO(img_bytes)
            _probe = Image.open(bio)
            _probe.verify()
            bio2 = io.BytesIO(img_bytes)
            img = Image.open(bio2)
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            return img.convert("RGB")
        except Exception as e:
            logger.warning("Lite frame decode failed: %s", e)
            return None

    async def get(self) -> Optional[Image.Image]:
        """Get the current frame image.

        :return: A copy of the current frame, or None if no frame available
                 or signaler is closed.
        :rtype: Optional[Image.Image]
        """
        async with self._lock:
            if self.signaler._closed.is_set():
                return None
            return self.image.copy() if self.image else None


# ----------------------
# Agent
# ----------------------
class NekoAgent:
    """Main agent class that orchestrates WebRTC signaling and AI-driven navigation.

    This agent connects to a Neko WebRTC server, receives video frames, processes
    them with an AI model (ShowUI-2B) to determine navigation actions, and
    executes those actions on the remote screen. It supports both offline
    (single-task) and online (multi-task via chat) operation modes.
    """

    def __init__(
        self,
        model,
        processor,
        ws_url: str,
        nav_task: str,
        nav_mode: str,
        *,
        settings: Settings,
        logger: logging.Logger,
        max_steps: Optional[int] = None,
        refinement_steps: Optional[int] = None,
        metrics_port: Optional[int] = None,  # kept for compatibility (not used)
        audio: Optional[bool] = None,
        online: bool = False,
    ):
        """Initialize the Neko agent.

        :param model: The AI model for processing images and generating actions.
        :type model: Qwen2VLForConditionalGeneration
        :param processor: The processor for preparing model inputs.
        :type processor: AutoProcessor
        :param ws_url: WebSocket URL for connecting to the Neko server.
        :type ws_url: str
        :param nav_task: Initial navigation task description.
        :type nav_task: str
        :param nav_mode: Navigation mode ('web' or 'phone').
        :type nav_mode: str
        :param max_steps: Maximum navigation steps per task.
        :type max_steps: int
        :param refinement_steps: Number of refinement iterations for actions.
        :type refinement_steps: int
        :param metrics_port: Port for Prometheus metrics server.
        :type metrics_port: int
        :param audio: Whether to enable audio streaming.
        :type audio: bool
        :param online: Whether to run in online mode (multi-task via chat).
        :type online: bool
        """
        self.signaler = Signaler(ws_url)
        self.frame_source: Optional[FrameSource] = None
        # In online mode the agent should start without a task and
        # wait for chat-provided tasks instead.
        self.nav_task = nav_task if not online else ""
        self.nav_mode = nav_mode
        self.settings = settings
        self.logger = logger
        self.max_steps = settings.max_steps if max_steps is None else max_steps
        self.refinement_steps = settings.refinement_steps if refinement_steps is None else refinement_steps
        self.audio = settings.audio_default if audio is None else audio
        self.online = online
        self.model = model
        self.processor = processor
        self.run_id = settings.run_id or str(uuid.uuid4())[:8]
        self.pc: Optional[RTCPeerConnection] = None
        self.shutdown = asyncio.Event()
        self.is_lite = False
        self._tg: Optional[asyncio.TaskGroup] = None
        self._current_inference_task: Optional[asyncio.Future] = None

        # Online task orchestration helpers
        self._is_running_task: bool = False
        self._pending_task: Optional[str] = None
        self._new_task_event: asyncio.Event = asyncio.Event()

        # Control intent/state (online on-demand host behavior)
        # We only want host control while actively running a task.
        self._want_control: bool = False
        self._is_host: bool = False
        self._is_host_event: asyncio.Event = asyncio.Event()
        self._last_control_action: float = 0.0

        self.session_id: Optional[str] = None
        self.screen_size: Tuple[int,int] = (1280, 720)

        self.sys_prompt = _NAV_SYSTEM.format(
            _APP=self.nav_mode,
            _ACTION_SPACE=ACTION_SPACE_DESC[self.nav_mode],
        )

        # Metrics server is started in main(); just placeholders here.
        self._metrics_server, self._metrics_thread = None, None

    async def run(self) -> None:
        """Main execution loop that handles connection, reconnection, and task processing.

        This method establishes the WebSocket connection, sets up media handling,
        and runs the navigation loop. It includes automatic reconnection logic
        and proper cleanup on shutdown.

        :return: None
        :rtype: None
        """
        loop = asyncio.get_running_loop()

        def signal_handler(signum, frame=None):
            """Handle shutdown signals by initiating clean shutdown.

            :param signum: Signal number received
            :param frame: Current stack frame (unused)
            """
            sig_name = signal.Signals(signum).name
            logger.info("Received %s - initiating clean shutdown...", sig_name)
            self.shutdown.set()
            if getattr(self, "_current_inference_task", None):
                self._current_inference_task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, signal_handler, sig)

        while not self.shutdown.is_set():
            reconnects.inc()
            try:
                await self.signaler.connect_with_backoff()

                # Request media on connect (host control gated by mode)
                # audio payload must explicitly disable when requested; default is enabled
                await self.signaler.send({
                    "event": "signal/request",
                    "payload": {
                        "video": {},
                        "audio": ({}) if self.audio else ({"disabled": True}),
                    },
                })
                # Offline: request control immediately and keep it until exit.
                # Online: start hands-off; only request when a task begins.
                if not self.online:
                    await self.signaler.send({"event": "control/request"})

                control_q = self.signaler.broker.topic_queue("control")
                offer_msg = None
                while not offer_msg:
                    msg = await control_q.get()
                    if msg.get("event") in ("signal/offer", "signal/provide"):
                        offer_msg = msg

                await self._setup_media(offer_msg)

                try:
                    async with asyncio.TaskGroup() as tg:
                        self._tg = tg
                        tg.create_task(self._shutdown_watcher(), name="shutdown-watcher")
                        tg.create_task(self._consume_system_topic())
                        tg.create_task(self._consume_chat_topic())
                        if not self.is_lite:
                            tg.create_task(self._consume_ice_topic())
                            tg.create_task(self._consume_control_topic())
                        else:
                            tg.create_task(self._consume_video_lite_topic())
                        tg.create_task(self._main_loop())
                except asyncio.CancelledError:
                    logger.info("Task group cancelled - cleaning up")
                except Exception as e:
                    logger.error("Task group error: %s", e, exc_info=True)

            except Exception as e:
                logger.error("Connect/RTC error: %s", e, exc_info=True)
            finally:
                await self._cleanup()
                if not self.shutdown.is_set():
                    logger.info("Disconnected - attempting to reconnect shortly.")
                    await asyncio.sleep(0.5)
                else:
                    logger.info("Shutdown requested - exiting reconnection loop.")
                    break

    # --- Media / Signaling
    async def _setup_media(self, offer_msg: Optional[Dict[str, Any]] = None) -> None:
        """Set up WebRTC media connection or lite mode frame source.

        This method handles both WebRTC peer connection setup (with SDP/ICE
        negotiation) and lite mode initialization. It includes early ICE
        candidate buffering to prevent race conditions during connection setup.

        :param offer_msg: Optional WebRTC offer message. If None, waits for one.
        :type offer_msg: Optional[Dict[str, Any]]
        :return: None
        :rtype: None
        """
        # Early ICE buffering to avoid addIceCandidate-before-SRD race
        early_ice_payloads: List[Dict[str, Any]] = []
        ice_q = self.signaler.broker.topic_queue("ice")
        buffer_running = True

        async def _buffer_ice():
            while buffer_running:
                try:
                    msg = await ice_q.get()
                    if msg.get("event") == "signal/candidate":
                        early_ice_payloads.append(msg.get("payload") or {})
                except asyncio.CancelledError:
                    break
                except Exception:
                    break

        buf_task = asyncio.create_task(_buffer_ice(), name="early-ice-buffer")

        if offer_msg is None:
            control_q = self.signaler.broker.topic_queue("control")
            while not offer_msg:
                msg = await control_q.get()
                if msg.get("event") in ("signal/offer", "signal/provide"):
                    offer_msg = msg

        payload = offer_msg.get("payload", offer_msg)

        # Lite mode detection
        self.is_lite = bool(offer_msg.get("lite") or payload.get("lite"))
        if self.is_lite:
            buffer_running = False
            buf_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await buf_task
            logger.info("Lite/WebSocket mode enabled.")
            self.frame_source = LiteFrameSource(self.signaler)
            await self.frame_source.start()
            return

        # Strict ICE mapping from payload
        ice_payload = (
            payload.get("ice")
            or payload.get("iceservers")
            or payload.get("iceServers")
            or payload.get("ice_servers")
            or []
        )
        ice_servers: List[RTCIceServer] = []
        for srv in ice_payload:
            if not isinstance(srv, dict):
                continue
            urls = srv.get("urls") or srv.get("url")
            username = srv.get("username")
            credential = srv.get("credential") or srv.get("password")
            if urls:
                ice_servers.append(RTCIceServer(urls=urls, username=username, credential=credential))

        if self.settings.neko_ice_policy != "strict":
            ice_servers.append(RTCIceServer(urls=[self.settings.neko_stun_url]))
            if self.settings.neko_turn_url:
                ice_servers.append(RTCIceServer(urls=[self.settings.neko_turn_url + "?transport=tcp"],
                                                username=self.settings.neko_turn_user, credential=self.settings.neko_turn_pass))

        config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(config)
        self.pc = pc
        self.frame_source = WebRTCFrameSource(self.settings, self.logger)

        @pc.on("iceconnectionstatechange")
        def _on_ics():
            """Handle ICE connection state changes."""
            logger.info("iceConnectionState -> %s", pc.iceConnectionState)
            # Stop keepalive if ICE is no longer viable
            if pc.iceConnectionState in ("disconnected", "failed", "closed") and hasattr(self, "_rtcp_task"):
                self._rtcp_task.cancel()
                with contextlib.suppress(Exception):
                    delattr(self, "_rtcp_task")

        @pc.on("connectionstatechange")
        def _on_cs():
            """Handle peer connection state changes."""
            logger.info("connectionState -> %s", pc.connectionState)
            # Optional RTCP keepalive once fully connected
            if self.settings.neko_rtcp_keepalive and pc.connectionState == "connected" and not hasattr(self, "_rtcp_task"):
                async def rtcp_keepalive():
                    logger.info("RTCP keepalive task starting")
                    count = 0
                    while self.pc and self.pc.connectionState == "connected":
                        try:
                            async with asyncio.timeout(1.0):
                                _ = await self.pc.getStats()
                            count += 1
                            logger.debug("RTCP keepalive #%d sent", count)
                        except asyncio.TimeoutError:
                            logger.warning("RTCP keepalive timeout")
                        except Exception as e:
                            logger.warning("RTCP keepalive error: %s", e)
                            break
                        await asyncio.sleep(5.0)
                    logger.info("RTCP keepalive task ended after %d packets", count)
                self._rtcp_task = asyncio.create_task(rtcp_keepalive())
            elif pc.connectionState in ("disconnected", "failed", "closed") and hasattr(self, "_rtcp_task"):
                self._rtcp_task.cancel()
                with contextlib.suppress(Exception):
                    delattr(self, "_rtcp_task")

        @pc.on("icecandidate")
        async def _on_ic(cand: RTCIceCandidate):
            await self._on_ice(cand)

        @pc.on("track")
        async def _on_tr(track: VideoStreamTrack):
            await self._on_track(track)

        remote_sdp = payload.get("sdp")
        remote_type = payload.get("type", "offer")
        if not remote_sdp:
            buffer_running = False
            buf_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await buf_task
            raise RuntimeError("Missing SDP in offer payload")

        await pc.setRemoteDescription(RTCSessionDescription(sdp=remote_sdp, type=remote_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self.signaler.send({
            "event": "signal/answer",
            "payload": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        })

        # --- Stop buffering & APPLY EARLY ICE ---
        buffer_running = False
        buf_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await buf_task

        # Drain any queued ICE quickly and append
        while True:
            try:
                msg = ice_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                if msg.get("event") == "signal/candidate":
                    early_ice_payloads.append(msg.get("payload") or {})

        for pay in early_ice_payloads:
            try:
                ice = self._parse_remote_candidate(pay)
                if ice:
                    await pc.addIceCandidate(ice)
            except Exception as e:
                logger.debug("Applying buffered ICE failed: %s", e)

    async def _consume_control_topic(self) -> None:
        """Consume control topic messages for WebRTC signaling events.

        This method processes control messages like signal/close which indicate
        remote disconnection and trigger reconnection.

        :return: None
        :rtype: None
        """
        q = self.signaler.broker.topic_queue("control")
        try:
            while not self.shutdown.is_set():
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if msg.get("event") == "signal/close":
                    logger.info("Remote close received; triggering reconnection.")
                    return
        except asyncio.CancelledError:
            return

    async def _consume_ice_topic(self) -> None:
        """Consume ICE candidate messages and apply them to the peer connection.

        This method processes incoming ICE candidates from the remote peer
        and adds them to the local RTCPeerConnection for connectivity.

        :return: None
        :rtype: None
        """
        q = self.signaler.broker.topic_queue("ice")
        try:
            while not self.shutdown.is_set():
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if msg.get("event") == "signal/candidate":
                    ice = self._parse_remote_candidate(msg.get("payload") or {})
                    if ice and self.pc:
                        with contextlib.suppress(Exception):
                            await self.pc.addIceCandidate(ice)
        except asyncio.CancelledError:
            return

    async def _consume_video_lite_topic(self) -> None:
        """Consume video topic messages in lite mode (WebSocket-based frames).

        This method handles video frame messages when operating in lite mode,
        where frames are sent as base64-encoded data over WebSocket instead
        of through WebRTC media streams.

        :return: None
        :rtype: None
        """
        ch = self.signaler.broker.topic_latest("video")
        try:
            while not self.shutdown.is_set():
                try:
                    _ = await asyncio.wait_for(ch.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            return

    # --- System events
    async def _consume_system_topic(self) -> None:
        """Consume system topic messages for server events and state updates.

        This method processes various system messages including heartbeats,
        initialization data, screen size changes, host control events,
        and keyboard mapping updates.

        :return: None
        :rtype: None
        """
        q = self.signaler.broker.topic_queue("system")
        try:
            while not self.shutdown.is_set():
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                ev = msg.get("event","")
                payload = msg.get("payload", {})

                if ev == "system/heartbeat":
                    try:
                        await self.signaler.send({"event": "client/heartbeat"})
                        logger.debug("Heartbeat response queued")
                    except Exception as e:
                        logger.warning("Failed to queue heartbeat response: %s", e)
                    continue

                if ev == "system/init" and isinstance(payload, dict):
                    self.session_id = payload.get("session_id") or self.session_id
                    if (size := payload.get("screen_size")) and isinstance(size, dict):
                        w = int(size.get("width", self.screen_size[0]))
                        h = int(size.get("height", self.screen_size[1]))
                        self.screen_size = (w,h)
                        logger.info("Initial screen size %dx%d", w, h)
                    continue

                if ev == "screen/updated" and isinstance(payload, dict):
                    if "width" in payload and "height" in payload:
                        self.screen_size = (int(payload["width"]), int(payload["height"]))
                        logger.info("Screen size changed to %dx%d", *self.screen_size)
                    continue

                if ev == "control/host" and isinstance(payload, dict):
                    # Update host state and react according to mode/intent.
                    host_id = payload.get("host_id")
                    has_host = bool(payload.get("has_host"))
                    was_host = self._is_host
                    self._is_host = bool(has_host and self.session_id and host_id == self.session_id)
                    if self._is_host and not was_host:
                        self._is_host_event.set()
                    if not self._is_host:
                        self._is_host_event.clear()
                    if self.online:
                        # Online: only act based on intent (on-demand host)
                        asyncio.create_task(self._maybe_adjust_control())
                    else:
                        # Offline: keep current behavior — hold controls
                        if not self._is_host:
                            logger.info("Host control lost/changed - re-requesting (offline mode).")
                            asyncio.create_task(self.signaler.send({"event": "control/request"}))
                    continue

                if ev == "keyboard/map":
                    mapping = None
                    if isinstance(payload, dict):
                        if "map" in payload and isinstance(payload["map"], dict):
                            mapping = payload["map"]
                        else:
                            mapping = payload
                    if isinstance(mapping, dict):
                        KEYSYM.update({str(k): int(v) for k,v in mapping.items() if isinstance(v, int)})
                        logger.info("Keyboard map updated (+%d entries).", len(mapping))
                    continue

                if ev.startswith("error/"):
                    logger.error("[server] %s :: %s", ev, json.dumps(payload, ensure_ascii=False))
                    continue

        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("System consumer error: %s", e, exc_info=True)

    async def _consume_chat_topic(self) -> None:
        """Consume chat topic messages for task updates in online mode.

        This method processes chat messages and send channel messages,
        looking for /task commands that provide new navigation instructions.
        It filters out self-echo messages and handles task queueing.

        :return: None
        :rtype: None
        """
        q = self.signaler.broker.topic_queue("chat")
        try:
            while not self.shutdown.is_set():
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                ev = msg.get("event", "")
                payload = msg.get("payload", {})

                # Ignore messages that originate from this agent's session (self-echo)
                sender_id = None
                if ev.startswith("chat/"):
                    sender_id = payload.get("id")
                elif ev.startswith("send/"):
                    sender_id = payload.get("sender")
                if sender_id and self.session_id and sender_id == self.session_id:
                    continue

                incoming_text: Optional[str] = None

                # Plugin chat: server→client payload: { id, created, content: { text } }
                # Client→server (not used here) would be { text }
                if ev == "chat/message":
                    content = payload.get("content")
                    if isinstance(content, dict):
                        incoming_text = content.get("text")
                    elif isinstance(content, str):
                        incoming_text = content
                    if not incoming_text:
                        incoming_text = payload.get("text") or payload.get("message")

                # Opaque send channel: treat subject "task" or body starting with "/task"
                elif ev in ("send/broadcast", "send/unicast"):
                    subject = payload.get("subject")
                    body = payload.get("body")
                    if isinstance(body, str):
                        incoming_text = body
                    elif isinstance(body, dict):
                        incoming_text = body.get("text") or body.get("message")
                    # Require explicit marker if not subject == "task"
                    if subject and isinstance(subject, str) and subject.lower() == "task":
                        pass  # accept
                    else:
                        # Only accept if text looks like a slash task command
                        if not (isinstance(incoming_text, str) and incoming_text.strip().startswith("/task")):
                            incoming_text = None

                if not (incoming_text and isinstance(incoming_text, str)):
                    continue

                # Only accept slash-task commands for task updates
                text = incoming_text.strip()
                if text.lower().startswith("/task"):
                    parts = text.split(" ", 1)
                    new_task = parts[1].strip() if len(parts) > 1 else ""
                    if not new_task:
                        await self._send_chat_safe("Usage: /task <instruction>")
                        continue

                    logger.info("Chat task update: %s", new_task)
                    if self.online and self._is_running_task:
                        self._pending_task = new_task
                        logger.info("Task queued until current run completes.")
                    else:
                        self.nav_task = new_task
                        if self.online:
                            self._new_task_event.set()
        except asyncio.CancelledError:
            return

    # --- Main loop: observe -> decide -> act
    async def _main_loop(self) -> None:
        """Run navigation cycles. In online mode, wait for chat tasks.

        Behavior:
        - Offline: run once with ``self.nav_task``, then request shutdown.
        - Online: wait for a task from chat, run it, report progress to chat,
          then wait for the next task without exiting.

        This method coordinates frame acquisition, AI model inference, action
        execution, and task lifecycle management.

        :return: None
        :rtype: None
        """
        try:
            # Wait for the first frame, but not forever
            if self.frame_source and hasattr(self.frame_source, "first_frame"):
                try:
                    await asyncio.wait_for(self.frame_source.first_frame.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.error("No video frames received within 5s after connect; aborting run.")
                    return

            while not self.shutdown.is_set():
                # In online mode, block until we have a task
                if self.online:
                    ready_announced = False
                    while not (self.nav_task and self.nav_task.strip()) and not self.shutdown.is_set():
                        if not ready_announced:
                            await self._send_chat_safe("Ready. Send a task in chat to begin.")
                            ready_announced = True
                        try:
                            await asyncio.wait_for(self._new_task_event.wait(), timeout=10.0)
                        except asyncio.TimeoutError:
                            continue
                        finally:
                            self._new_task_event.clear()
                    if self.shutdown.is_set():
                        break
                    await self._send_chat_safe(f"Starting task: {self.nav_task}")
                    # Online on-demand host: request control at task start
                    await self._set_control_intent(True)
                    # Best-effort: wait briefly to become host
                    await self._wait_until_host(timeout=10.0)

                history: List[Dict[str, Any]] = []
                step = 0
                no_img_since = time.monotonic()
                self._is_running_task = True

                while not self.shutdown.is_set() and step < self.max_steps:
                    navigation_steps.inc()
                    if not self.frame_source:
                        await asyncio.sleep(0.01); continue

                    # Stop promptly if the connection has closed (important for lite mode)
                    if self.signaler._closed.is_set():
                        logger.warning("Connection closed; stopping navigation loop")
                        break

                    img = await self.frame_source.get()
                    if img is None:
                        # Give up if we can't decode frames for a while
                        if time.monotonic() - no_img_since > 5.0:
                            logger.error("No decodable frames for 5s; aborting current run.")
                            break
                        await asyncio.sleep(0.01)
                        continue

                    no_img_since = time.monotonic()
                    img = resize_and_validate_image(img, self.settings, self.logger)
                    act = await self._navigate_once(img, history, step)
                    if not act or act.get("action") == "ANSWER":
                        break
                    history.append(act)
                    step += 1
                    await asyncio.sleep(0.01)

                self._is_running_task = False
                logger.info("Run complete: steps=%d", step)
                if self.online:
                    # Release controls while idle between tasks
                    await self._set_control_intent(False)
                    await self._send_chat_safe(f"Completed task: {self.nav_task} (steps={step}).")
                    # Rotate to any queued task or clear and wait again
                    if self._pending_task:
                        self.nav_task = self._pending_task
                        self._pending_task = None
                        # loop continues to start next task immediately
                    else:
                        self.nav_task = ""
                        # loop will wait for next task
                        continue
                else:
                    logger.info("Offline mode; requesting clean shutdown.")
                    self.shutdown.set()
                    return
        except asyncio.CancelledError:
            logger.info("Navigation loop cancelled - shutting down")
            raise
        except Exception as e:
            logger.error("Navigation loop error: %s", e, exc_info=True)
            return

    async def _send_chat_safe(self, text: str) -> None:
        """Send a chat/message event if the WS is open.

        This logs and suppresses any send errors so it never disrupts the
        primary navigation loop.
        """
        try:
            if getattr(self.signaler, "ws", None) and not self.signaler._closed.is_set():
                # Neko v3 chat plugin expects payload shape {"text": string}
                await self.signaler.send({"event": "chat/message", "payload": {"text": text}})
        except Exception:
            logger.debug("Failed to send chat message", exc_info=True)

    # --- Control helpers (on-demand host in online mode)
    async def _set_control_intent(self, want: bool) -> None:
        """Set whether we want host control and adjust accordingly.

        In online mode, we only want host while actively running a task.
        This is idempotent and defers to `_maybe_adjust_control()` which
        throttles actual WS traffic.
        """
        self._want_control = bool(want)
        await self._maybe_adjust_control()

    async def _maybe_adjust_control(self) -> None:
        """Send control/request or control/release based on intent/state.

        Throttles to avoid flooding the server if host state flaps.
        """
        now = time.monotonic()
        if now - self._last_control_action < 0.5:
            return
        self._last_control_action = now

        try:
            if self._want_control and not self._is_host:
                await self.signaler.send({"event": "control/request"})
            elif (not self._want_control) and self._is_host:
                await self.signaler.send({"event": "control/release"})
        except Exception:
            # Non-fatal; we may retry on next state change or intent flip
            logger.debug("Adjust control failed (ignored)", exc_info=True)

    async def _wait_until_host(self, timeout: float = 10.0) -> bool:
        """Wait until we become host, up to `timeout` seconds.

        Returns True if host; False if timeout or lost.
        """
        if self._is_host:
            return True
        try:
            async with asyncio.timeout(float(timeout)):
                await self._is_host_event.wait()
            return True
        except asyncio.TimeoutError:
            await self._send_chat_safe("Still waiting for host control…")
            return self._is_host

    async def _log_inference_progress(self, start_time: float, step: int) -> None:
        """Log periodic progress updates during model inference.

        This background task logs inference progress every 2 seconds,
        showing elapsed time and device information (GPU/CPU) for
        monitoring long-running model operations.

        :param start_time: Unix timestamp when inference started.
        :type start_time: float
        :param step: Current navigation step number for context.
        :type step: int
        :return: None
        :rtype: None
        """
        try:
            await asyncio.sleep(2.0)
            while True:
                elapsed = time.time() - start_time
                device_status = "GPU" if torch.cuda.is_available() and self.model.device.type == "cuda" else "CPU"
                logger.info("Model inference in progress (step=%d) | Elapsed: %.1fs | Device: %s",
                            step, elapsed, device_status)
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass

    def _crop_image(self, image: Image.Image, click_xy: Tuple[float,float], crop_factor: float=0.5) -> Tuple[Image.Image, Tuple[int,int,int,int]]:
        """Crop image around a click location for refinement.

        This method crops the input image to focus on a specific region
        around a click location, used for iterative refinement of click
        coordinates by providing a zoomed-in view to the AI model.

        :param image: The source image to crop.
        :type image: Image.Image
        :param click_xy: Normalized (x, y) coordinates of the click location.
        :type click_xy: Tuple[float, float]
        :param crop_factor: Factor determining crop size (0.5 = half image size).
        :type crop_factor: float
        :return: Tuple of (cropped image, bounding box coordinates).
        :rtype: Tuple[Image.Image, Tuple[int, int, int, int]]
        """
        width, height = image.size
        cw, ch = int(width*crop_factor), int(height*crop_factor)
        cx, cy = int(click_xy[0]*width), int(click_xy[1]*height)
        left = max(cx - cw//2, 0); top = max(cy - ch//2, 0)
        right = min(cx + cw//2, width); bottom = min(cy + ch//2, height)
        box = (left, top, right, bottom)
        return image.crop(box), box

    async def _navigate_once(self, img: Image.Image, history: List[Dict[str,Any]], step: int) -> Optional[Dict[str,Any]]:
        """Perform one navigation step using AI model inference.

        This method processes the current screen image with the AI model to
        determine the next action. It supports iterative refinement for click
        actions by cropping around the predicted location and re-inferring
        for more precise coordinates.

        :param img: The current screen image.
        :type img: Image.Image
        :param history: List of previously executed actions.
        :type history: List[Dict[str, Any]]
        :param step: Current step number.
        :type step: int
        :return: The predicted action dictionary, or None if inference fails.
        :rtype: Optional[Dict[str, Any]]
        """
        original_img = img
        current_img = img
        full_w, full_h = original_img.size
        crop_box = (0,0,full_w,full_h)

        act: Optional[Dict[str,Any]] = None

        for i in range(self.refinement_steps):
            content: List[Dict[str, Any]] = [
                {"type": "text", "text": self.sys_prompt},
                {"type": "text", "text": f"Task: {self.nav_task}"},
            ]
            if history:
                content.append({"type": "text", "text": f"Action history: {json.dumps(history)}"})
            content.append({
                "type": "image", "image": current_img,
                "size": {
                    "shortest_edge": self.settings.size_shortest_edge,
                    "longest_edge": self.settings.size_longest_edge
                }
            })
            msgs = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[current_img], videos=None, padding=True, return_tensors="pt").to(self.model.device)

            inference_start_time = time.time()
            logger.info("Model inference starting (step=%d)", step)

            if not hasattr(self, '_model_executor'):
                self._model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-inference")

            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                self._model_executor,
                lambda: self.model.generate(**inputs, max_new_tokens=128)
            )
            self._current_inference_task = future
            progress_task = asyncio.create_task(self._log_inference_progress(inference_start_time, step))

            try:
                with inference_latency.time():
                    async with asyncio.timeout(120.0):
                        gen = await future
                duration = time.time() - inference_start_time
                device_status = "GPU" if torch.cuda.is_available() and self.model.device.type == "cuda" else "CPU"
                logger.info("Model inference completed (step=%d) | Duration: %.2fs | Device: %s", step, duration, device_status)

            except asyncio.TimeoutError:
                future.cancel()
                logger.error("Model inference timeout (step=%d)", step)
                parse_errors.inc()
                return None
            except asyncio.CancelledError:
                logger.info("Model inference cancelled (step=%d)", step)
                return None
            finally:
                self._current_inference_task = None
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            out_ids = [o[len(i):] for o,i in zip(gen, inputs.input_ids)]
            raw_output = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            act = safe_parse_action(raw_output, nav_mode=self.nav_mode, logger=self.logger)
            logger.info("Raw model output: %s", raw_output)

            # Optional refinement: only if model emitted normalized 'x'/'y' fields for CLICK
            if not (act and act.get("action") == "CLICK" and act.get("x") is not None and act.get("y") is not None):
                break

            click_x, click_y = float(act["x"]), float(act["y"])
            crop_l, crop_t, crop_r, crop_b = crop_box
            crop_w = crop_r - crop_l
            crop_h = crop_b - crop_t
            abs_x = crop_l + click_x * crop_w
            abs_y = crop_t + click_y * crop_h
            final_x = abs_x / full_w
            final_y = abs_y / full_h
            act["x"], act["y"] = final_x, final_y
            # Ensure executor uses refined coordinates even if the model omitted 'position'
            if not isinstance(act.get("position"), list):
                act["position"] = [final_x, final_y]

            if i < self.refinement_steps - 1:
                current_img, crop_box = self._crop_image(original_img, (final_x, final_y))
                logger.info("Refinement %d -> crop=%s", i, crop_box)

        typ = (act or {}).get("action", "UNSUPPORTED")
        actions_executed.labels(action_type=typ if typ in ALLOWED_ACTIONS else "UNSUPPORTED").inc()
        logger.info("Chosen action (step=%d): %s", step, json.dumps(act or {}))

        if act and self.settings.click_save_path:
            with contextlib.suppress(Exception):
                marked = draw_action_markers(original_img, act, step)
                ts = time.monotonic()
                fname = f"action_step_{step:03d}_{ts:.3f}_{act.get('action','unknown')}.png"
                path = os.path.join(self.settings.click_save_path, fname) if os.path.isdir(self.settings.click_save_path) else f"{self.settings.click_save_path}_{fname}"
                dirpath = os.path.dirname(path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                save_atomic(marked, path, self.logger)
                self.logger.debug("Saved click action visualization to %s", path)
                logger.info("Saved action frame: %s", path)

        if act:
            await self._execute_action(act, original_img.size)
        return act

    async def _execute_action(self, action: Dict[str,Any], size: Tuple[int,int]) -> None:
        """Execute a predicted action on the remote screen.

        This method translates high-level action dictionaries into low-level
        control commands (mouse movements, clicks, keyboard input, etc.) and
        sends them via the WebSocket connection.

        :param action: Action dictionary with 'action', 'value', and 'position' keys.
        :type action: Dict[str, Any]
        :param size: Screen dimensions as (width, height).
        :type size: Tuple[int, int]
        :return: None
        :rtype: None
        """
        typ  = action.get("action")
        val  = action.get("value")
        pos  = action.get("position")

        def to_xy(norm_pt: List[float]) -> Tuple[int,int]:
            """Convert normalized coordinates to pixel coordinates."""
            x = int(float(norm_pt[0]) * size[0])
            y = int(float(norm_pt[1]) * size[1])
            return clamp_xy(x,y,size)

        async def move(x:int,y:int) -> None:
            """Move mouse cursor to coordinates."""
            await self.signaler.send({"event":"control/move","payload":{"x":x,"y":y}})

        async def button_press(x:int,y:int, button: str="left") -> None:
            """Press and release mouse button at coordinates."""
            code = BUTTON_CODES.get(button, 1)
            await self.signaler.send({"event":"control/buttonpress","payload":{"x":x,"y":y,"code":code}})

        async def button_down(x:int,y:int, button: str="left") -> None:
            """Press down mouse button at coordinates."""
            code = BUTTON_CODES.get(button, 1)
            await self.signaler.send({"event":"control/buttondown","payload":{"x":x,"y":y,"code":code}})

        async def button_up(x:int,y:int, button: str="left") -> None:
            """Release mouse button at coordinates."""
            code = BUTTON_CODES.get(button, 1)
            await self.signaler.send({"event":"control/buttonup","payload":{"x":x,"y":y,"code":code}})

        async def key_once(name_or_char: str) -> None:
            """Press and release a key."""
            ks = name_keysym(name_or_char)
            if ks:
                await self.signaler.send({"event":"control/keypress","payload":{"keysym":ks}})

        async def key_down(name_or_char: str) -> None:
            """Press down a key."""
            ks = name_keysym(name_or_char)
            if ks:
                await self.signaler.send({"event":"control/keydown","payload":{"keysym":ks}})

        async def key_up(name_or_char: str) -> None:
            """Release a key."""
            ks = name_keysym(name_or_char)
            if ks:
                await self.signaler.send({"event":"control/keyup","payload":{"keysym":ks}})

        try:
            # Announce action in chat (online or offline for visibility)
            action_preview = {
                "action": typ,
                "value": val if isinstance(val, (str, int, float)) else None,
                "position": pos,
            }
            await self._send_chat_safe(f"Action: {json.dumps(action_preview, ensure_ascii=False)}")

            if typ in {"CLICK","TAP","SELECT","HOVER"} and isinstance(pos, list) and len(pos) == 2:
                x,y = to_xy(pos)
                await move(x,y)
                if typ in {"CLICK","TAP","SELECT"}:
                    btn = "left"
                    if isinstance(val, str) and val.lower() in BUTTON_CODES:
                        btn = val.lower()
                    await button_press(x,y,btn)
                return

            if typ == "INPUT" and val and isinstance(pos, list) and len(pos) == 2:
                x,y = to_xy(pos)
                await move(x,y)
                await button_press(x,y,"left")
                for ch in str(val):
                    await key_once("Enter" if ch == "\n" else ch)
                return

            if typ == "ENTER":
                await key_once("Enter")
                return

            if typ == "SCROLL" and val:
                direction = str(val).lower()
                amount = action.get("amount", 1)
                try:
                    amount = int(amount)
                except Exception:
                    amount = 1
                delta_map = {
                    "down": (0, 120 * amount),
                    "up":   (0, -120 * amount),
                    "right":(120 * amount, 0),
                    "left": (-120 * amount, 0),
                }
                dx,dy = delta_map.get(direction, (0,0))
                if dx or dy:
                    await self.signaler.send({"event":"control/scroll","payload":{"delta_x":dx,"delta_y":dy}})
                return

            if typ == "SWIPE" and isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, list) and len(p) == 2 for p in pos):
                x1,y1 = to_xy(pos[0]); x2,y2 = to_xy(pos[1])
                await move(x1,y1)
                await button_down(x1,y1,"left")
                await asyncio.sleep(0.05)
                await move(x2,y2)
                await button_up(x2,y2,"left")
                return

            if typ == "SELECT_TEXT" and isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, list) and len(p) == 2 for p in pos):
                x1,y1 = to_xy(pos[0]); x2,y2 = to_xy(pos[1])
                await move(x1,y1)
                await button_down(x1,y1,"left")
                await move(x2,y2)
                await button_up(x2,y2,"left")
                return

            if typ == "COPY":
                await key_down("Control")
                await key_once("c")
                await key_up("Control")
                logger.info("[COPY] hint=%r", action.get("value"))
                await self._send_chat_safe("Copied selection to clipboard.")
                return

            if typ == "ANSWER":
                logger.info("[ANSWER] %r", val)
                await self._send_chat_safe(f"Answer: {val}")
                return

            logger.warning("Unsupported or malformed action: %r", action)
        except Exception as e:
            logger.error("Action execution failed: %s | action=%r", e, action, exc_info=True)

    # --- ICE & Tracks
    async def _on_ice(self, cand: Optional[RTCIceCandidate]) -> None:
        """Handle outgoing ICE candidates from the local peer connection.

        This callback is triggered when the local RTCPeerConnection generates
        ICE candidates that need to be sent to the remote peer via WebSocket.

        :param cand: The ICE candidate to send, or None if gathering is complete.
        :type cand: Optional[RTCIceCandidate]
        :return: None
        :rtype: None
        """
        if not cand or not self.signaler.ws:
            return
        await self.signaler.send({
            "event": "signal/candidate",
            "payload": {
                "candidate": cand.candidate,
                "sdpMid": cand.sdpMid,
                "sdpMLineIndex": cand.sdpMLineIndex,
            },
        })

    async def _on_track(self, track: VideoStreamTrack) -> None:
        """Handle incoming media tracks from the remote peer.

        This callback is triggered when a new media track (video/audio) is
        received from the remote peer. For video tracks, it starts the
        WebRTC frame source to begin processing frames.

        :param track: The received media track.
        :type track: VideoStreamTrack
        :return: None
        :rtype: None
        """
        logger.info("RTC: track=%s id=%s", track.kind, getattr(track, "id", "unknown"))
        if track.kind == "video" and isinstance(self.frame_source, WebRTCFrameSource):
            await self.frame_source.start(track)

    def _parse_remote_candidate(self, payload: Dict[str, Any]) -> Optional[RTCIceCandidate]:
        """Parse an ICE candidate from WebSocket payload data.

        This method converts ICE candidate information received via WebSocket
        into an RTCIceCandidate object that can be added to the peer connection.
        It handles various candidate string formats and extracts SDP metadata.

        :param payload: WebSocket payload containing candidate information.
        :type payload: Dict[str, Any]
        :return: Parsed ICE candidate, or None if parsing fails.
        :rtype: Optional[RTCIceCandidate]
        """
        cand_str = (payload or {}).get("candidate")
        if not cand_str:
            return None
        if cand_str.startswith("candidate:"):
            cand_str = cand_str.split(":",1)[1]
        ice = candidate_from_sdp(cand_str)
        ice.sdpMid = payload.get("sdpMid")
        sdp_mline = payload.get("sdpMLineIndex")
        if isinstance(sdp_mline, str) and sdp_mline.isdigit():
            sdp_mline = int(sdp_mline)
        ice.sdpMLineIndex = sdp_mline
        return ice

    # --- Cleanup
    async def _cleanup(self) -> None:
        """Clean up all resources and connections.

        This method performs comprehensive cleanup including stopping frame
        sources, closing WebRTC connections, shutting down executors, and
        optionally terminating the process if non-daemon threads linger.

        :return: None
        :rtype: None
        """
        if getattr(self, "_cleaning", False):
            return
        self._cleaning = True
        try:
            self._tg = None

            # Cancel WS loops promptly to reduce traffic during teardown
            sig = getattr(self, "signaler", None)
            if sig is not None:
                try:
                    for t in list(sig._tasks):
                        if not t.done():
                            t.cancel()
                    if sig._tasks:
                        await asyncio.gather(*sig._tasks, return_exceptions=True)
                finally:
                    sig._tasks.clear()

            # Stop model executor
            if hasattr(self, '_model_executor'):
                self._model_executor.shutdown(wait=True, cancel_futures=True)
                delattr(self, '_model_executor')

            # Cancel RTCP keepalive task if enabled and running
            if hasattr(self, "_rtcp_task"):
                self._rtcp_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._rtcp_task
                with contextlib.suppress(Exception):
                    delattr(self, "_rtcp_task")

            # Best-effort unhost
            try:
                if getattr(self.signaler, "ws", None) and not self.signaler._closed.is_set():
                    await self.signaler.send({"event":"control/release"})
            except Exception:
                pass

            # Stop frame source FIRST so the reader exits before the PC/track vanish.
            fs = getattr(self, "frame_source", None)
            if fs:
                with contextlib.suppress(Exception):
                    try:
                        async with asyncio.timeout(3):
                            await fs.stop()
                    except asyncio.TimeoutError:
                        pass
                self.frame_source = None

            # Close PC - stop receivers/transceivers & tracks before close
            pc = getattr(self, "pc", None)
            if pc:
                try:
                    # Stop receivers/transceivers; some .stop() are sync, some async
                    recv_tasks = []
                    for r in (pc.getReceivers() or []):
                        stop = getattr(r, "stop", None)
                        if callable(stop):
                            res = stop()
                            if asyncio.iscoroutine(res):
                                recv_tasks.append(res)
                        # also stop underlying track if present
                        tr = getattr(r, "track", None)
                        if tr and hasattr(tr, "stop"):
                            with contextlib.suppress(Exception):
                                tr.stop()
                    xcv_tasks = []
                    for t in (pc.getTransceivers() or []):
                        stop = getattr(t, "stop", None)
                        if callable(stop):
                            res = stop()
                            if asyncio.iscoroutine(res):
                                xcv_tasks.append(res)

                    for sender in pc.getSenders() or []:
                        track = getattr(sender, "track", None)
                        if track:
                            with contextlib.suppress(Exception):
                                await track.stop()

                    if recv_tasks or xcv_tasks:
                        try:
                            async with asyncio.timeout(3):
                                await asyncio.gather(*recv_tasks, *xcv_tasks, return_exceptions=True)
                        except asyncio.TimeoutError:
                            logger.warning("Timeout while stopping receivers/transceivers")
                except Exception:
                    pass

                with contextlib.suppress(Exception):
                    try:
                        async with asyncio.timeout(5):
                            await pc.close()
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for RTCPeerConnection.close()")
                self.pc = None

            # Close WS last
            sig = getattr(self, "signaler", None)
            if sig:
                with contextlib.suppress(Exception):
                    try:
                        async with asyncio.timeout(5):
                            await sig.close()
                    except asyncio.TimeoutError:
                        pass

            # Stop Prometheus HTTP server if we started one
            ms = getattr(self, "_metrics_server", None)
            mt = getattr(self, "_metrics_thread", None)
            if ms is not None:
                with contextlib.suppress(Exception):
                    if hasattr(ms, "shutdown"):
                        ms.shutdown()
                    if hasattr(ms, "server_close"):
                        ms.server_close()
            if mt is not None:
                with contextlib.suppress(Exception):
                    mt.join(timeout=1.0)
            for attr in ("_metrics_server","_metrics_thread"):
                if hasattr(self, attr):
                    with contextlib.suppress(Exception):
                        delattr(self, attr)

            # Small grace period for background libs
            await asyncio.sleep(0.2)

            # Optional hard-exit guard if non-daemon threads linger
            if self.settings.force_exit_guard_ms > 0:
                await asyncio.sleep(self.settings.force_exit_guard_ms / 1000.0)
                import threading, os as _os
                actives = [t for t in threading.enumerate()
                           if t.is_alive() and t is not threading.current_thread() and not t.daemon]
                if actives:
                    names = [t.name for t in actives]
                    logger.warning("Non-daemon threads still active: %s", names)
                    logger.info("Forcing process exit now to avoid hang")
                    _os._exit(0)

        finally:
            self._cleaning = False
            logger.info("Cleanup complete; agent resources released.")

    async def _shutdown_watcher(self) -> None:
        """Watch for shutdown signal and initiate cleanup.

        This background task waits for the shutdown event and then triggers
        cleanup by closing the signaler and cancelling related tasks.
        It raises CancelledError to terminate the task group.

        :return: None
        :rtype: None
        """
        await self.shutdown.wait()
        sig = getattr(self, "signaler", None)
        if sig is not None:
            try:
                sig._closed.set()
                for t in list(getattr(sig, "_tasks", [])):
                    if not t.done():
                        t.cancel()
            except Exception:
                pass
        raise asyncio.CancelledError

# ----------------------
# Entry Point / Boot
# ----------------------
def start_metrics_server(port: int, logger: logging.Logger) -> Tuple[Any, Any]:
    """Start Prometheus metrics server.

    Initializes a Prometheus HTTP server on the specified port to expose
    application metrics. Handles both single server and server/thread tuple
    return values from the underlying start_http_server function.

    :param port: Port number to bind the metrics server to
    :type port: int
    :param logger: Logger instance for recording server startup status
    :type logger: logging.Logger
    :return: Tuple of (server, thread) for clean shutdown. Thread may be None
             if the underlying implementation doesn't return a thread
    :rtype: Tuple[Any, Any]
    """
    try:
        ret = start_http_server(port)
        if isinstance(ret, tuple) and len(ret) == 2:
            server, thread = ret
        else:
            server, thread = ret, None
        logger.info("Metrics server started on port %d", port)
        return server, thread
    except Exception as e:
        logger.error("Failed to start metrics server on port %d: %s", port, e)
        return None, None

async def main() -> None:
    """Main entry point for the Neko agent application.

    This function handles command line argument parsing, configuration loading,
    logging setup, model loading, metrics server startup, authentication
    (REST login if needed), and agent initialization.
    It supports both direct WebSocket connections and REST-based login.
    Follows 12-Factor App principles with centralized configuration.

    :return: None
    :rtype: None
    """
    import argparse

    p = argparse.ArgumentParser("neko_agent", description="Production-ready Neko v3 WebRTC agent (ShowUI-2B)")
    p.add_argument("--ws",         default=os.environ.get("NEKO_WS", None), help="wss://…/api/ws?token=…  (direct; else REST)")
    p.add_argument("--task",       default=os.environ.get("NEKO_TASK", "Search the weather"), help="Navigation task")
    p.add_argument("--mode",       default=os.environ.get("NEKO_MODE", "web"), choices=list(ACTION_SPACES.keys()), help="Mode: web or phone")
    p.add_argument("--max-steps",  type=int, help="Max navigation steps (overrides env)")
    p.add_argument("--metrics-port", type=int, help="Prometheus metrics port (overrides env and $PORT)")
    p.add_argument("--loglevel",   default=os.environ.get("NEKO_LOGLEVEL","INFO"), help="Logging level")
    p.add_argument("--no-audio",   dest="audio", action="store_false", help="Disable audio stream")
    p.add_argument("--online",     action="store_true", help="Keep running after task completes and wait for more commands")
    p.add_argument("--neko-url",   default=os.environ.get("NEKO_URL",None), help="Base https://host for REST login")
    p.add_argument("--username",   default=os.environ.get("NEKO_USER",None), help="REST username")
    p.add_argument("--password",   default=os.environ.get("NEKO_PASS",None), help="REST password")
    p.add_argument("--healthcheck", action="store_true", help="Validate configuration and exit")
    p.set_defaults(audio=None)  # Will use settings.audio_default if None
    args = p.parse_args()

    # Load configuration from environment
    settings = Settings.from_env()

    # Handle --healthcheck flag
    if args.healthcheck:
        errors = settings.validate()
        if errors:
            print("Configuration validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
        else:
            print("ok")
            sys.exit(0)

    # Setup logging
    logger = setup_logging(settings)

    # Override log level if specified via CLI
    if args.loglevel:
        logging.getLogger().setLevel(args.loglevel.upper())

    # In online mode, ignore any provided task and start idle.
    if args.online:
        if args.task and args.task.strip():
            logger.info("--online specified; ignoring initial --task and waiting for chat tasks.")
        args.task = ""

    # Start metrics server
    metrics_port = args.metrics_port if args.metrics_port is not None else settings.metrics_port
    metrics_server, metrics_thread = start_metrics_server(metrics_port, logger)

    logger.info("Loading model/processor ...")
    device = "cpu"
    dtype = torch.float32

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("CUDA GPU detected: %s (%.1fGB)", gpu_name, gpu_memory)
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16, device="mps")
            dtype = torch.bfloat16
        except RuntimeError:
            dtype = torch.float32
        device = "mps"
        logger.info("Apple MPS detected")
        os.makedirs(settings.offload_folder, exist_ok=True)
    else:
        logger.warning("No GPU acceleration available - using CPU")

    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto"}
    if device == "mps":
        model_kwargs.update({"offload_folder": settings.offload_folder, "offload_state_dict": True})

    model = Qwen2VLForConditionalGeneration.from_pretrained(settings.repo_id, **model_kwargs).eval()
    processor = AutoProcessor.from_pretrained(
        settings.repo_id,
        size={"shortest_edge": settings.size_shortest_edge, "longest_edge": settings.size_longest_edge},
        trust_remote_code=True
    )

    logger.info("Model loaded successfully on device: %s (dtype: %s)", model.device, dtype)

    if device == "cuda":
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9
        logger.info("GPU memory allocated: %.2fGB", allocated_memory)

    ws_url = args.ws
    if not ws_url or ws_url == settings.default_ws:
        if not (args.neko_url and args.username and args.password):
            p.error("Provide --ws OR all of --neko-url, --username, --password")
        if any(a.startswith("--password") or a.startswith("--username") for a in sys.argv):
            print("[WARN] Consider env vars for secrets.", file=sys.stderr)
        try:
            base = args.neko_url.rstrip("/")
            r = requests.post(f"{base}/api/login", json={"username":args.username,"password":args.password}, timeout=10)
            r.raise_for_status()
            tok = r.json().get("token")
            if not tok:
                raise RuntimeError("REST login ok, but no token in response")
            host = base.split("://",1)[-1].rstrip("/")
            scheme = "wss" if base.startswith("https") else "ws"
            ws_url = f"{scheme}://{host}/api/ws?token={tok}"
            print(f"[INFO] REST login OK, WS host={host} path=/api/ws", file=sys.stderr)
        except Exception as e:
            print(f"REST login failed: {e}", file=sys.stderr); sys.exit(1)
    elif any((args.neko_url, args.username, args.password)):
        print("[WARN] --ws provided; ignoring REST args", file=sys.stderr)

    # Create agent with dependency injection (no side effects in constructor)
    agent = NekoAgent(
        model=model,
        processor=processor,
        ws_url=ws_url,
        nav_task=args.task,
        nav_mode=args.mode,
        settings=settings,
        logger=logger,
        max_steps=args.max_steps,
        metrics_port=metrics_port,
        audio=args.audio,
        online=args.online,
    )

    # Inject metrics server handles for clean shutdown
    agent._metrics_server = metrics_server
    agent._metrics_thread = metrics_thread
    try:
        await agent.run()
    finally:
        # Clean up metrics server
        if metrics_server and hasattr(metrics_server, 'shutdown'):
            try:
                metrics_server.shutdown()
                logger.info("Metrics server shut down")
            except Exception as e:
                logger.error("Error shutting down metrics server: %s", e)

if __name__ == "__main__":
    asyncio.run(main())

def cli() -> None:
    """Synchronous console entrypoint wrapper for packaging.

    This function provides a synchronous entry point that can be used
    as a console script in package installation. It simply runs the
    async main() function using asyncio.run().

    :return: None
    :rtype: None
    """
    asyncio.run(main())
