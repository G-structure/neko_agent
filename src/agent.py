#!/usr/bin/env python3
"""
agent.py — ShowUI-2B Neko v3 WebRTC GUI agent.

This agent connects to a Neko v3 server, performs the *correct* WebRTC
signaling handshake so that control events are accepted, receives screen
frames (WebRTC or "lite" WS base64), runs a VLM to decide actions, and
emits Neko control events (move/click/keys/scroll/etc).

Hardenings brought over from the production manual CLI:

- Quiet underlying library DEBUG noise (websockets, aiortc).
- Robust REST login to acquire a WebSocket token when needed.
- Proper heartbeat handling: reply `client/heartbeat` to `system/heartbeat`.
- Dynamic keyboard map learning via `keyboard/map` (accepts both payload shapes).
- Auto re-request host control if it is lost or given to another user.
- Strict ICE mapping from server payload (use only urls/username/credential).
- Correct candidate parsing (accept "candidate:..." and str/int sdpMLineIndex).
- Safe async task lifecycle and reconnection with cleanup.
- Align control protocol details (e.g., `control/scroll` uses delta_x/delta_y).
- Routes and logs `error/...` events.
- **Buffered early ICE**: candidates arriving before SRD are captured and applied post-SRD.
- **Backoff jitter**: randomized delay added to exponential reconnects to avoid thundering herd.

Environment flags used by this agent are compatible with prior versions
but this file should be considered the new canonical production agent.

"""

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
import multiprocessing
import threading
from typing import Any, Dict, List, Optional, Tuple, Set

# --- third-party
import torch
import requests
import websockets
from abc import ABC, abstractmethod
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
)
from aiortc.sdp import candidate_from_sdp

# Fail fast on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ----------------------
# Configuration / ENV
# ----------------------
MODEL_KEY            = os.environ.get("MODEL_KEY", "showui-2b")
REPO_ID              = os.environ.get("REPO_ID", "showlab/ShowUI-2B")
SIZE_SHORTEST_EDGE   = int(os.environ.get("SIZE_SHORTEST_EDGE", 224))
SIZE_LONGEST_EDGE    = int(os.environ.get("SIZE_LONGEST_EDGE", 1344))
DEFAULT_WS           = os.environ.get("NEKO_WS", "wss://neko.example.com/api/ws")
DEFAULT_METRIC_PORT  = int(os.environ.get("NEKO_METRICS_PORT", 9000))
MAX_STEPS            = int(os.environ.get("NEKO_MAX_STEPS", 8))
AUDIO_DEFAULT        = bool(int(os.environ.get("NEKO_AUDIO", "1")))
FRAME_SAVE_PATH      = os.environ.get("FRAME_SAVE_PATH", None)
CLICK_SAVE_PATH      = os.environ.get("CLICK_SAVE_PATH", None)
OFFLOAD_FOLDER       = os.environ.get("OFFLOAD_FOLDER", "./offload")
REFINEMENT_STEPS     = int(os.environ.get("REFINEMENT_STEPS", "3"))
NEKO_LOGFILE         = os.environ.get("NEKO_LOGFILE", None)
NEKO_LOGLEVEL        = os.environ.get("NEKO_LOGLEVEL", "INFO")

# Whether to append default STUN/TURN (legacy behavior); if "strict" we only use server ICE.
NEKO_ICE_POLICY      = os.environ.get("NEKO_ICE_POLICY", "strict")  # "strict" | "all"
NEKO_STUN_URL        = os.environ.get("NEKO_STUN_URL", "stun:stun.l.google.com:19302")
NEKO_TURN_URL        = os.environ.get("NEKO_TURN_URL")
NEKO_TURN_USER       = os.environ.get("NEKO_TURN_USER")
NEKO_TURN_PASS       = os.environ.get("NEKO_TURN_PASS")

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    level=NEKO_LOGLEVEL,
    format='[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logger = logging.getLogger("neko_agent")

# Optional file handler
if NEKO_LOGFILE:
    try:
        fh = logging.FileHandler(NEKO_LOGFILE)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s'))
        logging.getLogger().addHandler(fh)
        logger.info("File logging enabled: %s", NEKO_LOGFILE)
    except Exception as e:
        logger.error("Failed to set up file logging to %s: %s", NEKO_LOGFILE, e)

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
    if len(name) == 1:
        return KEYSYM.get(name, KEYSYM.get(name.lower(), ord(name)))
    return KEYSYM.get(name, KEYSYM.get(name.capitalize(), 0))

def clamp_xy(x:int,y:int,size:Tuple[int,int]) -> Tuple[int,int]:
    w,h = size
    return max(0,min(x,w-1)), max(0,min(y,h-1))

def resize_and_validate_image(image: Image.Image) -> Image.Image:
    ow,oh = image.size
    me = max(ow,oh)
    if me > SIZE_LONGEST_EDGE:
        scale = SIZE_LONGEST_EDGE / me
        nw,nh = int(ow*scale), int(oh*scale)
        t0 = time.monotonic()
        image = image.resize((nw,nh), Image.LANCZOS)
        resize_duration.observe(time.monotonic()-t0)
        logger.info("Resized %dx%d -> %dx%d", ow, oh, nw, nh)
    return image

def save_atomic(img: Image.Image, path: str) -> None:
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

def frame_to_pil_image(frame: VideoStreamTrack) -> Image.Image:
    try:
        img = frame.to_image()
        w,h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")
        rgb = img.convert("RGB").copy()
        if rgb.size != (w,h):
            raise ValueError(f"Image size changed during conversion: {img.size}->{rgb.size}")
        if FRAME_SAVE_PATH:
            with contextlib.suppress(Exception):
                save_atomic(rgb, FRAME_SAVE_PATH)
        return rgb
    except Exception as e:
        logger.error("Frame conversion failed: %s", e)
        raise

def draw_action_markers(img: Image.Image, action: Dict[str, Any], step: int) -> Image.Image:
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
    if font:
        bbox = d.textbbox((0,0), label, font=font)
        tw,th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    else:
        tw,th = len(label)*8, 16
    d.rectangle([10,10, 10+tw+10, 10+th+10], fill=(0,0,0,128))
    d.text((15,15), label, fill=(255,255,255), font=font if font else None)
    return out

# ----------------------
# Parsing LLM actions
# ----------------------
def safe_parse_action(output_text: str, nav_mode: str="web") -> Optional[Dict[str,Any]]:
    try:
        act = json.loads(output_text)
    except json.JSONDecodeError:
        try:
            act = ast.literal_eval(output_text)
        except (ValueError, SyntaxError) as e:
            logger.error("Parse error: %s | Raw=%r", e, output_text)
            parse_errors.inc()
            return None

    try:
        assert isinstance(act, dict)
        typ = act.get("action")
        if typ not in ACTION_SPACES.get(nav_mode, []):
            logger.warning("Non-whitelisted action: %r", typ)
            parse_errors.inc()
            return None
        for k in ("action","value","position"):
            assert k in act, f"Missing key {k}"
        return act
    except AssertionError as e:
        logger.error("Schema validation error: %s | Parsed=%r", e, act)
        parse_errors.inc()
        return None

# ----------------------
# Event bus + WS Signaler
# ----------------------
class LatestOnly:
    def __init__(self):
        self._val = None
        self._event = asyncio.Event()
    def set(self, v):
        self._val = v
        self._event.set()
    async def get(self):
        await self._event.wait()
        self._event.clear()
        return self._val

class Broker:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.latest: Dict[str, LatestOnly] = {}
        self.waiters: Dict[str, asyncio.Future] = {}

    def topic_queue(self, topic: str, maxsize: int = 512) -> asyncio.Queue:
        if topic not in self.queues:
            self.queues[topic] = asyncio.Queue(maxsize=maxsize)
        return self.queues[topic]

    def topic_latest(self, topic: str) -> LatestOnly:
        if topic not in self.latest:
            self.latest[topic] = LatestOnly()
        return self.latest[topic]

    def publish(self, msg: Dict[str, Any]) -> None:
        ev = msg.get("event","")
        # Match request/response if used
        if (rid := msg.get("reply_to")) and (fut := self.waiters.pop(rid, None)):
            if not fut.done():
                fut.set_result(msg)
            return
        # Route major prefixes
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
        else:
            self.topic_queue("misc").put_nowait(msg)

class Signaler:
    def __init__(self, url: str, **wsopts):
        self.url = url
        self.wsopts = dict(ping_interval=10, ping_timeout=20, max_queue=256, **wsopts)
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._tasks: Set[asyncio.Task] = set()
        self._sendq: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.broker = Broker()
        self._closed = asyncio.Event()

    async def connect_with_backoff(self):
        """Connect with exponential backoff + jitter."""
        backoff = 1
        while not self._closed.is_set():
            try:
                self.ws = await websockets.connect(self.url, **self.wsopts)
                logger.info("WebSocket connected: %s", self.url)
                self._tasks.add(asyncio.create_task(self._read_loop(), name="ws-read"))
                self._tasks.add(asyncio.create_task(self._send_loop(), name="ws-send"))
                self._closed.clear()
                return self
            except Exception as e:
                # ---- Regression fix: add small random jitter to avoid sync retries
                jitter = random.uniform(0, max(0.25, backoff * 0.25))
                delay = min(backoff + jitter, 30)
                logger.error("WS connect error: %s - retrying in %.2fs", e, delay)
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 30)

    async def close(self):
        self._closed.set()
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._tasks.clear()
        if self.ws and self.ws.open:
            with contextlib.suppress(Exception):
                await self.ws.close()

    async def _read_loop(self):
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
        try:
            while not self._closed.is_set():
                msg = await self._sendq.get()
                try:
                    await self.ws.send(json.dumps(msg))
                except websockets.ConnectionClosed:
                    logger.warning("Send failed: WS closed. Re-queue and exit send loop.")
                    await self._sendq.put(msg)
                    break
        except asyncio.CancelledError:
            logger.info("WS send loop cancelled")
        finally:
            self._closed.set()

    async def send(self, msg: Dict[str, Any]) -> None:
        await self._sendq.put(msg)

# ----------------------
# Frame Sources
# ----------------------
class FrameSource(ABC):
    @abstractmethod
    async def start(self, *args: Any) -> None: ...
    @abstractmethod
    async def stop(self) -> None: ...
    @abstractmethod
    async def get(self) -> Optional[Image.Image]: ...

class WebRTCFrameSource(FrameSource):
    def __init__(self) -> None:
        self.image: Optional[Image.Image] = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.first_frame = asyncio.Event()

    async def start(self, *args: Any) -> None:
        if not args:
            raise ValueError("WebRTCFrameSource.start(): need VideoStreamTrack")
        track = args[0]
        await self.stop()
        self.task = asyncio.create_task(self._reader(track))

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None
        async with self.lock:
            self.image = None

    async def _reader(self, track: VideoStreamTrack) -> None:
        frame_count = 0
        skip_initial = 3
        try:
            while True:
                try:
                    frame = await track.recv()
                    frame_count += 1
                    if frame_count <= skip_initial:
                        continue
                    if not hasattr(frame, "to_image"):
                        continue
                    img = frame_to_pil_image(frame)
                    w,h = img.size
                    if w < 32 or h < 32 or w > 8192 or h > 8192:
                        continue
                    async with self.lock:
                        self.image = img
                        if not self.first_frame.is_set():
                            self.first_frame.set()
                    frames_received.inc()
                except Exception as e:
                    logger.warning("Frame process failed (continuing): %s", e)
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Frame reader stopped: %s", e)

    async def get(self) -> Optional[Image.Image]:
        async with self.lock:
            # Return None if connection is closed to prevent broken state
            if hasattr(self, 'signaler') and hasattr(self.signaler, '_closed') and self.signaler._closed.is_set():
                return None
            return self.image.copy() if self.image else None

class LiteFrameSource(FrameSource):
    def __init__(self, signaler: Signaler) -> None:
        self.signaler = signaler
        self.image: Optional[Image.Image] = None
        self.first_frame = asyncio.Event()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, *args: Any) -> None:
        if self._task:
            await self.stop()
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
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
        ch = self.signaler.broker.topic_latest("video")
        try:
            while self._running and self.signaler.ws:
                msg = await ch.get()
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
            pass
        except Exception as e:
            logger.warning("LiteFrameSource consumer exiting: %s", e)

    def _decode_frame_base64(self, data: str) -> Optional[Image.Image]:
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
        async with self._lock:
            # Return None if connection is closed to prevent broken state
            if self.signaler._closed.is_set():
                return None
            return self.image.copy() if self.image else None


# ----------------------
# Agent
# ----------------------
class NekoAgent:
    def __init__(
        self,
        model,
        processor,
        ws_url: str,
        nav_task: str,
        nav_mode: str,
        max_steps: int = MAX_STEPS,
        refinement_steps: int = REFINEMENT_STEPS,
        metrics_port: int = DEFAULT_METRIC_PORT,
        audio: bool = AUDIO_DEFAULT,
    ):
        self.signaler = Signaler(ws_url)
        self.frame_source: Optional[FrameSource] = None
        self.nav_task = nav_task
        self.nav_mode = nav_mode
        self.max_steps = max_steps
        self.refinement_steps = refinement_steps
        self.audio = audio
        self.model = model
        self.processor = processor
        self.run_id = os.environ.get("NEKO_RUN_ID") or str(uuid.uuid4())[:8]
        self.pc: Optional[RTCPeerConnection] = None
        self.shutdown = asyncio.Event()
        self.loop = asyncio.get_event_loop()
        self.is_lite = False
        self._tg: Optional[asyncio.TaskGroup] = None
        self._current_inference_task: Optional[asyncio.Future] = None

        # server/session awareness
        self.session_id: Optional[str] = None
        self.screen_size: Tuple[int,int] = (1280, 720)

        self.sys_prompt = _NAV_SYSTEM.format(
            _APP=self.nav_mode,
            _ACTION_SPACE=ACTION_SPACE_DESC[self.nav_mode],
        )
        start_http_server(metrics_port)


    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        def signal_handler(signum, frame=None):
            sig_name = signal.Signals(signum).name
            logger.info("🛑 Received %s signal - initiating clean shutdown...", sig_name)
            self.shutdown.set()
            # Cancel inference if it's running
            if hasattr(self, '_current_inference_task') and self._current_inference_task:
                self._current_inference_task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler, sig)

        while not self.shutdown.is_set():
            reconnects.inc()
            try:
                await self.signaler.connect_with_backoff()

                # Request media and host on connect
                await self.signaler.send({"event": "signal/request", "payload": {"video": {}, "audio": {} if self.audio else {}}})
                await self.signaler.send({"event": "control/request"})
                await self.signaler.send({"event": "session/watch", "payload": {"id": "main"}})

                # Wait for offer/provide message before starting consumers and main loop
                control_q = self.signaler.broker.topic_queue("control")
                offer_msg = None
                while not offer_msg:
                    msg = await control_q.get()
                    if msg.get("event") in ("signal/offer", "signal/provide"):
                        offer_msg = msg

                # Setup media with the offer message
                await self._setup_media(offer_msg)

                # Now start consumers and main loop
                try:
                    async with asyncio.TaskGroup() as tg:
                        self._tg = tg
                        tg.create_task(self._consume_system_topic())   # heartbeat, keyboard map, host, screen size
                        tg.create_task(self._consume_chat_topic())     # updates to nav task
                        if not self.is_lite:
                            tg.create_task(self._consume_ice_topic())
                            tg.create_task(self._consume_control_topic())
                        else:
                            tg.create_task(self._consume_video_lite_topic())
                        tg.create_task(self._main_loop())
                except* asyncio.CancelledError:
                    logger.info("🚫 Task group cancelled - cleaning up")
                except* Exception as e:
                    logger.error("❌ Task group error: %s", e, exc_info=True)

            except Exception as e:
                logger.error("Connect/RTC error: %s", e, exc_info=True)
            finally:
                await self._cleanup()
                if not self.shutdown.is_set():
                    logger.info("Disconnected — attempting to reconnect shortly.")
                    await asyncio.sleep(0.5)

    # --- Media / Signaling
    async def _setup_media(self, offer_msg: Optional[Dict[str, Any]] = None) -> None:
        if offer_msg is None:
            control_q = self.signaler.broker.topic_queue("control")

            # ---- Regression fix: buffer any ICE candidates that might arrive early
            early_ice_payloads: List[Dict[str, Any]] = []
            ice_q = self.signaler.broker.topic_queue("ice")
            buffer_running = True

            async def _buffer_ice():
                # Collect ICE messages until we stop buffering after SRD
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

            # Wait for offer/provide
            offer_msg = None
            while not offer_msg:
                msg = await control_q.get()
                if msg.get("event") in ("signal/offer", "signal/provide"):
                    offer_msg = msg
        else:
            # If offer_msg provided, still need to buffer ICE candidates
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

        # Optionally append default STUN/TURN when policy != strict
        if NEKO_ICE_POLICY != "strict":
            ice_servers.append(RTCIceServer(urls=[NEKO_STUN_URL]))
            if NEKO_TURN_URL:
                ice_servers.append(RTCIceServer(urls=[NEKO_TURN_URL + "?transport=tcp"],
                                                username=NEKO_TURN_USER, credential=NEKO_TURN_PASS))

        config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(config)
        self.pc = pc
        self.frame_source = WebRTCFrameSource()

        @pc.on("iceconnectionstatechange")
        def _on_ics():
            logger.info("iceConnectionState -> %s", pc.iceConnectionState)

            # Also start keepalive on ICE connected as backup approach
            if (pc.iceConnectionState == "connected" and
                pc.connectionState in ("connecting", "connected") and
                not hasattr(self, '_rtcp_task')):
                logger.info("ICE connection established - starting RTCP keepalive (backup trigger)")

                async def rtcp_keepalive_ice():
                    """Send periodic RTCP packets to prevent ICE timeout (ICE-triggered)"""
                    logger.info("RTCP keepalive task starting (ICE trigger)")
                    keepalive_count = 0

                    # Run keepalive while ICE remains connected
                    while (self.pc and
                           self.pc.iceConnectionState == "connected" and
                           self.pc.connectionState in ("connecting", "connected")):
                        try:
                            # Get stats forces RTCP SR/RR packets to be sent - with timeout protection
                            stats = await asyncio.wait_for(self.pc.getStats(), timeout=1.0)
                            keepalive_count += 1
                            logger.info("RTCP keepalive (ICE) #%d sent (got %d stats)",
                                      keepalive_count, len(stats) if stats else 0)
                        except asyncio.TimeoutError:
                            logger.warning("RTCP keepalive (ICE) #%d timed out - getStats() blocked", keepalive_count + 1)
                            # Still increment count and continue - server expects regular timing
                            keepalive_count += 1
                        except Exception as e:
                            logger.warning("RTCP keepalive (ICE) failed: %s", e)
                            break

                        # Send every 1.5 seconds (must be < 2s server keepalive interval)
                        await asyncio.sleep(1.5)

                    logger.info("RTCP keepalive (ICE) task ended after %d packets (ice: %s, conn: %s)",
                               keepalive_count,
                               self.pc.iceConnectionState if self.pc else "None",
                               self.pc.connectionState if self.pc else "None")

                # Create and store the task
                try:
                    loop = asyncio.get_running_loop()
                    self._rtcp_task = loop.create_task(rtcp_keepalive_ice())
                    logger.info("RTCP keepalive task created (ICE trigger)")
                except RuntimeError:
                    self._rtcp_task = asyncio.create_task(rtcp_keepalive_ice())
                    logger.info("RTCP keepalive task created (ICE trigger, fallback)")

            # Clean up keepalive task if ICE is no longer connected
            elif (pc.iceConnectionState in ("disconnected", "failed", "closed") and
                  hasattr(self, '_rtcp_task')):
                logger.info("Stopping RTCP keepalive due to ICE state: %s", pc.iceConnectionState)
                self._rtcp_task.cancel()
                try:
                    delattr(self, '_rtcp_task')
                except AttributeError:
                    pass

        @pc.on("connectionstatechange")
        def _on_cs():
            logger.info("connectionState -> %s", pc.connectionState)

            # Start RTCP keepalive when connected
            if pc.connectionState == "connected" and not hasattr(self, '_rtcp_task'):
                logger.info("WebRTC connection established - starting RTCP keepalive")

                async def rtcp_keepalive():
                    """Send periodic RTCP packets to prevent ICE timeout"""
                    logger.info("RTCP keepalive task starting (connection established)")
                    keepalive_count = 0

                    # Run keepalive while connection remains connected
                    while self.pc and self.pc.connectionState == "connected":
                        try:
                            # Get stats forces RTCP SR/RR packets to be sent - with timeout protection
                            stats = await asyncio.wait_for(self.pc.getStats(), timeout=1.0)
                            keepalive_count += 1
                            logger.info("RTCP keepalive #%d sent (got %d stats)",
                                      keepalive_count, len(stats) if stats else 0)
                        except asyncio.TimeoutError:
                            logger.warning("RTCP keepalive #%d timed out - getStats() blocked", keepalive_count + 1)
                            # Still increment count and continue - server expects regular timing
                            keepalive_count += 1
                        except Exception as e:
                            logger.warning("RTCP keepalive failed: %s", e)
                            break

                        # Send every 1.5 seconds (must be < 2s server keepalive interval)
                        await asyncio.sleep(1.5)

                    logger.info("RTCP keepalive task ended after %d packets (connection state: %s)",
                               keepalive_count, self.pc.connectionState if self.pc else "None")

                # Create and store the task - schedule it on the event loop
                try:
                    loop = asyncio.get_running_loop()
                    self._rtcp_task = loop.create_task(rtcp_keepalive())
                    logger.info("RTCP keepalive task created and scheduled")
                except RuntimeError:
                    # Fallback if no loop running (shouldn't happen in this context)
                    self._rtcp_task = asyncio.create_task(rtcp_keepalive())
                    logger.info("RTCP keepalive task created (fallback)")

            # Clean up keepalive task if connection is no longer connected
            elif pc.connectionState in ("disconnected", "failed", "closed"):
                if hasattr(self, '_rtcp_task'):
                    logger.info("Stopping RTCP keepalive due to connection state: %s", pc.connectionState)
                    self._rtcp_task.cancel()
                    # Remove the attribute immediately to allow recreation
                    try:
                        delattr(self, '_rtcp_task')
                    except AttributeError:
                        pass

        @pc.on("icecandidate")
        async def _on_ic(cand):
            await self._on_ice(cand)

        @pc.on("track")
        async def _on_tr(track):
            await self._on_track(track)

        remote_sdp = payload.get("sdp")
        remote_type = payload.get("type", "offer")
        if not remote_sdp:
            # stop buffering before raising
            buffer_running = False
            buf_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await buf_task
            raise RuntimeError("Missing SDP in offer payload")

        # Apply remote SDP and answer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=remote_sdp, type=remote_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self.signaler.send({
            "event": "signal/answer",
            "payload": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        })

        # Stop buffering and apply any early candidates collected before SRD
        buffer_running = False
        buf_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await buf_task

        # Drain any remaining ICE already queued (non-blocking) and append to buffer
        while True:
            try:
                msg = ice_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                if msg.get("event") == "signal/candidate":
                    early_ice_payloads.append(msg.get("payload") or {})

        # Add buffered candidates now that SRD is set
        for pay in early_ice_payloads:
            try:
                ice = self._parse_remote_candidate(pay)
                if ice:
                    await pc.addIceCandidate(ice)
            except Exception as e:
                logger.debug("Applying buffered ICE failed: %s", e)

    async def _consume_control_topic(self) -> None:
        q = self.signaler.broker.topic_queue("control")
        try:
            while not self.shutdown.is_set():
                msg = await q.get()
                if msg.get("event") == "signal/close":
                    logger.info("Remote close received; triggering reconnection.")
                    # Don't set shutdown - let the connection restart
                    return
        except asyncio.CancelledError:
            pass

    async def _consume_ice_topic(self) -> None:
        q = self.signaler.broker.topic_queue("ice")
        try:
            while not self.shutdown.is_set():
                msg = await q.get()
                if msg.get("event") == "signal/candidate":
                    ice = self._parse_remote_candidate(msg.get("payload") or {})
                    if ice and self.pc:
                        with contextlib.suppress(Exception):
                            await self.pc.addIceCandidate(ice)
        except asyncio.CancelledError:
            pass

    async def _consume_video_lite_topic(self) -> None:
        # Lite frames are consumed by LiteFrameSource._run; keep stub here for symmetry
        ch = self.signaler.broker.topic_latest("video")
        try:
            while not self.shutdown.is_set():
                _ = await ch.get()
                # No-op: decoding handled by LiteFrameSource
        except asyncio.CancelledError:
            pass

    # --- System events (heartbeats, keyboard map, host changes, screen size)
    async def _consume_system_topic(self) -> None:
        q = self.signaler.broker.topic_queue("system")
        try:
            while not self.shutdown.is_set():
                msg = await q.get()
                ev = msg.get("event","")
                payload = msg.get("payload", {})

                # Heartbeat - respond via normal queue to maintain connection
                if ev == "system/heartbeat":
                    # Use normal send queue to avoid WebSocket write conflicts
                    try:
                        await self.signaler.send({"event": "client/heartbeat"})
                        logger.debug("Heartbeat response queued")
                    except Exception as e:
                        logger.warning("Failed to queue heartbeat response: %s", e)
                    continue

                # Initial session metadata
                if ev == "system/init" and isinstance(payload, dict):
                    self.session_id = payload.get("session_id") or self.session_id
                    if (size := payload.get("screen_size")) and isinstance(size, dict):
                        w = int(size.get("width", self.screen_size[0]))
                        h = int(size.get("height", self.screen_size[1]))
                        self.screen_size = (w,h)
                        logger.info("Initial screen size %dx%d", w, h)
                    continue

                # Screen size update
                if ev == "screen/updated" and isinstance(payload, dict):
                    if "width" in payload and "height" in payload:
                        self.screen_size = (int(payload["width"]), int(payload["height"]))
                        logger.info("Screen size changed to %dx%d", *self.screen_size)
                    continue

                # Host changes: re-request if lost
                if ev == "control/host" and isinstance(payload, dict):
                    host_id = payload.get("host_id")
                    has_host = payload.get("has_host")
                    if (not has_host) or (self.session_id and host_id != self.session_id):
                        logger.info("Host control lost/changed — re-requesting.")
                        asyncio.create_task(self.signaler.send({"event": "control/request"}))
                    continue

                # Keyboard map updates (two shapes supported)
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

                # Error routing
                if ev.startswith("error/"):
                    logger.error("[server] %s :: %s", ev, json.dumps(payload, ensure_ascii=False))
                    continue

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("System consumer error: %s", e, exc_info=True)

    async def _consume_chat_topic(self) -> None:
        q = self.signaler.broker.topic_queue("chat")
        try:
            while not self.shutdown.is_set():
                msg = await q.get()
                payload = msg.get("payload", {})
                task_update = None
                content = payload.get("content")
                if isinstance(content, dict):
                    task_update = content.get("text")
                elif isinstance(content, str):
                    task_update = content
                if not task_update:
                    task_update = payload.get("text") or payload.get("message")
                if task_update and isinstance(task_update, str):
                    logger.info("Chat task update: %s", task_update)
                    self.nav_task = task_update
        except asyncio.CancelledError:
            pass

    # Removed _respond_heartbeat_direct() - server ignores client/heartbeat responses
    # and direct WebSocket writes can conflict with the send queue causing connection issues

    # --- Main loop: observe -> decide -> act
    async def _main_loop(self) -> None:
        history: List[Dict[str, Any]] = []
        step = 0
        try:
            while not self.shutdown.is_set() and step < self.max_steps:
                navigation_steps.inc()
                if not self.frame_source:
                    await asyncio.sleep(0.01); continue

                # Check if connection is still alive before fetching frames
                if self.signaler._closed.is_set():
                    logger.warning("Connection closed, stopping navigation loop")
                    break

                img = await self.frame_source.get()
                if img is None:
                    await asyncio.sleep(0.01); continue
                # Resize for model while preserving action coords (we keep normalized)
                img = resize_and_validate_image(img)
                act = await self._navigate_once(img, history, step)
                if not act or act.get("action") == "ANSWER":
                    break
                history.append(act)
                step += 1
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("🚫 Navigation loop cancelled - shutting down gracefully")
        except Exception as e:
            logger.error("❌ Navigation loop error: %s", e, exc_info=True)
        finally:
            if self.shutdown.is_set():
                logger.info("🏁 Agent shutdown requested: steps completed=%d", step)
            else:
                logger.info("Run complete: steps=%d", step)

    async def _log_inference_progress(self, start_time: float, step: int) -> None:
        """Log inference progress every 2 seconds with elapsed time."""
        try:
            await asyncio.sleep(2.0)  # First progress log after 2 seconds
            while True:
                elapsed = time.time() - start_time
                current_time = time.strftime("%H:%M:%S", time.localtime())
                logger.info("⏳ Model inference IN PROGRESS (step=%d) at %s | Elapsed: %.1fs",
                          step, current_time, elapsed)
                await asyncio.sleep(2.0)  # Log progress every 2 seconds
        except asyncio.CancelledError:
            # Task was cancelled when inference completed
            pass

    def _crop_image(self, image: Image.Image, click_xy: Tuple[float,float], crop_factor: float=0.5) -> Tuple[Image.Image, Tuple[int,int,int,int]]:
        width, height = image.size
        cw, ch = int(width*crop_factor), int(height*crop_factor)
        cx, cy = int(click_xy[0]*width), int(click_xy[1]*height)
        left = max(cx - cw//2, 0); top = max(cy - ch//2, 0)
        right = min(cx + cw//2, width); bottom = min(cy + ch//2, height)
        box = (left, top, right, bottom)
        return image.crop(box), box

    async def _navigate_once(self, img: Image.Image, history: List[Dict[str,Any]], step: int) -> Optional[Dict[str,Any]]:
        original_img = img
        current_img = img
        full_w, full_h = original_img.size
        crop_box = (0,0,full_w,full_h)

        raw_output = ""
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
                "size": {"shortest_edge": SIZE_SHORTEST_EDGE, "longest_edge": SIZE_LONGEST_EDGE}
            })
            msgs = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[current_img], videos=None, padding=True, return_tensors="pt").to(self.model.device)

            # Enhanced inference logging with detailed timing
            inference_start_time = time.time()
            start_timestamp = time.strftime("%H:%M:%S", time.localtime(inference_start_time))
            logger.info("🧠 Model inference STARTING (step=%d) at %s", step, start_timestamp)

            future = self.loop.run_in_executor(None, lambda: self.model.generate(**inputs, max_new_tokens=128))

            # Store reference for signal handler cancellation
            self._current_inference_task = future

            # Create progress logging task
            progress_task = asyncio.create_task(self._log_inference_progress(inference_start_time, step))

            try:
                with inference_latency.time():
                    gen = await asyncio.wait_for(future, timeout=120.0)

                # Log successful completion
                inference_end_time = time.time()
                end_timestamp = time.strftime("%H:%M:%S", time.localtime(inference_end_time))
                duration = inference_end_time - inference_start_time
                logger.info("✅ Model inference COMPLETED (step=%d) at %s | Total duration: %.2fs",
                          step, end_timestamp, duration)

            except asyncio.TimeoutError:
                future.cancel()
                inference_end_time = time.time()
                end_timestamp = time.strftime("%H:%M:%S", time.localtime(inference_end_time))
                duration = inference_end_time - inference_start_time
                logger.error("⏰ Model inference TIMEOUT (step=%d) at %s | Duration before timeout: %.2fs",
                           step, end_timestamp, duration)
                parse_errors.inc()
                return None
            except asyncio.CancelledError:
                # Handle cancellation (from signal handler or timeout)
                inference_end_time = time.time()
                end_timestamp = time.strftime("%H:%M:%S", time.localtime(inference_end_time))
                duration = inference_end_time - inference_start_time
                logger.info("🚫 Model inference CANCELLED (step=%d) at %s | Duration: %.2fs",
                          step, end_timestamp, duration)
                return None
            finally:
                # Clear the task reference and cancel progress logging
                self._current_inference_task = None
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            out_ids = [o[len(i):] for o,i in zip(gen, inputs.input_ids)]
            raw_output = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            act = safe_parse_action(raw_output, nav_mode=self.nav_mode)
            logger.info("Raw model output: %s", raw_output)

            # Optional click refinement if the model emits {x,y} (legacy); current schema uses position
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

            if i < self.refinement_steps - 1:
                current_img, crop_box = self._crop_image(original_img, (final_x, final_y))
                logger.info("Refinement %d -> crop=%s", i, crop_box)

        typ = (act or {}).get("action", "UNSUPPORTED")
        actions_executed.labels(action_type=typ if typ in ALLOWED_ACTIONS else "UNSUPPORTED").inc()
        logger.info("Chosen action (step=%d): %s", step, json.dumps(act or {}))

        if act and CLICK_SAVE_PATH:
            with contextlib.suppress(Exception):
                marked = draw_action_markers(original_img, act, step)
                ts = asyncio.get_event_loop().time()
                fname = f"action_step_{step:03d}_{ts:.3f}_{act.get('action','unknown')}.png"
                path = os.path.join(CLICK_SAVE_PATH, fname) if os.path.isdir(CLICK_SAVE_PATH) else f"{CLICK_SAVE_PATH}_{fname}"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_atomic(marked, path)
                logger.info("Saved action frame: %s", path)

        if act:
            await self._execute_action(act, original_img.size)
        return act

    async def _execute_action(self, action: Dict[str,Any], size: Tuple[int,int]) -> None:
        typ  = action.get("action")
        val  = action.get("value")
        pos  = action.get("position")

        def to_xy(norm_pt: List[float]) -> Tuple[int,int]:
            x = int(float(norm_pt[0]) * size[0])
            y = int(float(norm_pt[1]) * size[1])
            return clamp_xy(x,y,size)

        async def move(x:int,y:int) -> None:
            await self.signaler.send({"event":"control/move","payload":{"x":x,"y":y}})

        async def button_press(x:int,y:int, button: str="left") -> None:
            code = BUTTON_CODES.get(button, 1)
            await self.signaler.send({"event":"control/buttonpress","payload":{"x":x,"y":y,"code":code}})

        async def button_down(x:int,y:int, button: str="left") -> None:
            code = BUTTON_CODES.get(button, 1)
            await self.signaler.send({"event":"control/buttondown","payload":{"x":x,"y":y,"code":code}})

        async def button_up(x:int,y:int, button: str="left") -> None:
            code = BUTTON_CODES.get(button, 1)
            await self.signaler.send({"event":"control/buttonup","payload":{"x":x,"y":y,"code":code}})

        async def key_once(name_or_char: str) -> None:
            ks = name_keysym(name_or_char)
            if ks:
                await self.signaler.send({"event":"control/keypress","payload":{"keysym":ks}})

        async def key_down(name_or_char: str) -> None:
            ks = name_keysym(name_or_char)
            if ks:
                await self.signaler.send({"event":"control/keydown","payload":{"keysym":ks}})

        async def key_up(name_or_char: str) -> None:
            ks = name_keysym(name_or_char)
            if ks:
                await self.signaler.send({"event":"control/keyup","payload":{"keysym":ks}})

        try:
            # CLICK/TAP/SELECT/HOVER
            if typ in {"CLICK","TAP","SELECT","HOVER"} and isinstance(pos, list) and len(pos) == 2:
                x,y = to_xy(pos)
                await move(x,y)
                if typ in {"CLICK","TAP","SELECT"}:
                    btn = "left"
                    if isinstance(val, str) and val.lower() in BUTTON_CODES:
                        btn = val.lower()
                    await button_press(x,y,btn)
                return

            # INPUT (focus then type)
            if typ == "INPUT" and val and isinstance(pos, list) and len(pos) == 2:
                x,y = to_xy(pos)
                await move(x,y)
                await button_press(x,y,"left")
                for ch in str(val):
                    if ch == "\n":
                        await key_once("Enter")
                    else:
                        await key_once(ch)
                return

            # ENTER
            if typ == "ENTER":
                await key_once("Enter")
                return

            # SCROLL using delta_x/delta_y like the manual client
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

            # SWIPE drag pos[0] -> pos[1]
            if typ == "SWIPE" and isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, list) and len(p) == 2 for p in pos):
                x1,y1 = to_xy(pos[0]); x2,y2 = to_xy(pos[1])
                await move(x1,y1)
                await button_down(x1,y1,"left")
                await asyncio.sleep(0.05)
                await move(x2,y2)
                await button_up(x2,y2,"left")
                return

            # SELECT_TEXT (drag)
            if typ == "SELECT_TEXT" and isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, list) and len(p) == 2 for p in pos):
                x1,y1 = to_xy(pos[0]); x2,y2 = to_xy(pos[1])
                await move(x1,y1)
                await button_down(x1,y1,"left")
                await move(x2,y2)
                await button_up(x2,y2,"left")
                return

            # COPY (Ctrl+C)
            if typ == "COPY":
                await key_down("Control")
                await key_once("c")
                await key_up("Control")
                logger.info("[COPY] hint=%r", action.get("value"))
                return

            # ANSWER is agent-internal (no control emission)
            if typ == "ANSWER":
                logger.info("[ANSWER] %r", val)
                return

            logger.warning("Unsupported or malformed action: %r", action)
        except Exception as e:
            logger.error("Action execution failed: %s | action=%r", e, action, exc_info=True)

    # --- ICE & Tracks
    async def _on_ice(self, cand: Optional[RTCIceCandidate]) -> None:
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
        logger.info("RTC: track=%s id=%s", track.kind, getattr(track, "id", "unknown"))
        if track.kind == "video" and isinstance(self.frame_source, WebRTCFrameSource):
            await self.frame_source.start(track)

    def _parse_remote_candidate(self, payload: Dict[str, Any]) -> Optional[RTCIceCandidate]:
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
        if getattr(self, "_cleaning", False):
            return
        self._cleaning = True
        try:
            # Cancel TaskGroup-managed tasks by dropping ref (context manager already cancels)
            self._tg = None


            # Cancel RTCP keepalive task
            if hasattr(self, '_rtcp_task'):
                self._rtcp_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._rtcp_task
                delattr(self, '_rtcp_task')  # Clean up the attribute

            # Best-effort unhost
            try:
                if getattr(self.signaler, "ws", None) and not self.signaler._closed.is_set():
                    await self.signaler.send({"event":"control/release"})
            except Exception:
                pass

            # Close PC
            pc = getattr(self, "pc", None)
            if pc:
                try:
                    for sender in pc.getSenders() or []:
                        track = getattr(sender, "track", None)
                        if track:
                            with contextlib.suppress(Exception):
                                await track.stop()
                except Exception:
                    pass
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(pc.close(), timeout=5)
                self.pc = None

            # Stop frame source
            fs = getattr(self, "frame_source", None)
            if fs:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(fs.stop(), timeout=3)
                self.frame_source = None

            # Close WS last
            sig = getattr(self, "signaler", None)
            if sig:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(sig.close(), timeout=5)
        finally:
            self._cleaning = False

# ----------------------
# Entry Point / Boot
# ----------------------
async def main() -> None:
    import argparse

    p = argparse.ArgumentParser("neko_agent", description="Production-ready Neko v3 WebRTC agent (ShowUI-2B)")
    p.add_argument("--ws",         default=os.environ.get("NEKO_WS", None), help="wss://…/api/ws?token=…  (direct; else REST)")
    p.add_argument("--task",       default=os.environ.get("NEKO_TASK", "Search the weather"), help="Navigation task")
    p.add_argument("--mode",       default=os.environ.get("NEKO_MODE", "web"), choices=list(ACTION_SPACES.keys()), help="Mode: web or phone")
    p.add_argument("--max-steps",  type=int, default=MAX_STEPS, help="Max navigation steps")
    p.add_argument("--metrics-port", type=int, default=DEFAULT_METRIC_PORT, help="Prometheus metrics port")
    p.add_argument("--loglevel",   default=os.environ.get("NEKO_LOGLEVEL","INFO"), help="Logging level")
    p.add_argument("--no-audio",   dest="audio", action="store_false", help="Disable audio stream")
    p.add_argument("--neko-url",   default=os.environ.get("NEKO_URL",None), help="Base https://host for REST login")
    p.add_argument("--username",   default=os.environ.get("NEKO_USER",None), help="REST username")
    p.add_argument("--password",   default=os.environ.get("NEKO_PASS",None), help="REST password")
    p.set_defaults(audio=AUDIO_DEFAULT)
    args = p.parse_args()

    logging.getLogger().setLevel(args.loglevel.upper())

    logger.info("Loading model/processor ...")
    # Determine device/dtype
    device = "cpu"
    dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16, device="mps")
            dtype = torch.bfloat16
        except RuntimeError:
            dtype = torch.float32
        device = "mps"
        os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto"}
    if device == "mps":
        model_kwargs.update({"offload_folder": OFFLOAD_FOLDER, "offload_state_dict": True})

    model = Qwen2VLForConditionalGeneration.from_pretrained(REPO_ID, **model_kwargs).eval()
    processor = AutoProcessor.from_pretrained(
        REPO_ID,
        size={"shortest_edge": SIZE_SHORTEST_EDGE, "longest_edge": SIZE_LONGEST_EDGE},
        trust_remote_code=True
    )

    ws_url = args.ws
    if not ws_url or ws_url == DEFAULT_WS:
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
            print(f"[INFO] REST login OK, WS={ws_url}", file=sys.stderr)
        except Exception as e:
            print(f"REST login failed: {e}", file=sys.stderr); sys.exit(1)
    elif any((args.neko_url, args.username, args.password)):
        print("[WARN] --ws provided; ignoring REST args", file=sys.stderr)

    agent = NekoAgent(
        model=model,
        processor=processor,
        ws_url=ws_url,
        nav_task=args.task,
        nav_mode=args.mode,
        max_steps=args.max_steps,
        metrics_port=args.metrics_port,
        audio=args.audio,
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
