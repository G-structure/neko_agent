#!/usr/bin/env python3
"""
neko.py — Robust ShowUI-2B Neko WebRTC GUI agent.
Usage: Provide either --ws (already have token) OR --neko-url + --username + --password for REST login.
Env vars: NEKO_URL, NEKO_USER, NEKO_PASS, NEKO_WS, etc.
"""
import os, sys, asyncio, json, signal, logging, random, uuid, contextlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import websockets
import requests
import base64
import io
from aiortc import (
    RTCConfiguration, RTCIceServer, RTCPeerConnection,
    RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
)
from aiortc.sdp import candidate_from_sdp
from PIL import Image, ImageFile
from abc import ABC, abstractmethod
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from prometheus_client import start_http_server, Counter, Histogram
import ast

logging.getLogger("aiortc").setLevel(logging.DEBUG)

# Configure PIL to fail fast on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = False

MODEL_KEY           = os.environ.get("MODEL_KEY", "showui-2b")
REPO_ID             = os.environ.get("REPO_ID", "showlab/ShowUI-2B")
SIZE_SHORTEST_EDGE  = int(os.environ.get("SIZE_SHORTEST_EDGE", 224))
SIZE_LONGEST_EDGE   = int(os.environ.get("SIZE_LONGEST_EDGE", 1344))
DEFAULT_WS          = os.environ.get("NEKO_WS", "wss://neko.example.com/api/ws")
DEFAULT_METRIC_PORT = int(os.environ.get("NEKO_METRICS_PORT", 9000))
MAX_STEPS           = int(os.environ.get("NEKO_MAX_STEPS", 8))
AUDIO_DEFAULT       = bool(int(os.environ.get("NEKO_AUDIO", "1")))
FRAME_SAVE_PATH     = os.environ.get("FRAME_SAVE_PATH", None)
OFFLOAD_FOLDER      = os.environ.get("OFFLOAD_FOLDER", "./offload")
NEKO_STUN_URL       = os.environ.get("NEKO_STUN_URL", "stun:stun.l.google.com:19302")
NEKO_TURN_URL       = os.environ.get("NEKO_TURN_URL")
NEKO_TURN_USER      = os.environ.get("NEKO_TURN_USER")
NEKO_TURN_PASS      = os.environ.get("NEKO_TURN_PASS")
NEKO_ICE_POLICY     = os.environ.get("NEKO_ICE_POLICY","all")

ALLOWED_ACTIONS = {
    "CLICK","INPUT","SELECT","HOVER","ANSWER","ENTER","SCROLL","SELECT_TEXT","COPY",
    "SWIPE","TAP"
}
frames_received    = Counter("neko_frames_received_total",    "Total video frames received")
actions_executed   = Counter("neko_actions_executed_total",   "Actions executed by type", ["action_type"])
parse_errors       = Counter("neko_parse_errors_total",       "Action parse errors")
navigation_steps   = Counter("neko_navigation_steps_total",   "Navigation step count")
inference_latency  = Histogram("neko_inference_latency_seconds","Inference latency")
reconnects         = Counter("neko_reconnects_total",         "WS reconnect attempts")
resize_duration    = Histogram("neko_resize_duration_seconds","Resize time")

logger = logging.getLogger("neko_agent")
logging.basicConfig(
    level=os.environ.get("NEKO_LOGLEVEL","INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
)

# Add a file handler if NEKO_LOGFILE environment variable is set
neko_logfile = os.environ.get("NEKO_LOGFILE")
if neko_logfile:
    try:
        # Use the same formatter for the file handler for consistency
        log_format = '{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
        formatter = logging.Formatter(log_format)

        file_handler = logging.FileHandler(neko_logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Log this info message to both console and the new file
        logger.info(f"Neko agent logging to file: {neko_logfile}")
    except Exception as e:
        logger.error(f"Failed to set up file logging to {neko_logfile}: {e}")


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

def safe_parse_action(output_text: str, nav_mode: str="web") -> Optional[Dict[str,Any]]:
    """Parses a JSON string into an action dictionary and validates it.

    This function attempts to parse the provided `output_text` as a JSON
    dictionary representing an action. It then validates the action's structure
    and ensures that the 'action' type is whitelisted for the given
    `nav_mode`. Invalid or malformed actions are logged, and a counter for
    parse errors is incremented.

    :param output_text: The raw string output, expected to be a JSON dictionary.
    :type output_text: str
    :param nav_mode: The navigation mode ("web" or "phone") to validate
        the action type against its allowed actions. Defaults to "web".
    :type nav_mode: str
    :returns: A validated action dictionary if parsing and validation succeed,
        otherwise `None`.
    :rtype: Optional[Dict[str,Any]]
    """
    try:
        # First, try to parse as strict JSON
        act = json.loads(output_text)
    except json.JSONDecodeError:
        # If JSON fails, try to evaluate as a Python literal
        try:
            act = ast.literal_eval(output_text)
        except (ValueError, SyntaxError) as e:
            logger.error("Failed to parse as JSON or Python literal: %s | Raw: %r", e, output_text)
            parse_errors.inc()
            return None

    try:
        assert isinstance(act, dict)
        typ = act.get("action")
        if typ not in ACTION_SPACES[nav_mode]:
            logger.warning("Security: Non-whitelisted action: %r", typ)
            parse_errors.inc()
            return None
        for k in ("action","value","position"):
            assert k in act, f"Missing key {k}"
        return act
    except AssertionError as e:
        logger.error("Schema validation error: %s | Parsed: %r", e, act)
        parse_errors.inc()
        return None

def clamp_xy(x:int,y:int,size:Tuple[int,int]) -> Tuple[int,int]:
    """Clamps X and Y coordinates to be within the bounds of a given size.

    This function ensures that the provided `x` and `y` coordinates do not
    exceed the width and height of the specified `size`. Coordinates are
    clamped to be between 0 and (width-1) for x, and 0 and (height-1) for y.

    :param x: The X-coordinate to clamp.
    :type x: int
    :param y: The Y-coordinate to clamp.
    :type y: int
    :param size: A tuple (width, height) representing the maximum bounds.
    :type size: Tuple[int,int]
    :returns: A new tuple (clamped_x, clamped_y) with coordinates
        guaranteed to be within the specified size.
    :rtype: Tuple[int,int]
    """
    w,h = size
    return max(0,min(x,w-1)), max(0,min(y,h-1))

def resize_and_validate_image(image:Image.Image) -> Image.Image:
    """Resizes an image if its longest edge exceeds a predefined maximum size.

    This function checks if the longest edge of the input image is greater than
    `SIZE_LONGEST_EDGE`. If it is, the image is scaled down proportionally so
    its longest edge matches `SIZE_LONGEST_EDGE`. The resizing operation uses
    the `Image.LANCZOS` filter for high-quality downsampling. The duration
    of the resize operation is measured and logged for performance monitoring.

    :param image: The Pillow Image object to be resized and validated.
    :type image: Image.Image
    :returns: The resized image if scaling was necessary, otherwise the
        original image.
    :rtype: Image.Image
    """
    import time
    ow,oh = image.size
    me = max(ow,oh)
    if me > SIZE_LONGEST_EDGE:
        scale = SIZE_LONGEST_EDGE / me
        nw,nh = int(ow*scale), int(oh*scale)
        t0 = time.monotonic()
        image = image.resize((nw,nh), Image.LANCZOS)
        resize_duration.observe(time.monotonic()-t0)
        logger.info(f"Resized {ow}×{oh}→{nw}×{nh}")
    return image

def pil_from_frame(frame):
    """Converts a video frame to a validated PIL Image.

    This function takes a video frame, converts it to a PIL Image,
    validates its integrity, and returns a copy to detach the buffer.

    :param frame: The video frame object received from an `aiortc.VideoStreamTrack`.
    :returns: A validated Pillow `Image.Image` object in "RGB" mode.
    """
    img = frame.to_image()
    # Validate image dimensions
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")

    # Convert to RGB and return a copy to detach buffer
    return img.convert("RGB").copy()

def save_atomic(img, path):
    """Save image atomically to prevent truncated files.

    This function saves an image to a temporary file first, then
    atomically moves it to the target path.

    :param img: The PIL Image to save.
    :param path: The target path to save the image to.
    """
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        img.save(tmp, format="PNG")
        os.replace(tmp, path)  # atomic on POSIX
        logger.info(f"Saved frame atomically to {path}")
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise e

def frame_to_pil_image(frame:VideoStreamTrack) -> Image.Image:
    """Converts a video frame from `aiortc.VideoStreamTrack` to a PIL Image.

    This function takes a video frame received from an WebRTC `VideoStreamTrack`,
    converts it into a Pillow `Image.Image` object, logs the process, and
    optionally saves the frame to a specified path. It ensures the resulting
    image is in RGB format.

    :param frame: The video frame object received from an `aiortc.VideoStreamTrack`.
    :type frame: VideoStreamTrack
    :returns: A Pillow `Image.Image` object representing the video frame,
        guaranteed to be in "RGB" mode.
    :rtype: Image.Image
    """
    logger.info("Decoding frame to PIL image...")
    try:
        img = pil_from_frame(frame)
        logger.info("Frame decoded and validated successfully.")
        logger.info(f"frame_to_pil_image: Got image {img.size}, mode {img.mode}")
        if FRAME_SAVE_PATH is not None:
            try:
                save_atomic(img, FRAME_SAVE_PATH)
            except Exception as e:
                logger.error(f"Error saving frame to {FRAME_SAVE_PATH}: {e}")
        return img
    except Exception as e:
        logger.error(f"Error decoding or validating frame: {e}")
        raise

class Signaler:
    """Manages WebSocket connections and signaling for the Neko agent.

    This class provides methods for connecting to a WebSocket URL with
    exponential backoff, and for sending/receiving JSON messages over the
    established connection. It abstracts the underlying `websockets` library
    for signaling purposes.
    """
    def __init__(self, url:str):
        """Initializes the Signaler with the WebSocket URL.

        :param url: The WebSocket URL to connect to (e.g., "wss://neko.example.com/api/ws?token=...").
        :type url: str
        """
        self.url = url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect_with_backoff(self) -> websockets.WebSocketClientProtocol:
        """Establishes a WebSocket connection with exponential backoff.

        This method attempts to connect to the `self.url`. If the connection
        fails, it retries with an increasing delay, up to a maximum of 60 seconds,
        plus a small random jitter to prevent thundering herd. It raises
        `websockets.InvalidStatusCode` for authentication or fatal errors.

        :raises websockets.InvalidStatusCode: If the WebSocket connection
            fails due to an invalid status code (e.g., authentication error).
        :returns: The established WebSocket client protocol.
        :rtype: websockets.WebSocketClientProtocol
        """
        backoff = 1
        while True:
            try:
                self.ws = await websockets.connect(self.url,
                                                   ping_interval=20,
                                                   ping_timeout=10,
                                                   max_queue=8)
                logger.info(f"WebSocket connected to {self.url}")
                return self.ws
            except websockets.InvalidStatusCode as e:
                logger.error("WS auth/fatal error: %s", e)
                raise
            except Exception as e:
                delay = min(backoff*2,60)
                logger.error("WS connect error: %s — retry %ss", e, delay)
                await asyncio.sleep(delay + random.uniform(0,delay*0.1))
                backoff = delay

    async def send(self, msg:Dict[str,Any]) -> None:
        """Sends a JSON message over the WebSocket connection.

        The message dictionary is serialized to a JSON string before sending.

        :param msg: The message dictionary to send.
        :type msg: Dict[str,Any]
        :raises websockets.ConnectionClosedOK: If the WebSocket connection is closed.
        """
        if not self.ws:
            logger.error("Attempted to send on a closed or uninitialized WebSocket.")
            return
        await self.ws.send(json.dumps(msg))

    async def recv(self, timeout:float=30) -> Dict[str,Any]:
        """Receives and parses a JSON message from the WebSocket connection.

        This method waits for a message from the WebSocket, with an optional timeout.
        The received data is expected to be a JSON string, which is then parsed
        into a Python dictionary.

        :param timeout: The maximum time in seconds to wait for a message. Defaults to 30.
        :type timeout: float
        :returns: The parsed message dictionary.
        :rtype: Dict[str,Any]
        :raises asyncio.TimeoutError: If no message is received within the specified timeout.
        :raises json.JSONDecodeError: If the received data is not valid JSON.
        :raises websockets.ConnectionClosedOK: If the WebSocket connection is closed.
        """
        if not self.ws:
            logger.error("Attempted to receive on a closed or uninitialized WebSocket.")
            # Depending on desired behavior, could raise an error or return an empty dict
            raise RuntimeError("WebSocket not connected.")
        async with asyncio.timeout(timeout):
            data = await self.ws.recv()
        return json.loads(data)


class FrameSource(ABC):
    """Abstract interface for screen/frame sources."""

    @abstractmethod
    async def start(self, *args: Any) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def get(self) -> Optional[Image.Image]:
        ...


class WebRTCFrameSource(FrameSource):
    """Continuously reads frames from a WebRTC video track and stores the latest.

    This implementation is a thin wrapper around a reader task that pulls
    frames from an `aiortc.VideoStreamTrack` and keeps only the most recent
    PIL Image in memory.

    :ivar image: The latest PIL Image received from the video track, or None if no frame has been received or the reader is stopped.
    :vartype image: Optional[Image.Image]
    :ivar task: The asyncio task running the frame reader, or None if not active.
    :vartype task: Optional[asyncio.Task]
    :ivar lock: An asyncio.Lock to protect access to the `image` attribute.
    :vartype lock: asyncio.Lock
    :ivar first_frame: An asyncio.Event that could signal when the first frame has been received (currently unused in snippet).
    :vartype first_frame: asyncio.Event
    """

    def __init__(self) -> None:
        """Initializes the source with no image, task, and sets up a lock and event."""
        self.image: Optional[Image.Image] = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.first_frame = asyncio.Event()

    async def start(self, *args: Any) -> None:
        """Starts or restarts the frame reader for a given video track.

        If a reader task is already running, it will be stopped before
        a new one is created for the provided `track`. This ensures
        only one reader is active at a time.

        :param args: The arguments. First argument should be the `aiortc.VideoStreamTrack` to read frames from.
        """
        if not args:
            raise ValueError("WebRTCFrameSource.start() requires a VideoStreamTrack argument")

        track = args[0]
        await self.stop()
        logger.info(f"WebRTCFrameSource: starting reader for track {track}")
        self.task = asyncio.create_task(self._reader(track))

    async def stop(self) -> None:
        """Stops the ongoing frame reader task and clears the stored image.

        If a reader task is active, it is cancelled and awaited. The `image`
        attribute is set to `None` to indicate no current frame is available.
        """
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            self.task = None
        async with self.lock:
            self.image = None

    async def _reader(self, track: VideoStreamTrack) -> None:
        """Internal asynchronous loop for reading frames from the track.

        Continuously receives frames from the `track`, converts them to PIL
        images, and stores the latest one. Increments a counter for each
        frame received. The loop runs until cancelled or an error occurs.

        :param track: The `aiortc.VideoStreamTrack` to receive frames from.
        :type track: VideoStreamTrack
        """
        try:
            while True:
                frame = await track.recv()
                img = frame_to_pil_image(frame)
                async with self.lock:
                    self.image = img
                frames_received.inc()
        except asyncio.CancelledError:
            # Expected on shutdown
            pass
        except Exception as e:
            logger.error(f"Frame reader stopped: {e}")

    async def get(self) -> Optional[Image.Image]:
        """Retrieves the most recently stored PIL Image.

        This method provides thread-safe access to the `image` attribute.

        :returns: The latest `Image.Image` object received, or `None` if no
            frame has been received or the reader is currently stopped.
        :rtype: Optional[Image.Image]
        """
        async with self.lock:
            return self.image


class LiteFrameSource(FrameSource):
    """Frame source for WebSocket-lite mode streaming base64 images."""

    def __init__(self, signaler: Signaler) -> None:
        self.signaler = signaler
        self.image: Optional[Image.Image] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    async def start(self, *args: Any) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while self._running and self.signaler.ws:
            try:
                msg = await self.signaler.recv()
            except Exception:
                break
            if msg.get("event") == "signal/video":
                data = msg.get("data")
                if not data:
                    continue
                try:
                    # Validate and decode base64 data
                    img_bytes = base64.b64decode(data, validate=True)

                    # Check for valid image headers
                    if not img_bytes.startswith(b"\x89PNG") and not img_bytes.startswith(b"\xff\xd8\xff"):
                        logger.warning("Bad image header, skipping frame")
                        continue

                    # Open and verify image integrity
                    img = Image.open(io.BytesIO(img_bytes))
                    img.verify()  # Ensure full file integrity

                    # Re-open after verify (PIL requirement)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    # Validate image dimensions
                    w, h = img.size
                    if w <= 0 or h <= 0:
                        logger.warning("Invalid image dimensions: %dx%d, skipping frame", w, h)
                        continue

                    async with self._lock:
                        self.image = img
                    frames_received.inc()
                except Exception as e:
                    logger.warning("Failed to decode lite frame: %s", e)

    async def get(self) -> Optional[Image.Image]:
        async with self._lock:
            return self.image

class NekoAgent:
    """Manages the Neko WebRTC agent's lifecycle, including signaling, WebRTC
    connection, video frame processing, and AI model inference for navigation.

    This class orchestrates the communication with the Neko server,
    receives video streams, applies a multimodal AI model to decide on actions,
    and sends control commands back to the server. It handles connection
    management, error recovery, and integrates with Prometheus for metrics.

    :ivar signaler: Handles WebSocket communication for signaling.
    :vartype signaler: Signaler
    :ivar frame_source: Active source for retrieving screen frames in either
        WebRTC or lite mode.
    :vartype frame_source: FrameSource
    :ivar nav_task: The specific task instruction for the navigation agent.
    :vartype nav_task: str
    :ivar nav_mode: The operational mode, either "web" or "phone", determining
        the allowed action space.
    :vartype nav_mode: str
    :ivar max_steps: Maximum number of navigation steps before terminating.
    :vartype max_steps: int
    :ivar audio: Whether audio stream is enabled for the WebRTC connection.
    :vartype audio: bool
    :ivar model: The loaded AI model for inference.
    :vartype model: Any
    :ivar processor: The processor for the AI model.
    :vartype processor: Any
    :ivar run_id: A unique identifier for the current agent run.
    :vartype run_id: str
    :ivar pc: The RTCPeerConnection object for WebRTC communication.
    :vartype pc: Optional[RTCPeerConnection]
    :ivar shutdown: An asyncio.Event to signal agent shutdown.
    :vartype shutdown: asyncio.Event
    :ivar loop: The asyncio event loop.
    :vartype loop: asyncio.AbstractEventLoop
    :ivar ice_task: The asyncio task for consuming remote ICE candidates.
    :vartype ice_task: Optional[asyncio.Task]
    :ivar sys_prompt: The system prompt for the AI model, dynamically formatted.
    :vartype sys_prompt: str
    """
    def __init__(self, model, processor, ws_url:str,
                 nav_task:str, nav_mode:str,
                 max_steps:int=MAX_STEPS,
                 metrics_port:int=DEFAULT_METRIC_PORT,
                 audio:bool=AUDIO_DEFAULT):
        """Initializes the NekoAgent with necessary components and configurations.

        :param model: The loaded multimodal AI model for generating actions.
        :param processor: The processor associated with the AI model for
            handling inputs (text and images).
        :param ws_url: The WebSocket URL for Neko signaling.
        :type ws_url: str
        :param nav_task: The high-level task description for the agent (e.g.,
            "Search for weather").
        :type nav_task: str
        :param nav_mode: The mode of navigation, "web" or "phone", which
            defines the available action space.
        :type nav_mode: str
        :param max_steps: The maximum number of navigation steps the agent
            will attempt before stopping. Defaults to `MAX_STEPS`.
        :type max_steps: int
        :param metrics_port: The port for the Prometheus metrics server.
            Defaults to `DEFAULT_METRIC_PORT`.
        :type metrics_port: int
        :param audio: A boolean indicating whether to enable audio streaming
            in the WebRTC connection. Defaults to `AUDIO_DEFAULT`.
        :type audio: bool
        """
        self.signaler   = Signaler(ws_url)
        self.frame_source: Optional[FrameSource] = None
        self.nav_task   = nav_task
        self.nav_mode   = nav_mode
        self.max_steps  = max_steps
        self.audio      = audio
        self.model      = model
        self.processor  = processor
        self.run_id     = os.environ.get("NEKO_RUN_ID") or str(uuid.uuid4())[:8]
        self.pc:Optional[RTCPeerConnection] = None
        self.shutdown   = asyncio.Event()
        self.chat_queue = asyncio.Queue()
        self.loop       = asyncio.get_event_loop()
        self.ice_task   = None
        self.is_lite    = False

        self.sys_prompt = _NAV_SYSTEM.format(
            _APP=self.nav_mode,
            _ACTION_SPACE=ACTION_SPACE_DESC[self.nav_mode]
        )
        start_http_server(metrics_port)

    async def run(self) -> None:
        """Runs the main loop of the Neko agent, handling WebSocket connections,
        WebRTC setup, and the navigation process.

        This method sets up signal handlers for graceful shutdown, then enters
        a continuous loop that attempts to connect to the Neko server,
        establish WebRTC, and execute the navigation sequence. It includes
        exponential backoff for reconnect attempts and logs critical errors.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.shutdown.set)
        #TODO: Fix chating with agent
        # start chat consumer
        self.chat_task = asyncio.create_task(self._consume_chat())
        while not self.shutdown.is_set():
            reconnects.inc()
            try:
                async with await self.signaler.connect_with_backoff() as ws:
                    self.signaler.ws = ws
                    await self._setup_media()
                    if not self.is_lite:
                        self.ice_task = asyncio.create_task(self._consume_remote_ice())
                    try:
                        await self._main_loop()
                    finally:
                        if self.ice_task:
                            self.ice_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await self.ice_task
            except Exception as e:
                logger.error(json.dumps({
                    "phase":"connect","run":self.run_id,"msg":f"WS/RTC error: {e}"
                }))
            finally:
                await self._cleanup()

# TODO: WebRTC not setting up correctly.
# Why do we ignore the info from "system/init"?
# Is the test neko server returning the correct ICE Candidates and are we able to fetch them with `ice_servers_payload`?
# Make sure the WebRTC setup flow is correct, webrtc is working in the webapp so it must be possible to establish connection
# with the python agent we have here.
    async def _setup_media(self) -> None:
        """Sets up media streaming with the Neko server (WebRTC or lite)."""
        req = {
            "video": {"width": 1280, "height": 720, "frameRate": 30}
        }
        if self.audio:
            req["audio"] = {}
        logger.info("Sending signal/request with payload: %r", req)
        await self.signaler.send({"event":"signal/request","payload":req})
        await self.signaler.send({"event": "session/watch", "payload": {"id": "main"}})

        offer_msg = None
        buffered_candidates = []
        while not offer_msg:
            msg = await self.signaler.recv()
            ev  = msg.get("event")
            if ev in ("signal/offer", "signal/provide"):
                offer_msg = msg
            elif ev == "signal/candidate":
                logger.info("Buffering early ICE candidate.")
                buffered_candidates.append(msg)
            else:
                logger.info("Ignoring event during setup: %s", ev)

        payload = offer_msg.get("payload", offer_msg)

        # TODO: Test this
        # Fallback to grab frames over websockets if neko is in lite mode
        self.is_lite = bool(offer_msg.get("lite") or payload.get("lite"))
        if self.is_lite:
            logger.info("Lite/WebSocket mode detected: using WebSocket image streaming.")
            self.frame_source = LiteFrameSource(self.signaler)
            await self.frame_source.start()
            return

        ice_payload = (
            payload.get("ice")
            or payload.get("iceservers")
            or payload.get("iceServers")
            or payload.get("ice_servers")
            or []
        )
        logger.info("Server ICE list: %r", ice_payload)

        ice_servers = [
            RTCIceServer(
                urls= srv.get("urls") or srv.get("url"),
                username=srv.get("username"),
                credential=srv.get("credential"),
            )
            for srv in ice_payload
        ]

        ice_servers.append(RTCIceServer(urls=[NEKO_STUN_URL]))
        if NEKO_TURN_URL:
            ice_servers.append(RTCIceServer(
                #TODO: hack for current usecase make more generic
                urls=[NEKO_TURN_URL + "?transport=tcp"],
                username=NEKO_TURN_USER,
                credential=NEKO_TURN_PASS,
            ))

        config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(config)
        self.pc = pc
        self.frame_source = WebRTCFrameSource()

        # Setup event listeners
        pc.on("iceconnectionstatechange", lambda: logger.info(f"iceConnectionState → {pc.iceConnectionState}"))
        pc.on("connectionstatechange", lambda: logger.info(f"connectionState → {pc.connectionState}"))
        pc.on("icecandidate", lambda c: asyncio.create_task(self._on_ice(c)))
        pc.on("track", lambda t: asyncio.create_task(self._on_track(t)))

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=payload["sdp"], type=payload.get("type","offer"))
        )

        # Process buffered candidates now that remote description is set
        for cand_msg in buffered_candidates:
            ice = self._parse_remote_candidate(cand_msg["payload"])
            if ice and self.pc:
                try:
                    await self.pc.addIceCandidate(ice)
                    logger.info("\u2705 Added buffered ICE candidate")
                except Exception as e:
                    logger.warning("\u26a0\ufe0f addIceCandidate (buffered) failed: %s", e)


        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self.signaler.send({
            "event":"signal/answer",
            "payload":{"sdp":pc.localDescription.sdp,"type":pc.localDescription.type}
        })

    # TODO: The WebRTC connection is still not fully reliable. We should monitor connectionState and ICE state closely here.
    async def _consume_remote_ice(self) -> None:
        """Asynchronously consumes remote ICE candidates from the WebSocket.

        This internal method continuously listens for "signal/candidate"
        messages on the WebSocket. Upon receiving a candidate, it constructs
        an `RTCIceCandidate` object and adds it to the `RTCPeerConnection`.
        The loop breaks if a "signal/close" message is received or the agent
        is shutting down.
        """
        while self.signaler.ws and not self.shutdown.is_set():
            try:
                msg = await self.signaler.recv(timeout=60)
            except asyncio.TimeoutError:
                continue

            if msg.get("event") == "signal/candidate":
                ice = self._parse_remote_candidate(msg["payload"]) # Use the helper
                if ice and self.pc:
                    try:
                        await self.pc.addIceCandidate(ice)
                        logger.info("\u2705 Added ICE candidate")
                    except Exception as e:
                        logger.warning("\u26a0\ufe0f addIceCandidate failed: %s", e)
            elif msg.get("event") == "signal/close":
                break

    async def _main_loop(self) -> None:
        """Executes the main navigation loop of the agent.

        This internal method repeatedly fetches the latest video frame,
        prepares it for inference, calls the AI model to determine the
        next action, and then executes that action. It continues until
        the maximum number of steps is reached, the agent is signaled
        to shut down, or an "ANSWER" action is generated, indicating
        task completion.
        """
        history: List[Dict[str,Any]] = []
        step = 0
        while not self.shutdown.is_set() and step < self.max_steps:
            navigation_steps.inc()
            if not self.frame_source:
                await asyncio.sleep(0.01)
                continue
            img = await self.frame_source.get()
            if img is None:
                await asyncio.sleep(0.01)
                continue
            #TODO: We need to account for resize in action position such as click
            img = resize_and_validate_image(img)
            act = await self._navigate_once(img, history, step)
            if not act or act.get("action") == "ANSWER":
                break
            history.append(act)
            step += 1
        logger.info(json.dumps({"phase":"complete","run":self.run_id,"steps":step}))

    async def _navigate_once(self, img:Image.Image, history:List[Dict[str,Any]], step:int) -> Optional[Dict[str,Any]]:
        """Performs a single navigation step, involving AI inference and action.

        This internal method constructs the prompt for the AI model using
        the current screen image, task, and action history. It then runs
        the model inference, parses the raw output into a structured action,
        and executes the action if valid.

        :param img: The current screen observation as a PIL Image.
        :type img: Image.Image
        :param history: A list of previously executed actions.
        :type history: List[Dict[str,Any]]
        :param step: The current navigation step number.
        :type step: int
        :returns: The parsed and executed action, or `None` if parsing or
            inference failed.
        :rtype: Optional[Dict[str,Any]]
        """
        content: List[Dict[str, Any]] = [
            {"type":"text","text":self.sys_prompt},
            {"type":"text","text":f"Task: {self.nav_task}"},
        ]
        if history:
            content.append({"type":"text","text":f"Action history: {json.dumps(history)}"})
        content.append({"type":"image","image":img,
                        "size":{"shortest_edge":SIZE_SHORTEST_EDGE,"longest_edge":SIZE_LONGEST_EDGE}})
        msgs = [{"role":"user","content":content}]
        text   = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], videos=None,
                                padding=True, return_tensors="pt").to(self.model.device)
        future = self.loop.run_in_executor(None, lambda: self.model.generate(**inputs, max_new_tokens=128))
        try:
            with inference_latency.time():
                gen = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            future.cancel()
            logger.error(json.dumps({"phase":"inference","run":self.run_id,"step":step,"msg":"timeout"}))
            parse_errors.inc()
            return None
        out_ids     = [o[len(i):] for o,i in zip(gen, inputs.input_ids)]
        raw_output  = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        logger.info(json.dumps({"phase":"navigate","run":self.run_id,"step":step,"raw":raw_output}))
        act = safe_parse_action(raw_output, nav_mode=self.nav_mode)
        typ = act["action"] if act and act["action"] in ALLOWED_ACTIONS else "UNSUPPORTED"
        actions_executed.labels(action_type=typ).inc()
        logger.info(json.dumps({"phase":"navigate","run":self.run_id,"step":step,"action":act}))
        if act:
            await self._execute_action(act, img.size)
        return act

    async def _execute_action(self, action:Dict[str,Any], size:Tuple[int,int]) -> None:
        """Executes a given action by sending control commands via WebSocket.

        This internal method translates a structured action dictionary
        (e.g., CLICK, INPUT, SWIPE) into a series of WebSocket control messages
        sent to the Neko server. Coordinates are clamped to the screen
        dimensions to prevent out-of-bounds errors.

        :param action: The action dictionary to execute.
        :type action: Dict[str,Any]
        :param size: The dimensions (width, height) of the screen image, used
            to scale and clamp coordinates.
        :type size: Tuple[int,int]
        """
        typ,val,pos = action.get("action"), action.get("value"), action.get("position")
        def xy(pt):
            x,y = int(pt[0]*size[0]), int(pt[1]*size[1])
            return clamp_xy(x,y,size)
        if typ in ("CLICK","TAP","SELECT","HOVER") and isinstance(pos,list) and len(pos)==2:
            x,y = xy(pos)
            await self.signaler.send({"event":"control/move","payload":{"x":x,"y":y}})
            if typ in ("CLICK","TAP"):
                await self.signaler.send({"event":"control/click","payload":{"button":"left","state":"down"}})
                await self.signaler.send({"event":"control/click","payload":{"button":"left","state":"up"}})
        elif typ=="INPUT" and val and isinstance(pos,list) and len(pos)==2:
            x,y = xy(pos)
            await self.signaler.send({"event":"control/move","payload":{"x":x,"y":y}})
            await self.signaler.send({"event":"control/click","payload":{"button":"left","state":"down"}})
            await self.signaler.send({"event":"control/click","payload":{"button":"left","state":"up"}})
            for ch in str(val):
                await self.signaler.send({"event":"control/key","payload":{"key":ch,"code":ord(ch),"state":"down"}})
                await self.signaler.send({"event":"control/key","payload":{"key":ch,"code":ord(ch),"state":"up"}})
        elif typ=="ENTER":
            for s in ({"key":"Enter","code":13,"state":"down"},{"key":"Enter","code":13,"state":"up"}):
                await self.signaler.send({"event":"control/key","payload":s})
        elif typ=="SCROLL" and val:
            await self.signaler.send({"event":"control/scroll","payload":{"direction":val}})
        elif typ=="SWIPE" and isinstance(pos,list) and len(pos)==2:
            (x1,y1),(x2,y2) = xy(pos[0]),xy(pos[1])
            await self.signaler.send({"event":"control/move","payload":{"x":x1,"y":y1}})
            await self.signaler.send({"event":"control/mouse","payload":{"button":"left","state":"down"}})
            await self.signaler.send({"event":"control/move","payload":{"x":x2,"y":y2}})
            await self.signaler.send({"event":"control/mouse","payload":{"button":"left","state":"up"}})
        elif typ=="SELECT_TEXT" and isinstance(pos,list) and len(pos)==2:
            (x1,y1),(x2,y2) = xy(pos[0]),xy(pos[1])
            await self.signaler.send({"event":"control/move","payload":{"x":x1,"y":y1}})
            await self.signaler.send({"event":"control/click","payload":{"button":"left","state":"down"}})
            await self.signaler.send({"event":"control/move","payload":{"x":x2,"y":y2}})
            await self.signaler.send({"event":"control/click","payload":{"button":"left","state":"up"}})
        elif typ=="COPY":
            logger.info("[COPY] to clipboard: %r", val)
            for k in (("Control",17),("c",67)):
                await self.signaler.send({"event":"control/key","payload":{"key":k[0],"code":k[1],"state":"down"}})
                await self.signaler.send({"event":"control/key","payload":{"key":k[0],"code":k[1],"state":"up"}})
        elif typ=="ANSWER":
            logger.info("[ANSWER] %r", val)
        else:
            logger.warning("Unsupported action: %r", action)
    
    #TODO: fix chat consumer
    async def _consume_chat(self):
        """Consume incoming chat messages and update the task."""
        try:
            while not self.shutdown.is_set():
                msg = await self.signaler.recv()
                if msg.get("event") == "chat/message":
                    payload = msg.get("payload", {})
                    task_update = None

                    # Case 1 & 2: Extract from `payload.content` which can be a string or a dict
                    content = payload.get("content")
                    if isinstance(content, dict):
                        task_update = content.get("text")  # `content: {"text": "..."}`
                    elif isinstance(content, str):
                        task_update = content              # `content: "..."`

                    # Case 3 & 4: Fallback to other keys if content is not present/valid
                    if not task_update:
                        task_update = payload.get("text") or payload.get("message")

                    if task_update and isinstance(task_update, str):
                        logger.info(f"Chat update received: {task_update}")
                        self.nav_task = task_update
        except asyncio.CancelledError:
            pass

    async def _on_ice(self, cand: RTCIceCandidate) -> None:
        """Handles the generation of local ICE candidates and sends them
        to the Neko server via the WebSocket.

        This internal method is called as a callback when `aiortc` generates
        a local ICE candidate. It constructs a payload with the candidate
        information and sends it over the signaling WebSocket.

        :param cand: The `RTCIceCandidate` object representing the local
            ICE candidate.
        :type cand: RTCIceCandidate
        """
        if not cand or not self.signaler.ws:
            return
        logger.info("SENDING local ICE candidate → %s", cand)
        await self.signaler.send({
            "event": "signal/candidate",
            "payload": {
                "candidate": cand.candidate,
                "sdpMid": cand.sdpMid,
                "sdpMLineIndex": cand.sdpMLineIndex
            }
        })

    async def _cleanup(self) -> None:
        """Performs cleanup of WebRTC connections and associated resources.

        This internal method is called to gracefully close the
        `RTCPeerConnection`, stop any active tracks (e.g., video frames),
        and close the WebSocket connection.
        """
        if self.pc:
            await self.pc.close()
            for s in self.pc.getSenders():
                if s.track: # Check if track exists before stopping
                    await s.track.stop()
            self.pc = None
        if self.signaler.ws:
            await self.signaler.ws.close()
        if self.frame_source:
            await self.frame_source.stop()
        if hasattr(self, 'chat_task'):
            self.chat_task.cancel()       # stop chat consumer

    async def _on_track(self, track: VideoStreamTrack) -> None:
        """Handles incoming media tracks from the WebRTC PeerConnection.

        This internal method is called as a callback when a new media track
        is received via WebRTC. If the track is a video track, it updates
        the active :class:`WebRTCFrameSource` to begin processing frames from
        this track. Other
        track kinds are currently ignored.

        :param track: The incoming `MediaStreamTrack` object.
        :type track: VideoStreamTrack
        """
        logger.info(f"RTC: Received track: kind={track.kind}, id={track.id}")
        if track.kind == "video" and isinstance(self.frame_source, WebRTCFrameSource):
            await self.frame_source.start(track)
        else:
            logger.info(f"RTC: Ignoring non-video track: kind={track.kind}")

    def _parse_remote_candidate(self, payload: Dict[str, Any]) -> Optional[RTCIceCandidate]:
        """Parses an ICE candidate from a WebSocket message payload."""
        cand_str = payload.get("candidate")
        if not cand_str:
            return None

        # The candidate_from_sdp helper expects the raw value, without the "candidate:" prefix.
        if cand_str.startswith("candidate:"):
            cand_str = cand_str.split(":", 1)[1]

        ice = candidate_from_sdp(cand_str)
        ice.sdpMid = payload.get("sdpMid")
        ice.sdpMLineIndex = payload.get("sdpMLineIndex")
        return ice

async def main() -> None:
    """Entry point for the Neko WebRTC agent.

    This asynchronous function parses command-line arguments and environment variables
    to configure the Neko agent. It handles the loading of the multimodal AI model
    (Qwen2VLForConditionalGeneration) to appropriate hardware (CUDA, MPS, or CPU)
    and resolves the WebSocket URL for connecting to the Neko server, either
    directly via a `--ws` argument/environment variable or by performing a REST
    login using `--neko-url`, `--username`, and `--password`.

    Once configured, it initializes and runs the `NekoAgent` instance, which
    manages the WebRTC connection, video stream processing, AI inference,
    and sending control commands to the Neko server based on the navigation task.

    Command-line Arguments:
        --ws (str): The direct WebSocket URL (e.g., "wss://neko.example.com/api/ws?token=...").
                    If not provided, attempts REST login.
        --task (str): The navigation task description for the agent (e.g., "Search the weather").
                      Defaults to "Search the weather" or `NEKO_TASK` env var.
        --mode (str): The navigation mode, either "web" or "phone", which determines
                      the allowed action space. Defaults to "web" or `NEKO_MODE` env var.
        --max-steps (int): Maximum number of navigation steps the agent will attempt
                           before terminating. Defaults to `MAX_STEPS`.
        --metrics-port (int): The port for the Prometheus metrics server.
                              Defaults to `DEFAULT_METRIC_PORT`.
        --loglevel (str): The logging level (e.g., "INFO", "DEBUG", "WARNING").
                          Defaults to "INFO" or `NEKO_LOGLEVEL` env var.
        --no-audio: Flag to disable the audio stream in the WebRTC connection.
                    By default, audio is enabled based on `AUDIO_DEFAULT`.
        --neko-url (str): Base HTTP URL for REST login (e.g., "https://host:port").
                          Used if `--ws` is not provided.
        --username (str): Username for REST login. Required with `--neko-url` and `--password`.
        --password (str): Password for REST login. Required with `--neko-url` and `--username`.

    Environment Variables:
        NEKO_WS: Overrides `--ws` argument if set.
        NEKO_TASK: Overrides `--task` argument if set.
        NEKO_MODE: Overrides `--mode` argument if set.
        NEKO_MAX_STEPS: Overrides `--max-steps` argument if set.
        NEKO_METRICS_PORT: Overrides `--metrics-port` argument if set.
        NEKO_LOGLEVEL: Overrides `--loglevel` argument if set.
        NEKO_AUDIO: Boolean (0 or 1) to enable/disable audio.
        NEKO_URL: Overrides `--neko-url` argument if set.
        NEKO_USER: Overrides `--username` argument if set.
        NEKO_PASS: Overrides `--password` argument if set.
        OFFLOAD_FOLDER: Directory for model offloading when using MPS.

    Raises:
        SystemExit: If required arguments for WebSocket URL or REST login are missing,
                    or if the 'requests' library is not installed for REST login,
                    or if REST login fails.
        RuntimeError: If REST login succeeds but no token is received.
    """
    import argparse

    p = argparse.ArgumentParser("neko_agent")
    p.add_argument("--ws",         default=os.environ.get("NEKO_WS",None),
                   help="WebSocket URL (wss://…?token=…); alternative to REST")
    p.add_argument("--task",       default=os.environ.get("NEKO_TASK","Search the weather"),
                   help="Navigation task")
    p.add_argument("--mode",       default=os.environ.get("NEKO_MODE","web"),
                   choices=list(ACTION_SPACES.keys()),
                   help="Navigation mode: web or phone")
    p.add_argument("--max-steps",  type=int, default=MAX_STEPS,
                   help="Max navigation steps per run")
    p.add_argument("--metrics-port",type=int,default=DEFAULT_METRIC_PORT,
                   help="Prometheus metrics port")
    p.add_argument("--loglevel",   default=os.environ.get("NEKO_LOGLEVEL","INFO"),
                   help="Logging level")
    p.add_argument("--no-audio",   dest="audio", action="store_false",
                   help="Disable audio stream")
    p.add_argument("--neko-url",   default=os.environ.get("NEKO_URL",None),
                   help="Base HTTP URL (https://host[:port]) for REST login")
    p.add_argument("--username",   default=os.environ.get("NEKO_USER",None),
                   help="REST login username")
    p.add_argument("--password",   default=os.environ.get("NEKO_PASS",None),
                   help="REST login password")
    p.set_defaults(audio=AUDIO_DEFAULT)
    args = p.parse_args()
    logging.getLogger().setLevel(args.loglevel.upper())

    logger.info(json.dumps({"phase":"setup","run":"startup","msg":"Loading model/processor"}))

    # Determine device and dtype for model loading
    device = "cpu"
    dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
        try:
            _ = torch.zeros(1,dtype=torch.bfloat16,device="mps")
            dtype = torch.bfloat16
        except RuntimeError:
            dtype = torch.float32
        device = "mps"
        # Ensure offload folder exists if MPS is used
        os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

    # Prepare model loading arguments
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",  # Let transformers handle device placement
    }

    # Add MPS-specific offloading parameters if device is MPS
    if device == "mps":
        model_kwargs["offload_folder"] = OFFLOAD_FOLDER
        model_kwargs["offload_state_dict"] = True

    # Load model and processor using the determined device and dtype
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        REPO_ID,
        **model_kwargs
    ).eval()

    processor = AutoProcessor.from_pretrained(
        REPO_ID,
        size={"shortest_edge": SIZE_SHORTEST_EDGE, "longest_edge": SIZE_LONGEST_EDGE},
        trust_remote_code=True
    )

    ws_url = args.ws
    if not ws_url or ws_url==DEFAULT_WS:
        if not (args.neko_url and args.username and args.password):
            p.error("Need --ws or all of --neko-url, --username, --password")
        if any(a.startswith("--password") or a.startswith("--username") for a in sys.argv):
            print("[WARN] Use env vars for secrets.",file=sys.stderr)

        login = args.neko_url.rstrip("/")+"/api/login"
        try:
            r = requests.post(login, json={"username":args.username,"password":args.password}, timeout=10)
            r.raise_for_status()
            tok = r.json().get("token")
            if not tok: raise RuntimeError("no token")
        except Exception as e:
            print(f"REST login failed: {e}",file=sys.stderr); sys.exit(1)
        host = args.neko_url.split("://",1)[-1].rstrip("/")
        scheme = "wss" if args.neko_url.startswith("https") else "ws"
        ws_url = f"{scheme}://{host}/api/ws?token={tok}"
        print(f"[INFO] REST login OK, WS={ws_url}",file=sys.stderr)
    elif any((args.neko_url,args.username,args.password)):
        print("[WARN] --ws provided, ignoring REST args",file=sys.stderr)

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
