#!/usr/bin/env python3
# src/capture.py
"""
capture.py — Training-data capture client for Neko that writes MosaicML Streaming (MDS) shards.

What this does
--------------
- Connects to a Neko server (same /api/login and /api/ws flow our agent uses).
- Listens to chat for task boundaries and action annotations (e.g., "Action: {...}" lines our agent emits).
- Pulls frames via the HTTP screenshot endpoint (/api/room/screen/shot.jpg) at a fixed FPS.
- On each task, packages an "episode" as a single sample using the *per-episode record* pattern:
  payload.zip = { meta.json, frames/<NNNNNN>.jpg, frames.ndjson, actions.ndjson }.
- Writes samples into **Mosaic Streaming (MDS)** shards with zstd compression, ready to **stream from S3**.

Why MDS?
--------
MosaicML’s MDSWriter can write locally *and* optionally upload shards to a remote URI like s3://...,
handling shard rotation and remote uploads for you. Configure S3/S3-compatible targets with standard
AWS_* envs and S3_ENDPOINT_URL for MinIO/Ceph style endpoints.

12-Factor friendly
------------------
- All configuration is via env vars or CLI flags (no local, baked-in config).
- Logs to stdout; stateless except for a temporary per-episode working dir.
- Can write to local shards and/or mirror to S3 by setting a remote URI.

Typical use
-----------
1) Send commands in the Neko chat:
   - "/start <task description>" (begin recording)
   - "Action: {…}" (step annotations)
   - "/stop" (end recording)

2) Run this **capture** alongside, pointed at the same Neko.
   It will observe chat, record frames at CAPTURE_FPS, and commit each task as an episode.

Environment / CLI (12-factor)
-----------------------------
NEKO_URL, NEKO_USER, NEKO_PASS       # OR provide --ws wss://.../api/ws?token=...
CAPTURE_OUT="./data/mds"             # local directory
CAPTURE_REMOTE=""                    # e.g. s3://bucket/prefix
CAPTURE_KEEP_LOCAL=0                 # set 1 to keep local copy when remote is used
CAPTURE_COMPRESSION="zstd:6"         # MDS shard compression
CAPTURE_SHARD_SIZE="512mb"           # shard size limit (before rolling)
CAPTURE_HASHES="sha1"                # comma-separated list for shard integrity
CAPTURE_FPS=2                        # screenshot polls per second
CAPTURE_JPEG_QUALITY=85              # /screen/shot.jpg quality (0-100)
CAPTURE_MIN_FRAMES=4                 # ignore runs shorter than this
CAPTURE_EPISODE_TIMEOUT=900          # safety cutoff in seconds

S3 auth (S3 or S3-compatible like MinIO):
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, (optional) AWS_SESSION_TOKEN
S3_ENDPOINT_URL (optional for S3-compatible endpoints)
"""

from __future__ import annotations

# stdlib
import os
import io
import re
import sys
import json
import time
import uuid
import queue
import shutil
import signal
import zipfile
import logging
import pathlib
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

# third-party (available in flake.nix shells)
import requests
import websockets

try:
    # MosaicML Streaming
    from streaming import MDSWriter
except Exception as e:
    print(
        "[FATAL] streaming (MosaicML) not importable. Install `mosaicml-streaming` (aka `streaming`).\n"
        "  pip install streaming\n\n"
        f"Import error: {e}",
        file=sys.stderr,
    )
    sys.exit(2)

# -----------------------
# Configuration / Env
# -----------------------
def env_bool(name: str, default: bool) -> bool:
    """Parse environment variable as boolean.
    
    Converts environment variable values to boolean, supporting both numeric
    (0/1) and string representations (true/false, yes/no, on/off).
    
    :param name: Environment variable name to read
    :param default: Default value if environment variable is not set
    :return: Boolean value parsed from environment variable or default
    """
    val = os.environ.get(name, str(int(default)))
    try:
        return bool(int(val))
    except Exception:
        return str(val).lower() in {"true","t","yes","y","on"}

NEKO_URL            = os.environ.get("NEKO_URL", "")
NEKO_USER           = os.environ.get("NEKO_USER", "")
NEKO_PASS           = os.environ.get("NEKO_PASS", "")
NEKO_WS             = os.environ.get("NEKO_WS", "")  # direct WS (optional)

CAPTURE_OUT         = os.environ.get("CAPTURE_OUT", "./data/mds")
CAPTURE_REMOTE      = os.environ.get("CAPTURE_REMOTE", "")  # e.g. s3://bucket/prefix or empty
CAPTURE_KEEP_LOCAL  = env_bool("CAPTURE_KEEP_LOCAL", False)
CAPTURE_COMPRESSION = os.environ.get("CAPTURE_COMPRESSION", "zstd:6")
CAPTURE_SHARD_SIZE  = os.environ.get("CAPTURE_SHARD_SIZE", "512mb")
CAPTURE_HASHES      = [h.strip() for h in os.environ.get("CAPTURE_HASHES", "sha1").split(",") if h.strip()]

CAPTURE_FPS         = float(os.environ.get("CAPTURE_FPS", "2.0"))
CAPTURE_JPEG_QUALITY= int(os.environ.get("CAPTURE_JPEG_QUALITY", "85"))
CAPTURE_MIN_FRAMES  = int(os.environ.get("CAPTURE_MIN_FRAMES", "4"))
CAPTURE_EP_TIMEOUT  = int(os.environ.get("CAPTURE_EPISODE_TIMEOUT", "900"))  # seconds

LOGLEVEL            = os.environ.get("CAPTURE_LOGLEVEL", "INFO").upper()

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=LOGLEVEL,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("capture")

# -----------------------
# Helpers / Types
# -----------------------
def now_iso() -> str:
    """Return current UTC time in ISO 8601 format.
    
    :return: Current UTC timestamp as ISO 8601 string (YYYY-MM-DDTHH:MM:SS)
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

def safe_mkdir(p: str) -> None:
    """Create directory and any missing parent directories.
    
    Safe directory creation that won't fail if the directory already exists
    and will create any missing parent directories in the path.
    
    :param p: Directory path to create
    """
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

@dataclass
class EpisodeBuffer:
    """Buffers one episode on disk, then materializes a ZIP bytes payload."""
    root: str
    episode_id: str
    task_text: str
    screen_size: Tuple[int, int] = (0, 0)
    started_at: str = field(default_factory=now_iso)
    frames_dir: str = field(init=False)
    actions_path: str = field(init=False)
    frames_idx: int = field(default=0)
    _closed: bool = field(default=False)

    def __post_init__(self):
        """Initialize episode buffer directory structure.
        
        Creates the root directory, frames subdirectory, and initializes
        the actions.ndjson file for storing action annotations.
        """
        safe_mkdir(self.root)
        self.frames_dir = os.path.join(self.root, "frames")
        safe_mkdir(self.frames_dir)
        self.actions_path = os.path.join(self.root, "actions.ndjson")
        with open(self.actions_path, "wb") as f:
            pass

    def add_frame(self, jpg_bytes: bytes, ts: float) -> None:
        """Add a frame to the episode buffer.
        
        Stores the JPEG frame data to disk and updates the frame index.
        Does nothing if the episode buffer has been closed.
        
        :param jpg_bytes: JPEG image data as bytes
        :param ts: Timestamp when the frame was captured
        """
        if self._closed:
            return
        name = f"{self.frames_idx:06d}.jpg"
        with open(os.path.join(self.frames_dir, name), "wb") as f:
            f.write(jpg_bytes)
        # minimal frame index
        with open(os.path.join(self.root, "frames.ndjson"), "ab") as f:
            rec = {"i": self.frames_idx, "ts": ts, "file": f"frames/{name}"}
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        self.frames_idx += 1

    def add_action(self, action: Dict[str, Any], ts: float, raw: Optional[str] = None) -> None:
        """Add an action record to the episode buffer.
        
        Stores an action annotation with timestamp to the actions.ndjson file.
        Does nothing if the episode buffer has been closed.
        
        :param action: Parsed action data as dictionary
        :param ts: Timestamp when the action occurred
        :param raw: Optional raw string representation of the action
        """
        if self._closed:
            return
        rec = {"ts": ts, "action": action}
        if raw:
            rec["raw"] = raw
        with open(self.actions_path, "ab") as f:
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))

    def finalize_zip_bytes(self, ended_at: Optional[str] = None, agent_meta: Optional[Dict[str,Any]] = None) -> bytes:
        """Finalize the episode and return ZIP archive as bytes.
        
        Creates a ZIP archive containing all episode data: metadata, frames,
        frame index, and actions. Cleans up temporary files after packaging.
        
        :param ended_at: Optional end timestamp (defaults to current time)
        :param agent_meta: Optional agent metadata to include
        :return: Complete episode data as ZIP archive bytes
        """
        if self._closed:
            return b""
        self._closed = True

        ended_at = ended_at or now_iso()
        meta = {
            "episode_id": self.episode_id,
            "task": self.task_text,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "screen": {"width": self.screen_size[0], "height": self.screen_size[1]},
            "num_frames": self.frames_idx,
            "agent": agent_meta or {},
            "schema": {
                "frames": "frames/<NNNNNN>.jpg (JPEG, RGB)",
                "frames_index": "frames.ndjson (one JSON object per line: {i, ts, file})",
                "actions": "actions.ndjson (one JSON object per line: {ts, action, raw?})",
            },
            "tool": "capture.py",
            "tool_version": "1.0",
        }

        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_STORED) as zf:
            # meta.json
            zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
            # frames index
            frames_index_path = os.path.join(self.root, "frames.ndjson")
            if os.path.exists(frames_index_path):
                zf.write(frames_index_path, arcname="frames.ndjson")
            # actions
            if os.path.exists(self.actions_path):
                zf.write(self.actions_path, arcname="actions.ndjson")
            # frames
            for name in sorted(os.listdir(self.frames_dir)):
                fp = os.path.join(self.frames_dir, name)
                if os.path.isfile(fp) and name.endswith(".jpg"):
                    zf.write(fp, arcname=f"frames/{name}")
        bio.seek(0)
        payload = bio.read()
        # Cleanup temp episode dir after packaging
        try:
            shutil.rmtree(self.root, ignore_errors=True)
        except Exception:
            pass
        return payload

# -----------------------
# MDS writer factory
# -----------------------
def make_mds_writer(local_out: str,
                    remote_out: Optional[str],
                    keep_local: bool,
                    compression: Optional[str],
                    shard_size: str,
                    hashes: List[str]) -> MDSWriter:
    """Configure an MDSWriter for per-episode training records.
    
    Creates a MosaicML Streaming writer configured for episode data
    with ZIP payloads. Supports local-only or local+remote storage
    with optional remote upload to S3-compatible endpoints.
    
    :param local_out: Local directory for MDS shards
    :param remote_out: Optional remote URI (e.g., s3://bucket/prefix)
    :param keep_local: Whether to keep local shards when uploading remotely
    :param compression: Compression algorithm (e.g., 'zstd:6')
    :param shard_size: Size limit before rolling shards (e.g., '512mb')
    :param hashes: List of hash algorithms for integrity checking
    :return: Configured MDSWriter instance
    
    .. note::
       For remote uploads, use standard AWS environment variables.
       S3-compatible endpoints can be configured with S3_ENDPOINT_URL.
    """
    columns = {
        "episode_id": "str",
        "task": "str",
        "payload": "bytes",     # zip archive
        "num_frames": "int",
        "num_actions": "int",
        "started_at": "str",
        "ended_at": "str",
        "screen_w": "int",
        "screen_h": "int",
        "agent": "json",
    }

    if remote_out:
        out: Any = (local_out, remote_out)
    else:
        out = local_out

    writer = MDSWriter(
        columns=columns,
        out=out,
        keep_local=keep_local,
        compression=compression,
        hashes=hashes or None,
        size_limit=shard_size,
        progress_bar=True,
        exist_ok=True,
    )
    return writer

# -----------------------
# Neko client (REST + WS)
# -----------------------
class NekoSession:
    """Neko WebSocket client session manager.
    
    Handles authentication, WebSocket connections, and message queuing
    for communication with a Neko server. Supports both REST login
    and direct WebSocket URL connection methods.
    
    :param base_url: Base URL for REST API (e.g., https://host)
    :param ws_url: Direct WebSocket URL (optional, alternative to REST)
    :param user: Username for REST authentication
    :param password: Password for REST authentication
    """
    
    def __init__(self, base_url: Optional[str], ws_url: Optional[str], user: Optional[str], password: Optional[str]):
        """Initialize Neko session with connection parameters.
        
        Sets up the session state, HTTP client, and message queues
        for WebSocket communication.
        """
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.ws_url = ws_url or ""
        self.user = user
        self.password = password
        self.session = requests.Session()
        self.token: Optional[str] = None
        self.session_id: Optional[str] = None
        self.screen_size: Tuple[int,int] = (0,0)
        self.stop_event = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

        # queues
        self.chat_q: "queue.Queue[Dict[str,Any]]" = queue.Queue(maxsize=2048)
        self.system_q: "queue.Queue[Dict[str,Any]]" = queue.Queue(maxsize=2048)

    def login_if_needed(self) -> None:
        """Perform REST login if WebSocket URL is not already set.
        
        Authenticates with the Neko server using username/password
        and constructs the WebSocket URL with the returned token.
        
        :raises RuntimeError: If credentials are missing or login fails
        """
        if self.ws_url:
            return
        if not (self.base_url and self.user and self.password):
            raise RuntimeError("Provide --ws OR all of --neko-url, --username, --password")
        r = self.session.post(
            f"{self.base_url}/api/login",
            json={"username": self.user, "password": self.password},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        tok = data.get("token")
        if not tok:
            raise RuntimeError("Login ok but no token in response (enable API token in the server?)")
        self.token = tok
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://",1)[-1]
        self.ws_url = f"{scheme}://{host}/api/ws?token={tok}"
        log.info("REST login ok; WS=%s", self.ws_url.split("?")[0] + "?token=***")

    def _auth_headers(self) -> Dict[str,str]:
        """Build HTTP headers with authentication token if available.
        
        :return: Dictionary of HTTP headers including authorization if token exists
        """
        hdrs = {"Accept": "*/*", "User-Agent": "capture.py/1.0"}
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        return hdrs

    def poll_screenshot(self, fps: float, quality: int, on_frame) -> None:
        """Poll screenshot endpoint at specified FPS and call frame callback.
        
        Continuously fetches JPEG screenshots from the Neko server's
        /api/room/screen/shot.jpg endpoint and calls the provided callback
        with the image data and timestamp.
        
        :param fps: Frames per second to poll at
        :param quality: JPEG quality (0-100) for the screenshots
        :param on_frame: Callback function(jpg_bytes, timestamp)
        """
        if not self.base_url:
            # Derive base from ws_url
            # ws(s)://host/api/ws?token= -> http(s)://host
            if not self.ws_url:
                raise RuntimeError("No NEKO_URL or WS specified.")
            scheme = "https" if self.ws_url.startswith("wss") else "http"
            host = self.ws_url.split("://",1)[-1].split("/",1)[0]
            self.base_url = f"{scheme}://{host}"

        period = max(1e-3, 1.0 / max(0.1, fps))
        last = 0.0
        log.info("Screenshot poller started at %.2f fps (quality=%d)", fps, quality)
        try:
            while not self.stop_event.is_set():
                t0 = time.time()
                try:
                    r = self.session.get(
                        f"{self.base_url}/api/room/screen/shot.jpg",
                        params={"quality": quality},
                        headers=self._auth_headers(),
                        timeout=10,
                    )
                    if r.ok and r.headers.get("content-type","").startswith("image/"):
                        on_frame(r.content, t0)
                    else:
                        log.warning("Shot.jpg non-OK: %s %s", r.status_code, r.text[:200])
                except Exception as e:
                    log.warning("Shot.jpg poll error: %s", e)
                # sleep remainder
                elapsed = time.time() - t0
                to_sleep = max(0.0, period - elapsed)
                time.sleep(to_sleep)
        finally:
            log.info("Screenshot poller exiting.")

    async def _ws_loop(self) -> None:
        """WebSocket message receiver loop.
        
        Connects to the WebSocket and continuously receives messages,
        routing them to appropriate queues based on event type.
        System events go to system_q, chat events go to chat_q.
        """
        async with websockets.connect(self.ws_url, ping_interval=30, ping_timeout=60, max_size=10_000_000) as ws:
            self._ws = ws
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                ev = msg.get("event","")
                if ev.startswith("system/"):
                    self._put(self.system_q, msg)
                elif ev.startswith("chat/") or ev.startswith("send/"):
                    self._put(self.chat_q, msg)

    def start_ws(self) -> None:
        """Start WebSocket connection in a background thread.
        
        Performs login if needed and starts the WebSocket receiver
        thread for continuous message processing.
        """
        self.login_if_needed()
        t = threading.Thread(target=self._run_ws_thread, name="ws-recv", daemon=True)
        t.start()
        self._ws_thread = t

    def _run_ws_thread(self):
        """WebSocket thread entry point with reconnection logic.
        
        Runs the async WebSocket loop in a synchronous thread context,
        automatically reconnecting on failures with exponential backoff.
        """
        import asyncio
        tries = 0
        while not self.stop_event.is_set():
            try:
                asyncio.run(self._ws_loop())
                break
            except Exception as e:
                tries += 1
                log.warning("WS loop error (%d): %s", tries, e)
                time.sleep(min(30, 1.5 * tries))

    def stop(self) -> None:
        """Signal all threads to stop gracefully."""
        self.stop_event.set()

    @staticmethod
    def _put(q: "queue.Queue[Dict[str,Any]]", msg: Dict[str,Any]) -> None:
        """Put message into queue with overflow protection.
        
        Attempts to put message in queue, dropping oldest message
        if queue is full to prevent blocking.
        
        :param q: Target queue for the message
        :param msg: Message dictionary to enqueue
        """
        try:
            q.put_nowait(msg)
        except queue.Full:
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(msg)
            except Exception:
                pass

# -----------------------
# Capture Orchestrator
# -----------------------
START_RE      = re.compile(r"^/start\s+(.+)$", re.IGNORECASE)
STOP_RE       = re.compile(r"^/stop\s*$", re.IGNORECASE)
ACTION_RE     = re.compile(r"^Action:\s*(\{.*\})\s*$", re.DOTALL)

class Capture:
    """Training data capture orchestrator for Neko sessions.
    
    Manages the complete capture workflow: listens for chat commands
    to start/stop episodes, captures screenshots at regular intervals,
    parses action annotations, and writes complete episodes to MDS shards.
    
    :param neko: NekoSession for WebSocket and screenshot access
    :param writer: MDSWriter for writing training data shards
    :param fps: Screenshot capture framerate
    :param jpeg_quality: JPEG quality for screenshots (0-100)
    :param min_frames: Minimum frames required to save an episode
    :param episode_timeout: Timeout in seconds to auto-end episodes
    """
    
    def __init__(self,
                 neko: NekoSession,
                 writer: MDSWriter,
                 fps: float = CAPTURE_FPS,
                 jpeg_quality: int = CAPTURE_JPEG_QUALITY,
                 min_frames: int = CAPTURE_MIN_FRAMES,
                 episode_timeout: int = CAPTURE_EP_TIMEOUT):
        """Initialize capture orchestrator with dependencies and configuration.
        
        Sets up the capture system with Neko session, MDS writer,
        and capture parameters for frame rate, quality, and timeouts.
        """
        self.neko = neko
        self.writer = writer
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.min_frames = min_frames
        self.episode_timeout = episode_timeout

        self._ep: Optional[EpisodeBuffer] = None
        self._ep_last_ts: float = 0.0
        self._poll_thread: Optional[threading.Thread] = None
        self._agent_meta = {"source": "neko+showui", "notes": "frames via shot.jpg; actions via chat"}

    # --- Episode lifecycle ---
    def _begin_episode(self, task_text: str) -> None:
        """Start a new episode capture session.
        
        Creates a new EpisodeBuffer with unique ID and temporary directory,
        then starts the screenshot polling thread to capture frames.
        
        :param task_text: Description of the task being performed
        """
        if self._ep:
            log.warning("Episode already running; ignoring new start.")
            return
        eid = str(uuid.uuid4())[:8]
        tmpdir = tempfile.mkdtemp(prefix=f"episode-{eid}-")
        self._ep = EpisodeBuffer(root=tmpdir, episode_id=eid, task_text=task_text, screen_size=self.neko.screen_size)
        self._ep_last_ts = time.time()
        log.info("Episode %s started: %s", eid, task_text)
        # Start screenshot poller thread
        self._poll_thread = threading.Thread(
            target=self.neko.poll_screenshot,
            args=(self.fps, self.jpeg_quality, self._on_frame),
            name="shot-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def _end_episode(self, reason: str) -> None:
        """End the current episode capture session.
        
        Stops the screenshot polling thread, finalizes the episode data,
        and writes it to the MDS shard if it meets minimum frame requirements.
        
        :param reason: Reason for ending (e.g., 'stopped', 'timeout', 'shutdown')
        """
        if not self._ep:
            return
        ep = self._ep
        self._ep = None
        if self._poll_thread:
            self.neko.stop()
            self._poll_thread.join(timeout=3.0)
            self._poll_thread = None
        num_actions = self._count_ndjson(ep.actions_path)
        payload = ep.finalize_zip_bytes(agent_meta=self._agent_meta)
        num_frames = self._frames_count_from_zip(payload)
        if num_frames >= self.min_frames:
            sample = {
                "episode_id": ep.episode_id,
                "task": ep.task_text,
                "payload": payload,
                "num_frames": num_frames,
                "num_actions": num_actions,
                "started_at": ep.started_at,
                "ended_at": now_iso(),
                "screen_w": ep.screen_size[0],
                "screen_h": ep.screen_size[1],
                "agent": self._agent_meta,
            }
            self.writer.write(sample)
            log.info("Episode %s committed (%d frames, %d actions) [%s]",
                     ep.episode_id, num_frames, num_actions, reason)
        else:
            log.warning("Episode %s discarded (too short: %d frames) [%s]", ep.episode_id, num_frames, reason)

    @staticmethod
    def _count_ndjson(path: str) -> int:
        """Count the number of lines in an NDJSON file.
        
        :param path: Path to the NDJSON file
        :return: Number of lines in the file, or 0 if file doesn't exist
        """
        if not os.path.exists(path):
            return 0
        n = 0
        with open(path, "rb") as f:
            for _ in f:
                n += 1
        return n

    @staticmethod
    def _frames_count_from_zip(zip_bytes: bytes) -> int:
        """Count the number of frame images in a ZIP archive.
        
        :param zip_bytes: ZIP archive bytes containing episode data
        :return: Number of .jpg files in the frames/ directory, or 0 on error
        """
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                names = [n for n in zf.namelist() if n.startswith("frames/") and n.endswith(".jpg")]
                return len(names)
        except Exception:
            return 0

    # --- Data sinks ---
    def _on_frame(self, jpg: bytes, ts: float) -> None:
        """Handle incoming frame data from screenshot polling.
        
        :param jpg: JPEG image data as bytes
        :param ts: Timestamp when the frame was captured
        """
        if not self._ep:
            return
        self._ep.add_frame(jpg, ts)
        self._ep_last_ts = ts

    def _on_action(self, act: Dict[str,Any], ts: float, raw: Optional[str]) -> None:
        """Handle incoming action annotation from chat messages.
        
        :param act: Parsed action dictionary
        :param ts: Timestamp when the action was received
        :param raw: Optional raw string representation of the action
        """
        if not self._ep:
            return
        self._ep.add_action(act, ts, raw)

    # --- Event loop (WS) ---
    def run(self) -> None:
        """Main capture event loop.
        
        Starts WebSocket connection, waits for screen size initialization,
        then processes chat messages to detect episode boundaries and
        action annotations. Handles timeouts and graceful shutdown.
        """
        self.neko.start_ws()
        log.info("Waiting for system/init to learn screen size…")
        screen_w, screen_h = 0, 0
        # Drain a few system messages to pick up screen size early
        t0 = time.time()
        while time.time() - t0 < 5.0:
            try:
                msg = self.neko.system_q.get(timeout=0.5)
            except queue.Empty:
                continue
            ev = msg.get("event","")
            if ev == "system/init":
                pay = msg.get("payload", {})
                size = pay.get("screen_size") or {}
                if isinstance(size, dict):
                    screen_w = int(size.get("width", 0))
                    screen_h = int(size.get("height", 0))
                    self.neko.screen_size = (screen_w, screen_h)
                    log.info("Screen size: %dx%d", screen_w, screen_h)
                    break

        # Main receive/process loop
        try:
            while True:
                # Process chat events
                try:
                    msg = self.neko.chat_q.get(timeout=0.5)
                except queue.Empty:
                    # timeout: check episode timeout
                    if self._ep and (time.time() - self._ep_last_ts > self.episode_timeout):
                        self._end_episode("timeout")
                    continue

                text = None
                ev = msg.get("event","")
                payload = msg.get("payload", {})
                # v3 chat plugin payloads carry content in a few shapes
                if ev == "chat/message":
                    content = payload.get("content")
                    if isinstance(content, dict) and "text" in content:
                        text = content.get("text")
                    elif isinstance(payload.get("text",""), str):
                        text = payload["text"]
                    elif isinstance(payload.get("message",""), str):
                        text = payload["message"]
                elif ev in ("send/broadcast","send/unicast"):
                    body = payload.get("body")
                    if isinstance(body, str):
                        text = body
                    elif isinstance(body, dict):
                        text = body.get("text") or body.get("message")

                if not isinstance(text, str):
                    continue
                text = text.strip()

                # Detect task boundaries
                start_match = START_RE.match(text)
                if start_match and not self._ep:
                    task_text = start_match.group(1).strip()
                    self._begin_episode(task_text)
                    continue

                if STOP_RE.match(text) and self._ep:
                    self._end_episode("stopped")
                    continue

                # Parse action annotations ("Action: {...}")
                ma = ACTION_RE.match(text)
                if ma:
                    raw = ma.group(1)
                    act = self._parse_action_json(raw)
                    if act is not None:
                        self._on_action(act, time.time(), raw=raw)
                    continue

        except KeyboardInterrupt:
            log.info("Ctrl-C pressed; shutting down…")
        finally:
            # End any in-flight episode
            if self._ep:
                self._end_episode("shutdown")

    @staticmethod
    def _parse_action_json(raw: str) -> Optional[Dict[str,Any]]:
        """Parse action JSON string with fallback to Python literal parsing.
        
        Attempts to parse JSON first, then falls back to ast.literal_eval
        for Python-style dictionaries with single quotes.
        
        :param raw: Raw action string to parse
        :return: Parsed action dictionary or None if parsing fails
        """
        try:
            return json.loads(raw)
        except Exception:
            # tolerate single quotes / python-style dicts
            try:
                import ast
                return ast.literal_eval(raw)
            except Exception:
                log.debug("Failed to parse action JSON: %s", raw[:200])
                return None

# -----------------------
# CLI
# -----------------------
def parse_args(argv: Optional[List[str]] = None):
    """Parse command line arguments for capture tool.
    
    Configures argument parser with connection, output, and capture
    parameter groups. Falls back to environment variables for defaults.
    
    :param argv: Optional argument list (defaults to sys.argv)
    :return: Parsed arguments namespace
    """
    import argparse
    p = argparse.ArgumentParser("capture", description="Neko training capture → Mosaic Streaming (MDS)")
    g_conn = p.add_argument_group("Connection")
    g_conn.add_argument("--ws", dest="ws", default=os.environ.get("NEKO_WS", ""), help="wss://host/api/ws?token=… (skips REST login)")
    g_conn.add_argument("--neko-url", dest="url", default=os.environ.get("NEKO_URL", ""), help="https://host (for REST login)")
    g_conn.add_argument("--username", dest="user", default=os.environ.get("NEKO_USER", ""), help="REST username")
    g_conn.add_argument("--password", dest="password", default=os.environ.get("NEKO_PASS", ""), help="REST password")

    g_out = p.add_argument_group("Output (MDS)")
    g_out.add_argument("--out", default=CAPTURE_OUT, help="Local MDS dir (default: %(default)s)")
    g_out.add_argument("--remote", default=CAPTURE_REMOTE, help="Optional remote URI (e.g., s3://bucket/prefix)")
    g_out.add_argument("--keep-local", action="store_true", default=CAPTURE_KEEP_LOCAL, help="Keep local shards when uploading to remote")
    g_out.add_argument("--compression", default=CAPTURE_COMPRESSION, help="Shard compression like 'zstd:6' (default: %(default)s)")
    g_out.add_argument("--shard-size", default=CAPTURE_SHARD_SIZE, help="Shard size limit before rolling (e.g., '512mb')")
    g_out.add_argument("--hashes", default=",".join(CAPTURE_HASHES), help="Comma-separated hashes for integrity (default: sha1)")

    g_cap = p.add_argument_group("Capture")
    g_cap.add_argument("--fps", type=float, default=CAPTURE_FPS, help="Screenshot polls per second (default: %(default)s)")
    g_cap.add_argument("--jpeg-quality", type=int, default=CAPTURE_JPEG_QUALITY, help="JPEG quality 0-100 (default: %(default)s)")
    g_cap.add_argument("--min-frames", type=int, default=CAPTURE_MIN_FRAMES, help="Min frames to keep an episode (default: %(default)s)")
    g_cap.add_argument("--episode-timeout", type=int, default=CAPTURE_EP_TIMEOUT, help="End an episode after N seconds of inactivity (default: %(default)s)")
    g_cap.add_argument("--loglevel", default=LOGLEVEL, help="Logging level (default: %(default)s)")

    args = p.parse_args(argv)
    return args

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for capture application.
    
    Parses command line arguments, configures logging and signal handlers,
    initializes Neko session and MDS writer, then runs the capture loop.
    
    :param argv: Optional command line arguments (defaults to sys.argv)
    :return: Exit code (0 for success)
    """
    args = parse_args(argv)
    logging.getLogger().setLevel(args.loglevel.upper())

    # Ctrl-C friendly shutdown for threads
    def _sig(*_a):
        raise KeyboardInterrupt()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except Exception:
            pass

    # Neko session
    neko = NekoSession(base_url=args.url, ws_url=args.ws, user=args.user, password=args.password)

    # MDS writer
    hashes = [h.strip() for h in (args.hashes or "").split(",") if h.strip()]
    writer = make_mds_writer(
        local_out=args.out,
        remote_out=args.remote or None,
        keep_local=bool(args.keep_local),
        compression=args.compression or None,
        shard_size=args.shard_size,
        hashes=hashes,
    )

    log.info("Capture config: out=%s remote=%s keep_local=%s compression=%s shard=%s hashes=%s fps=%.2f",
             args.out, args.remote or "-", bool(args.keep_local), args.compression, args.shard_size, hashes, args.fps)

    cap = Capture(
        neko=neko,
        writer=writer,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        min_frames=args.min_frames,
        episode_timeout=args.episode_timeout,
    )

    try:
        cap.run()
        return 0
    finally:
        try:
            writer.finish()
        except Exception:
            pass

if __name__ == "__main__":
    raise SystemExit(main())
