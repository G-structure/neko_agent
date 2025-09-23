"""
capture_refactored.py — Training-data capture client using HTTPNekoClient.

What this does
--------------
- Connects to a Neko server using the HTTPNekoClient from neko_comms
- Listens to chat for task boundaries and action annotations
- Pulls frames via HTTP screenshot endpoint at a fixed FPS
- Packages episodes as MDS shards for training

Refactored from original capture.py to use modular neko_comms library while
retaining the legacy capture functionality.
"""

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from streaming import MDSWriter

from neko_comms import HTTPNekoClient

# Environment-based configuration mirrors capture.py defaults
CAPTURE_OUT = os.environ.get("CAPTURE_OUT", "./data/mds")
CAPTURE_REMOTE = os.environ.get("CAPTURE_REMOTE", "")
CAPTURE_KEEP_LOCAL = bool(int(os.environ.get("CAPTURE_KEEP_LOCAL", "0")))
CAPTURE_COMPRESSION = os.environ.get("CAPTURE_COMPRESSION", "zstd:6") or None
CAPTURE_SHARD_SIZE = os.environ.get("CAPTURE_SHARD_SIZE", "512mb")
CAPTURE_HASHES = [h.strip() for h in os.environ.get("CAPTURE_HASHES", "sha1").split(",") if h.strip()]
CAPTURE_FPS = float(os.environ.get("CAPTURE_FPS", "2.0"))
CAPTURE_JPEG_QUALITY = int(os.environ.get("CAPTURE_JPEG_QUALITY", "85"))
CAPTURE_MIN_FRAMES = int(os.environ.get("CAPTURE_MIN_FRAMES", "4"))
CAPTURE_EP_TIMEOUT = int(os.environ.get("CAPTURE_EPISODE_TIMEOUT", "900"))

# Logging setup
log = logging.getLogger("neko_capture")

# Chat command patterns
START_RE = re.compile(r"^/start\s+(.+)$", re.IGNORECASE)
STOP_RE = re.compile(r"^/stop\s*$", re.IGNORECASE)
ACTION_RE = re.compile(r"^Action:\s*(\{.*\})\s*$", re.DOTALL)


# ----------------------
# Helpers
# ----------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_mds_writer(
    local_out: str,
    remote_out: Optional[str],
    keep_local: bool,
    compression: Optional[str],
    shard_size: str,
    hashes: List[str],
) -> MDSWriter:
    columns = {
        "episode_id": "str",
        "task": "str",
        "payload": "bytes",
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


# ----------------------
# Episode buffering (matches capture.py semantics)
# ----------------------
@dataclass
class EpisodeBuffer:
    root: str
    episode_id: str
    task_text: str
    screen_size: tuple[int, int] = (0, 0)
    started_at: str = field(default_factory=now_iso)
    frames_dir: str = field(init=False)
    actions_path: str = field(init=False)
    frames_idx: int = field(default=0)
    _closed: bool = field(default=False)

    def __post_init__(self) -> None:
        safe_mkdir(self.root)
        self.frames_dir = os.path.join(self.root, "frames")
        safe_mkdir(self.frames_dir)
        self.actions_path = os.path.join(self.root, "actions.ndjson")
        with open(self.actions_path, "wb") as fh:
            fh.write(b"")
        frames_index = os.path.join(self.root, "frames.ndjson")
        with open(frames_index, "wb") as fh:
            fh.write(b"")

    def add_frame(self, jpg_bytes: bytes, ts: float) -> None:
        if self._closed:
            return
        name = f"{self.frames_idx:06d}.jpg"
        frame_path = os.path.join(self.frames_dir, name)
        with open(frame_path, "wb") as fh:
            fh.write(jpg_bytes)
        record = {"i": self.frames_idx, "ts": ts, "file": f"frames/{name}"}
        frames_index = os.path.join(self.root, "frames.ndjson")
        with open(frames_index, "ab") as fh:
            fh.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        self.frames_idx += 1

    def add_action(self, action: Dict[str, Any], ts: float, raw: Optional[str] = None) -> None:
        if self._closed:
            return
        record = {"ts": ts, "action": action}
        if raw:
            record["raw"] = raw
        with open(self.actions_path, "ab") as fh:
            fh.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))

    def finalize_zip_bytes(self, ended_at: Optional[str], agent_meta: Dict[str, Any]) -> bytes:
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
            "agent": agent_meta,
            "schema": {
                "frames": "frames/<NNNNNN>.jpg (JPEG, RGB)",
                "frames_index": "frames.ndjson (one JSON object per line: {i, ts, file})",
                "actions": "actions.ndjson (one JSON object per line: {ts, action, raw?})",
            },
            "tool": "capture_refactored.py",
            "tool_version": "1.0",
        }

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
            frames_index = os.path.join(self.root, "frames.ndjson")
            if os.path.exists(frames_index):
                zf.write(frames_index, arcname="frames.ndjson")
            if os.path.exists(self.actions_path):
                zf.write(self.actions_path, arcname="actions.ndjson")
            for name in sorted(os.listdir(self.frames_dir)):
                path = os.path.join(self.frames_dir, name)
                if os.path.isfile(path) and name.endswith(".jpg"):
                    zf.write(path, arcname=f"frames/{name}")

        buffer.seek(0)
        payload = buffer.read()
        shutil.rmtree(self.root, ignore_errors=True)
        return payload


# ----------------------
# Capture orchestrator
# ----------------------
class NekoCapture:
    def __init__(
        self,
        neko_client: HTTPNekoClient,
        writer: MDSWriter,
        fps: float = CAPTURE_FPS,
        jpeg_quality: int = CAPTURE_JPEG_QUALITY,
        min_frames: int = CAPTURE_MIN_FRAMES,
        episode_timeout: int = CAPTURE_EP_TIMEOUT,
    ) -> None:
        self.neko = neko_client
        self.writer = writer
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.min_frames = min_frames
        self.episode_timeout = episode_timeout

        self._episode: Optional[EpisodeBuffer] = None
        self._episode_last_ts: float = 0.0
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop_event: Optional[threading.Event] = None
        self._stop_event = threading.Event()
        self._agent_meta = {"source": "neko+showui", "notes": "frames via shot.jpg; actions via chat"}

    async def run(self) -> None:
        log.info("Starting capture session")
        await self.neko.connect()
        self.neko.add_chat_listener(self._process_chat_message)

        try:
            await self._chat_listener()
        finally:
            await self._cleanup()

    async def _chat_listener(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)
            if self._episode and (time.time() - self._episode_last_ts > self.episode_timeout):
                log.warning("Episode timeout - auto-ending")
                self._end_episode("timeout")

    def _process_chat_message(self, _content: str, msg: Dict[str, Any]) -> None:
        text = self._extract_text(msg)
        if not text:
            return
        text = text.strip()

        if match := START_RE.match(text):
            task_text = match.group(1)
            self._begin_episode(task_text)
            return

        if STOP_RE.match(text):
            self._end_episode("stopped")
            return

        if match := ACTION_RE.match(text):
            raw = match.group(1)
            try:
                action_data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Failed to parse action: %s", raw)
                return
            if self._episode:
                ts = time.time()
                self._episode.add_action(action_data, ts, raw)
                self._episode_last_ts = ts
                log.debug("Recorded action: %s", action_data.get("action", "unknown"))

    def _begin_episode(self, task_text: str) -> None:
        if self._episode:
            log.warning("Episode already running; ignoring new start.")
            return

        episode_id = str(uuid.uuid4())[:8]
        tmpdir = tempfile.mkdtemp(prefix=f"episode-{episode_id}-")
        self._episode = EpisodeBuffer(
            root=tmpdir,
            episode_id=episode_id,
            task_text=task_text,
            screen_size=self.neko.frame_size,
        )
        self._episode_last_ts = time.time()
        log.info("Episode %s started: %s", episode_id, task_text)

        self._poll_stop_event = threading.Event()
        self._poll_thread = threading.Thread(
            target=self._screenshot_poller,
            args=(self._poll_stop_event,),
            name="screenshot-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def _end_episode(self, reason: str) -> None:
        if not self._episode:
            return

        episode = self._episode
        self._episode = None

        if self._poll_thread and self._poll_stop_event:
            self._poll_stop_event.set()
            self._poll_thread.join(timeout=3.0)
        self._poll_thread = None
        self._poll_stop_event = None

        num_actions = self._count_ndjson(episode.actions_path)
        ended_at = now_iso()
        payload = episode.finalize_zip_bytes(ended_at=ended_at, agent_meta=self._agent_meta)
        num_frames = self._frames_count_from_zip(payload)

        if num_frames >= self.min_frames:
            sample = {
                "episode_id": episode.episode_id,
                "task": episode.task_text,
                "payload": payload,
                "num_frames": num_frames,
                "num_actions": num_actions,
                "started_at": episode.started_at,
                "ended_at": ended_at,
                "screen_w": episode.screen_size[0],
                "screen_h": episode.screen_size[1],
                "agent": self._agent_meta,
            }
            self.writer.write(sample)
            log.info(
                "Episode %s committed (%d frames, %d actions) [%s]",
                episode.episode_id,
                num_frames,
                num_actions,
                reason,
            )
        else:
            log.warning(
                "Episode %s discarded (too short: %d frames) [%s]",
                episode.episode_id,
                num_frames,
                reason,
            )

    def _screenshot_poller(self, stop_event: threading.Event) -> None:
        interval = 1.0 / self.fps
        while not stop_event.is_set():
            start_time = time.time()
            episode = self._episode
            if not episode:
                break

            img = self.neko.get_screenshot(self.jpeg_quality)
            if img and self._episode:
                with io.BytesIO() as buffer:
                    img.save(buffer, "JPEG", quality=self.jpeg_quality)
                    frame_bytes = buffer.getvalue()
                if hasattr(img, "close"):
                    img.close()
                ts = time.time()
                self._episode.add_frame(frame_bytes, ts)
                self._episode_last_ts = ts

            elapsed = time.time() - start_time
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    async def _cleanup(self) -> None:
        self._stop_event.set()
        if self._episode:
            self._end_episode("shutdown")

        if self._poll_stop_event:
            self._poll_stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=3.0)
            self._poll_thread = None
        self._poll_stop_event = None

        with contextlib.suppress(Exception):
            self.neko.stop()
        await self.neko.disconnect()

        try:
            self.writer.finish()
        except Exception:
            pass

        log.info("Capture session ended")

    @staticmethod
    def _count_ndjson(path: str) -> int:
        if not os.path.exists(path):
            return 0
        count = 0
        with open(path, "rb") as fh:
            for _ in fh:
                count += 1
        return count

    @staticmethod
    def _frames_count_from_zip(zip_bytes: bytes) -> int:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                return len([n for n in zf.namelist() if n.startswith("frames/") and n.endswith(".jpg")])
        except Exception:
            return 0

    @staticmethod
    def _extract_text(msg: Dict[str, Any]) -> Optional[str]:
        event = msg.get("event", "")
        payload = msg.get("payload", {})

        if event == "chat/message":
            content = payload.get("content")
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    return text
            text = payload.get("text") or payload.get("message")
            if isinstance(text, str):
                return text
        elif event in ("send/broadcast", "send/unicast"):
            body = payload.get("body")
            if isinstance(body, str):
                return body
            if isinstance(body, dict):
                text = body.get("text") or body.get("message")
                if isinstance(text, str):
                    return text
        return None


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Neko training data capture")
    parser.add_argument("--neko-url", help="Neko server base URL")
    parser.add_argument("--username", help="Neko username")
    parser.add_argument("--password", help="Neko password")
    parser.add_argument("--ws", help="Direct WebSocket URL")
    parser.add_argument("--output", default=CAPTURE_OUT, help="Output directory")
    parser.add_argument("--remote", default=CAPTURE_REMOTE, help="Remote URI (e.g., s3://bucket)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s',
        datefmt='%H:%M:%S'
    )

    neko_client = HTTPNekoClient(
        base_url=args.neko_url or os.environ.get("NEKO_URL"),
        ws_url=args.ws or os.environ.get("NEKO_WS"),
        username=args.username or os.environ.get("NEKO_USER"),
        password=args.password or os.environ.get("NEKO_PASS"),
    )

    writer = make_mds_writer(
        local_out=args.output,
        remote_out=(args.remote or None) or (CAPTURE_REMOTE or None),
        keep_local=CAPTURE_KEEP_LOCAL,
        compression=CAPTURE_COMPRESSION,
        shard_size=CAPTURE_SHARD_SIZE,
        hashes=CAPTURE_HASHES,
    )

    capture = NekoCapture(
        neko_client,
        writer,
        fps=CAPTURE_FPS,
        jpeg_quality=CAPTURE_JPEG_QUALITY,
        min_frames=CAPTURE_MIN_FRAMES,
        episode_timeout=CAPTURE_EP_TIMEOUT,
    )

    try:
        await capture.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as exc:
        log.error("Capture failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
