#!/usr/bin/env python3
# src/capture_refactored.py
"""
capture_refactored.py — Training-data capture client using HTTPNekoClient.

What this does
--------------
- Connects to a Neko server using the HTTPNekoClient from neko_comms
- Listens to chat for task boundaries and action annotations
- Pulls frames via HTTP screenshot endpoint at a fixed FPS
- Packages episodes as MDS shards for training

Refactored from original capture.py to use modular neko_comms library.
"""

import asyncio
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
from pathlib import Path
from typing import Dict, List, Optional, Any

from streaming import MDSWriter
from PIL import Image

from neko_comms import HTTPNekoClient


# Environment-based configuration
CAPTURE_OUT = os.environ.get("CAPTURE_OUT", "./data/mds")
CAPTURE_REMOTE = os.environ.get("CAPTURE_REMOTE", "")
CAPTURE_KEEP_LOCAL = int(os.environ.get("CAPTURE_KEEP_LOCAL", "0"))
CAPTURE_COMPRESSION = os.environ.get("CAPTURE_COMPRESSION", "zstd:6")
CAPTURE_SHARD_SIZE = os.environ.get("CAPTURE_SHARD_SIZE", "512mb")
CAPTURE_HASHES = os.environ.get("CAPTURE_HASHES", "sha1").split(",")
CAPTURE_FPS = float(os.environ.get("CAPTURE_FPS", "2"))
CAPTURE_JPEG_QUALITY = int(os.environ.get("CAPTURE_JPEG_QUALITY", "85"))
CAPTURE_MIN_FRAMES = int(os.environ.get("CAPTURE_MIN_FRAMES", "4"))
CAPTURE_EP_TIMEOUT = int(os.environ.get("CAPTURE_EPISODE_TIMEOUT", "900"))

# Logging setup
log = logging.getLogger("neko_capture")

# Chat command patterns
START_RE = re.compile(r"^/start\s+(.+)$", re.IGNORECASE)
STOP_RE = re.compile(r"^/stop\s*$", re.IGNORECASE)
ACTION_RE = re.compile(r"^Action:\s*(\{.*\})\s*$", re.DOTALL)


class EpisodeBuffer:
    """Buffer for collecting episode data (frames, actions, metadata)."""

    def __init__(self, root: str, episode_id: str, task_text: str, screen_size: tuple):
        """Initialize episode buffer.

        :param root: Temporary directory for this episode
        :param episode_id: Unique episode identifier
        :param task_text: Task description
        :param screen_size: Screen dimensions (width, height)
        """
        self.root = Path(root)
        self.episode_id = episode_id
        self.task_text = task_text
        self.screen_size = screen_size

        # Create directory structure
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.frames_ndjson = self.root / "frames.ndjson"
        self.actions_ndjson = self.root / "actions.ndjson"
        self.meta_json = self.root / "meta.json"

        # Initialize files
        self.frames_ndjson.touch()
        self.actions_ndjson.touch()

        # Frame counter
        self.frame_count = 0

    def add_frame(self, img: Image.Image, jpeg_quality: int = 85) -> None:
        """Add a frame to the episode.

        :param img: PIL Image to add
        :param jpeg_quality: JPEG compression quality
        """
        frame_name = f"{self.frame_count:06d}.jpg"
        frame_path = self.frames_dir / frame_name

        # Save frame as JPEG
        img.save(frame_path, "JPEG", quality=jpeg_quality)

        # Add to frames.ndjson
        frame_record = {
            "frame_id": self.frame_count,
            "filename": frame_name,
            "timestamp": time.time(),
            "width": img.width,
            "height": img.height,
        }

        with open(self.frames_ndjson, "a") as f:
            f.write(json.dumps(frame_record) + "\n")

        self.frame_count += 1

    def add_action(self, action_data: Dict[str, Any]) -> None:
        """Add an action to the episode.

        :param action_data: Action dictionary
        """
        action_record = {
            "timestamp": time.time(),
            "action": action_data,
        }

        with open(self.actions_ndjson, "a") as f:
            f.write(json.dumps(action_record) + "\n")

    def finalize_zip_bytes(self, agent_meta: Dict[str, Any]) -> bytes:
        """Finalize episode and return as zip bytes.

        :param agent_meta: Agent metadata
        :return: Zip file contents as bytes
        """
        # Create meta.json
        meta = {
            "episode_id": self.episode_id,
            "task": self.task_text,
            "screen_size": self.screen_size,
            "frame_count": self.frame_count,
            "created_at": time.time(),
            **agent_meta,
        }

        with open(self.meta_json, "w") as f:
            json.dump(meta, f, indent=2)

        # Create zip file in memory
        import io
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add meta.json
            zf.write(self.meta_json, "meta.json")

            # Add frames.ndjson
            zf.write(self.frames_ndjson, "frames.ndjson")

            # Add actions.ndjson
            zf.write(self.actions_ndjson, "actions.ndjson")

            # Add all frame files
            for frame_file in self.frames_dir.glob("*.jpg"):
                zf.write(frame_file, f"frames/{frame_file.name}")

        return zip_buffer.getvalue()


class NekoCapture:
    """Training data capture orchestrator using HTTPNekoClient."""

    def __init__(self,
                 neko_client: HTTPNekoClient,
                 writer: MDSWriter,
                 fps: float = CAPTURE_FPS,
                 jpeg_quality: int = CAPTURE_JPEG_QUALITY,
                 min_frames: int = CAPTURE_MIN_FRAMES,
                 episode_timeout: int = CAPTURE_EP_TIMEOUT):
        """Initialize capture orchestrator.

        :param neko_client: HTTPNekoClient instance
        :param writer: MDSWriter for writing training data
        :param fps: Screenshot capture framerate
        :param jpeg_quality: JPEG quality for screenshots
        :param min_frames: Minimum frames required to save episode
        :param episode_timeout: Timeout in seconds to auto-end episodes
        """
        self.neko = neko_client
        self.writer = writer
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.min_frames = min_frames
        self.episode_timeout = episode_timeout

        # Episode state
        self._episode: Optional[EpisodeBuffer] = None
        self._episode_last_ts: float = 0.0
        self._poll_thread: Optional[threading.Thread] = None
        self._agent_meta = {"source": "neko+capture_refactored", "notes": "refactored using neko_comms"}

        # Stop event
        self._stop_event = threading.Event()

    async def run(self) -> None:
        """Main capture loop."""
        log.info("Starting capture session")

        # Connect to Neko server
        await self.neko.connect()

        try:
            # Listen for chat messages
            await self._chat_listener()
        finally:
            # Clean up
            await self._cleanup()

    async def _chat_listener(self) -> None:
        """Listen for chat commands and action annotations."""
        # Override the client's chat message handler
        original_handler = self.neko._handle_chat_message
        self.neko._handle_chat_message = self._process_chat_message

        # Keep running until stopped
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1)

            # Check for episode timeout
            if self._episode and time.time() - self._episode_last_ts > self.episode_timeout:
                log.warning("Episode timeout - auto-ending")
                self._end_episode("timeout")

        # Restore original handler
        self.neko._handle_chat_message = original_handler

    def _process_chat_message(self, content: str, msg: Dict[str, Any]) -> None:
        """Process chat messages for capture commands.

        :param content: Message content
        :param msg: Full message dictionary
        """
        content = content.strip()

        # Check for /start command
        if match := START_RE.match(content):
            task_text = match.group(1)
            self._begin_episode(task_text)
            return

        # Check for /stop command
        if STOP_RE.match(content):
            self._end_episode("stopped")
            return

        # Check for action annotation
        if match := ACTION_RE.match(content):
            try:
                action_data = json.loads(match.group(1))
                if self._episode:
                    self._episode.add_action(action_data)
                    self._episode_last_ts = time.time()
                    log.debug("Recorded action: %s", action_data.get("action", "unknown"))
            except json.JSONDecodeError:
                log.warning("Failed to parse action: %s", match.group(1))

    def _begin_episode(self, task_text: str) -> None:
        """Start a new episode capture session.

        :param task_text: Task description
        """
        if self._episode:
            log.warning("Episode already running; ending previous")
            self._end_episode("replaced")

        episode_id = str(uuid.uuid4())[:8]
        tmpdir = tempfile.mkdtemp(prefix=f"episode-{episode_id}-")

        self._episode = EpisodeBuffer(
            root=tmpdir,
            episode_id=episode_id,
            task_text=task_text,
            screen_size=self.neko.frame_size
        )
        self._episode_last_ts = time.time()

        log.info("Episode %s started: %s", episode_id, task_text)

        # Start screenshot polling thread
        self._poll_thread = threading.Thread(
            target=self._screenshot_poller,
            name="screenshot-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def _end_episode(self, reason: str) -> None:
        """End the current episode capture session.

        :param reason: Reason for ending
        """
        if not self._episode:
            return

        episode = self._episode
        self._episode = None

        # Stop screenshot polling
        if self._poll_thread:
            self.neko.stop()
            self._poll_thread.join(timeout=3.0)
            self._poll_thread = None

        # Check if episode meets minimum requirements
        if episode.frame_count >= self.min_frames:
            # Finalize and write episode
            payload = episode.finalize_zip_bytes(self._agent_meta)

            sample = {
                "episode_id": episode.episode_id,
                "task": episode.task_text,
                "frame_count": episode.frame_count,
                "screen_size": episode.screen_size,
                "payload": payload,
            }

            self.writer.write(sample)
            log.info("Episode %s completed (%d frames, %s)",
                    episode.episode_id, episode.frame_count, reason)
        else:
            log.info("Episode %s discarded (only %d frames, minimum %d)",
                    episode.episode_id, episode.frame_count, self.min_frames)

        # Clean up temporary directory
        try:
            shutil.rmtree(episode.root)
        except Exception as e:
            log.warning("Failed to clean up episode dir: %s", e)

    def _screenshot_poller(self) -> None:
        """Background thread that polls screenshots at fixed FPS."""
        interval = 1.0 / self.fps

        while not self.neko._stop_event.is_set() and self._episode:
            start_time = time.time()

            # Get screenshot
            img = self.neko.get_screenshot(self.jpeg_quality)
            if img and self._episode:
                self._episode.add_frame(img, self.jpeg_quality)

            # Sleep for remaining interval
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._stop_event.set()

        # End any active episode
        if self._episode:
            self._end_episode("shutdown")

        # Disconnect from Neko
        await self.neko.disconnect()

        # Close writer
        self.writer.finish()
        log.info("Capture session ended")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Neko training data capture")
    parser.add_argument("--neko-url", help="Neko server base URL")
    parser.add_argument("--username", help="Neko username")
    parser.add_argument("--password", help="Neko password")
    parser.add_argument("--ws", help="Direct WebSocket URL")
    parser.add_argument("--output", default=CAPTURE_OUT, help="Output directory")
    parser.add_argument("--remote", default=CAPTURE_REMOTE, help="Remote URI (e.g., s3://bucket)")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Create Neko client
    neko_client = HTTPNekoClient(
        base_url=args.neko_url or os.environ.get("NEKO_URL"),
        ws_url=args.ws or os.environ.get("NEKO_WS"),
        username=args.username or os.environ.get("NEKO_USER"),
        password=args.password or os.environ.get("NEKO_PASS")
    )

    # Create MDS writer
    columns = {
        "episode_id": "str",
        "task": "str",
        "frame_count": "int",
        "screen_size": "json",
        "payload": "bytes",
    }

    writer = MDSWriter(
        out=args.output,
        columns=columns,
        compression=CAPTURE_COMPRESSION,
        hashes=CAPTURE_HASHES,
        size_limit=CAPTURE_SHARD_SIZE,
    )

    if args.remote:
        writer = MDSWriter(
            out=args.remote,
            columns=columns,
            compression=CAPTURE_COMPRESSION,
            hashes=CAPTURE_HASHES,
            size_limit=CAPTURE_SHARD_SIZE,
            keep_local=bool(CAPTURE_KEEP_LOCAL),
        )

    # Create and run capture
    capture = NekoCapture(neko_client, writer)

    try:
        await capture.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error("Capture failed: %s", e, exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())