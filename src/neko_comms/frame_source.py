"""Frame source abstractions for Neko video handling.

This module provides abstract and concrete implementations for receiving
video frames from different sources (WebRTC tracks, WebSocket video_lite, etc.).
"""

import asyncio
import base64
import contextlib
import logging
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Optional

from PIL import Image
from webrtc import MediaStreamTrack

from .types import Frame

logger = logging.getLogger(__name__)


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

    @abstractmethod
    async def wait_for_frame(self, timeout: float = 5.0) -> Optional[Image.Image]:
        """Wait for a new frame with timeout.

        :param timeout: Maximum time to wait in seconds
        :type timeout: float
        :return: The frame or None if timeout
        :rtype: Optional[Image.Image]
        """
        ...


class WebRTCFrameSource(FrameSource):
    """Frame source that receives video frames from a WebRTC MediaStreamTrack.

    This implementation reads frames from a MediaStreamTrack in a
    background task, converting them to PIL Images and making them available
    for retrieval. It includes frame filtering and error handling.
    """

    def __init__(self, max_size: Optional[tuple] = None) -> None:
        """Initialize the WebRTC frame source.

        :param max_size: Optional maximum frame size as (width, height)
        :type max_size: Optional[tuple]
        """
        self.image: Optional[Image.Image] = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.first_frame = asyncio.Event()
        self.frame_event = asyncio.Event()
        self._running = False
        self.max_size = max_size

    async def start(self, *args: Any) -> None:
        """Start reading frames from a MediaStreamTrack.

        :param args: Must contain a MediaStreamTrack as the first argument.
        :type args: Any
        :raises ValueError: If no MediaStreamTrack is provided.
        :return: None
        :rtype: None
        """
        if not args:
            raise ValueError("WebRTCFrameSource.start(): need MediaStreamTrack")
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
            self.frame_event.clear()

    async def get(self) -> Optional[Image.Image]:
        """Retrieve the current frame image.

        :return: The current frame as a PIL Image, or None if unavailable.
        :rtype: Optional[Image.Image]
        """
        async with self.lock:
            return self.image.copy() if self.image else None

    async def wait_for_frame(self, timeout: float = 5.0) -> Optional[Image.Image]:
        """Wait for a new frame with timeout.

        :param timeout: Maximum time to wait in seconds
        :type timeout: float
        :return: The frame or None if timeout
        :rtype: Optional[Image.Image]
        """
        try:
            await asyncio.wait_for(self.frame_event.wait(), timeout=timeout)
            self.frame_event.clear()
            return await self.get()
        except asyncio.TimeoutError:
            return None

    def _convert_frame(self, frame: Any) -> Optional[Image.Image]:
        """Convert a WebRTC frame into a PIL Image with fallbacks."""
        try:
            img = frame.to_image()
            w, h = img.size
            if w <= 0 or h <= 0:
                raise ValueError(f"invalid image dimensions: {w}x{h}")
            return img.convert("RGB")
        except Exception as primary_exc:
            try:
                arr = frame.to_ndarray(format="rgb24")
                if getattr(arr, "size", 0) == 0:
                    raise ValueError("empty ndarray from frame")
                img = Image.fromarray(arr, "RGB")
                w, h = img.size
                if w <= 0 or h <= 0:
                    raise ValueError(f"invalid ndarray image dimensions: {w}x{h}")
                return img
            except Exception as fallback_exc:
                logger.warning(
                    "Frame conversion failed: %s; fallback error: %s",
                    primary_exc,
                    fallback_exc
                )
        return None

    async def _reader(self, track: MediaStreamTrack) -> None:
        """Background task that reads frames from the MediaStreamTrack."""
        try:
            while self._running:
                try:
                    frame = await track.recv()

                    img = self._convert_frame(frame)
                    if img is None:
                        await asyncio.sleep(0.05)
                        continue

                    if self.max_size and (img.width > self.max_size[0] or img.height > self.max_size[1]):
                        img = img.copy()
                        img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
                    else:
                        img = img.copy()

                    async with self.lock:
                        self.image = img
                        if not self.first_frame.is_set():
                            self.first_frame.set()
                        self.frame_event.set()

                except StopAsyncIteration:
                    logger.info("WebRTC video stream ended")
                    break
                except Exception as e:
                    logger.error("Error reading WebRTC frame: %s", e)
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.debug("WebRTC frame reader cancelled")
        except Exception as e:
            logger.error("WebRTC frame reader error: %s", e)
        finally:
            self._running = False

class LiteFrameSource(FrameSource):
    """Frame source for video_lite mode using base64-encoded images over WebSocket.

    This implementation receives base64-encoded image data via WebSocket messages
    and converts them to PIL Images. Used as fallback when WebRTC is not available.
    """

    def __init__(self) -> None:
        """Initialize the lite frame source."""
        self.image: Optional[Image.Image] = None
        self.lock = asyncio.Lock()
        self.frame_event = asyncio.Event()
        self._running = False

    async def start(self, *args: Any) -> None:
        """Start the lite frame source.

        :param args: Not used for lite frame source
        :type args: Any
        :return: None
        :rtype: None
        """
        self._running = True

    async def stop(self) -> None:
        """Stop the lite frame source and clean up resources.

        :return: None
        :rtype: None
        """
        self._running = False
        async with self.lock:
            self.image = None
            self.frame_event.clear()

    async def get(self) -> Optional[Image.Image]:
        """Retrieve the current frame image.

        :return: The current frame as a PIL Image, or None if unavailable.
        :rtype: Optional[Image.Image]
        """
        async with self.lock:
            return self.image.copy() if self.image else None

    async def wait_for_frame(self, timeout: float = 5.0) -> Optional[Image.Image]:
        """Wait for a new frame with timeout.

        :param timeout: Maximum time to wait in seconds
        :type timeout: float
        :return: The frame or None if timeout
        :rtype: Optional[Image.Image]
        """
        try:
            await asyncio.wait_for(self.frame_event.wait(), timeout=timeout)
            self.frame_event.clear()
            return await self.get()
        except asyncio.TimeoutError:
            return None

    async def update_frame(self, base64_data: str) -> None:
        """Update the current frame with base64-encoded image data.

        :param base64_data: Base64-encoded image data
        :type base64_data: str
        :return: None
        :rtype: None
        """
        try:
            # Decode base64 image data
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data))

            # Store the frame
            async with self.lock:
                self.image = img
                self.frame_event.set()

        except Exception as e:
            logger.error("Error decoding lite frame: %s", e)


class NoFrameSource(FrameSource):
    """Null frame source that returns no frames.

    Used when no video is needed or available.
    """

    async def start(self, *args: Any) -> None:
        """Start the null frame source (no-op).

        :param args: Ignored
        :type args: Any
        :return: None
        :rtype: None
        """
        pass

    async def stop(self) -> None:
        """Stop the null frame source (no-op).

        :return: None
        :rtype: None
        """
        pass

    async def get(self) -> Optional[Image.Image]:
        """Retrieve the current frame (always None).

        :return: Always None
        :rtype: Optional[Image.Image]
        """
        return None

    async def wait_for_frame(self, timeout: float = 5.0) -> Optional[Image.Image]:
        """Wait for a frame (always times out).

        :param timeout: Timeout in seconds
        :type timeout: float
        :return: Always None
        :rtype: Optional[Image.Image]
        """
        await asyncio.sleep(timeout)
        return None
