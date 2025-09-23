"""Audio broadcasting utilities for Neko WebRTC communication.

This module provides audio buffer management and WebRTC audio track
implementations for real-time audio streaming to Neko sessions.
"""

import logging
import threading
from fractions import Fraction
from typing import Optional

import numpy as np

try:
    import av
except ImportError as e:
    raise ImportError("PyAV (av) is required. pip install av") from e

from aiortc import MediaStreamTrack


class PCMQueue:
    """Bounded PCM jitter buffer with simple drop-oldest backpressure."""

    def __init__(self, sr: int, ch: int, max_sec: float, logger: logging.Logger):
        """Initialize PCM queue with audio parameters.

        :param sr: Sample rate in Hz
        :param ch: Number of audio channels
        :param max_sec: Maximum buffer duration in seconds
        :param logger: Logger instance
        """
        self.sr = sr
        self.ch = ch
        self.max_samples = int(sr * max_sec)
        self._buf = np.zeros((0, ch), dtype=np.int16)
        self._cv = threading.Condition()
        self._log = logger

    def clear(self):
        """Clear the PCM buffer."""
        with self._cv:
            self._buf = np.zeros((0, self.ch), dtype=np.int16)

    def duration(self) -> float:
        """Get current buffer duration in seconds.

        :return: Buffer duration in seconds
        """
        with self._cv:
            return float(len(self._buf)) / float(self.sr)

    def push(self, pcm_i16: np.ndarray):
        """Push PCM samples to the buffer with backpressure handling.

        :param pcm_i16: Int16 PCM samples with shape (N, C)
        """
        if pcm_i16.ndim == 1:
            pcm_i16 = pcm_i16[:, None]
        if pcm_i16.shape[1] != self.ch:
            # Mixdown or simple channel duplication
            if self.ch == 1:
                pcm_i16 = pcm_i16.astype(np.int32, copy=False)
                pcm_i16 = (pcm_i16.mean(axis=1).astype(np.int16, copy=False))[:, None]
            else:
                pcm_i16 = np.repeat(pcm_i16, 2, axis=1)[:, :2]

        with self._cv:
            if len(self._buf) == 0:
                self._buf = pcm_i16.copy()
            else:
                self._buf = np.concatenate([self._buf, pcm_i16], axis=0)
            if len(self._buf) > self.max_samples:
                drop = len(self._buf) - self.max_samples
                self._log.debug("PCM jitter overflow; dropping %d samples (%.2fs)", drop, drop / self.sr)
                self._buf = self._buf[drop:]
            self._cv.notify_all()

    def pull(self, n: int) -> np.ndarray:
        """Pull n samples from the buffer, padding with zeros if needed.

        :param n: Number of samples to pull
        :return: PCM samples with shape (n, channels)
        """
        with self._cv:
            # Wait briefly for data; if underrun, pad with zeros
            if len(self._buf) < n:
                self._cv.wait(timeout=0.02)
            take = min(n, len(self._buf))
            if take > 0:
                out = self._buf[:take].copy()
                self._buf = self._buf[take:]
            else:
                out = np.zeros((0, self.ch), dtype=np.int16)
            if take < n:
                pad = np.zeros((n - take, self.ch), dtype=np.int16)
                out = np.concatenate([out, pad], axis=0)
            return out


class YAPAudioTrack(MediaStreamTrack):
    """Custom aiortc audio track that pulls from a PCMQueue at fixed frame size."""
    kind = "audio"

    def __init__(self, pcmq: PCMQueue, frame_ms: int):
        """Initialize audio track with PCM queue and frame size.

        :param pcmq: PCM queue to pull audio from
        :param frame_ms: Frame size in milliseconds
        """
        super().__init__()
        self.q = pcmq
        self.frame_samples = int(pcmq.sr * frame_ms / 1000)
        self._pts = 0

    async def recv(self) -> av.AudioFrame:
        """Receive audio frame from PCM queue for WebRTC transmission.

        :return: Audio frame for aiortc
        """
        samples = self.q.pull(self.frame_samples)  # (N, C) int16
        layout = "mono" if samples.shape[1] == 1 else "stereo"
        # Reshape for packed format: (N, C) -> (1, N*C) with interleaved channels
        packed_samples = samples.reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(packed_samples, format="s16", layout=layout)
        frame.sample_rate = self.q.sr
        frame.time_base = Fraction(1, self.q.sr)
        frame.pts = self._pts
        self._pts += samples.shape[0]
        return frame