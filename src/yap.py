#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/yap.py — Neko WebRTC TTS broadcaster

What it does
------------
- Connects to the Neko server (same WS signaling as agent/capture).
- Adds one outbound WebRTC audio track (aiortc + PyAV).
- Listens to chat for /yap commands:
  - /yap <text>              → speak immediately
  - /yap:begin ... /yap:end  → streaming mode (incremental chunking)
  - /yap:stop                → cancel/flush current queue
  - /yap:voice set ...       → switch active voice/style/params live
  - /yap:voice add ...       → register a new voice in voices.json
  - /yap:voice reload        → hot-reload voices.json
- Low-latency pipeline: punctuation-aware segmentation → parallel F5-TTS
  workers → crossfade splicer → jitter-buffered WebRTC playout at 48kHz.

Dependencies
------------
- aiortc, av (PyAV), websockets, numpy
- F5-TTS: https://github.com/SWivid/F5-TTS
- Optional resampling backends (first that imports will be used):
  - torchaudio  OR  scipy.signal.resample_poly  OR fallback (linear)

12-Factor
---------
- All config via env/CLI (see Settings below).
- No side effects at import time.
- Structured logging (text/json).
- Graceful shutdown on SIGINT/SIGTERM.

"""

from __future__ import annotations

# stdlib
import os
import re
import io
import json
import sys
import time
import math
import signal
import asyncio
import logging
import contextlib
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

# third-party
import numpy as np

try:
    import av  # PyAV
except Exception as e:  # pragma: no cover
    print("PyAV (av) is required. pip install av", file=sys.stderr)
    raise

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack,
)
from aiortc.sdp import candidate_from_sdp

from neko.logging import setup_logging
from neko.websocket import Signaler


# ----------------------
# Resampling helpers
# ----------------------
_RESAMPLE_BACKEND = None
try:
    import torch
    import torchaudio  # type: ignore

    def _resample(wave: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        """Resample audio using torchaudio backend.
        
        :param wave: Input audio waveform with shape (N, C)
        :param sr_from: Source sample rate
        :param sr_to: Target sample rate
        :return: Resampled audio with shape (N, C)
        """
        global _RESAMPLE_BACKEND
        _RESAMPLE_BACKEND = _RESAMPLE_BACKEND or "torchaudio"
        t = torch.as_tensor(wave.T)  # [C, N]
        out = torchaudio.functional.resample(t, sr_from, sr_to)
        return out.T.contiguous().cpu().numpy()

except Exception:
    try:
        from scipy.signal import resample_poly  # type: ignore

        def _resample(wave: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
            """Resample audio using scipy backend.
            
            :param wave: Input audio waveform with shape (N, C)
            :param sr_from: Source sample rate
            :param sr_to: Target sample rate
            :return: Resampled audio with shape (N, C)
            """
            global _RESAMPLE_BACKEND
            _RESAMPLE_BACKEND = _RESAMPLE_BACKEND or "scipy"
            g = math.gcd(sr_from, sr_to)
            up, down = sr_to // g, sr_from // g
            return resample_poly(wave, up, down, axis=0)

    except Exception:
        def _resample(wave: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
            """Very simple linear resampler (fallback)."""
            global _RESAMPLE_BACKEND
            _RESAMPLE_BACKEND = _RESAMPLE_BACKEND or "linear"
            if sr_from == sr_to or wave.size == 0:
                return wave
            ratio = sr_to / sr_from
            n_src = wave.shape[0]
            n_dst = int(round(n_src * ratio))
            x = np.linspace(0.0, 1.0, n_src, endpoint=False)
            xi = np.linspace(0.0, 1.0, n_dst, endpoint=False)
            out = np.empty((n_dst, wave.shape[1]), dtype=wave.dtype)
            for ch in range(wave.shape[1]):
                out[:, ch] = np.interp(xi, x, wave[:, ch])
            return out


# ----------------------
# Settings (12-factor)
# ----------------------
@dataclass
class Settings:
    # Connection
    ws_url: str = field(default_factory=lambda: os.environ.get("YAP_WS", os.environ.get("NEKO_WS", "")))
    neko_url: Optional[str] = field(default_factory=lambda: os.environ.get("NEKO_URL", None))
    username: Optional[str] = field(default_factory=lambda: os.environ.get("NEKO_USER", None))
    password: Optional[str] = field(default_factory=lambda: os.environ.get("NEKO_PASS", None))

    # Audio / playout
    sr: int = int(os.environ.get("YAP_SR", "48000"))
    channels: int = int(os.environ.get("YAP_AUDIO_CHANNELS", "1"))
    frame_ms: int = int(os.environ.get("YAP_FRAME_MS", "20"))
    jitter_max_sec: float = float(os.environ.get("YAP_JITTER_MAX_SEC", "6.0"))

    # TTS / pipeline
    parallel: int = int(os.environ.get("YAP_PARALLEL", "2"))
    chunk_target_sec: float = float(os.environ.get("YAP_CHUNK_TARGET_SEC", "3.0"))
    max_chunk_chars: int = int(os.environ.get("YAP_MAX_CHARS", "350"))
    overlap_ms: int = int(os.environ.get("YAP_OVERLAP_MS", "30"))

    # Voice
    voices_dir: str = os.environ.get("YAP_VOICES_DIR", "./voices")
    default_speaker: str = os.environ.get("YAP_SPK_DEFAULT", "default")

    # Misc
    log_level: str = os.environ.get("YAP_LOGLEVEL", "INFO")
    log_format: str = os.environ.get("YAP_LOG_FORMAT", "text")  # 'text'|'json'
    metrics_port: int = int(os.environ.get("YAP_METRICS_PORT", os.environ.get("PORT", "0") or "0"))

    # ICE policy helpers (optional)
    stun_url: str = os.environ.get("YAP_STUN_URL", "stun:stun.l.google.com:19302")
    turn_url: Optional[str] = os.environ.get("YAP_TURN_URL", None)
    turn_user: Optional[str] = os.environ.get("YAP_TURN_USER", None)
    turn_pass: Optional[str] = os.environ.get("YAP_TURN_PASS", None)
    ice_policy: str = os.environ.get("YAP_ICE_POLICY", "strict")  # 'strict'|'all'

    def validate(self) -> List[str]:
        """Validate configuration settings.
        
        :return: List of validation error messages
        """
        errs = []
        if not (self.ws_url or (self.neko_url and self.username and self.password)):
            errs.append("Provide YAP_WS or (NEKO_URL, NEKO_USER, NEKO_PASS).")
        if self.channels not in (1, 2):
            errs.append("YAP_AUDIO_CHANNELS must be 1 or 2.")
        if self.frame_ms not in (10, 20, 30, 40, 60):
            errs.append("YAP_FRAME_MS must be typical Opus frame (10/20/30/40/60).")
        # WebRTC expects 48k; we allow others but strongly recommend 48k.
        return errs


# ----------------------
# Logging
# ----------------------


# ----------------------
# Voices
# ----------------------
@dataclass
class Voice:
    speaker: str
    ref_audio: str
    ref_text: Optional[str] = None
    styles: List[str] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)


class VoiceManager:
    """Hot-reloadable voice registry backed by voices.json in YAP_VOICES_DIR."""

    def __init__(self, root: str, default_speaker: str, logger: logging.Logger) -> None:
        """Initialize voice manager with directory and default speaker.
        
        :param root: Directory containing voices.json
        :param default_speaker: Default speaker name to use
        :param logger: Logger instance
        """
        self.root = root
        self.path = os.path.join(self.root, "voices.json")
        self.default = default_speaker
        self._voices: Dict[str, Voice] = {}
        self._logger = logger
        self.reload()

    def reload(self) -> None:
        """Reload voice configurations from voices.json file."""
        os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(self.path):
            # seed a default placeholder if none
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "default": {
                            "ref_audio": os.path.join(self.root, "default.wav"),
                            "ref_text": "This is a default reference.",
                            "styles": ["calm"],
                            "params": {"rate": 1.0, "pitch": 0.0},
                        }
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        try:
            data = json.load(open(self.path, "r", encoding="utf-8"))
            voices: Dict[str, Voice] = {}
            for spk, meta in data.items():
                if not isinstance(meta, dict) or "ref_audio" not in meta:
                    continue
                voices[spk] = Voice(
                    speaker=spk,
                    ref_audio=meta["ref_audio"],
                    ref_text=meta.get("ref_text"),
                    styles=list(meta.get("styles", [])),
                    params=dict(meta.get("params", {})),
                )
            self._voices = voices
            self._logger.info("Voices reloaded (%d entries)", len(self._voices))
        except Exception as e:
            self._logger.error("Failed to load voices.json: %s", e)

    def all(self) -> Dict[str, Voice]:
        """Get all registered voices.
        
        :return: Dictionary mapping speaker names to Voice objects
        """
        return dict(self._voices)

    def get(self, speaker: Optional[str]) -> Voice:
        """Get voice configuration for a speaker.
        
        :param speaker: Speaker name, or None for default
        :return: Voice configuration object
        """
        if speaker and speaker in self._voices:
            return self._voices[speaker]
        if self.default in self._voices:
            return self._voices[self.default]
        # last resort
        return next(iter(self._voices.values()))

    def add_or_update(self, spk: str, ref_audio: str, ref_text: Optional[str], styles: List[str], params: Dict[str, float]) -> None:
        """Add or update a voice configuration.
        
        :param spk: Speaker name/ID
        :param ref_audio: Path to reference audio file
        :param ref_text: Reference text for the audio
        :param styles: List of voice styles
        :param params: Voice parameters (rate, pitch, etc.)
        """
        data = {}
        try:
            data = json.load(open(self.path, "r", encoding="utf-8"))
        except Exception:
            pass
        data[spk] = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "styles": styles,
            "params": params,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.reload()


# ----------------------
# Text segmentation (punct-aware)
# ----------------------
_SENT_SPLIT = re.compile(r"(.+?[\.!\?;:]+)(\s+|$)")
_SOFT_COMMA = re.compile(r",\s+")


def segment_text(s: str, max_chars: int) -> List[str]:
    """Segment text into chunks based on punctuation and character limits.
    
    :param s: Input text to segment
    :param max_chars: Maximum characters per segment
    :return: List of text segments
    """
    s = (s or "").strip()
    if not s:
        return []
    out: List[str] = []
    cur = ""
    for m in _SENT_SPLIT.finditer(s):
        sent = (m.group(1) or "").strip()
        cand = (cur + " " + sent).strip() if cur else sent
        if len(cand) > max_chars:
            if cur:
                out.append(cur)
                cur = sent
            else:
                # try soft comma split
                parts = _SOFT_COMMA.split(sent)
                buf = ""
                for p in parts:
                    pc = (buf + ", " + p).strip(", ").strip()
                    if len(pc) > max_chars and buf:
                        out.append(buf)
                        buf = p
                    else:
                        buf = pc
                if buf:
                    out.append(buf)
                cur = ""
        else:
            cur = cand
    if cur:
        out.append(cur)
    if not out:
        out = [s[:max_chars]]
    return out


class StreamAssembler:
    """Accumulates text across messages and emits stable chunks opportunistically."""
    def __init__(self, max_chars: int):
        """Initialize stream assembler with character limit.
        
        :param max_chars: Maximum characters per text chunk
        """
        self.buf = ""
        self.max_chars = max_chars

    def feed(self, s: str) -> List[str]:
        """Feed new text and return ready chunks.
        
        :param s: New text to process
        :return: List of ready text chunks, leaving tail in buffer
        """
        self.buf += (" " + s.strip()) if self.buf else s.strip()
        parts = segment_text(self.buf, self.max_chars)
        if not parts:
            return []
        if self.buf.strip().endswith((".", "!", "?", ";", ":")):
            self.buf = ""
            return parts
        # keep only the last fragment as tail; emit the rest
        if len(parts) == 1:
            self.buf = parts[0]
            return []
        *ready, last = parts
        self.buf = last
        return ready

    def flush(self) -> List[str]:
        """Flush remaining text in buffer as final chunks.
        
        :return: List of remaining text segments
        """
        parts = segment_text(self.buf, self.max_chars)
        self.buf = ""
        return parts


# ----------------------
# PCM queue + Audio track
# ----------------------
def float_to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float32 audio to int16 format.
    
    :param x: Float32 audio array in range [-1.0, 1.0]
    :return: Int16 audio array
    """
    y = np.clip(x, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


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
            # mixdown or simple channel duplication
            if self.ch == 1:
                # FIX: avoid overflow by using higher dtype for averaging, then cast
                pcm_i16 = pcm_i16.astype(np.int32, copy=False)
                pcm_i16 = (pcm_i16.mean(axis=1).astype(np.int16, copy=False))[:, None]
            else:
                pcm_i16 = np.repeat(pcm_i16, 2, axis=1)[:, :2]

        has_audio = np.any(pcm_i16 != 0)
        if has_audio:
            print(f"[DEBUG] PCMQueue: received {len(pcm_i16)} samples with audio data")

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
            # wait briefly for data; if underrun, pad with zeros
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
        has_audio = np.any(samples != 0)
        if has_audio:
            print(f"[DEBUG] YAPAudioTrack: pulling {len(samples)} samples with audio data (non-zero)")
        layout = "mono" if samples.shape[1] == 1 else "stereo"
        # Reshape for packed format: (N, C) -> (1, N*C) with interleaved channels
        packed_samples = samples.reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(packed_samples, format="s16", layout=layout)
        frame.sample_rate = self.q.sr
        frame.time_base = Fraction(1, self.q.sr)
        frame.pts = self._pts
        self._pts += samples.shape[0]
        return frame


# ----------------------
# F5-TTS wrapper
# ----------------------
class F5Synth:
    def __init__(self, logger: logging.Logger):
        """Initialize F5-TTS synthesizer.
        
        :param logger: Logger instance
        :raises: Exception if F5-TTS is not available
        """
        self._log = logger
        try:
            from f5_tts.api import F5TTS  # type: ignore
            from f5_tts.infer.utils_infer import remove_silence_for_generated_wav  # type: ignore
        except Exception as e:  # pragma: no cover
            self._log.error("F5-TTS not importable: %s\nInstall SWivid/F5-TTS.", e)
            raise
        self.f5 = F5TTS()
        self._trim = remove_silence_for_generated_wav

    def infer(self, ref_audio: str, ref_text: Optional[str], text: str) -> Tuple[np.ndarray, int]:
        """Generate speech using F5-TTS.
        
        :param ref_audio: Path to reference audio file
        :param ref_text: Reference text for the audio
        :param text: Text to synthesize
        :return: Tuple of (waveform, sample_rate) - mono float32 in [-1,1]
        """
        wav, sr, _meta = self.f5.infer(
            ref_audio, ref_text, text, remove_silence=True
        )
        w = np.array(wav, dtype=np.float32)
        if w.ndim == 1:
            w = w[:, None]
        return w, int(sr)


def apply_rate_pitch(wave_f32: np.ndarray, sr: int, rate: float = 1.0, pitch: float = 0.0) -> Tuple[np.ndarray, int]:
    """Apply rate and pitch transformations to audio.
    
    Applies naive prosody transforms using resampling:
    - rate: time-stretch via resampling (changes duration)
    - pitch (semitones): simple resample pitch-shift that also alters duration
    
    These are lightweight approximations to keep dependencies small.
    
    :param wave_f32: Input audio waveform as float32
    :param sr: Sample rate
    :param rate: Rate factor (1.0 = normal speed)
    :param pitch: Pitch shift in semitones (0.0 = no change)
    :return: Tuple of (transformed_audio, sample_rate)
    """
    out = wave_f32
    cur_sr = sr
    if abs(pitch) > 1e-3:
        # shift by semitones: factor = 2^(semitones/12)
        f = 2.0 ** (pitch / 12.0)
        out = _resample(out, cur_sr, int(round(cur_sr * f)))
        cur_sr = int(round(cur_sr * f))
    if abs(rate - 1.0) > 1e-3:
        out = _resample(out, cur_sr, int(round(cur_sr * rate)))
        cur_sr = int(round(cur_sr * rate))
    if cur_sr != sr:
        out = _resample(out, cur_sr, sr)
        cur_sr = sr
    return out, cur_sr


# ----------------------
# TTS Pipeline (segment → parallel synth → splicer → queue)
# ----------------------
class TTSPipeline:
    def __init__(
        self,
        voices: VoiceManager,
        tts: F5Synth,
        pcmq: PCMQueue,
        sr_out: int,
        ch_out: int,
        overlap_ms: int,
        parallel: int,
        logger: logging.Logger,
        max_chars: int,
    ) -> None:
        """Initialize TTS pipeline with voice manager and audio parameters.
        
        :param voices: Voice manager instance
        :param tts: TTS synthesizer instance
        :param pcmq: PCM queue for audio output
        :param sr_out: Output sample rate
        :param ch_out: Output channel count
        :param overlap_ms: Crossfade overlap in milliseconds
        :param parallel: Number of parallel TTS workers
        :param logger: Logger instance
        :param max_chars: Maximum characters per chunk
        """
        self.voices = voices
        self.tts = tts
        self.q = pcmq
        self.sr_out = sr_out
        self.ch_out = ch_out
        self.overlap = int(sr_out * overlap_ms / 1000)
        self._logger = logger
        self._executor = None
        self._parallel = max(1, parallel)
        self._epoch = 0  # cancel token
        self.max_chars = max_chars

        # splicer tail carry (last overlap samples)
        self._carry = np.zeros((0, ch_out), dtype=np.float32)
        self._lock = asyncio.Lock()

    def _ensure_executor(self):
        """Initialize thread pool executor for TTS workers if needed."""
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(
                max_workers=self._parallel, thread_name_prefix="tts-worker"
            )

    def cancel_all(self):
        """Cancel all active TTS generation and clear buffers."""
        self._epoch += 1
        self.q.clear()
        with contextlib.suppress(Exception):
            self._carry = np.zeros((0, self.ch_out), dtype=np.float32)

    async def speak_text(
        self,
        text: str,
        speaker: Optional[str],
        flush_end: bool = True,
        params_override: Optional[Dict[str, float]] = None,
    ) -> None:
        """Segment and synthesize text using parallel TTS workers.
        
        :param text: Text to synthesize
        :param speaker: Speaker/voice to use
        :param flush_end: Whether to flush final audio chunks
        :param params_override: Override voice parameters
        """
        chunks = segment_text(text, max_chars=self.max_chars)
        if not chunks:
            return
        await self.speak_chunks(
            chunks, speaker=speaker, flush_end=flush_end, params_override=params_override
        )

    async def speak_chunks(
        self,
        chunks: List[str],
        speaker: Optional[str],
        flush_end: bool,
        params_override: Optional[Dict[str, float]] = None,
    ) -> None:
        """Synthesize a list of text chunks using parallel TTS workers.
        
        :param chunks: List of text chunks to synthesize
        :param speaker: Speaker/voice to use
        :param flush_end: Whether to flush final audio chunks
        :param params_override: Override voice parameters
        """
        self._ensure_executor()
        epoch = self._epoch
        v = self.voices.get(speaker)
        params = {**v.params, **(params_override or {})}
        loop = asyncio.get_running_loop()

        def _work(txt: str) -> np.ndarray:
            """Worker function to synthesize single text chunk.
            
            :param txt: Text to synthesize
            :return: Audio waveform as float32 array
            """
            # 1) F5-TTS
            wave, sr_in = self.tts.infer(v.ref_audio, v.ref_text, txt)
            # small edge fades to avoid clicks
            wave = self._tiny_fades(wave, sr_in)
            # 2) style shaping (rate/pitch) – naive
            wave, sr_a = apply_rate_pitch(
                wave, sr_in, float(params.get("rate", 1.0)), float(params.get("pitch", 0.0))
            )
            # 3) resample to output sr
            if sr_a != self.sr_out:
                wave = _resample(wave, sr_a, self.sr_out)
            # ensure float32 (N,C)
            if wave.ndim == 1:
                wave = wave[:, None]
            return wave.astype(np.float32, copy=False)

        # Bounded submission window to avoid runaway memory and allow cancel
        max_inflight = self._parallel * 2
        sem = asyncio.Semaphore(max_inflight)
        pending: Dict[int, asyncio.Future] = {}
        next_idx = 0

        for i, txt in enumerate(chunks):
            if epoch != self._epoch:
                return
            await sem.acquire()
            fut = loop.run_in_executor(self._executor, _work, txt)
            pending[i] = fut
            fut.add_done_callback(lambda _f: sem.release())

            # drain ready results in order
            while next_idx in pending and pending[next_idx].done():
                if epoch != self._epoch:
                    return
                try:
                    wave_f32 = await pending.pop(next_idx)
                except Exception as e:
                    self._logger.error("TTS chunk failed: %s", e)
                    next_idx += 1
                    continue
                async with self._lock:
                    out = self._splice_with_carry(wave_f32)
                    pcm_data = float_to_int16(out)
                    self._logger.debug("Pushing %d samples to PCM queue (%.2fs)", len(pcm_data), len(pcm_data) / self.sr_out)
                    self.q.push(pcm_data)
                next_idx += 1

        # drain the rest
        while next_idx in pending:
            if epoch != self._epoch:
                return
            try:
                wave_f32 = await pending.pop(next_idx)
            except Exception as e:
                self._logger.error("TTS chunk failed: %s", e)
                next_idx += 1
                continue
            async with self._lock:
                out = self._splice_with_carry(wave_f32)
                pcm_data = float_to_int16(out)
                self._logger.debug("Pushing %d samples to PCM queue (%.2fs)", len(pcm_data), len(pcm_data) / self.sr_out)
                self.q.push(pcm_data)
            next_idx += 1

        if flush_end:
            async with self._lock:
                if self._carry.size:
                    pcm_data = float_to_int16(self._carry)
                    self._logger.debug("Flushing final %d samples to PCM queue (%.2fs)", len(pcm_data), len(pcm_data) / self.sr_out)
                    self.q.push(pcm_data)
                    self._carry = np.zeros((0, self.ch_out), dtype=np.float32)

    @staticmethod
    def _tiny_fades(cur: np.ndarray, sr: int, ms: float = 4.0) -> np.ndarray:
        """Apply tiny fade in/out to avoid clicks at boundaries."""
        if cur.ndim == 1:
            cur = cur[:, None]
        n = min(cur.shape[0], max(1, int(sr * ms / 1000)))
        if n <= 1:
            return cur
        t = np.linspace(0.0, np.pi / 2, n, endpoint=False, dtype=np.float32)
        fade_in = np.sin(t)[:, None]
        fade_out = np.cos(t)[:, None]
        cur[:n, :] *= fade_in
        cur[-n:, :] *= fade_out[::-1]
        return cur

    def _splice_with_carry(self, cur: np.ndarray) -> np.ndarray:
        """Crossfade splice current chunk with carry from previous chunk.
        
        :param cur: Current audio chunk to splice
        :return: Processed audio ready for output
        """
        # Ensure channel count
        if cur.ndim == 1:
            cur = cur[:, None]
        if cur.shape[1] != self.ch_out:
            if self.ch_out == 1:
                cur = cur.mean(axis=1, keepdims=True)
            else:
                cur = np.repeat(cur, 2, axis=1)[:, :2]
        N = self.overlap
        if N <= 0 or cur.shape[0] < N:
            if self._carry.shape[0] == 0:
                if cur.shape[0] <= N:
                    self._carry = np.concatenate([self._carry, cur], axis=0)
                    return np.zeros((0, self.ch_out), dtype=np.float32)

        if self._carry.shape[0] == 0:
            if cur.shape[0] > N:
                out = cur[:-N, :]
                self._carry = cur[-N:, :]
                return out
            else:
                self._carry = np.concatenate([self._carry, cur], axis=0)
                return np.zeros((0, self.ch_out), dtype=np.float32)

        head = cur[:N, :]
        rest = cur[N:, :]
        Lc = self._carry.shape[0]
        if Lc < N:
            pad = np.zeros((N - Lc, self.ch_out), dtype=np.float32)
            carry = np.concatenate([pad, self._carry], axis=0)
        else:
            carry = self._carry[-N:, :]

        # Equal-power ramps to avoid level dip at the join
        t = np.linspace(0.0, np.pi / 2, N, endpoint=False, dtype=np.float32)
        fade_in = np.sin(t)[:, None]
        fade_out = np.cos(t)[:, None]
        cross = carry * fade_out + head * fade_in

        out = np.concatenate([cross, rest], axis=0)

        if out.shape[0] > N:
            self._carry = out[-N:, :].copy()
            return out[:-N, :]
        else:
            self._carry = out
            return np.zeros((0, self.ch_out), dtype=np.float32)


# ----------------------
# Chat command parsing
# ----------------------
RE_BEGIN = re.compile(r"^/yap:begin\s*$", re.I)
RE_END = re.compile(r"^/yap:end\s*$", re.I)
RE_STOP = re.compile(r"^/yap:stop\s*$", re.I)
RE_ONESHOT = re.compile(r"^/yap\s+(.+)$", re.I)
RE_VOICE_SET = re.compile(r"^/yap:voice\s+set\b(.*)$", re.I)
RE_VOICE_ADD = re.compile(r"^/yap:voice\s+add\b(.*)$", re.I)
RE_VOICE_RELOAD = re.compile(r"^/yap:voice\s+reload\s*$", re.I)


def _parse_kv_flags(rest: str) -> Dict[str, str]:
    """Parse command line style key-value flags.
    
    :param rest: String containing --key value or --key=value pairs
    :return: Dictionary of parsed key-value pairs
    """
    out: Dict[str, str] = {}
    tokens = re.findall(
        r"""--([a-zA-Z0-9_-]+)\s*=\s*("[^"]+"|'[^']+'|[^\s]+)|--([a-zA-Z0-9_-]+)\s+("[^"]+"|'[^']+'|[^\s]+)""",
        rest.strip(),
    )
    for a, b, c, d in tokens:
        if a:
            key, val = a.strip().lower(), b.strip()
        else:
            key, val = c.strip().lower(), d.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        out[key] = val
    return out


# ----------------------
# Neko Yap app
# ----------------------
class YapApp:
    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        """Initialize YAP application with settings and logger.
        
        :param settings: Application configuration settings
        :param logger: Logger instance
        """
        self.settings = settings
        self.log = logger
        self.signaler: Optional[Signaler] = None
        self.pc: Optional[RTCPeerConnection] = None
        self.session_id: Optional[str] = None
        self.audio_track: Optional[YAPAudioTrack] = None
        self.pcmq = PCMQueue(settings.sr, settings.channels, settings.jitter_max_sec, logger)
        self.stream_mode = False
        self.stream_asm = StreamAssembler(max_chars=settings.max_chunk_chars)
        self.current_speaker: Optional[str] = settings.default_speaker
        self.current_params: Dict[str, float] = {}
        self.voices = VoiceManager(settings.voices_dir, settings.default_speaker, logger)
        self.tts = F5Synth(logger)
        self.pipeline = TTSPipeline(
            self.voices,
            self.tts,
            self.pcmq,
            settings.sr,
            settings.channels,
            settings.overlap_ms,
            settings.parallel,
            logger,
            max_chars=settings.max_chunk_chars,
        )
        self._shutdown = asyncio.Event()
        # Idle flush to emit tail when user pauses mid-sentence
        self._idle_flush_task: Optional[asyncio.Task] = None
        self._idle_ms = 700

    # ---- Connection (REST optional) ----
    async def _login_if_needed(self) -> str:
        """Login via REST API if needed and return WebSocket URL.
        
        :return: WebSocket URL for connection
        """
        if self.settings.ws_url:
            return self.settings.ws_url
        base = (self.settings.neko_url or "").rstrip("/")
        if not (base and self.settings.username and self.settings.password):
            raise RuntimeError("Need YAP_WS or (NEKO_URL, NEKO_USER, NEKO_PASS)")
        import requests

        r = requests.post(
            f"{base}/api/login",
            json={"username": self.settings.username, "password": self.settings.password},
            timeout=10,
        )
        r.raise_for_status()
        tok = r.json().get("token")
        if not tok:
            raise RuntimeError("Login ok but no token in response.")
        scheme = "wss" if base.startswith("https") else "ws"
        host = base.split("://", 1)[-1]
        ws = f"{scheme}://{host}/api/ws?token={tok}"
        self.log.info("REST login ok; WS=%s", f"{scheme}://{host}/api/ws?token=***")
        return ws

    async def run(self) -> None:
        """Main application run loop with signal handling and task coordination."""
        loop = asyncio.get_running_loop()

        def _sig(_signum, _frame=None):
            """Signal handler for graceful shutdown.
            
            :param _signum: Signal number
            :param _frame: Current stack frame (unused)
            """
            self.log.info("Signal received; shutting down…")
            self._shutdown.set()

        for s in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(Exception):
                loop.add_signal_handler(s, _sig, s)

        ws_url = await self._login_if_needed()
        self.signaler = Signaler(ws_url)
        await self.signaler.connect_with_backoff()

        # Request media (audio send on, video disabled)
        await self.signaler.send({"event": "signal/request", "payload": {"video": {"disabled": True}, "audio": {"disabled": False}}})

        # Wait for offer/provide and set up RTC
        offer = None
        ctrl_q = self.signaler.broker.topic_queue("control")
        while not offer:
            msg = await ctrl_q.get()
            if msg.get("event") in ("signal/offer", "signal/provide"):
                offer = msg

        await self._setup_rtc(offer)

        # consumers
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._consume_ice())
                tg.create_task(self._consume_system())
                tg.create_task(self._consume_chat())
                tg.create_task(self._shutdown_watcher())
        except* asyncio.CancelledError:
            pass
        finally:
            await self._cleanup()

    async def _setup_rtc(self, offer_msg: Dict[str, Any]) -> None:
        """Set up WebRTC peer connection with ICE servers and audio track.
        
        :param offer_msg: WebRTC offer message from signaling server
        """
        payload = offer_msg.get("payload", offer_msg)
        # ICE servers (strict or append env)
        ice_payload = (
            payload.get("ice")
            or payload.get("iceservers")
            or payload.get("iceServers")
            or payload.get("ice_servers")
            or []
        )
        ice_servers: List[RTCIceServer] = []
        for srv in ice_payload:
            if isinstance(srv, dict) and srv.get("urls"):
                ice_servers.append(
                    RTCIceServer(urls=srv["urls"], username=srv.get("username"), credential=srv.get("credential"))
                )
        if self.settings.ice_policy != "strict":
            ice_servers.append(RTCIceServer(urls=[self.settings.stun_url]))
            if self.settings.turn_url:
                ice_servers.append(
                    RTCIceServer(urls=[self.settings.turn_url], username=self.settings.turn_user, credential=self.settings.turn_pass)
                )

        cfg = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(cfg)
        self.pc = pc

        # Add our audio track
        self.audio_track = YAPAudioTrack(self.pcmq, self.settings.frame_ms)
        pc.addTrack(self.audio_track)

        @pc.on("icecandidate")
        async def _on_icand(cand: RTCIceCandidate):
            """Handle ICE candidate events from WebRTC peer connection.
            
            :param cand: ICE candidate from WebRTC
            """
            if cand:
                await self.signaler.send({
                    "event": "signal/candidate",
                    "payload": {"candidate": cand.candidate, "sdpMid": cand.sdpMid, "sdpMLineIndex": cand.sdpMLineIndex},
                })

        remote_sdp = payload.get("sdp")
        typ = payload.get("type", "offer")
        await pc.setRemoteDescription(RTCSessionDescription(sdp=remote_sdp, type=typ))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self.signaler.send({"event": "signal/answer", "payload": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}})
        self.log.info("RTC answer sent; audio track live.")

    async def _consume_ice(self) -> None:
        """Process incoming ICE candidates for WebRTC connection."""
        q = self.signaler.broker.topic_queue("ice")
        while not self._shutdown.is_set():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if msg.get("event") == "signal/candidate":
                pay = msg.get("payload") or {}
                cand_str = pay.get("candidate")
                if not cand_str or not self.pc:
                    continue
                if cand_str.startswith("candidate:"):
                    cand_str = cand_str.split(":", 1)[1]
                c = candidate_from_sdp(cand_str)
                c.sdpMid = pay.get("sdpMid")
                c.sdpMLineIndex = pay.get("sdpMLineIndex")
                with contextlib.suppress(Exception):
                    await self.pc.addIceCandidate(c)

    async def _consume_system(self) -> None:
        """Process system messages like session initialization."""
        q = self.signaler.broker.topic_queue("system")
        while not self._shutdown.is_set():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if msg.get("event") == "system/init":
                payload = msg.get("payload") or {}
                self.session_id = payload.get("session_id")

    def _schedule_idle_flush(self):
        """Schedule delayed flush of streaming text buffer."""
        if self._idle_flush_task and not self._idle_flush_task.done():
            self._idle_flush_task.cancel()

        async def _later():
            """Delayed task to flush streaming text buffer."""
            try:
                await asyncio.sleep(self._idle_ms / 1000)
                if self.stream_mode:
                    chunks = self.stream_asm.flush()
                    await self._speak_chunks(chunks, flush_end=False)
            except asyncio.CancelledError:
                pass

        self._idle_flush_task = asyncio.create_task(_later())

    async def _consume_chat(self) -> None:
        """Process chat messages and handle /yap commands."""
        q = self.signaler.broker.topic_queue("chat")
        while not self._shutdown.is_set():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # Extract text (plugin shapes)
            ev = msg.get("event", "")
            payload = msg.get("payload") or {}
            text = None
            if ev == "chat/message":
                content = payload.get("content")
                if isinstance(content, dict):
                    text = content.get("text")
                if not text:
                    text = payload.get("text") or payload.get("message")
            elif ev in ("send/broadcast", "send/unicast"):
                b = payload.get("body")
                if isinstance(b, str):
                    text = b
                elif isinstance(b, dict):
                    text = b.get("text") or b.get("message")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            # Commands
            if RE_BEGIN.match(text):
                self.stream_mode = True
                self.stream_asm = StreamAssembler(max_chars=self.settings.max_chunk_chars)
                await self._chat("stream: begin")
                continue
            if RE_END.match(text):
                if self.stream_mode:
                    chunks = self.stream_asm.flush()
                    await self._speak_chunks(chunks, flush_end=True)
                self.stream_mode = False
                await self._chat("stream: end")
                continue
            if RE_STOP.match(text):
                self.pipeline.cancel_all()
                await self._chat("stopped / queue flushed")
                continue
            if (m := RE_ONESHOT.match(text)):
                phrase = m.group(1).strip()
                if phrase:
                    await self._speak_chunks(segment_text(phrase, self.settings.max_chunk_chars), flush_end=True)
                continue
            if (m := RE_VOICE_RELOAD.match(text)):
                self.voices.reload()
                await self._chat("voices reloaded")
                continue
            if (m := RE_VOICE_ADD.match(text)):
                rest = m.group(1) or ""
                flags = _parse_kv_flags(rest)
                spk = flags.get("spk") or flags.get("speaker")
                ref = flags.get("ref") or flags.get("ref_audio")
                rtext = flags.get("ref-text") or flags.get("ref_text")
                styles = [s.strip() for s in (flags.get("styles", "") or flags.get("style", "")).split(",") if s.strip()]
                params = {}
                for k in ("rate", "pitch"):
                    if k in flags:
                        try:
                            params[k] = float(flags[k])
                        except Exception:
                            pass
                if not (spk and ref):
                    await self._chat('usage: /yap:voice add --spk <id> --ref <wav> [--ref-text "…"] [--styles "a,b"] [--rate 1.0] [--pitch 0.0]')
                else:
                    self.voices.add_or_update(spk, ref, rtext, styles, params)
                    await self._chat(f"voice '{spk}' added/updated")
                continue
            if (m := RE_VOICE_SET.match(text)):
                rest = m.group(1) or ""
                flags = _parse_kv_flags(rest)
                if "spk" in flags or "speaker" in flags:
                    self.current_speaker = flags.get("spk") or flags.get("speaker")
                # override live params
                if "rate" in flags:
                    with contextlib.suppress(Exception):
                        self.current_params["rate"] = float(flags["rate"])
                if "pitch" in flags:
                    with contextlib.suppress(Exception):
                        self.current_params["pitch"] = float(flags["pitch"])
                await self._chat(f"voice set: spk={self.current_speaker} params={self.current_params}")
                continue

            # Stream mode incremental text
            if self.stream_mode:
                ready = self.stream_asm.feed(text)
                if ready:
                    await self._speak_chunks(ready, flush_end=False)
                self._schedule_idle_flush()

    async def _speak_chunks(self, chunks: List[str], flush_end: bool) -> None:
        """Internal helper to speak text chunks with current voice settings.
        
        :param chunks: List of text chunks to synthesize
        :param flush_end: Whether to flush final audio chunks
        """
        if not chunks:
            return
        await self.pipeline.speak_chunks(
            chunks,
            speaker=self.current_speaker,
            flush_end=flush_end,
            params_override=self.current_params,
        )

    async def _chat(self, text: str) -> None:
        """Send a chat message through the signaling connection.
        
        :param text: Message text to send
        """
        try:
            await self.signaler.send({"event": "chat/message", "payload": {"text": f"[yap] {text}"}})
        except Exception:
            pass

    async def _shutdown_watcher(self) -> None:
        """Wait for shutdown signal and trigger cancellation."""
        await self._shutdown.wait()
        raise asyncio.CancelledError

    async def _cleanup(self) -> None:
        """Clean up resources before application shutdown."""
        self.log.info("Cleaning up…")
        with contextlib.suppress(Exception):
            if self.pc:
                await self.pc.close()
                self.pc = None
        with contextlib.suppress(Exception):
            if self.signaler:
                await self.signaler.close()
                self.signaler = None
        # Ensure worker threads are shutdown cleanly
        with contextlib.suppress(Exception):
            if getattr(self.pipeline, "_executor", None):
                self.pipeline._executor.shutdown(cancel_futures=True)
        self.log.info("Yap shutdown complete.")


# ----------------------
# CLI / Entry
# ----------------------
def parse_args(argv: Optional[List[str]] = None):
    """Parse command line arguments.
    
    :param argv: Argument list, or None to use sys.argv
    :return: Parsed arguments namespace
    """
    import argparse

    p = argparse.ArgumentParser("yap", description="Neko WebRTC TTS broadcaster (F5-TTS → aiortc)")
    p.add_argument("--ws", default=os.environ.get("YAP_WS", os.environ.get("NEKO_WS", "")), help="Direct WS: wss://host/api/ws?token=…")
    p.add_argument("--neko-url", default=os.environ.get("NEKO_URL", ""), help="REST base URL (https://host) if --ws not provided")
    p.add_argument("--username", default=os.environ.get("NEKO_USER", ""), help="REST username")
    p.add_argument("--password", default=os.environ.get("NEKO_PASS", ""), help="REST password")

    p.add_argument("--sr", type=int, default=int(os.environ.get("YAP_SR", "48000")), help="Output sample rate (default 48000)")
    p.add_argument("--channels", type=int, default=int(os.environ.get("YAP_AUDIO_CHANNELS", "1")), help="1 or 2 (default 1)")
    p.add_argument("--frame-ms", type=int, default=int(os.environ.get("YAP_FRAME_MS", "20")), help="Frame size ms (default 20)")
    p.add_argument("--parallel", type=int, default=int(os.environ.get("YAP_PARALLEL", "2")), help="Parallel TTS workers")
    p.add_argument("--chunk-sec", type=float, default=float(os.environ.get("YAP_CHUNK_TARGET_SEC", "3.0")), help="Target chunk seconds (hint for segmentation)")
    p.add_argument("--max-chars", type=int, default=int(os.environ.get("YAP_MAX_CHARS", "350")), help="Hard limit per chunk")
    p.add_argument("--overlap-ms", type=int, default=int(os.environ.get("YAP_OVERLAP_MS", "30")), help="Crossfade overlap")
    p.add_argument("--jitter-max", type=float, default=float(os.environ.get("YAP_JITTER_MAX_SEC", "6.0")), help="PCM jitter buffer cap (sec)")

    p.add_argument("--voices-dir", default=os.environ.get("YAP_VOICES_DIR", "./voices"), help="Voice registry directory")
    p.add_argument("--default-speaker", default=os.environ.get("YAP_SPK_DEFAULT", "default"), help="Default speaker id")

    p.add_argument("--loglevel", default=os.environ.get("YAP_LOGLEVEL", "INFO"), help="Log level")
    p.add_argument("--log-format", default=os.environ.get("YAP_LOG_FORMAT", "text"), choices=("text", "json"))
    p.add_argument("--healthcheck", action="store_true", help="Validate configuration and exit 0/1")

    return p.parse_args(argv)


def build_settings(args) -> Settings:
    """Build Settings object from parsed command line arguments.
    
    :param args: Parsed command line arguments
    :return: Settings object with configuration
    """
    s = Settings(
        ws_url=args.ws,
        neko_url=args.neko_url or None,
        username=args.username or None,
        password=args.password or None,
        sr=args.sr,
        channels=args.channels,
        frame_ms=args.frame_ms,
        parallel=args.parallel,
        chunk_target_sec=args.chunk_sec,
        max_chunk_chars=args.max_chars,
        overlap_ms=args.overlap_ms,
        jitter_max_sec=args.jitter_max,
        voices_dir=args.voices_dir,
        default_speaker=args.default_speaker,
        log_level=args.loglevel,
        log_format=args.log_format,
    )
    return s


async def main_async(argv: Optional[List[str]] = None) -> int:
    """Main async entry point for the application.
    
    :param argv: Command line arguments, or None to use sys.argv
    :return: Exit code (0 for success, non-zero for errors)
    """
    args = parse_args(argv)
    global settings  # only for CLI build_settings; not used in pipeline anymore
    settings = build_settings(args)
    logger = setup_logging(settings.log_level, settings.log_format, name="yap")

    if args.healthcheck:
        errs = settings.validate()
        if errs:
            for e in errs:
                print("ERROR:", e, file=sys.stderr)
            return 1
        # Optional: existence of voices.json is ensured by VoiceManager.reload()
        print("ok")
        return 0

    errs = settings.validate()
    if errs:
        for e in errs:
            logger.error(e)
        return 2

    app = YapApp(settings, logger)
    try:
        await app.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error("Fatal: %s", e, exc_info=True)
        return 3


def main() -> None:
    """Synchronous entry point that runs the async main function."""
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
