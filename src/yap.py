#!/usr/bin/env python3
# src/yap_refactored.py
"""
yap_refactored.py — TTS broadcaster using WebRTCNekoClient.

What this does
--------------
- Connects to Neko server using WebRTCNekoClient from neko_comms
- Provides TTS audio streaming via WebRTC outbound audio track
- Listens to chat for /yap commands and generates speech using F5-TTS
- Features streaming mode, voice management, and real-time audio broadcasting

Refactored from original yap.py to use modular neko_comms library.
"""

import asyncio
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import av
except ImportError as e:
    print("PyAV (av) is required. pip install av", file=sys.stderr)
    raise

from webrtc import MediaStreamTrack

from neko_comms import WebRTCNekoClient, PCMQueue, YAPAudioTrack
from utils import setup_logging


# ----------------------
# Resampling helpers
# ----------------------
_RESAMPLE_BACKEND = None
try:
    import torch
    import torchaudio

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

except ImportError:
    try:
        from scipy.signal import resample_poly

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

    except ImportError:
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
# Settings
# ----------------------
@dataclass
class Settings:
    # Connection (reuse neko_comms patterns)
    ws_url: str = field(default_factory=lambda: os.environ.get("YAP_WS", os.environ.get("NEKO_WS", "")))
    neko_url: Optional[str] = field(default_factory=lambda: os.environ.get("NEKO_URL", None))
    username: Optional[str] = field(default_factory=lambda: os.environ.get("NEKO_USER", None))
    password: Optional[str] = field(default_factory=lambda: os.environ.get("NEKO_PASS", None))

    # Audio parameters
    sr: int = int(os.environ.get("YAP_SR", "48000"))
    channels: int = int(os.environ.get("YAP_AUDIO_CHANNELS", "1"))
    frame_ms: int = int(os.environ.get("YAP_FRAME_MS", "20"))
    jitter_max_sec: float = float(os.environ.get("YAP_JITTER_MAX_SEC", "6.0"))

    # TTS pipeline
    parallel: int = int(os.environ.get("YAP_PARALLEL", "2"))
    chunk_target_sec: float = float(os.environ.get("YAP_CHUNK_TARGET_SEC", "3.0"))
    max_chunk_chars: int = int(os.environ.get("YAP_MAX_CHARS", "350"))
    overlap_ms: int = int(os.environ.get("YAP_OVERLAP_MS", "30"))

    # Voice configuration
    voices_dir: str = os.environ.get("YAP_VOICES_DIR", "./voices")
    default_speaker: str = os.environ.get("YAP_SPK_DEFAULT", "default")

    # Debug/Testing
    save_audio_dir: Optional[str] = field(default_factory=lambda: os.environ.get("YAP_SAVE_AUDIO_DIR", None))

    # Logging
    log_level: str = os.environ.get("YAP_LOGLEVEL", "INFO")
    log_format: str = os.environ.get("YAP_LOG_FORMAT", "text")
    log_file: Optional[str] = os.environ.get("YAP_LOGFILE")

    # Metrics / ICE
    metrics_port: int = int(os.environ.get("YAP_METRICS_PORT", os.environ.get("PORT", "0") or "0"))
    stun_url: str = os.environ.get("YAP_STUN_URL", "stun:stun.l.google.com:19302")
    turn_url: Optional[str] = os.environ.get("YAP_TURN_URL")
    turn_user: Optional[str] = os.environ.get("YAP_TURN_USER")
    turn_pass: Optional[str] = os.environ.get("YAP_TURN_PASS")
    ice_policy: str = os.environ.get("YAP_ICE_POLICY", "strict")

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
        if self.ice_policy not in ("strict", "all"):
            errs.append("YAP_ICE_POLICY must be 'strict' or 'all'.")
        return errs


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
    """Hot-reloadable voice registry backed by voices.json."""

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
            # Create default voices.json
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

    def get(self, speaker: Optional[str]) -> Voice:
        """Get voice configuration for a speaker.

        :param speaker: Speaker name, or None for default
        :return: Voice configuration object
        """
        if speaker and speaker in self._voices:
            return self._voices[speaker]
        if self.default in self._voices:
            return self._voices[self.default]
        return next(iter(self._voices.values()))

    def add_or_update(self, spk: str, ref_audio: str, ref_text: Optional[str],
                     styles: List[str], params: Dict[str, float]) -> None:
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
# Text segmentation
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
                # Try soft comma split
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
        # Keep only the last fragment as tail; emit the rest
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


# PCMQueue and YAPAudioTrack are imported from neko_comms.audio


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
            from f5_tts.api import F5TTS
        except ImportError as e:
            self._log.error("F5-TTS not importable: %s\nInstall SWivid/F5-TTS.", e)
            raise
        self.f5 = F5TTS()
        self._lock = threading.Lock()

    def infer(self, ref_audio: str, ref_text: Optional[str], text: str) -> Tuple[np.ndarray, int]:
        """Generate speech using F5-TTS.

        :param ref_audio: Path to reference audio file
        :param ref_text: Reference text for the audio
        :param text: Text to synthesize
        :return: Tuple of (waveform, sample_rate) - mono float32 in [-1,1]
        """
        with self._lock:
            wav, sr, _meta = self.f5.infer(
                ref_audio, ref_text, text, remove_silence=True
            )
        w = np.array(wav, dtype=np.float32)
        if w.ndim == 1:
            w = w[:, None]
        return w, int(sr)


def apply_rate_pitch(wave_f32: np.ndarray, sr: int, rate: float = 1.0, pitch: float = 0.0) -> Tuple[np.ndarray, int]:
    """Apply rate and pitch transformations to audio.

    :param wave_f32: Input audio waveform as float32
    :param sr: Sample rate
    :param rate: Rate factor (1.0 = normal speed)
    :param pitch: Pitch shift in semitones (0.0 = no change)
    :return: Tuple of (transformed_audio, sample_rate)
    """
    out = wave_f32
    cur_sr = sr
    if abs(pitch) > 1e-3:
        # Shift by semitones: factor = 2^(semitones/12)
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
# TTS Pipeline
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
        save_audio_dir: Optional[str] = None,
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
        :param save_audio_dir: Optional directory to save audio chunks for debugging
        """
        self.voices = voices
        self.tts = tts
        self.q = pcmq
        self.sr_out = sr_out
        self.ch_out = ch_out
        self.overlap = int(sr_out * overlap_ms / 1000)
        self._logger = logger
        self._executor: Optional[ThreadPoolExecutor] = None
        self._parallel = max(1, parallel)
        self._epoch = 0  # Cancel token
        self.max_chars = max_chars
        self.save_audio_dir = save_audio_dir
        self._chunk_counter = 0

        # Create save directory if specified
        if self.save_audio_dir:
            os.makedirs(self.save_audio_dir, exist_ok=True)
            self._logger.info("Audio chunks will be saved to: %s", self.save_audio_dir)

        # Splicer tail carry (last overlap samples)
        self._carry = np.zeros((0, ch_out), dtype=np.float32)
        self._lock = asyncio.Lock()

    def _ensure_executor(self):
        """Initialize thread pool executor for TTS workers if needed."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._parallel, thread_name_prefix="tts-worker"
            )

    def cancel_all(self):
        """Cancel all active TTS generation and clear buffers."""
        self._epoch += 1
        self.q.clear()
        self._carry = np.zeros((0, self.ch_out), dtype=np.float32)

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
        if not chunks:
            return

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
            # Generate with F5-TTS
            wave, sr_in = self.tts.infer(v.ref_audio, v.ref_text, txt)
            # Apply rate/pitch transformations
            wave, sr_a = apply_rate_pitch(
                wave, sr_in, float(params.get("rate", 1.0)), float(params.get("pitch", 0.0))
            )
            # Resample to output sample rate
            if sr_a != self.sr_out:
                wave = _resample(wave, sr_a, self.sr_out)
            # Ensure float32 (N,C)
            if wave.ndim == 1:
                wave = wave[:, None]
            return wave.astype(np.float32, copy=False)

        # Process chunks with bounded parallelism
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

            # Drain ready results in order
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
                    self.q.push(pcm_data)
                    # Save audio chunk if debug directory is configured
                    if self.save_audio_dir and len(pcm_data) > 0:
                        self._save_audio_chunk(pcm_data, next_idx)
                next_idx += 1

        # Drain remaining results
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
                self.q.push(pcm_data)
                # Save audio chunk if debug directory is configured
                if self.save_audio_dir and len(pcm_data) > 0:
                    self._save_audio_chunk(pcm_data, next_idx)
            next_idx += 1

        if flush_end:
            async with self._lock:
                if self._carry.size:
                    pcm_data = float_to_int16(self._carry)
                    self.q.push(pcm_data)
                    self._carry = np.zeros((0, self.ch_out), dtype=np.float32)

    def _save_audio_chunk(self, pcm_data: np.ndarray, chunk_idx: int) -> None:
        """Save audio chunk to disk for debugging.

        :param pcm_data: PCM audio data as int16 array
        :param chunk_idx: Chunk index for filename
        """
        try:
            import wave
            timestamp = time.time()
            filename = os.path.join(
                self.save_audio_dir,
                f"chunk_{timestamp:.0f}_{chunk_idx:04d}.wav"
            )
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.ch_out)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.sr_out)
                wf.writeframes(pcm_data.tobytes())
            self._logger.debug("Saved audio chunk to: %s", filename)
        except Exception as e:
            self._logger.warning("Failed to save audio chunk: %s", e)

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

        # Equal-power crossfade
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
# Chat command patterns
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
# YAP application
# ----------------------
class YapApp:
    """TTS broadcasting application using WebRTCNekoClient."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        """Initialize YAP application with settings and logger.

        :param settings: Application configuration settings
        :param logger: Logger instance
        """
        self.settings = settings
        self.log = logger
        self.running = True

        # Configure optional ICE servers when policy allows it
        ice_servers: List[Dict[str, Any]] = []
        if settings.ice_policy != "strict":
            if settings.stun_url:
                ice_servers.append({"urls": [settings.stun_url]})
            if settings.turn_url:
                entry: Dict[str, Any] = {"urls": [settings.turn_url]}
                if settings.turn_user:
                    entry["username"] = settings.turn_user
                if settings.turn_pass:
                    entry["credential"] = settings.turn_pass
                ice_servers.append(entry)

        client_kwargs: Dict[str, Any] = {
            "auto_host": False,
            "request_media": False,
            "ice_policy": settings.ice_policy,
        }
        if ice_servers:
            client_kwargs["ice_servers"] = ice_servers

        # Initialize neko client (no media since we're audio-only)
        self.client = WebRTCNekoClient(
            ws_url=self._get_ws_url(),
            **client_kwargs,
        )

        # TTS components
        self.pcmq = PCMQueue(settings.sr, settings.channels, settings.jitter_max_sec, logger)
        self.audio_track = YAPAudioTrack(self.pcmq, settings.frame_ms)
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
            save_audio_dir=settings.save_audio_dir,
        )

        # State
        self.stream_mode = False
        self.stream_asm = StreamAssembler(max_chars=settings.max_chunk_chars)
        self.current_speaker: Optional[str] = settings.default_speaker
        self.current_params: Dict[str, float] = {}
        self._idle_flush_task: Optional[asyncio.Task] = None
        self._idle_ms = max(250, int(self.settings.chunk_target_sec * 500))

    def _get_ws_url(self) -> str:
        """Get WebSocket URL from settings, handling REST login if needed.

        :return: WebSocket URL for connection
        """
        if self.settings.ws_url:
            return self.settings.ws_url

        # Use REST login
        if not (self.settings.neko_url and self.settings.username and self.settings.password):
            raise RuntimeError("Need YAP_WS or (NEKO_URL, NEKO_USER, NEKO_PASS)")

        import requests

        base = self.settings.neko_url.rstrip("/")
        r = requests.post(
            f"{base}/api/login",
            json={"username": self.settings.username, "password": self.settings.password},
            timeout=10,
        )
        r.raise_for_status()
        token = r.json().get("token")
        if not token:
            raise RuntimeError("Login ok but no token in response.")

        scheme = "wss" if base.startswith("https") else "ws"
        host = base.split("://", 1)[-1]
        return f"{scheme}://{host}/api/ws?token={token}"

    async def run(self) -> None:
        """Main application run loop."""
        self.log.info("Starting YAP TTS broadcaster")

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown)
            except Exception:
                pass

        try:
            # Connect to Neko server
            await self.client.connect()
            self.log.info("Connected to Neko server")

            # Add our audio track to the peer connection
            self.client.add_outbound_audio_track(self.audio_track)
            self.log.info("Audio track added to WebRTC connection")

            # Request audio-only media from the server
            await self.client.request_media(audio=True, video=False)
            self.log.info("Requested audio-only media session")

            # Start chat listener
            await asyncio.gather(
                self._chat_listener(),
                return_exceptions=True
            )

        except KeyboardInterrupt:
            self.log.info("Interrupted by user")
        except Exception as e:
            self.log.error("YAP error: %s", e)
        finally:
            await self.client.disconnect()

    def _shutdown(self):
        """Signal handler for graceful shutdown."""
        self.log.info("Shutdown signal received")
        self.running = False

    async def _chat_listener(self) -> None:
        """Listen for chat messages and handle /yap commands."""
        while self.running and self.client.is_connected():
            try:
                # Subscribe to chat messages
                msg = await self.client.subscribe_topic("chat", timeout=1.0)
                await self._process_chat_message(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.log.error("Chat listener error: %s", e)
                break

    async def _process_chat_message(self, msg: Dict[str, Any]) -> None:
        """Process a chat message for /yap commands.

        :param msg: Chat message dictionary
        """
        # Extract text from message
        text = self._extract_text(msg)
        if not text:
            return

        # Handle commands
        if RE_BEGIN.match(text):
            self.stream_mode = True
            self.stream_asm = StreamAssembler(max_chars=self.settings.max_chunk_chars)
            await self._send_chat("stream: begin")
            return

        if RE_END.match(text):
            if self.stream_mode:
                chunks = self.stream_asm.flush()
                await self._speak_chunks(chunks, flush_end=True)
            self.stream_mode = False
            await self._send_chat("stream: end")
            return

        if RE_STOP.match(text):
            self.pipeline.cancel_all()
            await self._send_chat("stopped / queue flushed")
            return

        if (m := RE_ONESHOT.match(text)):
            phrase = m.group(1).strip()
            if phrase:
                await self._speak_chunks(segment_text(phrase, self.settings.max_chunk_chars), flush_end=True)
            return

        if RE_VOICE_RELOAD.match(text):
            self.voices.reload()
            await self._send_chat("voices reloaded")
            return

        if (m := RE_VOICE_ADD.match(text)):
            await self._handle_voice_add(m.group(1) or "")
            return

        if (m := RE_VOICE_SET.match(text)):
            await self._handle_voice_set(m.group(1) or "")
            return

        # Stream mode incremental text
        if self.stream_mode:
            ready = self.stream_asm.feed(text)
            if ready:
                await self._speak_chunks(ready, flush_end=False)
            self._schedule_idle_flush()

    def _extract_text(self, msg: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a chat message.

        :param msg: Message dictionary
        :return: Text content or None
        """
        event = msg.get("event", "")
        payload = msg.get("payload", {})

        if event == "chat/message":
            content = payload.get("content")
            if isinstance(content, dict):
                return content.get("text")
            return payload.get("text") or payload.get("message")
        elif event in ("send/broadcast", "send/unicast"):
            body = payload.get("body")
            if isinstance(body, str):
                return body
            elif isinstance(body, dict):
                return body.get("text") or body.get("message")

        return None

    async def _handle_voice_add(self, rest: str) -> None:
        """Handle /yap:voice add command.

        :param rest: Command arguments
        """
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
            await self._send_chat('usage: /yap:voice add --spk <id> --ref <wav> [--ref-text "…"] [--styles "a,b"] [--rate 1.0] [--pitch 0.0]')
        else:
            self.voices.add_or_update(spk, ref, rtext, styles, params)
            await self._send_chat(f"voice '{spk}' added/updated")

    async def _handle_voice_set(self, rest: str) -> None:
        """Handle /yap:voice set command.

        :param rest: Command arguments
        """
        flags = _parse_kv_flags(rest)
        if "spk" in flags or "speaker" in flags:
            self.current_speaker = flags.get("spk") or flags.get("speaker")

        # Override live params
        if "rate" in flags:
            try:
                self.current_params["rate"] = float(flags["rate"])
            except Exception:
                pass
        if "pitch" in flags:
            try:
                self.current_params["pitch"] = float(flags["pitch"])
            except Exception:
                pass

        await self._send_chat(f"voice set: spk={self.current_speaker} params={self.current_params}")

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

    async def _speak_chunks(self, chunks: List[str], flush_end: bool) -> None:
        """Speak text chunks with current voice settings.

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

    async def _send_chat(self, text: str) -> None:
        """Send a chat message through the client connection.

        :param text: Message text to send
        """
        try:
            await self.client.send_event("chat/message", {"text": f"[yap] {text}"})
        except Exception as exc:
            self.log.debug("Failed to send chat message: %s", exc)


# ----------------------
# Entry point
# ----------------------
# setup_logging is imported from utils


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Neko TTS broadcaster")
    parser.add_argument("--neko-url", help="Neko server base URL")
    parser.add_argument("--username", help="Neko username")
    parser.add_argument("--password", help="Neko password")
    parser.add_argument("--ws", help="Direct WebSocket URL")
    parser.add_argument("--voices-dir", help="Voice directory")
    parser.add_argument("--sr", type=int, help="Output sample rate")
    parser.add_argument("--channels", type=int, help="Audio channels (1 or 2)")
    parser.add_argument("--frame-ms", type=int, help="Audio frame size in milliseconds")
    parser.add_argument("--parallel", type=int, help="Parallel TTS workers")
    parser.add_argument("--chunk-sec", type=float, help="Target chunk duration in seconds")
    parser.add_argument("--max-chars", type=int, help="Maximum characters per chunk")
    parser.add_argument("--overlap-ms", type=int, help="Crossfade overlap in milliseconds")
    parser.add_argument("--jitter-max", type=float, help="PCM jitter buffer cap in seconds")
    parser.add_argument("--default-speaker", help="Default speaker identifier")
    parser.add_argument("--loglevel", help="Logging level")
    parser.add_argument("--log-format", choices=("text", "json"), help="Logging format")
    parser.add_argument("--log-file", help="Optional log file path")
    parser.add_argument("--stun-url", help="Additional STUN server URL")
    parser.add_argument("--turn-url", help="TURN server URL")
    parser.add_argument("--turn-user", help="TURN username")
    parser.add_argument("--turn-pass", help="TURN password")
    parser.add_argument("--ice-policy", choices=("strict", "all"), help="ICE transport policy")
    parser.add_argument("--metrics-port", type=int, help="Metrics server port")
    parser.add_argument("--healthcheck", action="store_true", help="Validate configuration and exit")

    args = parser.parse_args()

    # Build settings from args and environment
    settings = Settings()
    if args.neko_url:
        settings.neko_url = args.neko_url
    if args.username:
        settings.username = args.username
    if args.password:
        settings.password = args.password
    if args.ws:
        settings.ws_url = args.ws
    if args.voices_dir:
        settings.voices_dir = args.voices_dir
    if args.sr is not None:
        settings.sr = args.sr
    if args.channels is not None:
        settings.channels = args.channels
    if args.frame_ms is not None:
        settings.frame_ms = args.frame_ms
    if args.parallel is not None:
        settings.parallel = args.parallel
    if args.chunk_sec is not None:
        settings.chunk_target_sec = args.chunk_sec
    if args.max_chars is not None:
        settings.max_chunk_chars = args.max_chars
    if args.overlap_ms is not None:
        settings.overlap_ms = args.overlap_ms
    if args.jitter_max is not None:
        settings.jitter_max_sec = args.jitter_max
    if args.default_speaker:
        settings.default_speaker = args.default_speaker
    if args.loglevel:
        settings.log_level = args.loglevel
    if args.log_format:
        settings.log_format = args.log_format
    if args.log_file:
        settings.log_file = args.log_file
    if args.stun_url:
        settings.stun_url = args.stun_url
    if args.turn_url:
        settings.turn_url = args.turn_url
    if args.turn_user:
        settings.turn_user = args.turn_user
    if args.turn_pass:
        settings.turn_pass = args.turn_pass
    if args.ice_policy:
        settings.ice_policy = args.ice_policy
    if args.metrics_port is not None:
        settings.metrics_port = args.metrics_port

    # Validate settings
    errs = settings.validate()
    if errs:
        for err in errs:
            print(f"Error: {err}", file=sys.stderr)
        if args.healthcheck:
            raise SystemExit(1)
        raise SystemExit(1)

    if args.healthcheck:
        print("yap healthcheck ok")
        return

    # Set up logging and run app
    setup_logging(settings.log_level, settings.log_format, settings.log_file)
    logger = logging.getLogger("yap")
    app = YapApp(settings, logger)

    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("YAP failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())