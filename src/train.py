#!/usr/bin/env python3
"""
train.py — Finetune ShowUI/Qwen2VL on captured Neko episodes (MDS).

Overview
--------
This script consumes training data produced by ``src/capture.py`` (MosaicML
Streaming, aka MDS), turns each annotated action into a supervised example,
and fine‑tunes the ShowUI/Qwen2VL model used by ``src/agent.py``.

12‑Factor compliance
--------------------
- All configuration comes from environment variables (overridable by CLI).
- No side effects at import; main logic lives under ``main()``.
- Structured logging to stdout; optional JSON format via ``TRAIN_LOG_FORMAT``.
- Heavy libraries (``torch``, ``transformers``) are imported lazily.

Data assumptions
----------------
Each sample in the MDS dataset is an "episode" written by ``capture.py`` with
these columns: ``payload`` (zip bytes), ``task`` (str), and metadata. Inside
the zip, we expect:
- ``meta.json``
- ``frames/NNNNNN.jpg``
- ``frames.ndjson`` with entries ``{"i": int, "ts": float, "file": str}``
- ``actions.ndjson`` with entries ``{"ts": float, "action": {...}, "raw"?: str}``

For supervised learning, we create one training example per action:
- Input: chat template identical to the agent: system prompt + Task + image.
- Target: the action JSON string the agent is expected to emit.

We tokenize the concatenation of prompt + target and mask the prompt tokens to
compute loss only on the target portion (standard causal LM supervision).

Environment variables
---------------------
# Data paths
TRAIN_LOCAL                   # Local MDS directory (default: ./data/mds)
TRAIN_REMOTE                  # Optional remote MDS (e.g., s3://bucket/prefix)
TRAIN_CACHE                   # Local cache directory for StreamingDataset (default: ./data/cache)
TRAIN_OUTPUT                  # Output directory for checkpoints (default: ./checkpoints)

# Model configuration
REPO_ID                       # Model repository (default: showlab/ShowUI-2B)
SIZE_SHORTEST_EDGE            # Image preprocessing (default: 224)
SIZE_LONGEST_EDGE             # Image preprocessing (default: 1344)

# Training parameters
TRAIN_EPOCHS                  # Number of epochs (default: 1)
TRAIN_BATCH                   # Global batch size (default: 1)
TRAIN_ACCUM                   # Gradient accumulation steps (default: 1)
TRAIN_LR                      # Learning rate (default: 5e-6)
TRAIN_WD                      # Weight decay (default: 0.0)
TRAIN_MAX_STEPS               # Optional hard step cap (default: 0 = disabled)
TRAIN_MAX_SAMPLES_PER_EPOCH   # Optional cap for debug (default: 0 = disabled)
TRAIN_HISTORY_STEPS           # Include up to N previous actions in history text (default: 0)

# General
SEED                          # RNG seed (default: 1337)
TRAIN_LOGLEVEL                # DEBUG|INFO|WARNING|ERROR (default: INFO)
TRAIN_LOG_FORMAT              # text|json (default: text)

Typical use
-----------
# Basic training on local MDS data
uv run src/train.py --local ./data/mds --output ./checkpoints

# Train with remote S3 data source
uv run src/train.py --remote s3://bucket/training-data --cache ./data/cache \
    --output ./checkpoints/model-v2

# Fine-tune with specific hyperparameters
TRAIN_EPOCHS=3 TRAIN_BATCH=4 TRAIN_LR=1e-5 uv run src/train.py \
    --local ./data/mds --output ./checkpoints/fine-tuned

# Include action history in training examples
uv run src/train.py --local ./data/mds --history 3 --output ./checkpoints

# Using just command (preferred)
just train        # Train with default settings from .env
just uv-train     # Train with UV explicitly

# Quick debug run with limited samples
TRAIN_MAX_SAMPLES_PER_EPOCH=100 uv run src/train.py --local ./data/mds \
    --epochs 1 --output ./checkpoints/test

"""

from __future__ import annotations

# stdlib
import os
import io
import sys
import json
import math
import time
import signal
import zipfile
import logging
import random
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# --- Logging utilities (no heavy imports here) ---------------------------------

class JsonFormatter(logging.Formatter):
    """Format log records as JSON.

    :param record: Log record instance
    :returns: JSON string
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        data = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(level: str = "INFO", fmt: str = "text") -> logging.Logger:
    """Configure root logger with text or JSON formatting.

    :param level: Log level name
    :param fmt: 'text' or 'json'
    :returns: Module logger
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    if fmt.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s",
            datefmt="%H:%M:%S",
        )
    h = logging.StreamHandler()
    h.setFormatter(formatter)
    root.addHandler(h)
    root.setLevel(level.upper())
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)
    return logging.getLogger("train")


# --- Config --------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Configuration model for finetuning.

    All fields can be supplied via environment variables or CLI; CLI wins.
    """

    local: str
    remote: str
    cache: str
    output: str
    loglevel: str
    log_format: str
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    max_steps: int
    max_samples_per_epoch: int
    history_steps: int
    seed: int
    repo_id: str
    size_shortest_edge: int
    size_longest_edge: int

    @classmethod
    def from_env(cls) -> "TrainConfig":
        """Load configuration from environment variables.

        :returns: Populated TrainConfig instance
        """
        return cls(
            local=os.environ.get("TRAIN_LOCAL", os.environ.get("CAPTURE_OUT", "./data/mds")),
            remote=os.environ.get("TRAIN_REMOTE", os.environ.get("CAPTURE_REMOTE", "")),
            cache=os.environ.get("TRAIN_CACHE", "./data/cache"),
            output=os.environ.get("TRAIN_OUTPUT", "./checkpoints"),
            loglevel=os.environ.get("TRAIN_LOGLEVEL", "INFO"),
            log_format=os.environ.get("TRAIN_LOG_FORMAT", "text"),
            epochs=int(os.environ.get("TRAIN_EPOCHS", "1")),
            batch_size=int(os.environ.get("TRAIN_BATCH", "1")),
            grad_accum=int(os.environ.get("TRAIN_ACCUM", "1")),
            lr=float(os.environ.get("TRAIN_LR", "5e-6")),
            weight_decay=float(os.environ.get("TRAIN_WD", "0.0")),
            max_steps=int(os.environ.get("TRAIN_MAX_STEPS", "0")),
            max_samples_per_epoch=int(os.environ.get("TRAIN_MAX_SAMPLES_PER_EPOCH", "0")),
            history_steps=int(os.environ.get("TRAIN_HISTORY_STEPS", "0")),
            seed=int(os.environ.get("SEED", "1337")),
            repo_id=os.environ.get("REPO_ID", "showlab/ShowUI-2B"),
            size_shortest_edge=int(os.environ.get("SIZE_SHORTEST_EDGE", "224")),
            size_longest_edge=int(os.environ.get("SIZE_LONGEST_EDGE", "1344")),
        )

    def validate(self) -> List[str]:
        """Validate config values and return a list of error strings.

        :returns: List of validation errors (empty when valid)
        """
        errs: List[str] = []
        if self.batch_size <= 0:
            errs.append("TRAIN_BATCH must be > 0")
        if self.grad_accum <= 0:
            errs.append("TRAIN_ACCUM must be > 0")
        if self.epochs <= 0:
            errs.append("TRAIN_EPOCHS must be > 0")
        if self.lr <= 0:
            errs.append("TRAIN_LR must be > 0")
        return errs


# --- Small helpers --------------------------------------------------------------


def _load_episode_zip(payload: bytes) -> zipfile.ZipFile:
    """Return an in-memory ZipFile for the episode payload bytes.

    :param payload: Episode ZIP as raw bytes
    :returns: Opened ``ZipFile`` ready for reads
    :raises zipfile.BadZipFile: If bytes are not a valid ZIP archive
    """
    bio = io.BytesIO(payload)
    return zipfile.ZipFile(bio, mode="r")


def _read_lines(zf: zipfile.ZipFile, name: str) -> List[str]:
    """Read a text file from zip and return decoded lines.

    :param zf: Episode zipfile
    :param name: File name inside the zip
    :returns: List of lines decoded as UTF-8 without trailing newlines
    """
    try:
        with zf.open(name, "r") as f:
            return [ln.decode("utf-8").rstrip("\n") for ln in f]
    except KeyError:
        return []


def _nearest_frame_index(frames: List[Dict[str, Any]], ts: float) -> Optional[int]:
    """Find the nearest frame index at or before ``ts``.

    :param frames: List of frame index dicts from frames.ndjson
    :param ts: Action timestamp
    :returns: Index ``i`` for frame, or ``None`` if list is empty
    """
    if not frames:
        return None
    # frames are in chronological order; find rightmost <= ts
    best_i = frames[0]["i"]
    best_ts = frames[0].get("ts", 0.0)
    for rec in frames:
        rts = float(rec.get("ts", 0.0))
        if rts <= ts and rts >= best_ts:
            best_ts = rts
            best_i = int(rec.get("i", best_i))
    return best_i


def _open_image_from_zip(zf: zipfile.ZipFile, path: str):
    """Open a PIL image from a path inside the episode zip.

    Heavy imports are local to keep module import light.
    """
    from PIL import Image

    with zf.open(path, "r") as f:
        img = Image.open(io.BytesIO(f.read()))
        return img.convert("RGB")


# Match the agent's system prompt (kept in sync semantically; light duplication).
_NAV_SYSTEM = (
    "You are an assistant trained to navigate the {APP} screen. "
    "Given a task instruction, a screen observation, and an action history sequence, "
    "output the next action and wait for the next observation. "
    "Here is the action space:\n{ACTION_SPACE}\n"
    "Format the action as a dictionary with the following keys:\n"
    "{{'action': 'ACTION_TYPE', 'value': ..., 'position': ...}}\n"
    "If value or position is not applicable, set as None. "
    "Position might be [[x1,y1],[x2,y2]] for range actions. "
    "Do NOT output extra keys or commentary."
)

_ACTION_SPACE_DESC = {
    "web": (
        "1. CLICK: Click an element, value=None, position=[x, y].\n"
        "2. INPUT: Type a string into an element, value=string, position=[x, y].\n"
        "3. SELECT: Select a value for an element, value=None, position=[x, y].\n"
        "4. HOVER: Hover on an element, value=None, position=[x, y].\n"
        "5. ANSWER: Answer a question, value=string, position=None.\n"
        "6. ENTER: Enter, value=None, position=None.\n"
        "7. SCROLL: Scroll the screen, value=direction (e.g. \"down\"), position=None.\n"
        "8. SELECT_TEXT: Select text, value=None, position=[[x1, y1], [x2, y2]].\n"
        "9. COPY: Copy text, value=string, position=None.\n"
    ),
    "phone": (
        "1. INPUT: Type a string into an element, value=string, position=[x, y].\n"
        "2. SWIPE: Swipe the screen, value=None, position=[[x1, y1], [x2, y2]].\n"
        "3. TAP: Tap on an element, value=None, position=[x, y].\n"
        "4. ANSWER: Answer a question, value=string, position=None.\n"
        "5. ENTER: Enter, value=None, position=None.\n"
    ),
}


def _build_messages(task: str, img, history: List[Dict[str, Any]], *, mode: str = "web") -> List[Dict[str, Any]]:
    """Build chat messages for the processor's chat template.

    :param task: Task description
    :param img: PIL image for the current screen observation
    :param history: List of prior action dicts (serialized into a brief JSON)
    :param mode: Navigation mode ('web'|'phone')
    :returns: Messages list compatible with ``apply_chat_template``
    """
    sys_prompt = _NAV_SYSTEM.format(APP=mode, ACTION_SPACE=_ACTION_SPACE_DESC[mode])
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": sys_prompt},
        {"type": "text", "text": f"Task: {task}"},
    ]
    if history:
        content.append({"type": "text", "text": f"Action history: {json.dumps(history, ensure_ascii=False)}"})
    content.append({"type": "image", "image": img})
    return [{"role": "user", "content": content}]


# --- Dataset & Collate ----------------------------------------------------------


class ActionIterableDataset:
    """Iterable dataset that expands MDS episodes into per-action examples.

    This wraps a ``streaming.StreamingDataset`` and yields tuples
    ``(prompt_text, target_text, image, mode, task, history_list)``.
    """

    def __init__(
        self,
        ds,
        *,
        history_steps: int = 0,
        max_samples_per_epoch: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.ds = ds
        self.history_steps = max(0, int(history_steps))
        self.max_samples = max(0, int(max_samples_per_epoch))
        self.log = logger or logging.getLogger("train")

    def __iter__(self) -> Iterator[Tuple[str, str, Any, str, str, List[Dict[str, Any]]]]:
        import numpy as np  # noqa: F401  # used implicitly by Streaming when decoding

        produced = 0
        for sample in self.ds:
            try:
                payload = sample.get("payload")
                task = sample.get("task", "")
                mode = "web"  # episodes are web by default; adjust if your capture encodes mode
                if not isinstance(payload, (bytes, bytearray)):
                    continue
                with _load_episode_zip(payload) as zf:
                    # frames index
                    frame_index: List[Dict[str, Any]] = []
                    for ln in _read_lines(zf, "frames.ndjson"):
                        with contextlib.suppress(Exception):
                            frame_index.append(json.loads(ln))
                    # actions
                    actions: List[Dict[str, Any]] = []
                    for ln in _read_lines(zf, "actions.ndjson"):
                        with contextlib.suppress(Exception):
                            actions.append(json.loads(ln))
                    if not actions:
                        continue

                    prev_actions: List[Dict[str, Any]] = []
                    for a in actions:
                        ts = float(a.get("ts", 0.0))
                        act_obj = a.get("action") or {}
                        raw = a.get("raw")
                        # target text prefers raw (already JSON string) else canonical JSON
                        target_text = raw if isinstance(raw, str) else json.dumps(act_obj, ensure_ascii=False)

                        i = _nearest_frame_index(frame_index, ts)
                        if i is None:
                            continue
                        frame_name = f"frames/{i:06d}.jpg"
                        try:
                            img = _open_image_from_zip(zf, frame_name)
                        except Exception:
                            continue

                        # use last history_steps actions for context
                        history = prev_actions[-self.history_steps :] if self.history_steps else []
                        messages = _build_messages(task, img, history, mode=mode)
                        # We serialize prompt with processor later; here keep messages
                        yield (json.dumps(messages, ensure_ascii=False), target_text, img, mode, task, history)
                        produced += 1
                        prev_actions.append(act_obj)

                        if self.max_samples and produced >= self.max_samples:
                            return
            except Exception as e:  # keep streaming robust to bad samples
                self.log.debug("Skipping episode due to error: %s", e)


def make_collate(processor):
    """Create a collate function bound to an AutoProcessor instance.

    The batch items are tuples returned by ``ActionIterableDataset``.
    """

    tok = processor.tokenizer

    def collate(batch: Sequence[Tuple[str, str, Any, str, str, List[Dict[str, Any]]]]):
        # Deserialize messages JSON; build chat template text strings
        msgs_list: List[List[Dict[str, Any]]] = []
        targets: List[str] = []
        images: List[Any] = []
        prompt_texts: List[str] = []

        for (msgs_json, tgt, img, _mode, _task, _hist) in batch:
            msgs: List[Dict[str, Any]] = json.loads(msgs_json)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(text)
            msgs_list.append(msgs)
            targets.append(tgt)
            images.append(img)

        full_texts = [p + t for p, t in zip(prompt_texts, targets)]

        # Tokenize both prompt-only and full for label masking
        prompt_ids = tok(prompt_texts, padding=False, add_special_tokens=False, return_tensors=None)
        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")

        # Build labels tensor: -100 for prompt tokens, full ids elsewhere
        import torch  # local import

        labels = inputs["input_ids"].clone()
        labels[:] = -100
        for i, ids in enumerate(prompt_ids["input_ids"]):
            plen = len(ids)
            labels[i, :plen] = -100
            labels[i, plen : inputs["input_ids"].shape[1]] = inputs["input_ids"][i, plen : inputs["input_ids"].shape[1]]

        inputs["labels"] = labels
        return inputs

    return collate


# --- Training loop --------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed common RNGs for basic reproducibility."""
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


async def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for finetuning on captured episodes.

    Parse configuration, construct data streams, build model/processor, and
    run a minimal AMP training loop with gradient accumulation.
    """
    import argparse

    p = argparse.ArgumentParser(description="Finetune ShowUI/Qwen2VL on captured episodes (MDS)")
    p.add_argument("--local", dest="local", type=str, help="Local MDS dir", default=None)
    p.add_argument("--remote", dest="remote", type=str, help="Remote MDS URI", default=None)
    p.add_argument("--cache", dest="cache", type=str, help="Local cache dir", default=None)
    p.add_argument("--output", dest="output", type=str, help="Output directory", default=None)
    p.add_argument("--epochs", dest="epochs", type=int, default=None)
    p.add_argument("--batch", dest="batch", type=int, default=None)
    p.add_argument("--accum", dest="accum", type=int, default=None)
    p.add_argument("--lr", dest="lr", type=float, default=None)
    p.add_argument("--wd", dest="wd", type=float, default=None)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    p.add_argument("--max-samples-per-epoch", dest="max_samples", type=int, default=None)
    p.add_argument("--history", dest="history", type=int, default=None)
    p.add_argument("--loglevel", dest="loglevel", type=str, default=None)
    p.add_argument("--log-format", dest="log_format", type=str, default=None, choices=["text", "json"])
    p.add_argument("--repo-id", dest="repo_id", type=str, default=None)
    p.add_argument("--size-short", dest="size_short", type=int, default=None)
    p.add_argument("--size-long", dest="size_long", type=int, default=None)
    args = p.parse_args(argv)

    cfg = TrainConfig.from_env()
    # CLI overrides
    for k in ("local", "remote", "cache", "output", "epochs", "batch", "accum", "lr", "wd", "max_steps", "max_samples", "history", "loglevel", "log_format", "repo_id", "size_short", "size_long"):
        v = getattr(args, k, None)
        if v is not None:
            if k == "batch":
                cfg.batch_size = int(v)
            elif k == "accum":
                cfg.grad_accum = int(v)
            elif k == "wd":
                cfg.weight_decay = float(v)
            elif k == "max_samples":
                cfg.max_samples_per_epoch = int(v)
            elif k == "history":
                cfg.history_steps = int(v)
            elif k == "repo_id":
                cfg.repo_id = str(v)
            elif k == "size_short":
                cfg.size_shortest_edge = int(v)
            elif k == "size_long":
                cfg.size_longest_edge = int(v)
            else:
                setattr(cfg, k if k not in ("local", "remote", "cache", "output") else k, v)

    log = setup_logging(cfg.loglevel, cfg.log_format)
    errs = cfg.validate()
    if errs:
        for e in errs:
            log.error("Config error: %s", e)
        sys.exit(2)

    os.makedirs(cfg.output, exist_ok=True)
    set_seed(cfg.seed)

    # Lazy heavy imports
    from streaming import StreamingDataset
    import torch
    from torch.utils.data import DataLoader
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    # Build dataset stream
    ds = StreamingDataset(
        remote=cfg.remote or None,
        local=cfg.local,
        cache_dir=cfg.cache,
        shuffle=True,
        shuffle_seed=cfg.seed,
    )

    # Processor/model
    proc = AutoProcessor.from_pretrained(
        cfg.repo_id,
        size={"shortest_edge": cfg.size_shortest_edge, "longest_edge": cfg.size_longest_edge},
        trust_remote_code=True,
    )

    device = "cpu"
    dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda"
        # Prefer bf16 if supported
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = "mps"
        dtype = torch.bfloat16 if torch.backends.mps.is_built() else torch.float32

    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.repo_id, **model_kwargs)
    model.train()
    model.to(device)

    # Data pipeline
    it = ActionIterableDataset(
        ds,
        history_steps=cfg.history_steps,
        max_samples_per_epoch=cfg.max_samples_per_epoch,
        logger=log,
    )
    collate = make_collate(proc)
    loader = DataLoader(it, batch_size=cfg.batch_size, collate_fn=collate, num_workers=0)

    # Optimizer & AMP
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and dtype == torch.float16))

    total_steps_cap = cfg.max_steps if cfg.max_steps > 0 else math.inf
    step = 0
    global_start = time.time()

    def save_ckpt(tag: str) -> None:
        out = os.path.join(cfg.output, tag)
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)
        proc.save_pretrained(out)
        log.info("Saved checkpoint to %s", out)

    # Graceful stop
    stop_flag = {"stop": False}

    def _sigint(_signum, _frame):
        log.info("Received SIGINT - finishing current step and saving.")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint)

    for epoch in range(cfg.epochs):
        if stop_flag["stop"]:
            break
        log.info("Epoch %d/%d", epoch + 1, cfg.epochs)
        optim.zero_grad(set_to_none=True)

        for batch in loader:
            if stop_flag["stop"]:
                break
            if step >= total_steps_cap:
                break

            # Move to device
            for k in list(batch.keys()):
                if hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda" and dtype != torch.float32)):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)

            if step % 10 == 0:
                log.info("step=%d loss=%.4f", step, float(loss.detach().item() * cfg.grad_accum))

            step += 1

        # end epoch: save
        save_ckpt(f"epoch-{epoch+1}")

    save_ckpt("final")
    log.info("Training complete in %.1fs", time.time() - global_start)


def cli() -> None:
    """Synchronous entrypoint wrapper for console scripts."""
    import asyncio

    asyncio.run(main())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
