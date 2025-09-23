"""Common utility functions for Neko Agent.

This module provides shared utilities used across multiple components
including logging setup, image processing, file operations, and visualization.
"""

import json
import logging
import os
import sys
import tempfile
import time
from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont

from metrics import resize_duration


def setup_logging(log_level: str = "INFO", log_format: str = "text", log_file: str = None) -> logging.Logger:
    """Configure logging with flexible settings.

    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param log_format: Format style ('text' or 'json')
    :param log_file: Optional file path for log output
    :return: Configured logger instance
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure format based on settings
    if log_format == "json":
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_entry = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                }
                if record.exc_info:
                    log_entry['exception'] = self.formatException(record.exc_info)
                return json.dumps(log_entry)
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)-12s %(levelname)-7s - %(message)s',
            datefmt='%H:%M:%S'
        )

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level.upper())

    # Setup file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to set up file logging to {log_file}: {e}", file=sys.stderr)

    # Configure third-party logger levels
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.ERROR)

    return root_logger


def resize_and_validate_image(image: Image.Image, max_edge: int, logger: logging.Logger) -> Image.Image:
    """Resize image if it exceeds maximum dimensions while preserving aspect ratio.

    :param image: The PIL Image to resize and validate
    :param max_edge: Maximum allowed edge length in pixels
    :param logger: Logger instance for recording resize operations
    :return: The original image if within size limits, or a resized copy
    """
    ow, oh = image.size
    me = max(ow, oh)
    if me > max_edge:
        scale = max_edge / me
        nw, nh = int(ow * scale), int(oh * scale)
        t0 = time.monotonic()
        image = image.resize((nw, nh), Image.LANCZOS)
        resize_duration.observe(time.monotonic() - t0)
        logger.info("Resized %dx%d -> %dx%d", ow, oh, nw, nh)
    return image


def save_atomic(img: Image.Image, path: str, logger: logging.Logger) -> None:
    """Atomically save an image to disk using a temporary file and rename.

    :param img: The PIL Image to save
    :param path: The destination file path where the image should be saved
    :param logger: Logger instance for recording save operations
    """
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
        logger.debug("Saved image atomically to %s", path)
    except Exception as e:
        logger.error("Failed to save image to %s: %s", path, e)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def draw_action_markers(img: Image.Image, action: Dict[str, Any], step: int) -> Image.Image:
    """Draw visual markers on an image to indicate performed actions.

    :param img: The source image to annotate with action markers
    :param action: Dictionary containing action details
    :param step: The step number for labeling the action
    :return: A copy of the input image with action markers and labels drawn
    """
    out = img.copy()
    d = ImageDraw.Draw(out)
    action_type = action.get("action", "UNKNOWN")
    pos = action.get("position")
    value = action.get("value", "")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    if pos and isinstance(pos, list) and len(pos) >= 2:
        x = int(pos[0] * img.width)
        y = int(pos[1] * img.height)

        # Draw crosshair
        d.line([(x-10, y), (x+10, y)], fill="red", width=2)
        d.line([(x, y-10), (x, y+10)], fill="red", width=2)

        # Draw circle
        d.ellipse([x-5, y-5, x+5, y+5], outline="red", width=2)

        # Draw label
        label = f"{step}. {action_type}"
        if value:
            label += f": {str(value)[:20]}"

        if font:
            # Get text size
            bbox = d.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = len(label) * 6, 11

        # Position label to avoid going off screen
        label_x = max(0, min(x - text_width // 2, img.width - text_width))
        label_y = max(0, y - 25)

        # Draw background rectangle
        d.rectangle([label_x-2, label_y-2, label_x+text_width+2, label_y+text_height+2],
                   fill="white", outline="red")

        # Draw text
        d.text((label_x, label_y), label, fill="red", font=font)

    return out