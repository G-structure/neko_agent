"""Remote vision agent using OpenRouter API with tool calling support."""

import asyncio
import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import VisionAgent
from .parsing import parse_tool_call, parse_text_response

if TYPE_CHECKING:
    from ..agent_refactored import Settings


# Computer Use tool definition (Qwen-compatible format)
COMPUTER_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": (
            "Perform a computer action on the GUI. Use this to interact with "
            "screen elements by clicking, typing, or moving the mouse."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "type",
                        "key",
                    ],
                    "description": "The type of action to perform",
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Normalized [x, y] coordinate in 0-1000 range",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type (for 'type' action)",
                },
            },
            "required": ["action"],
        },
    },
}


class OpenRouterAgent(VisionAgent):
    """Remote vision agent using OpenRouter API.

    Supports:
    - Vision-language models via base64 image encoding
    - Tool calling for Qwen computer use
    - Structured output with JSON schema
    - Async HTTP with retry logic
    """

    def __init__(self, settings: 'Settings', logger: logging.Logger):
        """Initialize OpenRouter agent.

        :param settings: Configuration settings
        :param logger: Logger instance
        :raises ValueError: If API key is missing
        """
        self.settings = settings
        self.logger = logger

        # Validate API key
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouter agent. "
                "Get your key at https://openrouter.ai/keys"
            )

        # HTTP client configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.openrouter_timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        # Request headers
        self.headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if settings.openrouter_site_url:
            self.headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            self.headers["X-Title"] = settings.openrouter_app_name

        # Tool calling enabled by default
        self.use_tools = True

        self.logger.info(
            "OpenRouter agent initialized with model: %s",
            settings.openrouter_model
        )

    async def generate_action(
        self,
        image: Image.Image,
        task: str,
        system_prompt: str,
        action_history: List[Dict[str, Any]],
        crop_box: Tuple[int, int, int, int],
        iteration: int,
        full_size: Tuple[int, int],
    ) -> Optional[str]:
        """Generate action using OpenRouter API.

        :param image: Current screen image
        :param task: Task description
        :param system_prompt: System prompt defining action space
        :param action_history: Previous actions for context
        :param crop_box: Crop coordinates for refinement
        :param iteration: Refinement iteration number
        :param full_size: Original frame size
        :return: Raw action string or None on failure
        """
        try:
            # Encode image to base64
            base64_image = self._encode_image(image)

            # Build messages
            messages = self._build_messages(
                task, system_prompt, base64_image, action_history,
                crop_box, iteration, full_size
            )

            # Call API with retry logic
            response = await self._call_api_with_retry(messages)

            # Parse response
            return self._parse_response(response, crop_box, full_size)

        except Exception as e:
            self.logger.error("OpenRouter API error: %s", e, exc_info=True)
            return None

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.

        :param image: PIL Image to encode
        :return: Base64 encoded image string
        """
        buffer = io.BytesIO()
        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_messages(
        self,
        task: str,
        system_prompt: str,
        base64_image: str,
        action_history: List[Dict[str, Any]],
        crop_box: Tuple[int, int, int, int],
        iteration: int,
        full_size: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Build OpenRouter API messages.

        :param task: Task description
        :param system_prompt: System prompt
        :param base64_image: Base64 encoded image
        :param action_history: Previous actions
        :param crop_box: Current crop box
        :param iteration: Refinement iteration
        :param full_size: Full frame size
        :return: List of message dictionaries
        """
        # Build user prompt
        if iteration == 0:
            user_text = f"Task: {task}\n\nCurrent observation:"
        else:
            full_w, full_h = full_size
            crop_w = max(crop_box[2] - crop_box[0], 1)
            crop_h = max(crop_box[3] - crop_box[1], 1)
            cx = ((crop_box[0] + crop_box[2]) / 2) / full_w if full_w else 0.5
            cy = ((crop_box[1] + crop_box[3]) / 2) / full_h if full_h else 0.5
            span_x = crop_w / full_w if full_w else 1.0
            span_y = crop_h / full_h if full_h else 1.0
            user_text = (
                f"Task: {task}\n\n"
                f"Refinement pass {iteration + 1} "
                f"zoomed near normalized coords ({cx:.2f}, {cy:.2f}) "
                f"with approx span ({span_x:.2f}, {span_y:.2f}).\n\n"
                f"Current observation:"
            )

        # Add action history
        if action_history:
            history_lines = [
                f"{idx}. {json.dumps(act, ensure_ascii=False)}"
                for idx, act in enumerate(action_history[-5:], 1)
            ]
            user_text += "\n\nPrevious actions:\n" + "\n".join(history_lines)

        # Build messages
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]

        return messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _call_api_with_retry(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call OpenRouter API with retry logic.

        :param messages: Messages to send
        :return: API response dictionary
        :raises httpx.HTTPStatusError: On non-retryable HTTP errors
        :raises httpx.TimeoutException: On timeout
        """
        # Build request payload
        payload = {
            "model": self.settings.openrouter_model,
            "messages": messages,
            "max_tokens": self.settings.openrouter_max_tokens,
            "temperature": self.settings.openrouter_temperature,
            "top_p": self.settings.openrouter_top_p,
        }

        # Add tools if enabled
        if self.use_tools:
            payload["tools"] = [COMPUTER_USE_TOOL]
            payload["tool_choice"] = "auto"

        self.logger.debug(
            "Calling OpenRouter API with model: %s",
            self.settings.openrouter_model
        )

        response = await self.client.post(
            self.settings.openrouter_base_url,
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    def _parse_response(
        self,
        response: Dict[str, Any],
        crop_box: Tuple[int, int, int, int],
        full_size: Tuple[int, int],
    ) -> Optional[str]:
        """Parse OpenRouter API response.

        Handles both tool calling and text-based responses.

        :param response: API response dictionary
        :param crop_box: Current crop box for coordinate conversion
        :param full_size: Full frame size for coordinate conversion
        :return: Formatted action string or None
        """
        try:
            choice = response["choices"][0]
            message = choice["message"]

            # Check for tool calls (Qwen computer use)
            if "tool_calls" in message and message["tool_calls"]:
                return parse_tool_call(message["tool_calls"][0], self.logger)

            # Fallback: parse text content
            if "content" in message and message["content"]:
                return parse_text_response(message["content"], self.logger)

            self.logger.warning("No action found in API response")
            return None

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            self.logger.error("Failed to parse API response: %s", e)
            return None

    def get_device_info(self) -> Dict[str, Any]:
        """Return device and model information.

        :return: Dictionary with device info
        """
        return {
            "device": "remote",
            "api": "openrouter",
            "model": self.settings.openrouter_model,
            "features": ["vision", "tool_calling"],
        }

    async def cleanup(self) -> None:
        """Clean up HTTP client resources."""
        self.logger.info("Cleaning up OpenRouter agent resources")
        await self.client.aclose()