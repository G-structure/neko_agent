"""Remote vision agent using OpenRouter API with tool calling support."""

import asyncio
import base64
import io
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import ModelResponse, VisionAgent
from .parsing import parse_tool_call, parse_text_response
from .reasoning import extract_think_segments

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

    def __init__(self, settings: 'Settings', logger: logging.Logger, chat_callback: Optional[Callable[[str], Any]] = None):
        """Initialize OpenRouter agent.

        :param settings: Configuration settings
        :param logger: Logger instance
        :param chat_callback: Optional callback for sending chat messages (e.g., thinking tokens)
        :raises ValueError: If API key is missing
        """
        super().__init__(settings, logger, default_prompt_strategy="conversational_chain")
        self.chat_callback = chat_callback

        # Validate API key
        if not self.settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouter agent. "
                "Get your key at https://openrouter.ai/keys"
            )

        # HTTP client configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.openrouter_timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        # Request headers
        self.headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_site_url:
            self.headers["HTTP-Referer"] = self.settings.openrouter_site_url
        if self.settings.openrouter_app_name:
            self.headers["X-Title"] = self.settings.openrouter_app_name

        # Tool calling enabled by default
        self.use_tools = True

        self.logger.info(
            "OpenRouter agent initialized with model: %s",
            self.settings.openrouter_model
        )

    async def _invoke_model(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        image: Image.Image,
        task: str,
        nav_mode: str,
        crop_box: Tuple[int, int, int, int],
        iteration: int,
        full_size: Tuple[int, int],
        is_refinement: bool,
    ) -> ModelResponse:
        try:
            formatted_messages = self._format_messages_for_api(messages)
            response = await self._call_api_with_retry(formatted_messages)
            reasoning_emitted = await self._extract_and_send_thinking(response)
            action_text = self._parse_response(response, crop_box, full_size)
            reasoning = self._extract_reasoning_from_message(response)
            return ModelResponse(text=action_text, reasoning=reasoning, raw=response, reasoning_emitted=reasoning_emitted)
        except Exception as exc:
            self.logger.error("OpenRouter API error: %s", exc, exc_info=True)
            return ModelResponse(text=None)

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

    def _format_messages_for_api(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")
            if isinstance(content, list):
                formatted_content: List[Dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        formatted_content.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image" and item.get("image") is not None:
                        base64_image = self._encode_image(item["image"])
                        formatted_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            }
                        )
                formatted.append({"role": role, "content": formatted_content})
            else:
                formatted.append({"role": role, "content": content or ""})
        return formatted

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

        # Add reasoning configuration if enabled
        if self.settings.openrouter_reasoning_enabled:
            reasoning_config = {"enabled": True}
            if self.settings.openrouter_reasoning_effort:
                reasoning_config["effort"] = self.settings.openrouter_reasoning_effort
            payload["reasoning"] = reasoning_config
            self.logger.info("OpenRouter reasoning enabled with config: %s", reasoning_config)

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

    async def _extract_and_send_thinking(self, response: Dict[str, Any]) -> bool:
        """Extract thinking/reasoning tokens from response and forward via callback."""
        reasoning_segments: List[str] = []
        have_details = False
        emitted = False

        try:
            choice = response.get("choices", [{}])[0]
            reasoning_details = choice.get("reasoning_details")

            if not reasoning_details:
                if self.settings.openrouter_reasoning_enabled:
                    self.logger.info("OpenRouter reasoning enabled but no reasoning_details returned.")
                return False

            have_details = True
            callback = self.chat_callback

            for detail in reasoning_details:
                if not isinstance(detail, dict):
                    continue
                reasoning_text = detail.get("text") or detail.get("content")
                if not reasoning_text:
                    continue
                reasoning_segments.append(reasoning_text)
                if callback:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(reasoning_text)
                        else:
                            callback(reasoning_text)
                        emitted = True
                    except Exception as exc:
                        self.logger.debug("Reasoning callback failed: %s", exc, exc_info=True)
                self.logger.debug("Received reasoning token: %s chars", len(reasoning_text))

        except Exception as e:
            self.logger.debug("Failed to extract thinking tokens: %s", e, exc_info=True)
        finally:
            self._log_reasoning_details(response, reasoning_segments, have_details)

        return emitted


    def _extract_reasoning_from_message(self, response: Dict[str, Any]) -> Optional[str]:
        """Best-effort extraction of assistant reasoning text for history tracking."""
        try:
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                collected = [
                    segment.get("text", "")
                    for segment in content
                    if isinstance(segment, dict) and segment.get("type") == "text"
                ]
                merged = chr(10).join(t for t in collected if t)
                think = extract_think_segments(merged)
                if think:
                    return think
                return merged.strip() or None
            if isinstance(content, str):
                think = extract_think_segments(content)
                if think:
                    return think
                stripped = content.strip()
                return stripped or None
        except Exception:
            return None
        return None



    def _log_reasoning_details(
        self,
        response: Dict[str, Any],
        segments: List[str],
        have_details: bool,
    ) -> None:
        """Log reasoning metadata returned by OpenRouter."""
        if not self.settings.openrouter_reasoning_enabled:
            return

        try:
            if segments:
                preview_parts: List[str] = []
                for text in segments[:3]:
                    cleaned = " ".join(text.split())
                    if len(cleaned) > 200:
                        cleaned = f"{cleaned[:197]}..."
                    preview_parts.append(cleaned)
                preview = " | ".join(preview_parts)
                total_chars = sum(len(s) for s in segments)
                self.logger.info(
                    "OpenRouter reasoning returned %d segment(s) (%d chars total). Preview: %s",
                    len(segments),
                    total_chars,
                    preview,
                )
            elif have_details:
                self.logger.info(
                    "OpenRouter reasoning_details provided but no text content was found."
                )
            else:
                self.logger.info(
                    "OpenRouter reasoning enabled but response omitted reasoning_details."
                )

            if isinstance(response, dict):
                usage = response.get("usage")
                if isinstance(usage, dict):
                    if "reasoning_tokens" in usage:
                        self.logger.info(
                            "OpenRouter reasoning tokens reported: %s",
                            usage.get("reasoning_tokens"),
                        )
                    elif "reasoning" in usage:
                        self.logger.info(
                            "OpenRouter reasoning usage payload: %s",
                            usage.get("reasoning"),
                        )
                    else:
                        self.logger.debug("OpenRouter usage payload: %s", usage)

                choice = response.get("choices", [{}])[0]
                if isinstance(choice, dict) and choice.get("reasoning"):
                    self.logger.debug(
                        "OpenRouter choice reasoning metadata: %s",
                        choice.get("reasoning"),
                    )
        except Exception as exc:
            self.logger.debug("Failed to log reasoning metadata: %s", exc, exc_info=True)

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
