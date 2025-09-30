"""Remote API vision agent implementations (Claude Computer Use, Qwen3VL)."""

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from PIL import Image

from .base import VisionAgent

if TYPE_CHECKING:
    from ..agent_refactored import Settings


class ClaudeComputerUseAgent(VisionAgent):
    """Claude Computer Use API vision agent (placeholder for future implementation).

    This agent will use Anthropic's Claude Computer Use API for remote inference.
    Configuration via environment variables:
    - CLAUDE_API_KEY: Anthropic API key
    - CLAUDE_MODEL: Model identifier (e.g., "claude-3-5-sonnet-20241022")
    """

    def __init__(self, settings: 'Settings', logger: logging.Logger):
        """Initialize Claude Computer Use agent.

        :param settings: Configuration settings
        :param logger: Logger instance
        :raises NotImplementedError: This agent is not yet implemented
        """
        self.settings = settings
        self.logger = logger
        raise NotImplementedError(
            "Claude Computer Use agent is not yet implemented. "
            "Set NEKO_AGENT_TYPE=showui to use the local ShowUI model."
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
        """Generate action using Claude Computer Use API.

        :param image: Current screen image
        :param task: Task description
        :param system_prompt: System prompt
        :param action_history: Previous actions
        :param crop_box: Crop coordinates
        :param iteration: Refinement iteration
        :param full_size: Full frame size
        :return: Action string or None
        :raises NotImplementedError: Not yet implemented
        """
        raise NotImplementedError("Claude Computer Use API integration pending")

    def get_device_info(self) -> Dict[str, Any]:
        """Return device information.

        :return: Device info dictionary
        """
        return {
            "device": "remote",
            "api": "claude-computer-use",
            "implementation": "pending"
        }

    async def cleanup(self) -> None:
        """Clean up resources.

        No cleanup needed for API-based agent.
        """
        pass


class Qwen3VLAgent(VisionAgent):
    """Qwen3VL API vision agent (placeholder for future implementation).

    This agent will use Alibaba's Qwen3VL API for remote inference.
    Configuration via environment variables:
    - QWEN_API_KEY: Qwen API key
    - QWEN_API_ENDPOINT: API endpoint URL
    - QWEN_MODEL: Model identifier
    """

    def __init__(self, settings: 'Settings', logger: logging.Logger):
        """Initialize Qwen3VL agent.

        :param settings: Configuration settings
        :param logger: Logger instance
        :raises NotImplementedError: This agent is not yet implemented
        """
        self.settings = settings
        self.logger = logger
        raise NotImplementedError(
            "Qwen3VL agent is not yet implemented. "
            "Set NEKO_AGENT_TYPE=showui to use the local ShowUI model."
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
        """Generate action using Qwen3VL API.

        :param image: Current screen image
        :param task: Task description
        :param system_prompt: System prompt
        :param action_history: Previous actions
        :param crop_box: Crop coordinates
        :param iteration: Refinement iteration
        :param full_size: Full frame size
        :return: Action string or None
        :raises NotImplementedError: Not yet implemented
        """
        raise NotImplementedError("Qwen3VL API integration pending")

    def get_device_info(self) -> Dict[str, Any]:
        """Return device information.

        :return: Device info dictionary
        """
        return {
            "device": "remote",
            "api": "qwen3vl",
            "implementation": "pending"
        }

    async def cleanup(self) -> None:
        """Clean up resources.

        No cleanup needed for API-based agent.
        """
        pass