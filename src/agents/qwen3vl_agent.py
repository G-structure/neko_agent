"""Qwen2.5-VL vision agent wrapper for OpenRouter."""

import logging
import os
from typing import TYPE_CHECKING

from .remote_agent import OpenRouterAgent

if TYPE_CHECKING:
    from ..agent import Settings


class Qwen3VLAgent(OpenRouterAgent):
    """Qwen2.5-VL vision model via OpenRouter with computer use tools.

    This is a thin wrapper around OpenRouterAgent that provides
    Qwen-specific defaults and configuration.

    Qwen2.5-VL models available on OpenRouter:
    - qwen/qwen2.5-vl-3b-instruct:free (Free tier, good for testing)
    - qwen/qwen2.5-vl-32b-instruct:free (Free tier, larger model)
    - qwen/qwen2.5-vl-72b-instruct (Paid, best performance)
    """

    def __init__(self, settings: "Settings", logger: logging.Logger):
        """Initialize Qwen3VL agent with Qwen-specific defaults.

        :param settings: Configuration settings
        :param logger: Logger instance
        """
        # Override model with Qwen-specific default if not set
        if not settings.openrouter_model or settings.openrouter_model == "default":
            settings.openrouter_model = os.environ.get(
                "QWEN_VL_MODEL", "qwen/qwen2.5-vl-72b-instruct"
            )

        if not getattr(settings, "prompt_strategy", None):
            settings.prompt_strategy = "conversational_chain"

        # Initialize parent OpenRouter agent
        super().__init__(settings, logger)

        # Enable tool calling by default for Qwen (always use computer use tools)
        self.use_tools = True

        self.logger.info(
            "Qwen3VL agent initialized with model: %s", settings.openrouter_model
        )
